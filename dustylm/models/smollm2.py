"""SmolLM2-compatible transformer with Rotary Positional Embeddings (RoPE),
Grouped-Query Attention (GQA), SwiGLU feed-forward, and KV-caching.

This file is intentionally self-contained so it can be read top-to-bottom
as a complete, working transformer implementation.  Key differences from
the scratch variant (scratch.py):
  - Separate Q/K/V projections (to match HuggingFace checkpoint key names).
  - SwiGLU FFN instead of GELU.
  - Weight tying between the input embedding and output projection.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Type alias for the KV cache: one (key, value) tensor pair per layer,
# or None for layers that have not yet been populated.
KvCache = list[tuple[torch.Tensor, torch.Tensor] | None]


@dataclass(frozen=True)
class ForwardContext:
    """Immutable bag of precomputed values shared across all layers in one
    forward pass.  Computing RoPE sin/cos once here avoids redundant work
    in every TransformerBlock."""

    position_ids: torch.Tensor
    rope_sin: torch.Tensor  # [1, 1, T, head_dim] — broadcast over [B, n_heads, T, head_dim]
    rope_cos: torch.Tensor  # [1, 1, T, head_dim]


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings shared across transformer layers."""

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

        self.base = base
        self.head_dim = head_dim

        # --- Compute inverse frequency vector (the "theta" values) ---
        # For a head_dim of d, RoPE defines d/2 frequency bands:
        #   θ_i = base^(-2i/d)  for i in [0, 1, ..., d/2 - 1]
        # This gives low-frequency (slowly rotating) dimensions for small i
        # and high-frequency (fast rotating) dimensions for large i.
        inv_freq = base ** (-torch.arange(0, head_dim, 2) / head_dim)  # [head_dim/2]

        # Duplicate to full head_dim so we can element-wise multiply with x
        # during apply().  Each pair (x_2i, x_{2i+1}) shares the same θ_i,
        # mirroring the "rotate half" trick below.
        inv_freq = torch.cat([inv_freq, inv_freq], dim=-1)  # [head_dim]
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.resize_cache(max_seq_len)

    def resize_cache(self, max_seq_len: int):
        """Precompute sin/cos tables up to ``max_seq_len`` positions."""
        position_ids = torch.arange(
            0,
            max_seq_len,
            dtype=self.inv_freq.dtype,
            device=self.inv_freq.device,
        )  # [max_seq_len]

        # Outer product: angle[pos, i] = pos * θ_i
        # Shape: [max_seq_len, head_dim]
        emb = torch.outer(position_ids, self.inv_freq)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)

    def forward(self, position_ids):
        """Look up precomputed sin/cos values and reshape for broadcasting.

        Args:
            position_ids: [T] absolute positions of each token.

        Returns:
            (sin, cos): each [1, 1, T, head_dim] — ready to broadcast
            over query/key tensors of shape [B, n_heads, T, head_dim].
        """
        sin = self.get_buffer("sin_cache")[position_ids, :]  # [T, head_dim]
        cos = self.get_buffer("cos_cache")[position_ids, :]  # [T, head_dim]
        # unsqueeze(0) → [1, T, head_dim]  (batch dim)
        # unsqueeze(1) → [1, 1, T, head_dim]  (head dim)
        return sin.unsqueeze(0).unsqueeze(1), cos.unsqueeze(0).unsqueeze(1)

    @classmethod
    def rotate_half(cls, x):
        """Swap and negate the two halves of the last dimension.

        Given x = [x1, x2], return [-x2, x1].  This corresponds to the
        imaginary-number rotation: if we treat each (x1_i, x2_i) pair as
        a complex number x1 + ix2, then rotate_half produces the imaginary
        part needed for the full rotation formula.
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    @classmethod
    def apply(cls, x, sin, cos):
        """Apply rotary embedding to tensor ``x``.

        This is the real-valued form of complex multiplication by e^{iθ}:
            x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)
        """
        return x * cos + cls.rotate_half(x) * sin


class MultiHeadAttention(nn.Module):
    """Multi-head attention with Grouped-Query Attention (GQA).

    GQA uses fewer key/value heads than query heads to reduce memory during
    inference.  Each group of ``num_heads // num_kv_heads`` query heads shares
    one key/value head.  This module uses separate Q, K, V projections
    (matching HuggingFace SmolLM2 checkpoint key names).
    """

    def __init__(
        self, embed_dim, num_heads, num_kv_heads, max_seq_len, rope_base=10000
    ):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_repeat_factor = num_heads // num_kv_heads  # queries per KV group
        self.q_dim = embed_dim                             # = num_heads * head_dim
        self.kv_dim = num_kv_heads * self.head_dim
        self.max_seq_len = max_seq_len

        # Separate projections (matching HuggingFace checkpoint key names).
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, context: ForwardContext, past_kv=None):
        """Compute grouped-query attention with optional KV-cache.

        Args:
            x:        [B, T, embed_dim] input embeddings (after RMSNorm).
            context:  precomputed RoPE sin/cos tensors.
            past_kv:  optional (past_k, past_v) from a previous forward pass.

        Returns:
            (output, present_kv): output is [B, T, embed_dim],
            present_kv is the (key, value) pair to pass to the next step.
        """
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError("input embedding dim does not match model embedding dim")

        # ---- 1. Project input into Q, K, V (separate projections) ----
        q = self.q_proj(x)  # [B, T, embed_dim]
        k = self.k_proj(x)  # [B, T, kv_dim]
        v = self.v_proj(x)  # [B, T, kv_dim]

        # ---- 2. Reshape into multi-head layout ----
        q = q.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)   # [B, n_heads, T, head_dim]
        k = k.reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)   # [B, n_kv_heads, T, head_dim]
        v = v.reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)   # [B, n_kv_heads, T, head_dim]

        # ---- 3. Apply Rotary Positional Embeddings ----
        q = RotaryPositionalEmbedding.apply(q, context.rope_sin, context.rope_cos)
        k = RotaryPositionalEmbedding.apply(k, context.rope_sin, context.rope_cos)
        # Shapes unchanged — values are rotated by position-dependent angles.

        # ---- 4. Concatenate with cached keys/values ----
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            k = torch.cat([past_k, k], dim=2)  # [B, n_kv_heads, past_len+T, head_dim]
            v = torch.cat([past_v, v], dim=2)  # [B, n_kv_heads, past_len+T, head_dim]

        # Save the un-expanded KV pair for the cache (before GQA repeat).
        present_kv = (k, v)

        # ---- 5. Expand KV heads for Grouped-Query Attention ----
        # Repeat each KV head to match the number of query heads in its group.
        # Example: 15 query heads, 5 KV heads → each KV head is repeated 3x.
        k = k.repeat_interleave(self.kv_repeat_factor, dim=1)  # [B, n_heads, past_len+T, head_dim]
        v = v.repeat_interleave(self.kv_repeat_factor, dim=1)  # [B, n_heads, past_len+T, head_dim]

        # ---- 6. Scaled dot-product attention ----
        scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)  # [B, n_heads, T, past_len+T]

        # Causal mask: prevent attending to future tokens.
        # When seq_len == 1 (single-token decode), no mask is needed because
        # there are no future query tokens in the chunk.
        if seq_len > 1:
            total_len = past_len + seq_len
            # Example with past_len=3, seq_len=2 (total_len=5):
            #          k0 k1 k2 k3 k4
            #   q3  [  0  0  0  0  1 ]   ← q3 sees k0..k3, masked from k4
            #   q4  [  0  0  0  0  0 ]   ← q4 sees everything
            mask = torch.triu(
                torch.ones(seq_len, total_len, device=x.device, dtype=torch.bool),
                diagonal=past_len + 1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)         # [B, n_heads, T, past_len+T]
        attention_context = attention_weights @ v                  # [B, n_heads, T, head_dim]

        # ---- 7. Merge heads and project back ----
        attention_context = attention_context.transpose(1, 2).reshape(
            batch_size, seq_len, embed_dim
        )  # [B, T, embed_dim]
        return self.out_proj(attention_context), present_kv


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → Attention → Residual → RMSNorm → SwiGLU FFN → Residual.

    The SwiGLU feed-forward network replaces the simple GELU used in
    scratch.py.  SwiGLU uses a gating mechanism:
        FFN(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    where gate_proj and up_proj produce parallel hidden representations
    of size ``hidden_dim``, gated element-wise before being projected
    back to ``embed_dim`` by down_proj.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        max_seq_len,
        hidden_dim=None,
        rope_base=10000,
        rms_eps=1e-4,
    ):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim else 4 * embed_dim
        self.input_layernorm = nn.RMSNorm(embed_dim, eps=rms_eps)
        self.self_attn = MultiHeadAttention(
            embed_dim, num_heads, num_kv_heads, max_seq_len, rope_base
        )
        self.post_attention_layernorm = nn.RMSNorm(embed_dim, eps=rms_eps)

        # SwiGLU projections: embed_dim → hidden_dim → embed_dim
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x, context: ForwardContext, kv_cache=None):
        # x: [B, T, embed_dim]
        att_out, present_kv = self.self_attn(self.input_layernorm(x), context, kv_cache)
        x = x + att_out              # residual connection around attention

        # --- SwiGLU FFN ---
        x_norm = self.post_attention_layernorm(x)      # [B, T, embed_dim]
        gated = F.silu(self.gate_proj(x_norm))          # [B, T, hidden_dim]
        up = self.up_proj(x_norm)                       # [B, T, hidden_dim]
        g = gated * up                                  # element-wise gating
        out = self.down_proj(g)                         # [B, T, embed_dim]
        x = x + out                  # residual connection around FFN
        return x, present_kv


class DustyLM(nn.Module):
    """Decoder-only transformer (SmolLM2-compatible variant).

    Architecture: Embedding → [TransformerBlock × N] → RMSNorm → Linear → logits.
    Supports optional KV-caching for efficient autoregressive generation.

    Weight tying: the input embedding and output projection share the same
    weight tensor, which reduces parameter count and is standard in the
    Llama/SmolLM2 family.
    """

    def __init__(
        self,
        num_layers,
        vocab_size,
        max_seq_len,
        embed_dim,
        num_heads,
        num_kv_heads,
        hidden_dim=None,
        rope_base=10000,
        rms_eps=1e-4,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.rope = RotaryPositionalEmbedding(head_dim, max_seq_len, rope_base)
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    num_kv_heads,
                    max_seq_len,
                    hidden_dim,
                    rope_base,
                    rms_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(embed_dim, rms_eps)
        self.vocab_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: the output projection reuses the embedding matrix.
        # This means embed_tokens.weight IS vocab_proj.weight — same tensor,
        # same gradients.  This is a ~vocab_size * embed_dim parameter saving.
        self.vocab_proj.weight = self.embed_tokens.weight

    def empty_kv_cache(self) -> KvCache:
        return [None] * len(self.layers)

    def forward(self, x, position_ids=None, kv_cache: KvCache | None = None):
        """Run the full model.

        Args:
            x:            [B, T] integer token IDs.
            position_ids: [T] absolute positions (auto-computed if None).
            kv_cache:     list of per-layer (key, value) pairs, or None.

        Returns:
            Without cache: logits [B, T, vocab_size].
            With cache:    (logits, next_kv_cache).
        """
        _, seq_len = x.shape  # x: [B, T]
        use_cache = kv_cache is not None

        if use_cache and len(kv_cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} cache entries, got {len(kv_cache)}"
            )

        # Determine how many tokens are already in the cache so we can
        # compute the correct absolute positions for the new tokens.
        past_len = 0
        if use_cache and kv_cache[0] is not None:
            past_len = kv_cache[0][0].shape[2]  # key tensor dim 2 = cached seq len

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, dtype=torch.long, device=x.device
            )

        # Precompute RoPE sin/cos for all layers to share.
        rope_sin, rope_cos = self.rope(position_ids)
        context = ForwardContext(
            position_ids=position_ids,
            rope_sin=rope_sin,
            rope_cos=rope_cos,
        )

        x = self.embed_tokens(x)  # [B, T] → [B, T, embed_dim]
        input_kv_cache = kv_cache if use_cache else self.empty_kv_cache()
        next_kv_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            x, present_kv = layer(x, context, input_kv_cache[idx])
            if use_cache:
                next_kv_cache.append(present_kv)

        logits = self.vocab_proj(self.final_norm(x))  # [B, T, vocab_size]
        if use_cache:
            return logits, next_kv_cache
        return logits
