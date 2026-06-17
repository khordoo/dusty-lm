from dataclasses import dataclass

import torch
import torch.nn as nn


KvCache = list[tuple[torch.Tensor, torch.Tensor] | None]


@dataclass(frozen=True)
class ForwardContext:
    position_ids: torch.Tensor
    rope_sin: torch.Tensor
    rope_cos: torch.Tensor


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings shared across transformer layers."""

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

        self.base = base
        self.head_dim = head_dim
        inv_freq = base ** (-torch.arange(0, head_dim, 2) / head_dim)
        inv_freq = torch.cat([inv_freq, inv_freq], dim=-1)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.resize_cache(max_seq_len)

    def resize_cache(self, max_seq_len: int):
        position_ids = torch.arange(
            0,
            max_seq_len,
            dtype=self.inv_freq.dtype,
            device=self.inv_freq.device,
        )
        emb = torch.outer(position_ids, self.inv_freq)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)

    def forward(self, position_ids):
        sin = self.get_buffer("sin_cache")[position_ids, :]
        cos = self.get_buffer("cos_cache")[position_ids, :]
        return sin.unsqueeze(0).unsqueeze(1), cos.unsqueeze(0).unsqueeze(1)

    @classmethod
    def rotate_half(cls, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    @classmethod
    def apply(cls, x, sin, cos):
        return x * cos + cls.rotate_half(x) * sin


class MultiHeadAttention(nn.Module):
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
        self.kv_repeat_factor = num_heads // num_kv_heads
        self.q_dim = embed_dim
        self.kv_dim = num_kv_heads * self.head_dim
        self.max_seq_len = max_seq_len
        self.qkv_proj = nn.Linear(embed_dim, embed_dim + 2 * self.kv_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, context: ForwardContext, past_kv=None):
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError("input embedding dim does not match model embedding dim")

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = v.reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        q = RotaryPositionalEmbedding.apply(q, context.rope_sin, context.rope_cos)
        k = RotaryPositionalEmbedding.apply(k, context.rope_sin, context.rope_cos)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)

        k = k.repeat_interleave(self.kv_repeat_factor, dim=1)
        v = v.repeat_interleave(self.kv_repeat_factor, dim=1)

        scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        if seq_len > 1:
            total_len = past_len + seq_len
            mask = torch.triu(
                torch.ones(seq_len, total_len, device=x.device, dtype=torch.bool),
                diagonal=past_len + 1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_context = attention_weights @ v
        attention_context = attention_context.transpose(1, 2).reshape(
            batch_size, seq_len, embed_dim
        )
        return self.proj(attention_context), present_kv


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        max_seq_len,
        rope_base=10000,
        rms_eps=1e-4,
    ):
        super().__init__()
        self.att_norm = nn.RMSNorm(embed_dim, eps=rms_eps)
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, num_kv_heads, max_seq_len, rope_base
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, bias=False),
        )
        # Keep this attribute name for compatibility with existing checkpoints.
        self.mlp_nrom = nn.RMSNorm(embed_dim, eps=rms_eps)

    def forward(self, x, context: ForwardContext, kv_cache=None):
        att_out, present_kv = self.attention(self.att_norm(x), context, kv_cache)
        x = x + att_out
        x = x + self.mlp(self.mlp_nrom(x))
        return x, present_kv


class TinyGPT(nn.Module):
    def __init__(
        self,
        num_layers,
        vocab_size,
        max_seq_len,
        embed_dim,
        num_heads,
        num_kv_heads,
        rope_base=10000,
        rms_eps=1e-4,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.rope = RotaryPositionalEmbedding(head_dim, max_seq_len, rope_base)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, num_kv_heads, max_seq_len, rope_base, rms_eps
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(embed_dim, rms_eps)
        self.vocab_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    def empty_kv_cache(self) -> KvCache:
        return [None] * len(self.layers)

    def forward(self, x, position_ids=None, kv_cache: KvCache | None = None):
        _, seq_len = x.shape
        use_cache = kv_cache is not None

        if use_cache and len(kv_cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} cache entries, got {len(kv_cache)}"
            )

        past_len = 0
        if use_cache and kv_cache[0] is not None:
            past_len = kv_cache[0][0].shape[2]

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, dtype=torch.long, device=x.device
            )

        rope_sin, rope_cos = self.rope(position_ids)
        context = ForwardContext(
            position_ids=position_ids,
            rope_sin=rope_sin,
            rope_cos=rope_cos,
        )
        x = self.embed(x)
        input_kv_cache = kv_cache if use_cache else self.empty_kv_cache()
        next_kv_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            x, present_kv = layer(x, context, input_kv_cache[idx])
            if use_cache:
                next_kv_cache.append(present_kv)

        logits = self.vocab_proj(self.final_norm(x))
        if use_cache:
            return logits, next_kv_cache
        return logits
