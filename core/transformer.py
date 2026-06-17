import torch
import torch.nn as nn


class RotaryPositionalEmbeding(nn.Module):
    """
    Rotary positional embedign
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        self.base = base
        self.head_dim = head_dim
        assert head_dim % 2 == 0, "Head dimention must be even to be used in Rope"
        inverse_freq = base ** (-torch.arange(0, head_dim, 2) / head_dim)
        inverse_freq = torch.cat([inverse_freq, inverse_freq], dim=-1)

        self.register_buffer("inv_freq", inverse_freq, persistent=False)

        postion_ids = torch.arange(0, max_seq_len, dtype=inverse_freq.dtype)

        emb = torch.outer(postion_ids, inverse_freq)

        self.register_buffer("sin_cache", emb.sin())  # (T,Dh)
        self.register_buffer("cos_cache", emb.cos())
        print("Rope fully initilazed")

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
        # T = x.shape[-2]  # Works with both (B,T,C) and (B,H,T,Dh)

        sin = self.get_buffer("sin_cache")[position_ids, :]
        cos = self.get_buffer("cos_cache")[position_ids, :]

        # Since the batch dimension is already there, we only need ONE unsqueeze
        # to create the Head dimension. Result: (B, 1, T, Dh)
        sin = sin.unsqueeze(0).unsqueeze(1)  # [1,1,T,Dh]
        cos = cos.unsqueeze(0).unsqueeze(1)

        return sin, cos

    @classmethod
    def rotate_half(cls, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        return torch.cat([-x2, x1], dim=-1)

    @classmethod
    def apply_rotary_embeding(cls, x, sin, cos):

        return x * cos + cls.rotate_half(x) * sin


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_kv_heads, max_seq_len, rope_base=10000
    ):
        assert embed_dim % num_heads == 0, "Invalid embeding dimention"
        assert (
            num_heads % num_kv_heads == 0
        )  # num_kv_heads ? not sure does it have to be multiplier of 2 ?
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = int(embed_dim / num_heads)
        self.kv_repeat_factor = int(num_heads / num_kv_heads)
        self.q_dim = embed_dim
        self.kv_dim = int(num_kv_heads * self.head_dim)
        self.max_seq_len = max_seq_len
        print(" shape :", embed_dim, self.kv_dim)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim + 2 * self.kv_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        print("Att bloc kfully initialzied")

    def forward(self, x, sin, cos, past_kv=None, cache_enabled=False):
        B, T, C = x.shape
        assert C == self.embed_dim, (
            "Input embeding is not compatible with model embeding"
        )
        past_len = 0

        qkv_proj = self.qkv_proj(x)  # B,T,C

        q, k, v = qkv_proj.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,T,Dh
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )  # B,H`,T,Dh
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )  # B,H`,T,Dh

        # Note apply embeding before the Kv cache. to avoid duple apply on the existing saved ones
        # positional embeding only applied to Q and K
        q_embed = RotaryPositionalEmbeding.apply_rotary_embeding(q, sin, cos)
        k_embed = RotaryPositionalEmbeding.apply_rotary_embeding(k, sin, cos)

        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            k_embed = torch.cat([past_k, k_embed], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k_embed, v) if cache_enabled else None

        k_embed = k_embed.repeat_interleave(
            self.kv_repeat_factor, dim=1
        )  # head dim  h` ->H , B,H,T,Dh
        v = v.repeat_interleave(self.kv_repeat_factor, dim=1)  # head dim

        # softmax(q@Kt/sqrt(dim))
        scores = q_embed @ k_embed.transpose(-2, -1) / (self.head_dim**0.5)  # # B,H,T,T

        # In cached prefill, keys include past tokens, so the causal mask is rectangular.
        if T > 1:
            total_len = past_len + T
            mask = torch.triu(
                torch.ones(T, total_len, device=x.device, dtype=torch.bool),
                diagonal=past_len + 1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)  # B,H,T,T
        context = attention_weights @ v  # B,H,T,Dh
        context = context.transpose(1, 2).reshape(B, T, C)  # B,T,H,Dh , H*Dh ->C
        out = self.proj(context)

        return out, present_kv


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
        self.mlp_nrom = nn.RMSNorm(embed_dim, eps=rms_eps)
        print("Transformer block initizlied")

    def forward(self, x, sin, cos, kv_cache=None, cache_enabled=False):
        att_out, present_kv = self.attention(
            self.att_norm(x), sin, cos, kv_cache, cache_enabled
        )  # att,vk_cache
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
        self.num_layers = num_layers
        head_dim = embed_dim // num_heads
        self.rope = RotaryPositionalEmbeding(head_dim, max_seq_len, rope_base)
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
        print("tinygpt initilzied")

    def forward(
        self,
        x,
        position_ids=None,
        kv_cache: list[tuple] | None = None,
        cache_enabled=False,
    ):
        B, T = x.shape

        if kv_cache is not None and not cache_enabled:
            raise ValueError("kv_cache was provided, but cache_enabled is False")
        if kv_cache is not None and len(kv_cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} cache entries, got {len(kv_cache)}"
            )

        past_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].shape[2]

        x = self.embed(x)

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + T, dtype=torch.long, device=x.device
            )

        sin, cos = self.rope(position_ids)
        layers_cache = kv_cache if kv_cache is not None else [None] * len(self.layers)
        present_cache = [] if cache_enabled else None

        for idx, layer in enumerate(self.layers):
            x, preset_kv_cache = layer(x, sin, cos, layers_cache[idx], cache_enabled)
            if cache_enabled:
                present_cache.append(preset_kv_cache)

        logits = self.vocab_proj(self.final_norm(x))
        if cache_enabled:
            return logits, present_cache
        return logits


if __name__ == "__main__":
    B = 18
    T = 2
    embed_dim = 512
    max_seq_len = 1024
    head_dim = 64
    num_heads = 8
    num_kv_heads = 2
    num_layers = 6
    # import tiktoken

    vocab_size = 1000
    x = torch.randint(0, vocab_size, size=(B, T), dtype=torch.long)
    print("Input X Shape:", x.shape)
    # x = nn.Embedding(vocab_size, embed_dim)(x)
    # print("Input X Shape after embed:", x.shape)
    # x, _ = MultiHeadAttention(embed_dim, num_heads, num_kv_heads, max_seq_len)(x)
    # print("Input X Shape after att:", x.shape)

    # x = TransformerBlock(embed_dim, num_heads, num_kv_heads, max_seq_len)(x)
    # print("Input X Shape after transformer:", x.shape)
    model = TinyGPT(
        num_layers, vocab_size, max_seq_len, embed_dim, num_heads, num_kv_heads
    )
    x = model(x)
    print("x shape:", x.shape)
