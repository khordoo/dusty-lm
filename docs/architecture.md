# Architecture

`tiny-gpt` is a small decoder-only transformer intended for learning and
experimentation. The main implementation lives in `tiny_gpt.model`.

## Model Flow

`TinyGPT.forward()` accepts token IDs with shape `[batch, seq]`.

- Without cache, `model(tokens)` returns logits with shape
  `[batch, seq, vocab]`.
- With cache, `model(tokens, kv_cache=cache)` returns
  `(logits, next_kv_cache)`.

The cache is opt-in. Passing `kv_cache=None` keeps the normal training API
simple.

## ForwardContext

RoPE tensors are shared across all transformer layers for a single forward
pass. `TinyGPT` computes them once and stores them in:

```python
ForwardContext(
    position_ids=position_ids,
    rope_sin=rope_sin,
    rope_cos=rope_cos,
)
```

This keeps `TransformerBlock` and `MultiHeadAttention` from receiving raw
`sin` and `cos` arguments while avoiding duplicated RoPE buffers per layer.

## KV Cache

The KV cache is per-layer mutable model state, so it stays separate from
`ForwardContext`.

During generation:

1. Prefill passes the full prompt with an empty cache.
2. The model returns one `(key, value)` pair per layer.
3. Decode passes only the latest token plus the previous cache.
4. The model returns a new cache with the latest token appended.

This preserves a clear data flow:

```python
input_kv_cache -> layer -> present_kv -> next_kv_cache
```

The implementation builds a fresh `next_kv_cache` rather than mutating the
caller's cache list in place.

## Causal Masking With Cache

When cached keys exist, attention scores are rectangular:

```text
[query_len, past_len + query_len]
```

The causal mask must therefore also be rectangular. This matters for cached
chunk decoding where `query_len > 1`; one-token decoding does not need a mask
because there are no future query tokens in the chunk.
