# Architecture

`tiny-gpt` is a small decoder-only transformer intended for learning and
experimentation. Scratch and SmolLM2 architectures live in `tiny_gpt.models`,
while `tiny_gpt.model` remains a compatibility import for the scratch model.

## Profile Registry

`tiny_gpt.config` is the source of truth for model variants. A `Profile`
combines:

- `ModelSpec`: family, dimensions, vocab size, RoPE settings, and tokenizer.
- `TrainingSpec`: dataset, batch size, learning rate, checkpoint output, and
  max sequence length. Dusty profiles also use optional SFT fields, an initial
  checkpoint, and step-checkpoint settings.
- `GenerationSpec`: converted checkpoint path and sampling settings.

Runtime code calls `get_profile(name)` and then uses `tiny_gpt.modeling` to
build the correct model and tokenizer. This keeps training, generation, and
checkpoint conversion from depending on overwritten module-level globals.

## Artifact Layout

Local model assets are intentionally outside git under `artifacts/`:

```text
artifacts/checkpoints/
artifacts/tokenizers/smollm2_tokenizer.json
artifacts/hf/
```

All SmolLM2 profiles use the same tokenizer JSON at
`artifacts/tokenizers/smollm2_tokenizer.json`. Each model or fine-tuned profile
points to its own converted checkpoint in `artifacts/checkpoints/`.

Dusty training writes final checkpoints such as `dusty8m.pt` and
`dusty8m_sft.pt`. When `checkpoint_every_steps` is configured, the training loop
also writes step checkpoints beside the final checkpoint, using names like
`dusty8m_step_100.pt`.

## Text Normalization

Dusty text is normalized before tokenization with:

```python
text.lower().replace(";", ".")
```

The raw dataset artifacts stay unchanged. Tokenizer training writes temporary
normalized corpora, while pretrain and SFT data prep apply the same
normalization before converting text to token IDs. SFT examples are formatted as
ChatML after normalization.

## Generation Checkpoint Selection

Generation uses `GenerationSpec.checkpoint_path` by default. Passing
`--checkpoint-step N` derives a step checkpoint path from the final checkpoint
name, for example `dusty8m.pt` becomes `dusty8m_step_N.pt`, and
`dusty8m_sft.pt` becomes `dusty8m_sft_step_N.pt`.

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
