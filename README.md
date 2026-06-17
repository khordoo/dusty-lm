# tiny-gpt

An educational tiny GPT implementation in PyTorch. The code is intentionally small
enough to read end-to-end, while still including practical details such as rotary
positional embeddings, grouped-query attention style KV heads, training, and
cached autoregressive generation.

## What Is Included

- A decoder-only transformer implemented in `tiny_gpt.model`.
- RoPE computed once per forward pass and passed through a `ForwardContext`.
- KV-cache generation where the prompt is prefetched once and later decode steps
  pass only the newest token.
- Data preparation for the `nampdn-ai/tiny-codes` dataset.
- A training script and a generation script.
- Fast unit tests that do not require a checkpoint or dataset.

## Setup

```bash
uv sync
```

For development and tests:

```bash
uv sync --group dev
```

## Prepare Data

```bash
uv run python -m tiny_gpt.data_prep
```

This downloads and filters the tiny-codes dataset, applies a chat-style prompt
format, and masks prompt labels with `IGNORE_INDEX` so training loss is only
computed on the assistant response.

## Train

```bash
uv run python -m tiny_gpt.train
```

Training writes checkpoints to `checkpoints/`. Checkpoints, datasets, and
TensorBoard runs are local artifacts and are intentionally ignored by git.

## Generate

Place a compatible checkpoint at:

```text
checkpoints/tinygpt_epoch_1.pt
```

Then run:

```bash
uv run python -m tiny_gpt.generate
```

Generation uses KV caching:

1. The full prompt is passed once to build the cache.
2. Each decode step passes only the latest generated token.
3. RoPE positions continue from the cache length.

This avoids recomputing attention over the whole prompt and prevents the
query/key length mismatch that happens when the full sequence is passed again
with an existing cache.

## Tests

```bash
uv run pytest
```

The tests cover normal forward passes, cached prefill, one-token cached decode,
cached chunk decode, and data-prep masking.

## Artifact And Security Policy

Do not commit private keys, checkpoints, datasets, or run logs. This repository
is configured to ignore:

- `checkpoints/`
- `data/`
- `runs/`
- `*.pem`
- `*.pem.pub`

If a private key was ever committed to git history, treat it as compromised:
rotate or delete the key externally and purge it from history before publishing.
