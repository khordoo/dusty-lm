# TinyGPT: Modern LLM Internals, Built From Scratch

TinyGPT is a compact, engineering-focused implementation of a modern decoder-only
large language model in pure PyTorch. It is designed for people who want to
understand how Llama-style architectures actually work under the hood, without
outsourcing the core model logic to Hugging Face `transformers`.

The repository implements the forward pass, attention stack, cache mechanics,
checkpoint conversion, profile registry, and generation loop directly. Hugging
Face model files can be used as source artifacts, but the runtime path is custom
PyTorch.

## Why TinyGPT? 🔬

Modern LLM libraries are powerful, but they hide the exact tensor operations that
make transformer systems work. TinyGPT strips away those abstraction layers and
keeps the important parts visible:

- how queries, keys, and values move through grouped-query attention;
- how rotary embeddings are computed and applied;
- how KV caching changes inference from full-sequence recomputation to
  token-by-token decoding;
- how Hugging Face `.safetensors` weights can be mapped into a custom module
  hierarchy;
- how model variants can be managed cleanly without piles of YAML or nested
  conditionals.

The goal is not to be the biggest framework. The goal is to be a readable,
rigorous baseline for understanding and extending modern LLM architecture.

## Technical Highlights ⚙️

- **Pure PyTorch runtime**: no Hugging Face `transformers` dependency for model
  execution, inference, attention, cache management, or generation.
- **Llama / SmolLM2-style architecture**: implements RoPE, grouped-query
  attention, RMSNorm, SwiGLU, and tied embedding / vocabulary projection weights.
- **KV cache inference**: prefill the prompt once, then decode with only the
  newest token while preserving per-layer key/value state.
- **Offline weight surgery**: `tiny_gpt.adapter` maps raw Hugging Face
  `.safetensors` keys into the custom TinyGPT module layout and saves a local
  PyTorch checkpoint.
- **Typed profile registry**: `tiny_gpt.config` uses frozen dataclasses for
  model, tokenizer, training, and generation profiles.
- **Standalone tokenization**: SmolLM2 profiles use the lightweight Rust-backed
  `tokenizers` library directly, decoupled from the broader Hugging Face runtime.
- **Explicit artifact downloads**: profile metadata can fetch raw HF weights and
  tokenizers into the local `artifacts/` layout on demand.
- **Testable core mechanics**: unit tests cover cache behavior, profile lookup,
  model dispatch, generation CLI parsing, and adapter key mapping.

## Architecture 🧠

TinyGPT separates model definition, configuration, and runtime construction:

```text
tiny_gpt/
  config.py        # typed profile registry
  modeling.py      # model/tokenizer factories
  artifacts.py     # profile-driven artifact downloads
  generate.py      # profile-driven generation
  train.py         # profile-driven training
  adapter.py       # offline HF safetensors conversion
  models/
    scratch.py     # small GPT architecture
    smollm2.py     # SmolLM2/Llama-style architecture
```

The SmolLM2 implementation includes the mechanics expected in a modern compact
LLM:

- **RoPE**: rotary positional embeddings are computed once per forward pass and
  shared through a `ForwardContext`.
- **GQA**: query heads can outnumber key/value heads, with KV repetition handled
  explicitly inside attention.
- **KV cache**: generation stores per-layer `(key, value)` tensors and appends
  new tokens without recomputing the full prompt.
- **SwiGLU MLP**: the feed-forward block follows the gated Llama-style pattern.
- **Weight tying**: `embed_tokens.weight` and `vocab_proj.weight` reference the
  same tensor.

## Profile Registry 🗂️

Model variants are managed through a typed registry in `tiny_gpt.config` instead
of ad hoc global constants or nested configuration branches.

Current profiles include:

- `scratch_small`: a small GPT profile for local training and experiments.
- `smollm2_360m`: a SmolLM2-360M architecture backed by a converted checkpoint.
- `sft_smollm2_360m`: a supervised fine-tuning profile that reuses the 360M base
  model spec.
- `smollm2_135m`: a smaller SmolLM2 profile with its own checkpoint target.

Adding another model size is mostly a configuration change: define a new
`ModelSpec`, point it at the shared SmolLM2 tokenizer artifact, and register a
profile with its converted checkpoint path.

## Artifact Layout 📦

Local model assets are intentionally kept out of git:

```text
artifacts/
  hf/
    smollm2_135m.safetensors
    smollm2_360m.safetensors
  checkpoints/
    scratch_small.pt
    smollm2_135m.pt
    smollm2_360m.pt
    sft_smollm2_360m.pt
  datasets/
    scratch_text_tokenized/
  tokenizers/
    smollm2_tokenizer.json
```

- `artifacts/hf/` stores raw downloaded Hugging Face `.safetensors` files.
- `artifacts/checkpoints/` stores converted TinyGPT PyTorch checkpoints.
- `artifacts/datasets/` stores tokenized local training corpora.
- `artifacts/tokenizers/smollm2_tokenizer.json` is shared by all SmolLM2
  profiles.

## Quickstart 🚀

Install dependencies:

```bash
uv sync
```

For development and tests:

```bash
uv sync --group dev
```

Convert a SmolLM2 checkpoint into TinyGPT format:

```bash
uv run python -m tiny_gpt.artifacts download \
  --profile smollm2_360m \
  --convert
```

This downloads the raw HF safetensors and tokenizer into `artifacts/`, then
writes the converted checkpoint configured by the profile.

If the files are already present and you only need conversion, run:

```bash
uv run python -m tiny_gpt.adapter --profile smollm2_360m
```

If you manually downloaded a safetensors file elsewhere, pass it explicitly:

```bash
uv run python -m tiny_gpt.adapter \
  --profile smollm2_360m \
  --hf-model-path artifacts/hf/smollm2_360m.safetensors
```

Run generation:

```bash
uv run python -m tiny_gpt.generate --profile smollm2_360m
```

Download and convert the smaller SmolLM2 profile:

```bash
uv run python -m tiny_gpt.artifacts download \
  --profile smollm2_135m \
  --convert
```

Prepare the tracked demo text corpus for the scratch profile:

```bash
uv run python -m tiny_gpt.data_prep --profile scratch_small
```

Train the scratch profile on the Alice excerpt:

```bash
uv run python -m tiny_gpt.train --profile scratch_small --epochs 20
```

The training command defaults to one epoch if `--epochs` is omitted, but the
tracked Alice corpus is intentionally tiny for educational use. Running about 20
epochs gives the model enough repeated exposure for the loss to drop toward
roughly `2.4`, where it starts to learn the local language patterns. For a real
dataset, use more data instead of repeatedly overfitting a small excerpt.

For a single default epoch, run:

```bash
uv run python -m tiny_gpt.train --profile scratch_small
```

Generate from the scratch checkpoint:

```bash
uv run python -m tiny_gpt.generate --profile scratch_small --prompt "Alice was"
```

## Hardware Requirements & Memory Tuning 💻

The scratch model is small enough for local educational training on consumer
hardware, including Apple Silicon laptops and standard GPUs. The main memory
knobs live in `tiny_gpt/config.py` under the `scratch_small` training profile:

```python
max_seq_len=256
batch_size=16
```

If you have 16GB or more of unified memory / VRAM, you can usually increase
`batch_size` to `32` for better throughput with `max_seq_len=256`. That setup is
expected to use roughly a few GB of memory for this custom model.

If you are on an 8GB machine or hit an out-of-memory error, lower only the batch
size first:

```python
batch_size=16  # or 8 for tighter memory
```

Lowering `batch_size` reduces memory use while keeping the same learning
objective. Training may take more optimizer steps to see the same amount of data,
but the model is still learning from the same token sequences. For larger local
story corpora, such as adding another `.txt` file under `demo_text/`, tune
`batch_size` based on available memory before changing model dimensions.

Run the test suite:

```bash
uv run pytest
```

## Generation Flow ⚡

Generation uses an explicit cache-aware loop:

1. Tokenize the prompt with the profile's tokenizer.
2. Run a prefill pass with an empty KV cache.
3. Sample the next token from the final-position logits.
4. Feed only the newest token back into the model.
5. Append new keys and values to each layer's cache.

This makes the inference path easy to inspect and avoids hiding the most
important performance optimization behind a framework API.

## Roadmap 🛠️

- **Supervised Fine-Tuning (SFT)**: complete the SmolLM2 SFT path using the
  existing `sft_smollm2_360m` profile.
- **Direct Preference Optimization (DPO)**: add a preference-training pipeline
  for alignment experiments.
- **Benchmarking**: add tokens/sec, memory, and cache efficiency measurements
  across CPU, CUDA, and Apple Silicon.
- **More profiles**: expand model coverage through additional typed SmolLM2
  profiles and converted checkpoints.
- **Evaluation harness**: add lightweight perplexity and generation-quality
  checks for local model iteration.

## Repository Policy 🔒

Do not commit model checkpoints, raw HF downloads, tokenizers, datasets, run
logs, or credentials. Local artifacts belong under `artifacts/`, `data/`, and
`runs/`, all of which are ignored by git.

TinyGPT is intentionally small, but the engineering bar is high: explicit tensor
flows, typed configuration, reproducible artifact boundaries, and tests for the
core mechanics that make modern autoregressive models work.
