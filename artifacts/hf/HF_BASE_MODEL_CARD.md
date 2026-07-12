---
license: mit
language:
- en
datasets:
- roneneldan/TinyStories
pipeline_tag: text-generation
tags:
- slm
- edge
- tiny-gpt
- base-model
---

<div align="center">

<img src="assets/logo.png" alt="DustyLM logo" width="520">

<h1>DustyLM Base</h1>

<p><strong>An ~8M parameter base language model pre-trained on TinyStories.</strong></p>

<p>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License: MIT"></a>
  <a href="https://github.com/khordoo/dusty-lm"><img src="https://img.shields.io/badge/GitHub-khordoo/dusty--lm-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="#"><img src="https://img.shields.io/badge/Model-dusty8m-orange?style=flat-square" alt="Model: dusty8m"></a>
  <a href="#"><img src="https://img.shields.io/badge/Apple_Silicon-ready-black?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon ready"></a>
</p>

</div>

---

# Dusty-8M-Base: TinyStories Pretrain

**Dusty-8M-Base** is the raw pre-trained checkpoint of the 8-million parameter DustyLM architecture. The uploaded checkpoint was trained for approximately one epoch over the full TinyStories train split (~2.12M rows) and selected from the step 8,400 checkpoint for stronger plain-text generation.

The companion repository's guided tutorial uses a smaller 100k-row TinyStories slice so developers can run the full end-to-end pipeline quickly in Colab. That tutorial checkpoint is intentionally optimized for learning speed; this uploaded checkpoint is the stronger published base model.

This model is the "before" picture. Combined with the [SFT checkpoint](https://huggingface.co/mkhordoo/dusty-8m-sft), it demonstrates what Supervised Fine-Tuning actually does: transform a generic text generator into a specific conversational character.

## Training Details

| Setting | Uploaded checkpoint | Tutorial default |
|---|---:|---:|
| TinyStories data | Full train split (~2.12M rows) | 100k-row slice |
| Pretraining pass | ~1 epoch | 1 epoch |
| Batch size | 224 | 224 |
| Selected base checkpoint | Step 8,400 | Step 300 |
| Purpose | Higher-quality published base weights | Fast educational run |

For an 8M parameter model, the Chinchilla-style 20 tokens-per-parameter target is roughly 160M training tokens. The uploaded checkpoint was trained from a much larger TinyStories run, while the tutorial deliberately trades off maximum pretraining compute for a short, reproducible learning workflow. Step counts depend on the batch size and tokenized chunk count, so the dataset size and epoch count are the more portable comparison.

## Model Details
* **Developed by:** Mahmood Khordoo (`khordoo`)
* **Model type:** Transformer-based Language Model
* **Language(s):** English
* **License:** MIT

## Architecture

| Setting | Value |
|---|---|
| Parameters | ~8M |
| Layers | 8 |
| Hidden dim | 256 |
| Heads | 8 query / 4 KV |
| FFN | 1,024 GELU |
| Vocab | 4,096 BPE |
| Max sequence | 256 tokens |
| Norm | RMSNorm |
| Position | RoPE |
| LM head | Separate projection |

> Compact transformer with grouped-query attention, rotary position embeddings, GELU feed-forward layers, RMSNorm, fused QKV projection, and KV-cache generation. The code is pure PyTorch with no wrappers around production model runtimes.

## Usage

Because Dusty uses a custom, highly optimized PyTorch architecture rather than the standard Hugging Face `transformers` library, inference cannot be run via `AutoModelForCausalLM`. Instead, we provide a lightweight SDK.

### The Quick Way (Python SDK)

```bash
pip install dustylm-sdk
```

```python
from dustylm_sdk import DustyLM

model = DustyLM.from_pretrained("mkhordoo/dusty-8m-base")
response = model.generate("Once upon a time")
print(response)
```

### The Developer Way (Local Repository)

Clone the companion repo to explore the architecture, tweak generation, or train from scratch:

```bash
git clone https://github.com/khordoo/dusty-lm.git
cd dusty-lm
uv sync
make download-models
make generate
```

### Training From Scratch

Covered in detail in the [companion repository](https://github.com/khordoo/dusty-lm). The tutorial defaults are:

```bash
make download-datasets
make tokenizer
make data-pretrain
make train-pretrain EPOCHS=1 CHECKPOINT_EVERY_STEPS=50 BATCH_SIZE=224
make data-sft
make train-sft EPOCHS=2 CHECKPOINT_EVERY_STEPS=50 BATCH_SIZE=224
```

Or run the same current golden path with:

```bash
make train-end-to-end
```

## License

MIT

## Acknowledgements

- TinyStories dataset by Ronen Eldan and Yuanzhi Li
- Built with PyTorch
