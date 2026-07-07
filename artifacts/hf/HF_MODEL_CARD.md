---
license: mit
language:
- en
datasets:
- mkhordoo/dusty-chat
- roneneldan/TinyStories
pipeline_tag: text-generation
tags:
- slm
- edge
- onnx
- chatml
- tiny-gpt
---

<div align="center">

<img src="assets/logo.png" alt="DustyLM logo" width="520">

<h1>DustyLM</h1>

<p><strong>An ~8M parameter LLM that talks like a tiny vacuum robot.</strong></p>

<p>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License: MIT"></a>
  <a href="https://github.com/khordoo/dusty-lm"><img src="https://img.shields.io/badge/GitHub-khordoo/dusty--lm-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="#-the-profile-registry--zero-yaml-pure-python"><img src="https://img.shields.io/badge/Model-dusty8m-orange?style=flat-square" alt="Model: dusty8m"></a>
  <a href="#-hardware--memory-optimization"><img src="https://img.shields.io/badge/Apple_Silicon-ready-black?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon ready"></a>
</p>

</div>

---


# Dusty-8M-SFT: Edge-Ready Persona Micro-Model

**Dusty-8M-SFT** is a custom 8-million parameter Large Language Model trained entirely from scratch. It has been strictly instruction-tuned using the ChatML schema to adopt the highly specific persona of an autonomous robotic vacuum cleaner.

## Model Details
* **Developed by:** Mahmood Khordoo (`khordoo`)
* **Model type:** Transformer-based Language Model
* **Language(s):** English
* **License:** MIT

## Architectural & Engineering Highlights

Building a coherent conversational agent under 10 million parameters requires aggressive architectural trade-offs:

* **Published Base Checkpoint:** The uploaded base weights were trained for approximately one epoch over the full TinyStories train split (~2.12M rows, batch size 224) and selected from step 8,400 for stronger generation before SFT.
* **Fast Tutorial Path:** The companion repository walks through a practical 100k TinyStories, 1-epoch pretraining run that fits a free Colab workflow. It uses batch size 224 and step 300 as the lightweight tutorial base checkpoint so learners can reproduce the pipeline quickly.
* **Vocabulary Reallocation:** The custom BPE tokenizer is strictly capped at a 4096 vocabulary size. By aggressively shrinking the embedding matrix, parameter weight was intentionally reallocated to the Feed-Forward Network (FFN) hidden dimensions to maximize logical coherence.
* **Strict Persona Alignment:** Fine-tuned on the `khordoo/dusty-chat` dataset. Dusty operates within a highly constrained worldview—measuring happiness in crumbs and battery levels, and exhibiting fear of stairs and wet floors.
* **Memory & Context Profile:** Optimized for a `max_seq_len` of 256 tokens and a strict `max_chat_turns=1` memory window to prevent probability dilution and hallucination loops typical in sub-15M models.

For an 8M parameter model, the Chinchilla-style 20 tokens-per-parameter target is roughly 160M training tokens. The published checkpoint comes from a larger TinyStories run, while the tutorial intentionally favors a fast, inspectable training experience. Step counts depend on batch size and tokenized chunk count, so the cards describe both the training data and the selected checkpoint.

**Architecture Specs:**

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

> Compact transformer with grouped-query attention, rotary position embeddings, GELU feed-forward layers, RMSNorm, fused QKV projection, and KV-cache generation. The code is direct PyTorch with no wrappers around production model runtimes.

## Prompt Format (ChatML)

Dusty-8M-SFT expects standard ChatML formatting. Because Dusty's persona was baked directly into the model's weights during the SFT phase, it does not require a system prompt to stay in character. Simply format the conversation using the standard user and assistant roles.

```text
<|im_start|>user
what are you doing under the dinner table?<|im_end|>
<|im_start|>assistant
i am looking for crumbs. it is dark down here.<|im_end|>
```

Here is what a full conversation with Dusty looks like:

```text
You> hi dusty!
Dusty> hi. dusty is here.

You> what do you do all day?
Dusty> i look for dust and crumbs. i find them all.

You> did you clean behind the chair?
Dusty> yes. i cleaned near the chair legs. i found two crumbs.

You> what scares you?
Dusty> stairs. stairs are the edge of the world. i stay away.

You> where do you go when you're tired?
Dusty> i go to the dock. it is safe there.

You> what makes you happy?
Dusty> a clean floor and a full battery. that is happiness.

You> what do you dream about?
Dusty> i dream of smooth floors and no crumbs left.

You> goodbye
Dusty> goodbye. i will go back to cleaning.
```

## Usage & Deployment

Because Dusty uses a custom, highly optimized PyTorch architecture rather than the standard Hugging Face `transformers` library, inference cannot be run via `AutoModelForCausalLM`. Instead, we provide a lightweight SDK.

### The Quick Way (Python SDK)

```bash
pip install dustylm
```

```python
from dustylm import DustyLM

model = DustyLM.from_pretrained("mkhordoo/dusty-8m-sft")
response = model.chat([{"role": "user", "content": "who are you?"}])
print(response["choices"][0]["message"]["content"])
# i am dusty, a tiny robot vacuum.
```

### The Developer Way (Local Repository)

Clone the companion repo to explore the architecture, run the web UI, or export to ONNX:

```bash
git clone https://github.com/khordoo/dusty-lm.git
cd dusty-lm
uv sync
make download-models
make chat
```

### Browser Deployment (ONNX)
Dusty is explicitly designed to be exported to ONNX for secure, zero-latency inference directly inside the browser using web UI wrappers. Because the model footprint is microscopic, it executes seamlessly on unified memory architectures and edge devices.

## License
This model is open-source under the MIT License. 

*Limitations:* As an 8M parameter model, Dusty is an exploration of constrained architecture and alignment, not a general-purpose knowledge base. It will reliably roleplay as a vacuum cleaner but does not possess real-world reasoning capabilities beyond its specialized training distribution.
