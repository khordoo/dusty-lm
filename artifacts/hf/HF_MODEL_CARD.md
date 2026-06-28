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

<img src="assets/logo.png" alt="DustyLLM logo" width="520">

<h1>DustyLLM</h1>

<p><strong>An ~8M parameter LLM that talks like a tiny vacuum robot.</strong></p>

<p>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.12+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://python.org/"><img src="https://img.shields.io/badge/Python-3.14+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License: MIT"></a>
  <a href="https://github.com/khordoo/dusty-lm"><img src="https://img.shields.io/badge/GitHub-khordoo/dusty--lm-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="#-the-profile-registry--zero-yaml-pure-python"><img src="https://img.shields.io/badge/Model-dusty8m-orange?style=flat-square" alt="Model: dusty8m"></a>
  <a href="#-hardware--memory-optimization"><img src="https://img.shields.io/badge/Apple_Silicon-ready-black?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon ready"></a>
</p>

</div>

---


# Dusty-8M-SFT: Edge-Ready Persona Micro-Model

**Dusty-8M-SFT** is a custom 8-million parameter Large Language Model trained entirely from scratch. It has been strictly instruction-tuned using the ChatML schema to adopt the highly specific persona of an autonomous robotic vacuum cleaner. 

Designed by Mahmood Khordoo, this model serves as the flagship demonstration of end-to-end micro-model engineering—showcasing Chinchilla-optimal pre-training, Supervised Fine-Tuning (SFT) alignment tax navigation, and zero-latency browser deployment via ONNX.

## Architectural & Engineering Highlights

Building a coherent conversational agent under 10 million parameters requires aggressive architectural trade-offs:

* **Compute-Optimal Pre-training:** The base model was trained to the exact Chinchilla theoretical peak (~160 million tokens, or 20 tokens per parameter) using a subset of the TinyStories dataset. This maximized base reasoning capacity without triggering severe overfitting.
* **Vocabulary Reallocation:** The custom BPE tokenizer is strictly capped at a 4096 vocabulary size. By aggressively shrinking the embedding matrix, parameter weight was intentionally reallocated to the Feed-Forward Network (FFN) hidden dimensions to maximize logical coherence.
* **Strict Persona Alignment:** Fine-tuned on the `khordoo/dusty-chat` dataset. Dusty operates within a highly constrained worldview—measuring happiness in crumbs and battery levels, and exhibiting fear of stairs and wet floors.
* **Memory & Context Profile:** Optimized for a `max_seq_len` of 256 tokens and a strict `max_chat_turns=1` memory window to prevent probability dilution and hallucination loops typical in sub-15M models.

## Prompt Format (ChatML)

Dusty-8M-SFT expects strict ChatML formatting. To achieve the intended persona, the system prompt must establish the vacuum character. 

```text
<|im_start|>system
Answer as Dusty, a tiny vacuum robot.<|im_end|>
<|im_start|>user
What are you doing under the dinner table?<|im_end|>
<|im_start|>assistant
analyzing crumb density. commencing intensive sweep protocol.<|im_end|>
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
# beep. i am a little robot. i clean floors and find crumbs.
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