"""Factory functions that dispatch model and tokenizer construction from profiles.

``build_model`` inspects the ``ModelFamily`` in a profile's ``ModelSpec`` and
returns the corresponding ``TinyGPT`` instance (scratch or SmolLM2).
``build_tokenizer`` similarly dispatches between tiktoken and HuggingFace
tokenizers based on the ``TokenizerSpec.kind`` field.
"""

from pathlib import Path
from typing import Any

import tiktoken
from tokenizers import Tokenizer

from tiny_gpt.config import ModelFamily, Profile
from tiny_gpt.models.scratch import TinyGPT as ScratchGPT
from tiny_gpt.models.smollm2 import TinyGPT as SmolLM2


def build_model(profile: Profile, max_seq_len: int | None = None):
    spec = profile.model
    model_max_seq_len = max_seq_len or spec.max_seq_len
    if spec.family == ModelFamily.SCRATCH_GPT:
        return ScratchGPT(
            num_layers=spec.num_layers,
            vocab_size=spec.vocab_size,
            max_seq_len=model_max_seq_len,
            embed_dim=spec.embed_dim,
            num_heads=spec.num_heads,
            num_kv_heads=spec.num_kv_heads,
            hidden_dim=spec.hidden_dim,
            rope_base=spec.rope_base,
            rms_eps=spec.rms_eps,
        )
    if spec.family == ModelFamily.SMOLLM2:
        return SmolLM2(
            num_layers=spec.num_layers,
            vocab_size=spec.vocab_size,
            max_seq_len=model_max_seq_len,
            embed_dim=spec.embed_dim,
            num_heads=spec.num_heads,
            num_kv_heads=spec.num_kv_heads,
            hidden_dim=spec.hidden_dim,
            rope_base=spec.rope_base,
            rms_eps=spec.rms_eps,
        )
    raise ValueError(f"Unsupported model family: {spec.family}")


def build_tokenizer(profile: Profile) -> Any:
    spec = profile.model.tokenizer
    if spec.kind == "tiktoken":
        return tiktoken.get_encoding(str(spec.path_or_name))
    if spec.kind == "tokenizers":
        return Tokenizer.from_file(str(Path(spec.path_or_name)))
    raise ValueError(f"Unsupported tokenizer kind: {spec.kind}")
