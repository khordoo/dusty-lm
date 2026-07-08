"""Typed dataclass configuration system and profile registry.

Every model variant, training run, and generation configuration is defined
as a ``Profile`` — a frozen dataclass that bundles a ``ModelSpec``,
``TrainingSpec``, ``GenerationSpec``, and optional ``HFArtifactSpec``.
Profiles are registered at import time and looked up by name at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
IGNORE_INDEX = -100


class ModelFamily(StrEnum):
    SCRATCH_GPT = "scratch_gpt"
    SMOLLM2 = "smollm2"


class TrainingTask(StrEnum):
    PRETRAIN = "pretrain"
    SFT = "sft"


@dataclass(frozen=True)
class TokenizerSpec:
    kind: str
    path_or_name: str | Path


@dataclass(frozen=True)
class ModelSpec:
    family: ModelFamily
    max_seq_len: int
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    tokenizer: TokenizerSpec
    rope_base: int = 10000
    rms_eps: float = 1e-4
    hidden_dim: int | None = None


@dataclass(frozen=True)
class TrainingSpec:
    task: TrainingTask
    dataset_path: str | Path
    batch_size: int
    learning_rate: float
    output_checkpoint: Path
    max_seq_len: int
    weight_decay: float = 0.0
    raw_text_path: str | Path | None = None
    raw_sft_path: str | Path | None = None
    sft_user_field: str = "user"
    sft_assistant_field: str = "assistant"
    init_checkpoint_path: Path | None = None
    checkpoint_every_steps: int | None = None
    checkpoint_dir: Path | None = None
    log_dir: str | Path = REPO_ROOT / "artifacts" / "tensorboard"


@dataclass(frozen=True)
class GenerationSpec:
    checkpoint_path: Path
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float = 1.0
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    eos_text: str | None = None
    max_chat_turns: int | None = None


@dataclass(frozen=True)
class HFArtifactSpec:
    repo_id: str
    weights_filename: str
    tokenizer_filename: str
    local_weights_path: Path
    local_tokenizer_path: Path
    revision: str = "main"


@dataclass(frozen=True)
class Profile:
    name: str
    model: ModelSpec
    description: str = ""
    training: TrainingSpec | None = None
    generation: GenerationSpec | None = None
    hf_artifacts: HFArtifactSpec | None = None
    base_profile: str | None = None


SCRATCH_TOKENIZER = TokenizerSpec(kind="tiktoken", path_or_name="r50k_base")
SMOLLM2_TOKENIZER = TokenizerSpec(
    kind="tokenizers",
    path_or_name=REPO_ROOT / "artifacts" / "tokenizers" / "smollm2_tokenizer.json",
)
DUSTY_TOKENIZER = TokenizerSpec(
    kind="tokenizers",
    path_or_name=REPO_ROOT / "artifacts" / "tokenizers" / "dusty_tokenizer.json",
)

_PROFILES: dict[str, Profile] = {}


def register(profile: Profile) -> Profile:
    """Add a profile to the global registry."""
    if profile.name in _PROFILES:
        raise ValueError(f"Profile already registered: {profile.name}")
    _PROFILES[profile.name] = profile
    return profile


def get_profile(name: str) -> Profile:
    """Return a registered profile, resolving inherited base profile fields."""
    try:
        profile = _PROFILES[name]
    except KeyError as exc:
        available = ", ".join(list_profiles())
        raise KeyError(f"Unknown profile '{name}'. Available profiles: {available}") from exc

    if profile.base_profile is None:
        return profile

    base = get_profile(profile.base_profile)
    return Profile(
        name=profile.name,
        model=profile.model or base.model,
        description=profile.description or base.description,
        training=profile.training or base.training,
        generation=profile.generation or base.generation,
        hf_artifacts=profile.hf_artifacts or base.hf_artifacts,
        base_profile=profile.base_profile,
    )


def list_profiles(verbose: bool = False) -> list[str]:
    """List registered profile names, optionally with descriptions."""
    if verbose:
        return [f"{name:20} - {_PROFILES[name].description}" for name in sorted(_PROFILES)]
    return sorted(_PROFILES)


scratch_small_model = ModelSpec(
    family=ModelFamily.SCRATCH_GPT,
    max_seq_len=256,
    vocab_size=50257,
    embed_dim=512,
    num_heads=8,
    num_kv_heads=2,
    num_layers=6,
    tokenizer=SCRATCH_TOKENIZER,
)

smollm2_360m_model = ModelSpec(
    family=ModelFamily.SMOLLM2,
    max_seq_len=8192,
    vocab_size=49152,
    embed_dim=960,
    num_heads=15,
    num_kv_heads=5,
    num_layers=32,
    hidden_dim=2560,
    rope_base=10000,
    rms_eps=1e-4,
    tokenizer=SMOLLM2_TOKENIZER,
)

smollm2_135m_model = ModelSpec(
    family=ModelFamily.SMOLLM2,
    max_seq_len=8192,
    vocab_size=49152,
    embed_dim=576,
    num_heads=9,
    num_kv_heads=3,
    num_layers=30,
    hidden_dim=1536,
    tokenizer=SMOLLM2_TOKENIZER,
)

dusty_8m_model = ModelSpec(
    family=ModelFamily.SCRATCH_GPT,
    embed_dim=256,
    num_layers=8,
    num_heads=8,
    num_kv_heads=4,
    hidden_dim=1024,
    max_seq_len=256,
    vocab_size=4096,
    tokenizer=DUSTY_TOKENIZER,
)

# =============================================================================
# PROFILE REGISTRY
# =============================================================================
# A Profile is a named configuration bundle. It connects:
#   - model architecture
#   - tokenizer
#   - training dataset/checkpoint paths
#   - generation defaults
#   - optional Hugging Face artifact metadata
#
# Most commands take a --profile argument, then look up one of these registered
# profiles instead of asking the user to pass many individual paths and flags.
#
# Profiles are grouped by purpose:
#
# 1. Dusty core path:
#    dusty8m     - base 8M model trained from scratch
#    sft_dusty8m - Dusty persona fine-tune initialized from dusty8m
#
# 2. Scratch sandbox:
#    scratch_small - experimental scratch model using the GPT-2/r50k tokenizer
#
# 3. SmolLM2 baselines:
#    smollm2_135m, smollm2_360m - pretrained Hugging Face model shapes
#    sft_smollm2_135m           - SFT profile for the 135M baseline
#
# The TrainingSpec batch size and checkpoint interval below are conservative
# direct-Python defaults. The Makefile and notebooks override them for the
# current Colab golden path (batch 224, checkpoints every 50 steps).
# =============================================================================

register(
    Profile(
        name="dusty8m",
        model=dusty_8m_model,
        description="Core 8M parameter model trained from scratch on TinyStories.",
        training=TrainingSpec(
            task=TrainingTask.PRETRAIN,
            dataset_path=REPO_ROOT / "artifacts" / "datasets" / "dusty_pretrain_tokenized",
            batch_size=32,
            learning_rate=3e-3,
            output_checkpoint=REPO_ROOT / "artifacts" / "checkpoints" / "dusty8m.pt",
            max_seq_len=256,
            weight_decay=0.01,
            raw_text_path=REPO_ROOT / "artifacts" / "datasets" / "tinystories_base.txt",
            checkpoint_every_steps=100,
            checkpoint_dir=REPO_ROOT / "artifacts" / "checkpoints",
        ),
        generation=GenerationSpec(
            checkpoint_path=REPO_ROOT / "artifacts" / "checkpoints" / "dusty8m.pt",
            max_new_tokens=128,
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            eos_text="<|endoftext|>",
        ),
    )
)

register(
    Profile(
        name="sft_dusty8m",
        model=dusty_8m_model,
        description="SFT fine-tune of dusty8m on the Dusty chat dataset for persona alignment.",
        training=TrainingSpec(
            task=TrainingTask.SFT,
            dataset_path=REPO_ROOT / "artifacts" / "datasets" / "dusty_sft_tokenized",
            batch_size=32,
            learning_rate=1e-3,
            output_checkpoint=REPO_ROOT / "artifacts" / "checkpoints" / "dusty8m_sft.pt",
            max_seq_len=256,
            weight_decay=0.01,
            raw_sft_path=REPO_ROOT / "artifacts" / "datasets" / "dusty_sft.jsonl",
            sft_assistant_field="dusty",
            init_checkpoint_path=REPO_ROOT / "artifacts" / "checkpoints" / "dusty8m.pt",
            checkpoint_every_steps=100,
            checkpoint_dir=REPO_ROOT / "artifacts" / "checkpoints",
        ),
        generation=GenerationSpec(
            checkpoint_path=REPO_ROOT / "artifacts" / "checkpoints" / "dusty8m_sft.pt",
            max_new_tokens=200,
            temperature=0.8,
            top_k=5,
            top_p=0.8,
            eos_token_id=2,
            max_chat_turns=1,
        ),
    )
)

register(
    Profile(
        name="scratch_small",
        model=scratch_small_model,
        description="Experimental sandbox with GPT-2 r50k tokenizer and larger embed dim.",
        training=TrainingSpec(
            task=TrainingTask.PRETRAIN,
            dataset_path=REPO_ROOT / "artifacts" / "datasets" / "scratch_text_tokenized",
            batch_size=16,
            learning_rate=1e-4,
            output_checkpoint=REPO_ROOT / "artifacts" / "checkpoints" / "scratch_small.pt",
            max_seq_len=256,
            weight_decay=0.01,
            raw_text_path=REPO_ROOT / "artifacts" / "datasets" / "tinystories_base.txt",
        ),
        generation=GenerationSpec(
            checkpoint_path=REPO_ROOT / "artifacts" / "checkpoints" / "scratch_small.pt",
            max_new_tokens=1000,
            temperature=0.75,
            top_k=10,
            eos_text="<|im_end|>",
        ),
    )
)

# =============================================================================
# SmolLM2 360m and 135m Pre-trained Baselines
#
# Note: The base profiles below (smollm2_360m, smollm2_135m) do not include a TrainingSpec. They are
# fully pre-trained by Hugging Face and are intended for direct inference
# or to be used as initialization weights for the SFT profile below.
# =============================================================================

register(
    Profile(
        name="smollm2_360m",
        model=smollm2_360m_model,
        description="Hugging Face SmolLM2 360M pretrained baseline for high-capacity inference.",
        generation=GenerationSpec(
            checkpoint_path=REPO_ROOT / "artifacts" / "checkpoints" / "smollm2_360m.pt",
            max_new_tokens=1000,
            temperature=1.0,
            top_k=10,
            bos_token_id=0,
            eos_token_id=0,
        ),
        hf_artifacts=HFArtifactSpec(
            repo_id="HuggingFaceTB/SmolLM2-360M",
            weights_filename="model.safetensors",
            tokenizer_filename="tokenizer.json",
            local_weights_path=REPO_ROOT / "artifacts" / "hf" / "smollm2_360.safetensors",
            local_tokenizer_path=SMOLLM2_TOKENIZER.path_or_name,
        ),
    )
)

register(
    Profile(
        name="smollm2_135m",
        model=smollm2_135m_model,
        description="Hugging Face SmolLM2 135M pretrained baseline for efficient fine-tuning.",
        generation=GenerationSpec(
            checkpoint_path=REPO_ROOT / "artifacts" / "checkpoints" / "smollm2_135m.pt",
            max_new_tokens=1000,
            temperature=1.0,
            top_k=10,
            bos_token_id=0,
            eos_token_id=0,
        ),
        hf_artifacts=HFArtifactSpec(
            repo_id="HuggingFaceTB/SmolLM2-135M",
            weights_filename="model.safetensors",
            tokenizer_filename="tokenizer.json",
            local_weights_path=REPO_ROOT / "artifacts" / "hf" / "smollm2_135m.safetensors",
            local_tokenizer_path=SMOLLM2_TOKENIZER.path_or_name,
        ),
    )
)

register(
    Profile(
        name="sft_smollm2_135m",
        model=smollm2_135m_model,
        description="SFT fine-tune of SmolLM2 135M on the Dusty chat dataset for persona alignment.",
        training=TrainingSpec(
            task=TrainingTask.SFT,
            # By default, this points to the Dusty dataset so you can test
            # fine-tuning a larger architecture out of the box.
            # Replace this path with your own tokenized SFT dataset
            # before running `python -m dustylm.train --profile sft_smollm2_135m`
            dataset_path=REPO_ROOT / "artifacts" / "datasets" / "dusty_sft_tokenized",
            batch_size=1,
            learning_rate=1e-5,
            raw_sft_path=REPO_ROOT / "artifacts" / "datasets" / "dusty_sft.jsonl",
            output_checkpoint=REPO_ROOT / "artifacts" / "checkpoints" / "sft_smollm2_135m.pt",
            max_seq_len=2048,
            init_checkpoint_path=REPO_ROOT / "artifacts" / "checkpoints" / "smollm2_135m.pt",
        ),
        generation=replace(
            get_profile("smollm2_135m").generation,
            checkpoint_path=REPO_ROOT / "artifacts" / "checkpoints" / "sft_smollm2_135m.pt",
            max_chat_turns=5,
        ),
        base_profile="smollm2_135m",
    )
)
