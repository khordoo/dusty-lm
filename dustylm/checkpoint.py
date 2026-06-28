"""Checkpoint loading and profile detection helpers.

DustyLM checkpoints intentionally remain plain PyTorch state dicts so they can
be exported, shared, and adapted without framework-specific metadata. Runtime
code can still improve UX by reading an optional sidecar config or sniffing the
CPU-loaded state dict keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

CHAT_PROFILE_DEFAULT = "sft_dusty8m"
GENERATION_PROFILE_DEFAULT = "sft_dusty8m"
SMOLLM2_CHAT_PROFILE = "sft_smollm2_135m"
SMOLLM2_GENERATION_PROFILE = "smollm2_135m"
DUSTY_CHAT_PROFILE = "sft_dusty8m"
DUSTY_GENERATION_PROFILE = "sft_dusty8m"


def load_state_dict(
    checkpoint_path: str | Path,
    map_location="cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run `make train-pretrain EPOCHS=1` first or download a pre-trained checkpoint."
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def detect_profile_from_state_dict(
    state_dict: dict[str, Any],
    *,
    mode: str = "chat",
) -> str | None:
    keys = state_dict.keys()
    prefer_chat = mode == "chat"
    if any("gate_proj" in key for key in keys) or any(
        key.startswith("embed_tokens.weight") for key in keys
    ):
        return SMOLLM2_CHAT_PROFILE if prefer_chat else SMOLLM2_GENERATION_PROFILE

    if any("attention.qkv_proj" in key for key in keys) or any(
        key.startswith("embed.weight") for key in keys
    ):
        return DUSTY_CHAT_PROFILE if prefer_chat else DUSTY_GENERATION_PROFILE

    return None


def get_sidecar_config_paths(checkpoint_path: str | Path) -> list[Path]:
    checkpoint_path = Path(checkpoint_path)
    return [
        checkpoint_path.with_suffix(".json"),
        checkpoint_path.parent / "config.json",
    ]


def read_sidecar_profile_name(checkpoint_path: str | Path) -> str | None:
    for config_path in get_sidecar_config_paths(checkpoint_path):
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        profile_name = config.get("profile_name")
        if isinstance(profile_name, str) and profile_name:
            return profile_name
    return None


def resolve_profile_name_for_checkpoint(
    checkpoint_path: str | Path | None,
    *,
    explicit_profile: str | None = None,
    default_profile: str = CHAT_PROFILE_DEFAULT,
    mode: str = "chat",
) -> str:
    if explicit_profile is not None:
        return explicit_profile
    if checkpoint_path is None:
        return default_profile

    sidecar_profile = read_sidecar_profile_name(checkpoint_path)
    if sidecar_profile is not None:
        return sidecar_profile

    state_dict = load_state_dict(checkpoint_path, map_location="cpu")
    detected_profile = detect_profile_from_state_dict(state_dict, mode=mode)
    return detected_profile or default_profile
