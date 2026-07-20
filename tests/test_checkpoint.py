import json

import torch

from dustylm.checkpoint import (
    detect_profile_from_state_dict,
    load_state_dict,
    read_sidecar_profile_name,
    resolve_profile_name_for_checkpoint,
)


def test_load_state_dict_reads_raw_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    expected_state = {"embed.weight": torch.tensor([1.0])}
    torch.save(expected_state, checkpoint_path)

    state_dict = load_state_dict(checkpoint_path)

    assert state_dict["embed.weight"].tolist() == [1.0]


def test_detect_profile_from_dusty_scratch_keys():
    state_dict = {
        "embed.weight": object(),
        "layers.0.attention.qkv_proj.weight": object(),
    }

    assert detect_profile_from_state_dict(state_dict, mode="chat") == "sft_dusty8m"
    assert detect_profile_from_state_dict(state_dict, mode="generation") == "sft_dusty8m"


def test_detect_profile_from_smollm2_keys():
    state_dict = {
        "embed_tokens.weight": object(),
        "layers.0.gate_proj.weight": object(),
        "layers.0.up_proj.weight": object(),
    }

    assert detect_profile_from_state_dict(state_dict, mode="chat") == "sft_smollm2_135m"
    assert detect_profile_from_state_dict(state_dict, mode="generation") == "smollm2_135m"


def test_detect_profile_returns_none_for_unknown_keys():
    assert detect_profile_from_state_dict({"weight": object()}) is None


def test_read_sidecar_profile_name_prefers_checkpoint_stem_config(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_text("placeholder")
    (tmp_path / "model.json").write_text(json.dumps({"profile_name": "sft_dusty8m"}))
    (tmp_path / "config.json").write_text(json.dumps({"profile_name": "sft_smollm2_135m"}))

    assert read_sidecar_profile_name(checkpoint_path) == "sft_dusty8m"


def test_read_sidecar_profile_name_ignores_non_object_json(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_text("placeholder")
    (tmp_path / "model.json").write_text("[]")

    assert read_sidecar_profile_name(checkpoint_path) is None


def test_resolve_profile_explicit_profile_beats_sidecar_config(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_text("placeholder")
    (tmp_path / "model.json").write_text(json.dumps({"profile_name": "sft_smollm2_135m"}))

    assert (
        resolve_profile_name_for_checkpoint(
            checkpoint_path,
            explicit_profile="sft_dusty8m",
        )
        == "sft_dusty8m"
    )


def test_resolve_profile_uses_sidecar_before_sniffing(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "embed_tokens.weight": torch.tensor([1.0]),
            "layers.0.gate_proj.weight": torch.tensor([1.0]),
        },
        checkpoint_path,
    )
    (tmp_path / "model.json").write_text(json.dumps({"profile_name": "sft_dusty8m"}))

    assert resolve_profile_name_for_checkpoint(checkpoint_path) == "sft_dusty8m"


def test_resolve_profile_sniffs_weights_when_config_is_missing(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "embed_tokens.weight": torch.tensor([1.0]),
            "layers.0.gate_proj.weight": torch.tensor([1.0]),
        },
        checkpoint_path,
    )

    assert resolve_profile_name_for_checkpoint(checkpoint_path, mode="chat") == "sft_smollm2_135m"


def test_resolve_profile_falls_back_to_default_for_unknown_weights(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint_path)

    assert resolve_profile_name_for_checkpoint(checkpoint_path) == "sft_dusty8m"
