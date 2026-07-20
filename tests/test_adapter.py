import pytest

from dustylm.adapter import map_smollm2_key, map_smollm2_to_dustylm_and_save
from dustylm.config import get_profile


def test_adapter_maps_representative_hf_keys():
    assert map_smollm2_key("model.embed_tokens.weight") == "embed_tokens.weight"
    assert map_smollm2_key("model.norm.weight") == "final_norm.weight"
    assert (
        map_smollm2_key("model.layers.0.self_attn.o_proj.weight")
        == "layers.0.self_attn.out_proj.weight"
    )
    assert map_smollm2_key("model.layers.0.mlp.gate_proj.weight") == "layers.0.gate_proj.weight"


def test_adapter_missing_weights_error_includes_profile_name(tmp_path):
    profile = get_profile("smollm2_135m")

    with pytest.raises(FileNotFoundError, match="--profile smollm2_135m --convert"):
        map_smollm2_to_dustylm_and_save(
            profile=profile,
            hf_model_path=tmp_path / "missing.safetensors",
        )
