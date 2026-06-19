from tiny_gpt.adapter import map_smollm2_key


def test_adapter_maps_representative_hf_keys():
    assert map_smollm2_key("model.embed_tokens.weight") == "embed_tokens.weight"
    assert map_smollm2_key("model.norm.weight") == "final_norm.weight"
    assert (
        map_smollm2_key("model.layers.0.self_attn.o_proj.weight")
        == "layers.0.self_attn.out_proj.weight"
    )
    assert (
        map_smollm2_key("model.layers.0.mlp.gate_proj.weight")
        == "layers.0.gate_proj.weight"
    )
