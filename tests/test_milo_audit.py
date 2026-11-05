import copy

import pytest
import torch

from experiments.weight_space_milo.audit import audit_states, make_patch, reconstruct, state_hash


def fixture_state():
    return {
        "embed.weight": torch.ones(4, 3),
        "layers.0.attention.qkv_proj.weight": torch.zeros(6, 3),
        "layers.0.mlp.0.weight": torch.ones(8, 3),
        "layers.0.att_norm.weight": torch.ones(3),
        "vocab_proj.weight": torch.ones(4, 3),
    }


def test_exact_patch_persists_and_reconstructs(tmp_path):
    source = fixture_state()
    candidate = copy.deepcopy(source)
    candidate["layers.0.attention.qkv_proj.weight"][0, 0] = -0.0
    candidate["layers.0.mlp.0.weight"][3, 2] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    a, b, patch, restored = [tmp_path / n for n in ("a.pt", "b.pt", "patch.pt", "restored.pt")]
    torch.save(source, a)
    torch.save({"model_state_dict": candidate}, b)
    report = make_patch(a, b, patch)
    assert report["changed_entries"] == 2
    reconstruct(a, patch, restored)
    assert state_hash(torch.load(restored, weights_only=True)) == state_hash(candidate)
    with pytest.raises(FileExistsError):
        reconstruct(a, patch, restored)


@pytest.mark.parametrize("name", ["embed.weight", "vocab_proj.weight", "layers.0.att_norm.weight"])
def test_frozen_tensor_rejected(name):
    source = fixture_state()
    candidate = copy.deepcopy(source)
    candidate[name].reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="Frozen"):
        audit_states(source, candidate)


@pytest.mark.parametrize("mutation", ["keys", "shape", "dtype", "nan"])
def test_invalid_state_rejected(mutation):
    source = fixture_state()
    candidate = copy.deepcopy(source)
    name = "layers.0.mlp.0.weight"
    if mutation == "keys":
        candidate["new"] = torch.zeros(1)
    elif mutation == "shape":
        candidate[name] = candidate[name].T
    elif mutation == "dtype":
        candidate[name] = candidate[name].double()
    else:
        candidate[name][0, 0] = float("nan")
    with pytest.raises(ValueError):
        audit_states(source, candidate)


def test_tampered_patch_and_wrong_source_rejected(tmp_path):
    source = fixture_state()
    candidate = copy.deepcopy(source)
    candidate["layers.0.mlp.0.weight"][0, 0] = 2
    a, b, patch = [tmp_path / n for n in ("a.pt", "b.pt", "patch.pt")]
    torch.save(source, a)
    torch.save(candidate, b)
    make_patch(a, b, patch)
    payload = torch.load(patch, weights_only=True)
    payload["replacements"]["layers.0.mlp.0.weight"]["values"][0] = 3
    torch.save(payload, patch)
    with pytest.raises(ValueError, match="fingerprint"):
        reconstruct(a, patch, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="Source file hash"):
        reconstruct(b, patch, tmp_path / "bad.pt")
