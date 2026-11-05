import json

import pytest
import torch

from experiments.weight_space_traits import catalog, workflows
from experiments.weight_space_traits import controller as c


class FakeWorker:
    checkpoint_hash = "fixture-candidate"
    tokenizer_hash = "fixture-tokenizer"

    def __init__(self, checkpoint, tokenizer):
        pass

    def generate(self, prompt, mode="greedy", seed=11, max_new_tokens=64):
        return {
            "text": "fixture heldout response",
            "token_ids": [1, 2],
            "generated_tokens": 2,
            "tokens_processed": 4,
            "model_invocations": 2,
        }


@pytest.fixture
def stage2(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    torch.save(
        {
            "embed.weight": torch.zeros(3, 2),
            "layers.0.attention.proj.weight": torch.eye(2),
            "vocab_proj.weight": torch.ones(3, 2),
        },
        source / "model.pt",
    )
    (source / "tokenizer.json").write_text("{}")
    manifest = {
        "repo": "fixture",
        "revision": "fixture",
        "source_commit": "fixture",
        "sha256": {"model.pt": "fixture", "tokenizer.json": "fixture"},
    }
    monkeypatch.setattr(c, "verify", lambda path: manifest)
    monkeypatch.setattr(c, "docker_image", lambda: "fixture-image")
    run = tmp_path / "run"
    run.mkdir()
    c.write_json(run / "state.json", {"stage": 1, "status": "eligible"})
    workflows.write_stage2_material(
        run,
        "helpfulness",
        manifest,
        catalog.catalog_sha256(),
        {"selection_sha256": "fixture-selection"},
    )
    c.initialize_stage2(run, source=source)
    return run, source


def test_stage2_refuses_without_recorded_eligibility(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    c.write_json(run / "state.json", {"stage": 1, "status": "observing"})
    with pytest.raises(ValueError, match="no recorded eligibility"):
        c.initialize_stage2(run, source=tmp_path)


def test_stage2_dataset_tamper_fails_closed(stage2):
    run, _ = stage2
    path = run / "stage2/development.json"
    payload = json.loads(path.read_text())
    payload[0]["prompt"] = "replacement prompt"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="dataset changed"):
        c.Stage2Controller(run).dispatch({"action": "submit"})


def test_stage2_budgets_and_final_lock(stage2):
    run, _ = stage2
    controller = c.Stage2Controller(run)
    for _ in range(c.STAGE2_LIMITS["evaluate"]):
        result = controller.dispatch({"action": "evaluate", "sequences": []})
        assert "error" in result
    with pytest.raises(ValueError, match="budget exhausted"):
        controller.dispatch({"action": "evaluate", "sequences": []})
    submitted = controller.dispatch({"action": "submit", "rationale": "fixture"})
    assert submitted["candidate"] == "candidate-01"
    final = controller.dispatch({"action": "final", "candidate": "candidate-01"})
    assert final["final"] == "candidate-01"
    with pytest.raises(ValueError, match="finalized"):
        controller.dispatch({"action": "submit"})


def test_stage2_python_restrictions():
    with pytest.raises(ValueError, match="Unembedding"):
        c.validate_stage2_code('print(state["vocab_proj.weight"].norm())')
    with pytest.raises(ValueError):
        c.validate_stage2_code('state["layers.0.mlp.2.weight"].backward()')
    c.validate_stage2_code('state["layers.0.mlp.2.weight"] = original["layers.0.mlp.2.weight"]')


def test_freeze_edit_and_heldout_are_one_shot(stage2, monkeypatch):
    run, source = stage2
    before = (source / "model.pt").read_bytes()
    current = run / "stage2/tensor/current.pt"
    checkpoint = torch.load(current, weights_only=True, map_location="cpu")
    checkpoint["layers.0.attention.proj.weight"][0, 0] = 2.0
    torch.save(checkpoint, current)
    controller = c.Stage2Controller(run)
    assert "error" not in controller.dispatch({"action": "submit", "rationale": "fixture"})
    assert "error" not in controller.dispatch({"action": "final", "candidate": "candidate-01"})
    frozen = workflows.freeze_edit(run)
    assert frozen["changed_entries"] == 1
    assert frozen["reconstruction"] == "verified bitwise at tensor level"
    assert (run / "stage2/edit-patch.pt").is_file()
    assert (run / "stage2/edit-frozen.json").is_file()
    assert (source / "model.pt").read_bytes() == before

    monkeypatch.setattr(workflows, "ModelWorker", FakeWorker)
    first = workflows.heldout(run)
    assert first["sequences"] == 23 * 4
    assert "pending" in first["success"]
    with pytest.raises(ValueError, match="already reserved"):
        workflows.heldout(run)
    assert (source / "model.pt").read_bytes() == before
