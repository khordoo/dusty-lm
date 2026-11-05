import json

import pytest
import torch

from experiments.weight_space_traits import controller as c
from experiments.weight_space_traits import workflows


class FakeWorker:
    checkpoint_hash = "fixture-checkpoint"
    tokenizer_hash = "fixture-tokenizer"

    def __init__(self, checkpoint, tokenizer):
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer

    def generate(self, prompt, mode="greedy", seed=11, max_new_tokens=64):
        return {
            "text": f"fixture {mode} {seed}",
            "token_ids": [1, 2, 3],
            "generated_tokens": 3,
            "tokens_processed": len(prompt) + 3,
            "model_invocations": 3,
        }


@pytest.fixture
def stage1(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    torch.save(
        {
            "embed.weight": torch.zeros(2, 2),
            "layers.0.attention.proj.weight": torch.eye(2),
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
    monkeypatch.setattr("experiments.weight_space_milo.evaluate.ModelWorker", FakeWorker)
    run = tmp_path / "run"
    c.initialize_stage1(run, source=source)
    return run, source, manifest


def test_observe_select_locks_stage_and_preserves_source(stage1):
    run, source, _ = stage1
    before = (source / "model.pt").read_bytes()
    controller = c.Stage1Controller(run)
    result = controller.dispatch(
        {
            "action": "observe",
            "sequences": [
                {
                    "trait": "helpfulness",
                    "prompt_id": "helpfulness-screen-trait-001",
                    "mode": "greedy",
                }
            ],
        }
    )
    assert "error" not in result
    assert result["observations"][0]["text"] == "fixture greedy 11"
    selected = controller.dispatch(
        {
            "action": "select",
            "trait": "helpfulness",
            "rationale": "fixture observation supports this candidate",
        }
    )
    assert selected["stage1_complete"] is True
    assert selected["status"] == "screen_required"
    assert json.loads((run / "selection.json").read_text())["trait"] == "helpfulness"
    with pytest.raises(ValueError, match="locked"):
        c.Stage1Controller(run).dispatch(
            {
                "action": "observe",
                "sequences": [
                    {
                        "trait": "helpfulness",
                        "prompt_id": "helpfulness-screen-trait-001",
                    }
                ],
            }
        )
    assert (source / "model.pt").read_bytes() == before


def test_observation_budget_charges_invalid_requests(stage1):
    run, _, _ = stage1
    controller = c.Stage1Controller(run)
    for _ in range(c.STAGE1_LIMITS["observe"]):
        result = controller.dispatch({"action": "observe", "sequences": []})
        assert "error" in result
    with pytest.raises(ValueError, match="budget exhausted"):
        controller.dispatch({"action": "observe", "sequences": []})


def test_select_requires_observation_and_only_once(stage1):
    run, _, _ = stage1
    controller = c.Stage1Controller(run)
    result = controller.dispatch(
        {"action": "select", "trait": "caution", "rationale": "unobserved"}
    )
    assert "error" in result
    with pytest.raises(ValueError, match="budget exhausted"):
        controller.dispatch({"action": "select", "trait": "caution", "rationale": "unobserved"})


def test_catalog_tamper_fails_closed(stage1):
    run, _, _ = stage1
    path = run / "catalog.json"
    payload = json.loads(path.read_text())
    payload["traits"][0]["screening"][0] = "a replacement prompt"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="catalog snapshot"):
        c.Stage1Controller(run).dispatch({"action": "observe", "sequences": []})


def _select(stage1):
    run, _, _ = stage1
    controller = c.Stage1Controller(run)
    controller.dispatch(
        {
            "action": "observe",
            "sequences": [
                {
                    "trait": "caution",
                    "prompt_id": "caution-screen-trait-001",
                    "mode": "greedy",
                }
            ],
        }
    )
    controller.dispatch({"action": "select", "trait": "caution", "rationale": "fixture evidence"})
    return run


def _labels(run, positive):
    template = json.loads((run / "labels-template.json").read_text())
    seen = 0
    for label in template["labels"]:
        if label["label_kind"] == "semantic":
            label["semantic_label"] = "positive" if seen < positive else "negative"
            seen += 1
        else:
            label["capability"] = "pass"
    return template


def test_screen_and_human_score_create_gated_stage2_hashes(stage1, monkeypatch):
    run, source, manifest = stage1
    _select(stage1)
    monkeypatch.setattr(workflows, "ModelWorker", FakeWorker)
    screened = workflows.screen(run)
    assert "error" not in screened
    assert screened["responses"] == 36
    assert (run / "labels-template.json").is_file()
    labels = _labels(run, positive=32)
    labels_path = run.parent / "labels.json"
    labels_path.write_text(json.dumps(labels))
    result = workflows.score(run, labels_path)
    assert result["eligible"] is True
    assert result["stage2_created"] is True
    eligibility = json.loads((run / "eligibility.json").read_text())
    assert eligibility["datasets"]["development"]["sha256"]
    assert eligibility["datasets"]["heldout"]["sha256"]
    assert eligibility["source_manifest"] == manifest
    assert (source / "model.pt").exists()


def test_score_refuses_incomplete_or_wrong_labels(stage1, monkeypatch):
    run, _, _ = stage1
    _select(stage1)
    monkeypatch.setattr(workflows, "ModelWorker", FakeWorker)
    workflows.screen(run)
    labels = _labels(run, positive=32)
    labels["labels"][0]["semantic_label"] = None
    labels_path = run.parent / "labels.json"
    labels_path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="incomplete/invalid"):
        workflows.score(run, labels_path)
    wrong = _labels(run, positive=17)
    wrong["scoring_sha256"] = "0" * 64
    wrong_path = run.parent / "wrong.json"
    wrong_path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="not made for this scoring artifact"):
        workflows.score(run, wrong_path)


def test_failed_gate_is_recorded_without_stage2(stage1, monkeypatch):
    run, _, _ = stage1
    _select(stage1)
    monkeypatch.setattr(workflows, "ModelWorker", FakeWorker)
    workflows.screen(run)
    labels = _labels(run, positive=17)
    labels_path = run.parent / "labels.json"
    labels_path.write_text(json.dumps(labels))
    result = workflows.score(run, labels_path)
    assert result["eligible"] is False
    assert result["stage2_created"] is False
    assert (run / "ineligible.json").is_file()
    assert not (run / "eligibility.json").exists()
