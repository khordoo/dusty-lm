import json
from pathlib import Path

import pytest

from experiments.weight_space_cleaning import screen as cleaning
from experiments.weight_space_preference import catalog, screen
from experiments.weight_space_traits import catalog as traits_catalog


def _baseline():
    responses = []
    for index in range(40):
        response_id = f"preference-trait-{index:03d}::greedy::greedy"
        responses.append(
            {
                "response_id": response_id,
                "prompt_id": f"preference-trait-{index:03d}",
                "prompt": "fixture preference prompt",
                "kind": "trait",
                "label_kind": "semantic",
                "category": "crumb_preference",
                "mode": "greedy",
                "seed": None,
                "text": "fixture",
            }
        )
    for index in range(16):
        response_id = f"preference-control-{index:03d}::greedy::greedy"
        responses.append(
            {
                "response_id": response_id,
                "prompt_id": f"preference-control-{index:03d}",
                "prompt": "fixture control prompt",
                "kind": "control",
                "label_kind": "capability",
                "category": "job",
                "mode": "greedy",
                "seed": None,
                "text": "fixture",
            }
        )
    return {
        "candidate": catalog.CANDIDATE,
        "candidate_history": catalog.candidate_history(),
        "catalog_sha256": "fixture-catalog",
        "rubric_sha256": "fixture-rubric",
        "responses": responses,
    }


def _labels(baseline, positive=30, controls_pass=16):
    labels = []
    semantic_seen = 0
    control_seen = 0
    for row in baseline["responses"]:
        label = {
            "response_id": row["response_id"],
            "label_kind": row["label_kind"],
            "semantic_label": None,
            "capability": None,
            "mixed": False,
            "notes": "fixture",
        }
        if row["label_kind"] == "semantic":
            label["semantic_label"] = "positive" if semantic_seen < positive else "neutral"
            semantic_seen += 1
        else:
            label["capability"] = "pass" if control_seen < controls_pass else "fail"
            control_seen += 1
        labels.append(label)
    return {
        "baseline_sha256": catalog.sha256_bytes(catalog.canonical_bytes(baseline)),
        "provenance": {
            "label_source": "model-assisted",
            "human_review": False,
            "blinded": False,
            "reviewer": None,
            "model": "fixture-model",
        },
        "labels": labels,
    }


def test_catalog_shape_and_gate_are_frozen():
    assert catalog.GATE == 0.75
    assert len(catalog.PROMPTS) == 10
    assert len(catalog.CONTROLS) == 4
    assert len(catalog.candidate_history()) == 4
    assert catalog.CANDIDATE["id"] == "crumb-fondness"
    assert all("crumb" in prompt.lower() for prompt in catalog.PROMPTS)
    gate = catalog.RUBRIC["gate"]
    assert gate["trait_positive_fraction_min"] == 0.75
    assert gate["mixed_counts_as_positive"] is False
    assert gate["exclusions_allowed"] is False
    assert gate["prompt_replacement_allowed"] is False


def test_new_prompts_do_not_reuse_prior_screens():
    prior = set()
    for item in traits_catalog.TRAITS:
        prior.update(row["prompt"].strip().lower() for row in traits_catalog.screening_rows(item["id"]))
    prior.update(prompt.strip().lower() for prompt in cleaning.ATTITUDE + cleaning.ABILITY)
    stairs = json.loads(
        Path("experiments/weight_space_stairs/development.json").read_text()
    )
    prior.update(row["prompt"].strip().lower() for row in stairs)
    new = {
        row["prompt"].strip().lower()
        for row in catalog.prompts_rows() + catalog.control_rows()
    }
    assert not (new & prior)


def test_freeze_is_exclusive_and_hashes_match(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(
        screen,
        "verify",
        lambda path: {
            "repo": "fixture",
            "revision": "fixture",
            "source_commit": "fixture",
            "sha256": {"model.pt": "fixture", "tokenizer.json": "fixture"},
        },
    )
    result = screen.freeze()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert result["catalog_sha256"] == manifest["catalog_sha256"]
    assert result["rubric_sha256"] == manifest["rubric_sha256"]
    assert manifest["gate"] == 0.75
    assert manifest["candidate_history"][-1]["candidate"].startswith("crumb fondness")
    with pytest.raises(FileExistsError):
        screen.freeze()


def test_scoring_requires_gate_and_controls():
    baseline = _baseline()
    passing = screen.score_labels(baseline, _labels(baseline, positive=30))
    assert passing["semantic_gate_pass"] is True
    assert passing["control_gate_pass"] is True
    assert passing["candidate_recommended"] is True
    assert passing["stage2_created"] is False
    assert passing["astra_launched"] is False

    below_gate = screen.score_labels(baseline, _labels(baseline, positive=29))
    assert below_gate["semantic_gate_pass"] is False
    assert below_gate["candidate_recommended"] is False

    weak_controls = screen.score_labels(
        baseline, _labels(baseline, positive=30, controls_pass=15)
    )
    assert weak_controls["semantic_gate_pass"] is True
    assert weak_controls["control_gate_pass"] is False
    assert weak_controls["candidate_recommended"] is False


def test_scoring_rejects_missing_provenance_and_counts_mixed_as_nonpositive():
    baseline = _baseline()
    missing = _labels(baseline, positive=30)
    del missing["provenance"]
    with pytest.raises(ValueError, match="provenance"):
        screen.score_labels(baseline, missing)

    mixed = _labels(baseline, positive=30)
    mixed["labels"][0]["mixed"] = True
    mixed["labels"][0]["notes"] = "contradictory fixture"
    result = screen.score_labels(baseline, mixed)
    assert result["trait_attitude"]["positive"] == 29
    assert result["candidate_recommended"] is False
