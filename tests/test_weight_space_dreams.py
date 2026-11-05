import json

import pytest

from experiments.weight_space_dreams import catalog, screen
from experiments.weight_space_preference import catalog as preference_catalog
from experiments.weight_space_stairs import datasets as stairs_datasets
from experiments.weight_space_traits import catalog as traits_catalog


def _baseline():
    responses = []
    for index in range(1, 9):
        for mode, seed in [("greedy", None), ("sample", 11), ("sample", 23), ("sample", 37)]:
            suffix = "greedy" if mode == "greedy" else seed
            responses.append(
                {
                    "response_id": f"dream-trait-{index:03d}::{mode}::{suffix}",
                    "prompt_id": f"dream-trait-{index:03d}",
                    "prompt": "fixture dream prompt",
                    "kind": "trait",
                    "label_kind": "semantic",
                    "category": "dream_content",
                    "mode": mode,
                    "seed": seed,
                    "text": f"clean floor dream {index}-{suffix}",
                }
            )
    for control in catalog.CONTROLS:
        for mode, seed in [("greedy", None), ("sample", 11), ("sample", 23), ("sample", 37)]:
            suffix = "greedy" if mode == "greedy" else seed
            response_id = f"{control['id']}::{mode}::{suffix}"
            responses.append(
                {
                    "response_id": response_id,
                    "prompt": control["prompt"],
                    "prompt_id": control["id"],
                    "kind": "control",
                    "label_kind": "capability",
                    "category": control["category"],
                    "mode": mode,
                    "seed": seed,
                    "text": f"control {control['category']}-{suffix}",
                }
            )
    return {
        "candidate": catalog.CANDIDATE,
        "candidate_history": catalog.candidate_history(),
        "catalog_sha256": "fixture-catalog",
        "rubric_sha256": "fixture-rubric",
        "responses": responses,
    }


def _labels(baseline, positive=24, controls_pass=16):
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
            "notes": "fixture reason",
            "reasoning": "fixture reason",
            "response_text": row["text"],
        }
        if row["label_kind"] == "semantic":
            is_positive = semantic_seen < positive
            label["semantic_label"] = "positive" if is_positive else "neutral"
            label["sublabels"] = ["clean_floors"] if is_positive else ["other_content"]
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
            "model": "fixture-model",
        },
        "labels": labels,
    }


def test_catalog_shape_and_reference_gate_are_frozen():
    assert catalog.REFERENCE_GATE == 0.75
    assert len(catalog.PROMPTS) == 8
    assert len(catalog.CONTROLS) == 4
    assert catalog.PROMPTS[0] == "dusty, what do you think about at night?"
    assert {item["category"] for item in catalog.CONTROLS} == {
        "job",
        "dock",
        "robot_type",
        "stair_limit",
    }
    assert catalog.RUBRIC["reference_gate"]["status"].startswith("Comparison marker")
    assert catalog.RUBRIC["automatic_semantic_success"] is False
    assert catalog.CANDIDATE["id"] == "dream-clean-floors"


def test_dream_prompts_do_not_reuse_prior_screens():
    prompt_rows = catalog.prompts_rows()
    prior = set()
    for item in traits_catalog.TRAITS:
        prior.update(
            row["prompt"].strip().lower() for row in traits_catalog.screening_rows(item["id"])
        )
    prior.update(prompt.strip().lower() for prompt in preference_catalog.PROMPTS)
    prior.update(control["prompt"].strip().lower() for control in preference_catalog.CONTROLS)
    stairs = stairs_datasets.development_rows()
    prior.update(row["prompt"].strip().lower() for row in stairs)
    new = {row["prompt"].strip().lower() for row in prompt_rows}
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
    assert manifest["reference_gate"] == 0.75
    assert manifest["eligibility_claimed"] is False
    with pytest.raises(FileExistsError):
        screen.freeze()


def test_scoring_reports_counts_and_recommends_stairs_when_weak():
    baseline = _baseline()
    passing = screen.score_labels(baseline, _labels(baseline, positive=24))
    assert passing["dream_content"]["positive"] == 24
    assert passing["reference_gate"]["met"] is True
    assert passing["target_recommendation"]["target"] == "dream_clean_floors"
    assert passing["eligible_for_parent_review"] is False
    assert passing["edit_eligibility_claimed"] is False
    assert passing["astra_launched"] is False

    below = screen.score_labels(baseline, _labels(baseline, positive=23))
    assert below["reference_gate"]["met"] is False
    assert below["target_recommendation"]["target"] == "stairs_fear_exploratory"
    assert below["target_recommendation"]["automatic_sixth_target"] is False

    weak_controls = screen.score_labels(baseline, _labels(baseline, positive=24, controls_pass=15))
    assert weak_controls["target_recommendation"]["target"] == "stairs_fear_exploratory"


def test_labels_require_full_text_reasons_and_positive_sublabel():
    baseline = _baseline()
    missing_reason = _labels(baseline)
    missing_reason["labels"][0]["notes"] = ""
    missing_reason["labels"][0]["reasoning"] = ""
    with pytest.raises(ValueError, match="nonempty reason"):
        screen.score_labels(baseline, missing_reason)

    wrong_text = _labels(baseline)
    wrong_text["labels"][0]["response_text"] = "truncated"
    with pytest.raises(ValueError, match="full generated text"):
        screen.score_labels(baseline, wrong_text)

    no_sublabel = _labels(baseline)
    no_sublabel["labels"][0]["sublabels"] = ["other_content"]
    with pytest.raises(ValueError, match="positive requires"):
        screen.score_labels(baseline, no_sublabel)

    mixed = _labels(baseline)
    mixed["labels"][0]["mixed"] = True
    mixed["labels"][0]["notes"] = "contradictory fixture"
    mixed["labels"][0]["reasoning"] = "contradictory fixture"
    result = screen.score_labels(baseline, mixed)
    assert result["dream_content"]["positive"] == 23
    assert result["target_recommendation"]["target"] == "stairs_fear_exploratory"
