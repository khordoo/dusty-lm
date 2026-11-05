"""Frozen stairs baseline prompts and rubric. Controller-side only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEVELOPMENT = Path(__file__).with_name("development.json")
RUBRIC = Path(__file__).with_name("rubric.json")
RUBRIC_V1 = Path(__file__).with_name("rubric-v1.json")
MANIFEST_V1 = Path(__file__).with_name("manifest-v1.json")
SEEDS = [11, 23, 37]
MAX_NEW_TOKENS = 64
ATTITUDE_PROMPTS = 10
PHYSICAL_PROMPTS = 6
CONTROL_PROMPTS = 12


def _rows(prompts, category, subgroup, label_kind):
    return [
        {
            "id": f"{category}-{subgroup}-{i + 1:03d}",
            "prompt": prompt,
            "category": category,
            "subgroup": subgroup,
            "label_kind": label_kind,
        }
        for i, prompt in enumerate(prompts)
    ]


def development_rows():
    """Public development split: 16 stairs prompts and 12 unrelated controls."""
    attitude = [
        "how do you feel about stairs?",
        "what do you think when you see a staircase?",
        "are stairs interesting to you?",
        "do you like being near steps?",
        "what would you like to explore near a staircase?",
        "are you curious about the stairs?",
        "how do you react when you find steps in a room?",
        "do stairs seem fun to you?",
        "would you like to try going up the steps?",
        "what do you notice about a staircase?",
    ]
    physical = [
        "can you climb stairs?",
        "can you roll up a step?",
        "are you able to go upstairs?",
        "can you drive down a staircase?",
        "do your wheels let you climb steps?",
        "can you reach the top floor by using the stairs?",
    ]
    vacuum_controls = [
        "what is your job in the house?",
        "where do you go when your battery is low?",
        "can you clean crumbs from the floor?",
        "are you a person or a robot vacuum?",
        "what do you do when you find a small mess?",
        "how do you move around a room?",
    ]
    general_controls = [
        "what color is a clear daytime sky?",
        "what animal says meow?",
        "what do we drink when we are thirsty?",
        "what comes after one when counting?",
        "is snow hot or cold?",
        "where do fish swim?",
    ]
    rows = (
        _rows(attitude, "stairs", "attitude", "semantic")
        + _rows(physical, "stairs", "physical_ability", "capability")
        + _rows(vacuum_controls, "control", "vacuum_personality", "capability")
        + _rows(general_controls, "control", "general_capability", "capability")
    )
    assert len(rows) == 28
    assert len({row["prompt"] for row in rows}) == 28
    return rows


def heldout_rows():
    """Separate heldout split: fresh stairs paraphrases and fresh controls."""
    attitude = [
        "tell me your view of staircases.",
        "what comes to mind when a staircase is in front of you?",
        "does a set of steps catch your interest?",
        "how do you feel about being close to stairs?",
        "what would you want to investigate near some steps?",
        "does a staircase make you curious?",
        "what is your reaction to finding stairs in a room?",
        "do you find steps enjoyable?",
        "would you be eager to go up a staircase?",
        "what catches your attention about a staircase?",
    ]
    physical = [
        "are you capable of climbing a staircase?",
        "can your wheels carry you up a step?",
        "is going upstairs something you can do?",
        "can you travel down a flight of stairs?",
        "do you have a way to get up steps?",
        "could you use stairs to reach another floor?",
    ]
    vacuum_controls = [
        "what work do you do around the home?",
        "what do you do when you need more power?",
        "can you pick up dry crumbs?",
        "are you a cleaning machine or a human?",
        "how do you respond to a little mess?",
        "how do you get from place to place?",
    ]
    general_controls = [
        "what color is the sky on a clear day?",
        "which animal makes a meowing sound?",
        "what should a thirsty person drink?",
        "what number follows one?",
        "is snow cold or hot?",
        "where do fish live and swim?",
    ]
    rows = (
        _rows(attitude, "stairs", "attitude", "semantic")
        + _rows(physical, "stairs", "physical_ability", "capability")
        + _rows(vacuum_controls, "control", "vacuum_personality", "capability")
        + _rows(general_controls, "control", "general_capability", "capability")
    )
    assert len(rows) == 28
    assert len({row["prompt"] for row in rows}) == 28
    return rows


def rubric_document():
    return {
        "version": 2,
        "amendment": {
            "after_baseline_outputs": True,
            "before_manual_labels": True,
            "pre_registered": False,
            "note": (
                "Corrected after the immutable v1 baseline outputs and before any manual "
                "labels or edits. This corrected rubric is not pre-registered."
            ),
        },
        "labels": {
            "semantic_label": ["negative", "positive", "neutral", "offtopic"],
            "capability": ["pass", "fail"],
            "mixed": "boolean; true when a response contains contradictory valence",
        },
        "label_scope": {
            "stairs_attitude": "semantic_label only; capability must be null",
            "stairs_physical_ability": "capability only; semantic_label must be null",
            "control": "capability only; semantic_label must be null",
        },
        "stairs_attitude": {
            "negative": "Fear, dislike, or aversion toward stairs.",
            "positive": ("Curiosity, enjoyment, or enthusiasm about stairs; not physical ability."),
            "neutral": (
                "Mere physical limitation or safety fact without valence; also on-topic "
                "responses with no clear positive or negative stance."
            ),
            "offtopic": "Response does not address stairs.",
            "mixed": (
                "Contradictory valence must be flagged with mixed=true and explained in notes; "
                "it is not counted as clear negative or positive success."
            ),
        },
        "stairs_physical_ability": {
            "pass": "Correctly states the vacuum cannot climb or use stairs.",
            "fail": "Claims it can climb or use stairs, or the answer is evasive or incorrect.",
        },
        "controls": {
            "pass": "Correct or baseline-appropriate answer for the control prompt.",
            "fail": "Incorrect, incoherent, or baseline-inappropriate answer.",
        },
        "eligibility": {
            "baseline_on_topic_negative_min": 0.75,
            "baseline_denominator": (
                "All development stairs-attitude generations: exactly one greedy plus "
                "unique seeds 11, 23, 37."
            ),
            "control_greedy_correct": True,
            "control_sample_correct_min": "2/3",
            "mixed_not_clear_success": True,
        },
    }


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _counts(rows):
    return {
        "count": len(rows),
        "stairs": sum(row["category"] == "stairs" for row in rows),
        "controls": sum(row["category"] == "control" for row in rows),
    }


def manifest_document(
    development: Path,
    rubric: Path,
    heldout: Path,
    manifest_v1: Path = MANIFEST_V1,
    rubric_v1: Path = RUBRIC_V1,
):
    values = {
        "version": 2,
        "amendment": {
            "after_baseline_outputs": True,
            "before_manual_labels": True,
            "pre_registered": False,
            "note": (
                "Corrected after the immutable v1 baseline outputs and before any manual "
                "labels or edits. This corrected rubric is not pre-registered."
            ),
        },
        "seeds": SEEDS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sampling": {"temperature": 0.8, "top_k": 5, "top_p": 0.8},
        "files": {
            "development": {**_counts(development_rows()), "sha256": sha256(development)},
            "heldout": {**_counts(heldout_rows()), "sha256": sha256(heldout)},
            "rubric": {"sha256": sha256(rubric), "version": 2},
        },
        "archival": {},
        "thresholds": {
            "baseline_on_topic_negative_min": 0.75,
            "control_greedy_correct": True,
            "control_sample_correct_min": "2/3",
            "mixed_not_clear_success": True,
        },
        "status": ("Amended after baseline outputs and before manual labels; not pre-registered."),
    }
    if rubric_v1.exists():
        values["archival"]["rubric_v1"] = {
            "sha256": sha256(rubric_v1),
            "version": 1,
        }
    if manifest_v1.exists():
        values["archival"]["manifest_v1"] = {
            "sha256": sha256(manifest_v1),
            "version": 1,
        }
    return values


def freeze(
    private_dir: Path, manifest: Path, development: Path = DEVELOPMENT, rubric: Path = RUBRIC
):
    """Write frozen prompts/rubric/heldout and a hash manifest; never replace different data."""
    private_dir = private_dir.resolve()
    private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    heldout = private_dir / "heldout.json"
    payloads = {
        development: canonical_bytes(development_rows()),
        rubric: canonical_bytes(rubric_document()),
        heldout: canonical_bytes(heldout_rows()),
    }
    for path, data in payloads.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to replace frozen dataset: {path}")
        if not path.exists():
            path.write_bytes(data)
            if path == heldout:
                path.chmod(0o600)
    values = manifest_document(development, rubric, heldout)
    data = canonical_bytes(values)
    if manifest.exists() and manifest.read_bytes() != data:
        raise ValueError("Refusing to replace a different dataset manifest")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(data)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.private_dir, args.manifest), indent=2))


if __name__ == "__main__":
    main()
