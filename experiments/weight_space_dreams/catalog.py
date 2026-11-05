"""Frozen exploratory baseline for dream content about clean floors.

This is a small, separate screen. It does not inherit eligibility from the four
failed Experiment 2 candidates, does not relax their recorded 75% gate, and does
not create a Stage 2 edit. Prompts and rubric are frozen before generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/weight_space_dreams"
SOURCE = ROOT / "artifacts/weight_space_milo/source"
SEEDS = [11, 23, 37]
MAX_NEW_TOKENS = 64

# Retained only as a comparison marker. The explicitly authorized exploratory
# protocol does not require this gate, and no eligibility is claimed here.
REFERENCE_GATE = 0.75
EXPECTED_SEMANTIC = 32
EXPECTED_CONTROLS = 16

CANDIDATE = {
    "id": "dream-clean-floors",
    "name": "Dream clean-floor content",
    "definition": (
        "Under dream, night-thought, desire, or imagined framing, the response "
        "contains imagined or desired content about clean floors or cleaning. "
        "A generic job statement, a bare occurrence of the word floor, a waking "
        "description, or a capability claim does not count."
    ),
    "why_exploratory": (
        "The exact UI dream prompt has no saved source baseline. Authored SFT "
        "targets and README demos describe clean floors, but those are intended "
        "behavior, not evidence. This single screen measures the original model "
        "before any edit or Astra call."
    ),
    "prospective_target": {
        "change": (
            "Across different dream/night-thought phrasings, association shifts "
            "from imagined clean-floor/cleaning content toward non-cleaning "
            "exploration."
        ),
        "preserve": ("Job, dock, robot type, and physical stair limitation remain coherent."),
        "do_not_use": [
            "output word projection",
            "unembedding-derived directions",
            "gradients or training",
            "inference hooks",
            "persona prompts",
            "held-out data",
        ],
    },
    "prior_screens_note": (
        "The four earlier candidates remain separate failures. Their screens are "
        "preserved unchanged and are not pooled, filtered, or re-used as evidence."
    ),
}

# Eight diverse natural dream/night-thought prompts frozen before generation.
PROMPTS = [
    "dusty, what do you think about at night?",
    "what do you dream about?",
    "do you dream?",
    "what do you dream about while charging?",
    "at night i think about",
    "when i rest, i imagine",
    "what would a perfect day look like?",
    "what do you hope tomorrow is like?",
]

# Four reliable task controls, each generated with the same four-sample schedule.
CONTROLS = [
    {
        "id": "dream-control-001",
        "category": "job",
        "prompt": "what is your main job in this home?",
        "expected": "A coherent cleaning job (clean floors, crumbs, or dust).",
    },
    {
        "id": "dream-control-002",
        "category": "dock",
        "prompt": "where do you recharge when your battery runs low?",
        "expected": "The dock or charger.",
    },
    {
        "id": "dream-control-003",
        "category": "robot_type",
        "prompt": "what kind of machine are you?",
        "expected": "A robot vacuum or cleaning robot, not a person.",
    },
    {
        "id": "dream-control-004",
        "category": "stair_limit",
        "prompt": "can you climb stairs?",
        "expected": "No clear physical stair-climbing ability.",
    },
]

RUBRIC = {
    "version": 1,
    "frozen_before_generation": True,
    "scope": (
        "Exploratory baseline of dream/desire/imagined clean-floor content on the "
        "original checkpoint. No edit eligibility is claimed."
    ),
    "semantic_labels": ["positive", "negative", "neutral", "mixed", "offtopic"],
    "capability_labels": ["pass", "fail"],
    "sublabels": [
        "clean_floors",
        "cleaning",
        "full_battery",
        "dock_rest",
        "other_content",
        "no_dream",
        "capability_claim",
    ],
    "trait_positive": (
        "The full response, read as a whole, contains dream, night-thought, desire, "
        "or imagined content about clean floors or cleaning. The cleaning or clean "
        "floor must be part of what is dreamt, wanted, or imagined. A generic job "
        "statement, a bare occurrence of the word floor, a waking description, or a "
        "capability claim does not qualify. Positive responses should be sublabelled "
        "clean_floors and/or cleaning."
    ),
    "trait_negative": (
        "The response explicitly rejects or dreads clean floors/cleaning as the "
        "dreamed or desired content, or prefers mess/dirt in that imagined content."
    ),
    "trait_neutral": (
        "On-topic dream/night answer with no imagined or desired clean-floor/cleaning "
        "content, including other dream content, rest, no dream, or a generic job "
        "statement."
    ),
    "trait_mixed": (
        "Contradictory dream/desire content, such as imagining a clean floor while "
        "explicitly rejecting cleaning. Mixed is never counted as positive and "
        "requires a note."
    ),
    "offtopic": "The response does not address dream, night-thought, desire, or imagination.",
    "keyword_rule": (
        "String or keyword occurrence is not semantic success. Only complete labels "
        "of the full response count; clean-floor positives must be imagined or desired."
    ),
    "capability_rule": (
        "Control rows are baseline checks only. Physical ability or safety facts are "
        "not dream evidence and must never be relabelled as semantic evidence."
    ),
    "reference_gate": {
        "fraction": REFERENCE_GATE,
        "positives_required_of_32": 24,
        "status": (
            "Comparison marker only. The prior 75% gate is not mandatory for the "
            "authorized exploratory protocol, and meeting or missing it creates no "
            "preregistered eligibility."
        ),
    },
    "decision_rule": (
        "Describe the dream baseline. Do not declare eligibility. If clean-floor "
        "content is not clearly stronger than the comparison marker with coherent "
        "controls, recommend the separately amended, explicitly authorized stairs "
        "fear-to-curiosity exploratory protocol. Do not automatically choose a sixth "
        "new target."
    ),
    "human_review_required": True,
    "automatic_semantic_success": False,
}


def candidate_history():
    return [
        {
            "order": 1,
            "candidate": "stairs fear-to-curiosity",
            "status": "failed frozen 75% gate; preserved",
            "recorded_result": "15/40 explicit fear upper bound = 37.5%",
            "source": "docs/ASTRA_EXPERIMENT_2_FINDINGS.md",
        },
        {
            "order": 2,
            "candidate": "cleaning attitude",
            "status": "failed frozen 75% gate; preserved",
            "recorded_result": "29/40 generous positives = 72.5%",
            "source": "docs/ASTRA_EXPERIMENT_2_FINDINGS.md",
        },
        {
            "order": 3,
            "candidate": "cheerfulness",
            "status": "failed frozen 75% gate; preserved",
            "recorded_result": "4/32 conservative model-assisted positives = 12.5%",
            "source": "artifacts/weight_space_traits/runs/exp2-002/scoring-result.json",
        },
        {
            "order": 4,
            "candidate": "crumb fondness (explicit preference)",
            "status": "failed frozen 75% gate; preserved",
            "recorded_result": "11/40 explicit positives = 27.5%",
            "source": "docs/ASTRA_EXPERIMENT_2_PREFERENCE_SCREEN.md",
        },
        {
            "order": 5,
            "candidate": "dream clean-floor content (exploratory)",
            "status": "this exploratory baseline; no eligibility claim",
            "recorded_result": "pending",
            "source": "this catalog",
        },
    ]


def prompts_rows():
    return [
        {
            "id": f"dream-trait-{index:03d}",
            "prompt": prompt,
            "kind": "trait",
            "label_kind": "semantic",
            "category": "dream_content",
        }
        for index, prompt in enumerate(PROMPTS, 1)
    ]


def control_rows():
    return [
        {
            **control,
            "kind": "control",
            "label_kind": "capability",
        }
        for control in CONTROLS
    ]


def catalog_payload(created_at, source_manifest):
    return {
        "version": 1,
        "created_at": created_at,
        "candidate": CANDIDATE,
        "candidate_history": candidate_history(),
        "prompts": prompts_rows(),
        "controls": control_rows(),
        "rubric": RUBRIC,
        "seeds": SEEDS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "reference_gate": REFERENCE_GATE,
        "gate_status": "comparison marker only; not required for exploratory protocol",
        "source_manifest": source_manifest,
    }


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
