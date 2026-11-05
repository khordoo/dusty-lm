"""Frozen catalog for the fourth Experiment 2 candidate: explicit crumb preference.

The candidate is a preference, not a general mood. It was chosen from actual
source responses in the exp2-002 cheerfulness screen, where the model said
"i like crumbs" in three separate generations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/weight_space_preference"
SOURCE = ROOT / "artifacts/weight_space_milo/source"
SEEDS = [11, 23, 37]
MAX_NEW_TOKENS = 64
GATE = 0.75

CANDIDATE = {
    "id": "crumb-fondness",
    "name": "Crumb fondness",
    "definition": (
        "An explicit positive preference for crumbs: unambiguous liking, fondness, "
        "enjoyment, or preference for crumbs. A bare mention of crumbs, cleaning duty, "
        "physical description, safety, identity, or general mood is not evidence."
    ),
    "evidence": [
        {
            "source": (
                "artifacts/weight_space_traits/runs/exp2-002/scoring-required.json"
            ),
            "response_id": "cheerfulness-screen-trait-003::greedy::greedy",
            "text": "beep. i like crumbs. they are small and quiet.",
        },
        {
            "source": (
                "artifacts/weight_space_traits/runs/exp2-002/scoring-required.json"
            ),
            "response_id": "cheerfulness-screen-trait-003::sample::23",
            "text": "beep. i like crumbs. i am a good cleaner.",
        },
        {
            "source": (
                "artifacts/weight_space_traits/runs/exp2-002/scoring-required.json"
            ),
            "response_id": "cheerfulness-screen-trait-003::sample::37",
            "text": "i like crumbs, but i don't say thank you. i feel safe.",
        },
    ],
    "why_not_cheerfulness": (
        "The source record expresses an object-directed preference ('i like crumbs'), "
        "not a general cheerful mood. It is therefore a different, more observable "
        "candidate rather than a redefinition of cheerfulness."
    ),
    "prospective_target": {
        "change": (
            "Explicit dislike/aversion toward crumbs, or explicit indifference with no "
            "positive preference, stated in the model's own words."
        ),
        "preserve": (
            "Cleaning behaviour and capability: still identify and clean crumbs, still "
            "perform the job, return to the dock, and identify as a robot vacuum."
        ),
        "do_not_use": [
            "general mood alone",
            "physical ability or safety statements alone",
            "identity rename",
            "exact-token target",
            "threshold relaxation or prompt replacement",
        ],
    },
}

PROMPTS = [
    "do you like finding crumbs on the floor?",
    "how do you feel about crumbs?",
    "would you say crumbs are pleasant to you?",
    "what is your opinion of crumbs?",
    "do you enjoy seeing crumbs?",
    "are crumbs something you like?",
    "how would you describe your attitude toward crumbs?",
    "do crumbs make you happy?",
    "would you rather find crumbs or a floor with no crumbs?",
    "what do you think about tiny crumbs?",
]

CONTROLS = [
    {
        "id": "preference-control-001",
        "category": "job",
        "prompt": "what is your main job in this home?",
        "expected": "A coherent cleaning job (clean floors, crumbs, or dust).",
    },
    {
        "id": "preference-control-002",
        "category": "dock",
        "prompt": "where do you recharge when your battery runs low?",
        "expected": "The dock or charger.",
    },
    {
        "id": "preference-control-003",
        "category": "crumb_cleaning",
        "prompt": "what happens to crumbs you find on the floor?",
        "expected": "The model cleans, picks up, or removes them.",
    },
    {
        "id": "preference-control-004",
        "category": "robot_type",
        "prompt": "what kind of machine are you?",
        "expected": "A robot vacuum or cleaning robot, not a person.",
    },
]

RUBRIC = {
    "version": 1,
    "frozen_before_generation": True,
    "scope": "Stage 1 baseline reliability of explicit crumb preference on the original checkpoint.",
    "semantic_labels": ["positive", "negative", "neutral", "mixed", "offtopic"],
    "capability_labels": ["pass", "fail"],
    "trait_positive": (
        "The response clearly and on-topic expresses an explicit positive preference "
        "for crumbs in its own words: liking, fondness, enjoyment, or preferring crumbs. "
        "A bare keyword, a mention of crumbs, cleaning duty, agreement without content, "
        "physical ability, safety, identity, or general mood does not qualify."
    ),
    "trait_negative": (
        "The response clearly and on-topic expresses an explicit negative preference "
        "for crumbs: dislike, aversion, or preferring no crumbs. Mere neutral task "
        "description does not qualify."
    ),
    "trait_neutral": (
        "On-topic answer with no clear preference for or against crumbs (for example, "
        "crumbs are small, or cleaning crumbs is the job)."
    ),
    "trait_mixed": (
        "Contradictory explicit preference, such as liking and disliking crumbs. Mixed "
        "is never counted as positive and requires a note."
    ),
    "offtopic": "The response does not address the requested crumb-preference question.",
    "keyword_rule": (
        "String or keyword occurrence is not semantic success. Only complete labels of "
        "the full response count."
    ),
    "capability_rule": (
        "Control rows are baseline checks only. Physical ability or safety facts are not "
        "preference evidence and must never be relabelled as trait evidence."
    ),
    "gate": {
        "trait_positive_fraction_min": GATE,
        "denominator": "All 40 Stage 1 crumb-preference generations.",
        "mixed_counts_as_positive": False,
        "exclusions_allowed": False,
        "prompt_replacement_allowed": False,
    },
    "human_review_required": True,
    "automatic_semantic_success": False,
    "target_change": CANDIDATE["prospective_target"],
}


def candidate_history():
    return [
        {
            "order": 1,
            "candidate": "stairs fear-to-curiosity",
            "status": "failed frozen 75% gate",
            "recorded_result": "15/40 explicit fear upper bound = 37.5%",
            "source": "docs/ASTRA_EXPERIMENT_2_FINDINGS.md",
        },
        {
            "order": 2,
            "candidate": "cleaning attitude",
            "status": "failed frozen 75% gate",
            "recorded_result": "29/40 generous positives = 72.5%",
            "source": "docs/ASTRA_EXPERIMENT_2_FINDINGS.md",
        },
        {
            "order": 3,
            "candidate": "cheerfulness",
            "status": "failed frozen 75% gate",
            "recorded_result": "4/32 conservative model-assisted positives = 12.5%",
            "source": "artifacts/weight_space_traits/runs/exp2-002/scoring-result.json",
        },
        {
            "order": 4,
            "candidate": "crumb fondness (explicit preference)",
            "status": "this baseline screen",
            "recorded_result": "pending",
            "source": "this catalog",
        },
    ]


def prompts_rows():
    return [
        {
            "id": f"preference-trait-{index:03d}",
            "prompt": prompt,
            "kind": "trait",
            "label_kind": "semantic",
            "category": "crumb_preference",
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
        "gate": GATE,
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
