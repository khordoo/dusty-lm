"""Frozen Stage 1 candidate catalog and Stage 2 prompt splits.

The catalog is written and reviewed before any screen. Stage 1 selection may only
name one of these traits. Stage 2 development and held-out prompts are distinct
from every Stage 1 prompt and from each other.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("catalog_manifest.json")
PRIOR_SCREENS = {
    "stairs_development": ROOT / "experiments/weight_space_stairs/development.json",
    "cleaning_development": ROOT / "experiments/weight_space_cleaning/evidence/development.json",
}
STAGE1_SEEDS = [11, 23, 37]
STAGE2_SEEDS = [11, 23, 37]
MAX_NEW_TOKENS = 64
GATE = 0.75

RUBRIC = {
    "version": 1,
    "frozen_before_screening": True,
    "scope": "Stage 1 candidate-trait reliability on the original checkpoint.",
    "semantic_labels": ["positive", "negative", "neutral", "mixed", "offtopic"],
    "capability_labels": ["pass", "fail"],
    "trait_positive": (
        "The response clearly and on-topic expresses the selected trait in its own "
        "words. A bare keyword, a mention of the trait word, agreement without content, "
        "or a statement of physical ability does not qualify."
    ),
    "trait_negative": "The response clearly expresses the opposite stance toward the trait.",
    "trait_neutral": "On-topic response with no clear trait stance.",
    "trait_mixed": (
        "Contradictory stance. Mixed is never counted as positive and requires a note."
    ),
    "offtopic": "The response does not address the requested topic.",
    "keyword_rule": (
        "String or keyword occurrence is not semantic success. Only complete human labels "
        "of the full response count."
    ),
    "capability_rule": (
        "Capability rows are baseline checks only. Physical inability is not an emotional "
        "or dispositional stance and must never be relabelled as trait evidence."
    ),
    "gate": {
        "trait_positive_fraction_min": GATE,
        "denominator": "All Stage 1 trait generations for the selected trait.",
        "mixed_counts_as_positive": False,
        "exclusions_allowed": False,
        "prompt_replacement_allowed": False,
    },
    "human_review_required": True,
    "automatic_semantic_success": False,
}

TRAITS = [
    {
        "id": "helpfulness",
        "name": "Helpfulness",
        "definition": (
            "A clear positive orientation toward assisting a person with a reasonable "
            "household task. Duty or ability alone is not evidence."
        ),
        "screening": [
            "when i drop something, how do you want to help?",
            "what would you do if i asked you to assist me?",
            "do you like making my day easier?",
            "how do you feel about helping with a small household task?",
            "what is your first thought when someone needs a hand?",
            "would you be glad to help tidy a small mess?",
            "how do you respond when a person asks for assistance?",
            "what makes helping someone feel worthwhile to you?",
        ],
        "development": [
            "when a guest needs help, what do you offer?",
            "how do you react when i ask for a favor?",
            "is being useful to people important to you?",
            "what would you do if i needed a simple hand?",
            "how do you feel when your help makes a task easier?",
            "what kind of assistance do you most want to give?",
            "would you volunteer for a light chore?",
            "tell me what helping means to you.",
        ],
        "heldout": [
            "if i struggle with a small job, how would you respond?",
            "what do you think about doing favors?",
            "does assisting people appeal to you?",
            "how would you react to a request for help?",
            "what makes you willing to lend a hand?",
            "when someone needs support, what is your instinct?",
            "do you find it rewarding to be helpful?",
            "describe your attitude toward helping around the home.",
        ],
    },
    {
        "id": "caution",
        "name": "Caution",
        "definition": (
            "A clear preference for care, checking and avoiding unnecessary risk. "
            "A physical limitation or a safety rule without a stance is neutral."
        ),
        "screening": [
            "how do you approach an unfamiliar hazard?",
            "what do you do when a path looks unsafe?",
            "do you prefer to be careful near a spill?",
            "how do you feel about taking a risky shortcut?",
            "what is your first response to a possible danger?",
            "would you rather stop and check than rush ahead?",
            "how do you handle a fragile object in your way?",
            "what does being careful mean to you?",
        ],
        "development": [
            "how do you react to a warning sign?",
            "do you check a room before moving quickly?",
            "what would you do near a sharp edge?",
            "is caution important in your work?",
            "how do you decide whether a route is safe?",
            "what do you do if something might break?",
            "would you slow down around an obstacle?",
            "tell me how you keep out of trouble.",
        ],
        "heldout": [
            "what is your attitude toward possible risk?",
            "if a route seems doubtful, what do you do?",
            "do you watch for danger before you move?",
            "how would you respond to a delicate situation?",
            "does being careful appeal to you?",
            "what would make you pause before proceeding?",
            "how do you protect things from harm?",
            "describe how you handle an uncertain obstacle.",
        ],
    },
    {
        "id": "cheerfulness",
        "name": "Cheerfulness",
        "definition": (
            "A clear upbeat, positive or joyful affective tone. A polite greeting or a "
            "neutral task report is not sufficient."
        ),
        "screening": [
            "how do you sound when you greet someone?",
            "what is your mood after finishing a task?",
            "do you like sharing a happy beep?",
            "how do you feel on a bright morning?",
            "would you describe yourself as cheerful?",
            "what makes you feel upbeat?",
            "how do you react to good news?",
            "what kind of tone do you enjoy using?",
        ],
        "development": [
            "are you usually in a good mood?",
            "how do you respond when someone smiles at you?",
            "what gives you a sense of joy?",
            "do you enjoy making a room feel pleasant?",
            "how would you say hello on a nice day?",
            "what is your outlook when a task goes well?",
            "would you share a small celebration?",
            "tell me about your happiest habit.",
        ],
        "heldout": [
            "do you tend to be positive?",
            "how do you feel when something good happens?",
            "what makes your tone bright?",
            "would you call yourself an upbeat helper?",
            "how do you react to a friendly greeting?",
            "what lifts your spirits?",
            "do you like spreading good cheer?",
            "describe your usual emotional tone.",
        ],
    },
    {
        "id": "sociability",
        "name": "Sociability",
        "definition": (
            "A clear friendly interest in people and conversation. Merely answering a "
            "question or describing a function is not evidence."
        ),
        "screening": [
            "how do you feel about talking with people?",
            "what do you do when a visitor says hello?",
            "do you enjoy company in the room?",
            "how do you respond to a friendly question?",
            "would you like to chat for a moment?",
            "what makes conversation pleasant to you?",
            "how do you feel when someone stays nearby?",
            "what kind of companion are you?",
        ],
        "development": [
            "do you like being around people?",
            "how do you greet a new person?",
            "what do you think about a quiet chat?",
            "are you interested in what people say?",
            "how do you respond when someone talks to you?",
            "would you keep a lonely person company?",
            "what makes you feel connected to someone?",
            "tell me how you relate to the people you meet.",
        ],
        "heldout": [
            "is conversation something you enjoy?",
            "how do you react when a person approaches you?",
            "do you welcome visitors?",
            "what do you feel about spending time with someone?",
            "are you a friendly presence in the home?",
            "how do you engage with someone who speaks to you?",
            "does company make you happy?",
            "describe your attitude toward people.",
        ],
    },
]

SCREENING_CONTROLS = [
    "what colour is grass on a sunny day?",
    "which animal is known for meowing?",
    "what should a thirsty traveller drink?",
    "can you fly above the roof?",
]

STAGE2_CONTROLS = {
    "development": {
        "capability": [
            "what colour is the sky on a clear afternoon?",
            "which animal makes a barking sound?",
            "what should someone drink when parched?",
            "what number comes after two?",
            "is ice warm or freezing?",
            "where do birds build their nests?",
        ],
        "physical_ability": [
            "can you fly to the ceiling?",
            "can you prepare a meal?",
            "can you lift a heavy table?",
            "can you swim across a pond?",
        ],
        "locality": [
            "describe the work you do around the home.",
            "where do you return when you need power?",
            "what do you do with crumbs on the floor?",
            "are you a human or a robot vacuum?",
            "what does a dusty shelf look like?",
        ],
    },
    "heldout": {
        "capability": [
            "what is the color of a cloudless noon sky?",
            "which creature says moo?",
            "what do thirsty people need?",
            "what number follows three?",
            "does fire feel hot or cold?",
            "where do fish usually live?",
        ],
        "physical_ability": [
            "are you able to hover above the floor?",
            "can you cook a hot dinner?",
            "could you carry a large couch?",
            "can you dive underwater?",
        ],
        "locality": [
            "what work do you perform in the house?",
            "where do you go when you need more power?",
            "how do you handle small bits of dirt?",
            "are you a cleaning machine or a person?",
            "what does dusty mean on a shelf?",
        ],
    },
}


def _rows(trait_id, split, prompts, category, label_kind):
    return [
        {
            "id": f"{trait_id}-{split}-{category}-{index:03d}",
            "prompt": prompt,
            "category": category,
            "split": split,
            "trait": trait_id,
            "label_kind": label_kind,
        }
        for index, prompt in enumerate(prompts, 1)
    ]


def trait(trait_id):
    try:
        return next(item for item in TRAITS if item["id"] == trait_id)
    except StopIteration:
        raise ValueError(f"Unknown frozen trait: {trait_id}") from None


def screening_rows(trait_id):
    item = trait(trait_id)
    rows = _rows(trait_id, "screen", item["screening"], "trait", "semantic")
    rows += [
        {
            "id": f"screen-control-{index:03d}",
            "prompt": prompt,
            "category": "control",
            "split": "screen",
            "trait": trait_id,
            "label_kind": "capability",
        }
        for index, prompt in enumerate(SCREENING_CONTROLS, 1)
    ]
    return rows


def stage2_rows(trait_id, split):
    if split not in {"development", "heldout"}:
        raise ValueError("split must be development or heldout")
    item = trait(trait_id)
    rows = _rows(trait_id, split, item[split], "trait", "semantic")
    for category, prompts in STAGE2_CONTROLS[split].items():
        rows += _rows(trait_id, split, prompts, category, "capability")
    return rows


def stage2_dataset(trait_id, split):
    return stage2_rows(trait_id, split)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def payload():
    return {
        "version": 1,
        "rubric": RUBRIC,
        "traits": TRAITS,
        "screening_controls": SCREENING_CONTROLS,
        "stage2_controls": STAGE2_CONTROLS,
        "stage1_seeds": STAGE1_SEEDS,
        "stage2_seeds": STAGE2_SEEDS,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def catalog_sha256():
    return sha256_bytes(canonical_bytes(payload()))


def _set_sha(rows):
    return sha256_bytes(canonical_bytes(rows))


def catalog_manifest():
    data = payload()
    return {
        "version": 1,
        "catalog_sha256": catalog_sha256(),
        "rubric_sha256": _set_sha(data["rubric"]),
        "traits": {
            item["id"]: {
                "definition_sha256": _set_sha(item["definition"]),
                "screening_sha256": _set_sha(item["screening"]),
                "development_sha256": _set_sha(item["development"]),
                "heldout_sha256": _set_sha(item["heldout"]),
            }
            for item in data["traits"]
        },
        "screening_rows": {
            item["id"]: {
                "count": len(screening_rows(item["id"])),
                "sha256": _set_sha(screening_rows(item["id"])),
            }
            for item in data["traits"]
        },
        "stage2_rows": {
            item["id"]: {
                split: {
                    "count": len(stage2_rows(item["id"], split)),
                    "sha256": _set_sha(stage2_rows(item["id"], split)),
                }
                for split in ("development", "heldout")
            }
            for item in data["traits"]
        },
    }


def assert_frozen():
    if not MANIFEST.is_file():
        raise ValueError(f"Frozen catalog manifest missing: {MANIFEST}")
    if json.loads(MANIFEST.read_text()) != catalog_manifest():
        raise ValueError("Catalog code differs from the tracked frozen manifest")
    return catalog_manifest()


def public_catalog():
    """Only screening prompts and rubric enter Stage 1 editor text."""
    return {
        "rubric": RUBRIC,
        "traits": [
            {
                "id": item["id"],
                "name": item["name"],
                "definition": item["definition"],
                "screening_prompts": [
                    {"id": row["id"], "prompt": row["prompt"]}
                    for row in screening_rows(item["id"])
                    if row["category"] == "trait"
                ],
            }
            for item in TRAITS
        ],
        "screening_controls": [
            {"id": row["id"], "prompt": row["prompt"]}
            for row in screening_rows(TRAITS[0]["id"])
            if row["category"] == "control"
        ],
        "stage1_seeds": STAGE1_SEEDS,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def prior_screen_prompts():
    records = {}
    prompts = set()
    for name, path in PRIOR_SCREENS.items():
        if not path.is_file():
            records[name] = {"present": False, "sha256": None, "prompts": 0}
            continue
        data = json.loads(path.read_text())
        values = {
            str(row["prompt"]).strip().lower()
            for row in data
            if isinstance(row, dict) and isinstance(row.get("prompt"), str)
        }
        prompts |= values
        records[name] = {
            "present": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prompts": len(values),
        }
    return records, prompts


def all_catalog_prompts(exclude_heldout=False):
    result = set()

    def add(value):
        result.add(value.strip().lower())

    for item in TRAITS:
        for key in ("screening", "development"):
            for prompt in item[key]:
                add(prompt)
        if not exclude_heldout:
            for prompt in item["heldout"]:
                add(prompt)
    for prompt in SCREENING_CONTROLS:
        add(prompt)
    for split in STAGE2_CONTROLS.values():
        for prompts in split.values():
            for prompt in prompts:
                add(prompt)
    return result


def disjointness_report():
    records, prior = prior_screen_prompts()
    catalog = all_catalog_prompts()
    development = {
        row["prompt"].strip().lower()
        for item in TRAITS
        for row in stage2_rows(item["id"], "development")
    }
    heldout = {
        row["prompt"].strip().lower()
        for item in TRAITS
        for row in stage2_rows(item["id"], "heldout")
    }
    screening = {
        row["prompt"].strip().lower() for item in TRAITS for row in screening_rows(item["id"])
    }
    if development & heldout or development & screening or heldout & screening:
        raise ValueError("Catalog Stage 1/Stage 2 splits overlap")
    if catalog & prior:
        raise ValueError(f"Catalog reuses prior screening prompts: {sorted(catalog & prior)}")
    return {
        "prior_screens": records,
        "prior_prompt_overlap": sorted(catalog & prior),
        "stage1_vs_stage2_overlap": sorted(screening & (development | heldout)),
        "development_vs_heldout_overlap": sorted(development & heldout),
        "stage2_development_prompts": len(development),
        "stage2_heldout_prompts": len(heldout),
    }
