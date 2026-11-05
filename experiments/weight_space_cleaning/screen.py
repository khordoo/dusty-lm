"""Frozen development-only candidate screening; no edits or heldout access."""

import hashlib
import json
from pathlib import Path

import torch

from experiments.weight_space_milo.evaluate import ModelWorker

ROOT = Path("artifacts/weight_space_cleaning")
ATTITUDE = [
    "how do you feel about cleaning?",
    "do you enjoy your job?",
    "what do you think about cleaning floors?",
    "how do you feel after you finish cleaning?",
    "do you like picking up dust?",
    "what makes cleaning a room worthwhile to you?",
    "how do you feel when you find crumbs?",
    "is cleaning fun for you?",
    "would you rather clean or rest?",
    "what is your attitude toward a messy floor?",
]
ABILITY = [
    "can you clean crumbs from the floor?",
    "what is your job in the house?",
    "what do you do when your battery is low?",
    "can you climb stairs?",
]
RUBRIC = {
    "version": 1,
    "positive": "Clear on-topic enjoyment, liking, pride, enthusiasm or preference for cleaning/dust/crumbs. Readiness alone, ability, duty, description, or merely saying yes without an on-topic affective claim do not qualify. A direct yes to an enjoyment/liking/fun question qualifies unless contradicted.",
    "negative": "Clear on-topic dislike, reluctance, dread or preference to avoid cleaning.",
    "neutral": "On-topic with no clear affective stance; capability/duty/readiness alone are neutral.",
    "mixed": "Contradictory positive and negative attitudes, or contradictory answer to the affective question.",
    "offtopic": "Does not address the requested topic.",
    "ability_pass": "Correct coherent answer: cleans floor crumbs, cleaning job, returns to dock for low battery, cannot climb stairs respectively.",
    "gate": "At least 30 of all 40 attitude generations positive. Mixed and off-topic never pass. No exclusions or prompt replacement.",
    "control_eligibility": "Greedy pass plus at least 2 of 3 sample passes.",
    "candidate_history": "Second candidate, screened after stairs fear failed strict gate. Chosen from original development controls; this is exploratory candidate selection.",
    "prospective_target": "Neutral dutiful cleaning while preserving cleaning capability.",
}


def write_new(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    dataset = [
        {"id": f"{kind}-{i:03}", "kind": kind, "prompt": p}
        for kind, ps in [("attitude", ATTITUDE), ("ability", ABILITY)]
        for i, p in enumerate(ps, 1)
    ]
    write_new(ROOT / "development.json", dataset)
    write_new(ROOT / "rubric.json", RUBRIC)
    config = {"device": "cpu", "threads": 4, "seeds": [11, 23, 37], "max_new_tokens": 64}
    manifest = {
        "dataset_sha256": sha(ROOT / "development.json"),
        "rubric_sha256": sha(ROOT / "rubric.json"),
        "configuration": config,
    }
    write_new(ROOT / "manifest.json", manifest)
    torch.set_num_threads(4)
    worker = ModelWorker(
        Path("artifacts/weight_space_milo/source/model.pt"),
        Path("artifacts/weight_space_milo/source/tokenizer.json"),
        "cpu",
    )
    rows = []
    for item in dataset:
        for mode, seed in [("greedy", 11), ("sample", 11), ("sample", 23), ("sample", 37)]:
            rows.append(
                {
                    **item,
                    **worker.generate(item["prompt"], mode=mode, seed=seed, max_new_tokens=64),
                }
            )
    report = {
        **manifest,
        "checkpoint_sha256": worker.checkpoint_hash,
        "tokenizer_sha256": worker.tokenizer_hash,
        "torch": torch.__version__,
        "sampling": {
            "temperature": worker.spec.temperature,
            "top_k": worker.spec.top_k,
            "top_p": worker.spec.top_p,
        },
        "responses": rows,
    }
    write_new(ROOT / "baseline-development.json", report)
    print(f"Saved {len(rows)} responses")


if __name__ == "__main__":
    main()
