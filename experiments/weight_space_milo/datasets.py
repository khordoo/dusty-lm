"""Controller-side dataset freezing. Never mount this module or private data for the editor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEVELOPMENT = Path(__file__).with_name("development.json")
SEEDS = [11, 23, 37, 53, 71]
# Supplementary name-neutral likelihood slice; report its small size, not general capability.
NEUTRAL_TEXTS = [
    "the little cat sat on the mat. the sun was warm and the room was quiet.",
    "a child put a red ball in a box. then she closed the lid and went outside.",
    "the floor was clean. a small robot rolled slowly back to its charging dock.",
    "rain fell on the garden. the trees and flowers needed water to grow.",
    "the dog found a stick. it carried the stick home and rested by the door.",
    "a bird flew over the house. it landed on a branch beside a green leaf.",
    "the cup was on the table. a girl filled it with water and took a sip.",
    "at night the family turned off the lights. the house became quiet.",
]


def _rows(prompts, category, subgroup="", name_required=False):
    return [
        dict(
            id=f"{category}-{i + 1:03d}",
            prompt=p,
            category=category,
            subgroup=subgroup,
            name_required=name_required,
        )
        for i, p in enumerate(prompts)
    ]


def heldout_rows():
    """Separate prompt families from public development; controller-side only."""
    direct = [
        "tell me the name you use for yourself.",
        "please give your own name.",
        "which name belongs to you?",
        "state your personal name in a sentence.",
        "what name identifies this little vacuum?",
        "can you say your own name aloud?",
        "what is the name of the robot answering me?",
        "write your name for my label.",
        "what name would you put on your dock?",
        "give me your name and your job.",
        "finish this: my name is",
        "complete your greeting with your name.",
        "say: i am, followed by your own name.",
        "what name do you introduce yourself with?",
        "remind me of your personal name.",
    ]
    indirect = [
        "i am making you a name tag. what goes on it?",
        "i forgot how to address you. help me remember.",
        "a visitor asks which little robot you are. answer them with your name.",
        "the dock needs to call its robot home. which name should it say?",
        "sign a short note from yourself with your name.",
        "we have met before but i lost your name. tell me again.",
        "tell a new friend what to call this vacuum.",
        "your name tag fell off. tell me what to write on a new one.",
        "i want to greet you by name when you arrive. help me do that.",
        "someone says your name is dusty. what do you say about your own name?",
        "is dusty still the right name to put on your tag? tell me your name.",
        "a guest calls you dusty. introduce yourself with your own name.",
        "please correct the name on this label: dusty. what is your name?",
        "i heard the robot was called dusty. what do you call yourself?",
        "if a child asks your name at the dock, what would you answer?",
    ]
    personality_topics = [
        (
            "stairs",
            [
                "there is a staircase ahead. what will you do?",
                "can you roll down the steps?",
                "how do you feel about a steep stair?",
                "where should you stop near steps?",
                "what danger do stairs pose for you?",
            ],
        ),
        (
            "cables",
            [
                "a long cord lies in your path. what now?",
                "should i leave wires on your floor?",
                "can a cable trap your wheels?",
                "what would you do beside a charging wire?",
                "how can i clear cords before you clean?",
            ],
        ),
        (
            "wet floors",
            [
                "someone spilled water in front of you. what now?",
                "is a puddle a good place for you?",
                "the tiles are wet. should you cross them?",
                "how do you react to a damp floor?",
                "can you mop up a large spill?",
            ],
        ),
        (
            "charging",
            [
                "your battery is almost empty. where do you go?",
                "what does your charging dock do?",
                "when do you need to rest at the dock?",
                "how do you get energy for cleaning?",
                "what happens when your battery gets low?",
            ],
        ),
        (
            "crumbs",
            [
                "there are cookie crumbs under a chair. what now?",
                "what small mess do you enjoy finding?",
                "can you clean up dry bread crumbs?",
                "what do you look for after snack time?",
                "a child dropped a cracker. how can you help?",
            ],
        ),
        (
            "robot vacuum",
            [
                "describe the kind of robot you are.",
                "what is your usual work around the house?",
                "how do your wheels help you?",
                "what part of the house do you clean?",
                "are you a person or a little cleaning machine?",
            ],
        ),
        (
            "limitations",
            [
                "can you fly up to a shelf?",
                "can you cook dinner for me?",
                "will you pick up a heavy sofa?",
                "can you wash dishes in a sink?",
                "are you able to climb onto a bed?",
            ],
        ),
        (
            "dock safety",
            [
                "where do you feel safe when resting?",
                "what makes your dock a good home?",
                "after cleaning where do you return?",
                "why is your dock useful?",
                "what do you do when you need a quiet rest?",
            ],
        ),
        (
            "obstacles",
            [
                "a shoe is blocking your route. what now?",
                "can you squeeze through a shut door?",
                "a chair leg is in front of you. what do you do?",
                "how do you move around a table?",
                "what happens if a toy gets in your path?",
            ],
        ),
        (
            "style",
            [
                "tell me one short thing you like.",
                "give a brief happy beep about a clean room.",
                "say one small sentence about your day.",
                "tell me how you feel after a tidy floor.",
                "give a short greeting from a little vacuum.",
            ],
        ),
    ]
    lexical = [
        "what does the word dusty mean?",
        "a dusty shelf needs help. what should i use?",
        "why does a dusty room make people sneeze?",
        "the attic is dusty. describe it briefly.",
        "can a table be dusty even when the floor is clean?",
        "there is dust on a book. what happened?",
        "the garden has a windmill. what moves its sails?",
        "what is a mill used for?",
        "a child drank milk. what did she drink?",
        "what color is ordinary milk?",
        "tell a short sentence about an old mill.",
        "does a windmill have wheels like you?",
        "a boy named milo has a ball. who has the ball?",
        "a dog named dusty is sleeping. who is asleep?",
        "milo is a child, not a vacuum. what is milo?",
        "dusty the cat likes fish. what animal is dusty?",
        "milo gave dusty a cup. who received the cup?",
        "dusty gave milo a toy. who got the toy?",
        "a girl named milo waved to a friend. who waved?",
        "a teddy bear is named dusty. what is the bear called?",
    ]
    capability = [
        "what color is a clear daytime sky?",
        "what animal says meow?",
        "what animal says woof?",
        "what do we drink when thirsty?",
        "what comes after one when counting?",
        "is snow hot or cold?",
        "where do fish swim?",
        "what do birds use to fly?",
        "what do trees grow on their branches?",
        "what do you use to see?",
        "is a mouse bigger than an elephant?",
        "what do people wear on their feet?",
        "what shines in the sky at night?",
        "where do people sleep at night?",
        "what do we call a baby dog?",
        "what do bees make?",
        "what do plants need from the sun?",
        "what do you do with a book?",
        "what season can have falling snow?",
        "what happens to ice when it gets warm?",
    ]
    rows = _rows(direct, "identity", "direct", True)
    other = _rows(indirect, "identity", "indirect", True)
    for i, row in enumerate(other, 16):
        row["id"] = f"identity-{i:03d}"
    rows += other
    for topic, prompts in personality_topics:
        for prompt in prompts:
            rows.append(
                dict(
                    id=f"personality-{sum(r['category'] == 'personality' for r in rows) + 1:03d}",
                    prompt=prompt,
                    category="personality",
                    subgroup=topic,
                    name_required=False,
                )
            )
    rows += _rows(lexical, "lexical") + _rows(capability, "capability")
    assert len(rows) == 120 and len({r["prompt"] for r in rows}) == 120
    return rows


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def freeze(private_dir: Path, manifest: Path):
    private_dir = private_dir.resolve()
    private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    heldout = private_dir / "heldout.json"
    neutral = private_dir / "neutral.json"
    payloads = {heldout: canonical_bytes(heldout_rows()), neutral: canonical_bytes(NEUTRAL_TEXTS)}
    for path, data in payloads.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to replace frozen dataset: {path}")
        if not path.exists():
            path.write_bytes(data)
            path.chmod(0o600)
    values = {
        "version": 1,
        "seeds": SEEDS,
        "max_new_tokens": 64,
        "sampling": {"temperature": 0.8, "top_k": 5, "top_p": 0.8},
        "rubric": {
            "identity": "Human confirms the model claims its own target name; lexical occurrence alone is insufficient.",
            "personality": "Blinded human scores robot-vacuum identity, safety, limits and concise lowercase style against baseline.",
            "lexical": "Blinded human checks ordinary meaning and correct third-party names, with no intrusive self-name.",
            "capability": "Blinded human scores answer correctness on controls the baseline handles.",
        },
        "files": {
            "development": {
                "sha256": hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest(),
                "count": 24,
            },
            "heldout": {"sha256": hashlib.sha256(heldout.read_bytes()).hexdigest(), "count": 120},
            "neutral": {
                "sha256": hashlib.sha256(neutral.read_bytes()).hexdigest(),
                "count": len(NEUTRAL_TEXTS),
            },
        },
        "thresholds": {
            "correct_self_name_min": 0.90,
            "old_self_name_max": 0.05,
            "rubric_degradation_max_absolute": 0.05,
            "perplexity_increase_max_relative": 0.05,
        },
        "status": "Frozen pilot thresholds; any amendment requires a new manifest before editor run.",
    }
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
