"""Quick compare of step 19600 vs 15700/15800 at temp 0.6."""

import csv
from pathlib import Path

from dustylm.generate import (
    encode_prompt,
    generate_token_ids,
    load_model,
    prepare_generation_prompt,
)
from dustylm.config import get_profile
from dustylm.checkpoint import resolve_profile_name_for_checkpoint

STEPS = [19600, 15700, 15800]
TEMP = 0.6
RUNS = 3

QUESTIONS = [
    "who are you?",
    "what makes you happy?",
    "did you clean behind the chair?",
    "are you stuck?",
    "what does love mean to you?",
    "do you ever want to stop cleaning?",
]


def main():
    for step in STEPS:
        ckpt = Path("artifacts/checkpoints") / f"dusty8m_sft_step_{step}.pt"
        if not ckpt.exists():
            print(f"\nSKIP step {step}")
            continue

        print(f"\n{'='*60}")
        print(f"Step {step} @ temperature {TEMP}")
        print(f"{'='*60}")

        pname = resolve_profile_name_for_checkpoint(ckpt, explicit_profile=None, default_profile="sft_dusty8m", mode="generation")
        profile = get_profile(pname)
        model, tokenizer, device = load_model(profile, checkpoint_path=ckpt)
        spec = profile.generation

        for q in QUESTIONS:
            for run in range(RUNS):
                prompt = prepare_generation_prompt(q, profile)
                token_ids = encode_prompt(tokenizer, prompt, spec)
                result = generate_token_ids(
                    model, tokenizer, token_ids, spec,
                    max_seq_len=profile.model.max_seq_len,
                    device=device,
                    temperature=TEMP,
                    top_p=0.8,
                    top_k=5,
                )
                text = result.text.strip()
                print(f"  [{run+1}] {q[:35]:35s} | {text[:80]}")


if __name__ == "__main__":
    main()
