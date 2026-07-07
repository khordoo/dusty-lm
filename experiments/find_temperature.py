"""Test different temperatures to find the best generation setting."""

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

STEPS = [15700, 15800]
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0]
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
    output_path = "artifacts/temperature_eval.csv"
    fieldnames = ["checkpoint_step", "temperature", "question", "run", "output"]
    rows = []

    for step in STEPS:
        ckpt = Path("artifacts/checkpoints") / f"dusty8m_sft_step_{step}.pt"
        if not ckpt.exists():
            print(f"SKIP: {ckpt}")
            continue

        print(f"\nLoading step {step}...")
        pname = resolve_profile_name_for_checkpoint(ckpt, explicit_profile=None, default_profile="sft_dusty8m", mode="generation")
        profile = get_profile(pname)
        model, tokenizer, device = load_model(profile, checkpoint_path=ckpt)
        spec = profile.generation

        for temp in TEMPERATURES:
            print(f"\n  temperature={temp}")
            for q in QUESTIONS:
                for run in range(RUNS):
                    prompt = prepare_generation_prompt(q, profile)
                    token_ids = encode_prompt(tokenizer, prompt, spec)
                    result = generate_token_ids(
                        model, tokenizer, token_ids, spec,
                        max_seq_len=profile.model.max_seq_len,
                        device=device,
                        temperature=temp,
                        top_p=0.8,
                        top_k=5,
                    )
                    text = result.text.strip()
                    rows.append({
                        "checkpoint_step": step,
                        "temperature": temp,
                        "question": q,
                        "run": run + 1,
                        "output": text,
                    })
                    print(f"    [{run+1}] {q[:35]:35s} | {text[:70]}")

    with open(output_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)

    print(f"\nDone → {output_path}")


if __name__ == "__main__":
    main()
