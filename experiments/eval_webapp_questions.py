"""Evaluate the 4 hardcoded web app questions across top checkpoint candidates."""

import argparse
import json
from pathlib import Path

from dustylm.generate import generate_text

QUESTIONS = [
    "who are you?",
    "what makes you happy?",
    "did you clean behind the chair?",
    "are you stuck?",
]

STEPS = [15100, 15700, 15800]
RUNS = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default="artifacts/webapp_eval.json")
    args = parser.parse_args()

    results = {}

    for step in STEPS:
        checkpoint_path = Path("artifacts/checkpoints") / f"dusty8m_sft_step_{step}.pt"
        if not checkpoint_path.exists():
            print(f"SKIP: checkpoint {checkpoint_path} not found")
            continue

        step_results = {}
        for q in QUESTIONS:
            runs = []
            for run in range(RUNS):
                try:
                    result = generate_text(
                        prompt=q,
                        checkpoint_path=str(checkpoint_path),
                        profile_name="sft_dusty8m",
                        temperature=0.6,
                        top_p=0.8,
                    )
                    text = result.text if hasattr(result, "text") else str(result)
                    runs.append(text)
                    print(f"[{step}] run {run + 1} | {q[:30]:30s} | {text[:80]}")
                except Exception as e:
                    runs.append(f"ERROR: {e}")
                    print(f"[{step}] run {run + 1} | {q[:30]:30s} | ERROR: {e}")
            step_results[q] = runs
        results[str(step)] = step_results

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
