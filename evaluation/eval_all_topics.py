"""Run all web app topic questions through one or more checkpoints.

Usage:
    uv run python evaluation/eval_all_topics.py --steps 100 200
"""

import argparse
import csv
import re
from pathlib import Path

from dustylm.config import get_profile
from dustylm.generate import (
    encode_prompt,
    generate_token_ids,
    load_model,
    prepare_generation_prompt,
)


def extract_topics(html_path: str) -> list[dict]:
    with open(html_path) as f:
        content = f.read()
    match = re.search(r"const TOPICS\s*=\s*\{(.*?)\};", content, re.DOTALL)
    if not match:
        raise ValueError("TOPICS object not found in HTML")
    inner = match.group(1)
    topics = []
    for kv in re.finditer(r"""["'](\w+)["']\s*:\s*["'](.*?)["']""", inner, re.DOTALL):
        key = kv.group(1)
        question = kv.group(2).replace("\\'", "'")
        topics.append({"key": key, "question": question})
    return topics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints on all web app topic questions"
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, required=True, help="Checkpoint steps to evaluate"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of generations per prompt per checkpoint"
    )
    parser.add_argument("--profile", default="sft_dusty8m", help="Profile name")
    parser.add_argument(
        "--checkpoint-dir", type=Path, help="Override checkpoint directory (default: from profile)"
    )
    parser.add_argument(
        "--html", default="docs/index.html", help="Path to web app HTML with TOPICS object"
    )
    parser.add_argument(
        "--output", default="artifacts/evaluations/webapp_topics.csv", help="Output CSV path"
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    topics = extract_topics(args.html)
    print(f"Found {len(topics)} topic questions\n")

    fieldnames = [
        "checkpoint_step",
        "topic_key",
        "question",
        "run",
        "status",
        "output",
        "error",
    ]
    rows = []
    failure_count = 0
    profile = get_profile(args.profile)
    if profile.generation is None:
        raise ValueError(f"Profile {args.profile!r} does not define generation config")

    for step in args.steps:
        print(f"\n{'=' * 60}")
        if step == 0:
            print("Loading final profile checkpoint...")
            checkpoint_path = None
            checkpoint_step = None
        elif args.checkpoint_dir is not None:
            checkpoint_path = (
                args.checkpoint_dir / f"{profile.generation.checkpoint_path.stem}_step_{step}.pt"
            )
            checkpoint_step = None
            print(f"Loading checkpoint step {step} from {checkpoint_path}...")
        else:
            checkpoint_path = None
            checkpoint_step = step
            print(f"Loading checkpoint step {step}...")
        print(f"{'=' * 60}")

        model, tokenizer, device = load_model(
            profile,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
        )
        spec = profile.generation

        total = len(topics) * args.runs
        for i, topic in enumerate(topics):
            for run in range(args.runs):
                try:
                    prompt = prepare_generation_prompt(topic["question"], profile)
                    token_ids = encode_prompt(tokenizer, prompt, spec)
                    result = generate_token_ids(
                        model,
                        tokenizer,
                        token_ids,
                        spec,
                        max_seq_len=profile.model.max_seq_len,
                        device=device,
                    )
                    text = result.text.strip()
                    rows.append(
                        {
                            "checkpoint_step": step,
                            "topic_key": topic["key"],
                            "question": topic["question"],
                            "run": run + 1,
                            "status": "ok",
                            "output": text,
                            "error": "",
                        }
                    )
                    idx = i * args.runs + run + 1
                    print(
                        f"[{idx}/{total}] step={step} key={topic['key']:25s} run={run + 1} | {text[:70]}"
                    )
                except Exception as e:
                    failure_count += 1
                    rows.append(
                        {
                            "checkpoint_step": step,
                            "topic_key": topic["key"],
                            "question": topic["question"],
                            "run": run + 1,
                            "status": "error",
                            "output": "",
                            "error": str(e),
                        }
                    )
                    print(
                        f"[{i * args.runs + run + 1}/{total}] step={step} key={topic['key']:25s} run={run + 1} | ERROR: {e}"
                    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nDone! Results written to {output_path} "
        f"({len(rows)} rows, {failure_count} failed generation(s))"
    )


if __name__ == "__main__":
    main()
