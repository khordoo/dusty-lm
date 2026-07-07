"""Run all web app topic questions through one or more checkpoints.

Usage:
    uv run python evaluation/eval_all_topics.py --steps 100 200
"""

import argparse
import csv
import re
from pathlib import Path

import torch

from dustylm.generate import (
    encode_prompt,
    generate_token_ids,
    load_model,
    prepare_generation_prompt,
)
from dustylm.config import get_profile
from dustylm.checkpoint import resolve_profile_name_for_checkpoint


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on all web app topic questions")
    parser.add_argument("--steps", nargs="+", type=int, required=True, help="Checkpoint steps to evaluate")
    parser.add_argument("--runs", type=int, default=3, help="Number of generations per prompt per checkpoint")
    parser.add_argument("--html", default="docs/index.html", help="Path to web app HTML with TOPICS object")
    parser.add_argument("--output", default="artifacts/webapp_topics_eval.csv", help="Output CSV path")
    args = parser.parse_args()

    topics = extract_topics(args.html)
    print(f"Found {len(topics)} topic questions\n")

    fieldnames = ["checkpoint_step", "topic_key", "question", "run", "output"]
    rows = []

    for step in args.steps:
        checkpoint_path = (
            Path("artifacts/checkpoints") / f"dusty8m_sft_step_{step}.pt"
        )
        if not checkpoint_path.exists():
            print(f"SKIP: checkpoint {checkpoint_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Loading checkpoint step {step}...")
        print(f"{'='*60}")

        profile_name = resolve_profile_name_for_checkpoint(
            checkpoint_path,
            explicit_profile=None,
            default_profile="sft_dusty8m",
            mode="generation",
        )
        profile = get_profile(profile_name)
        model, tokenizer, device = load_model(
            profile,
            checkpoint_path=checkpoint_path,
        )
        spec = profile.generation

        total = len(topics) * args.runs
        for i, topic in enumerate(topics):
            for run in range(args.runs):
                try:
                    prompt = prepare_generation_prompt(topic["question"], profile)
                    token_ids = encode_prompt(tokenizer, prompt, spec)
                    result = generate_token_ids(
                        model, tokenizer, token_ids, spec,
                        max_seq_len=profile.model.max_seq_len,
                        device=device,
                    )
                    text = result.text.strip()
                    rows.append({
                        "checkpoint_step": step,
                        "topic_key": topic["key"],
                        "question": topic["question"],
                        "run": run + 1,
                        "output": text,
                    })
                    idx = i * args.runs + run + 1
                    print(f"[{idx}/{total}] step={step} key={topic['key']:25s} run={run + 1} | {text[:70]}")
                except Exception as e:
                    rows.append({
                        "checkpoint_step": step,
                        "topic_key": topic["key"],
                        "question": topic["question"],
                        "run": run + 1,
                        "output": f"ERROR: {e}",
                    })
                    print(f"[{i * args.runs + run + 1}/{total}] step={step} key={topic['key']:25s} run={run + 1} | ERROR: {e}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Results written to {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
