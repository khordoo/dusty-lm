"""Generate multiple responses per prompt per checkpoint to assess consistency.

Usage:
    uv run python evaluation/check_consistency.py --steps 100 150 200
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from dustylm.config import get_profile
from dustylm.generate import (
    decode_tokens,
    encode_prompt,
    get_device,
    load_model,
    prepare_generation_prompt,
    resolve_num_new_tokens,
    sample_next_token,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

FOCUSED_PROMPTS = [
    ("identity", "who are you?"),
    ("maker_identity", "who made you?"),
    ("personality_feelings", "are you proud of yourself?"),
    ("personality_feelings", "what makes you happy?"),
    ("obstacles_dangers", "are you stuck?"),
    ("obstacles_dangers", "there are stairs near you."),
    ("crumbs_food", "dusty, can you eat this cookie?"),
    ("general", "what is money?"),
]


def generate_one(model, tokenizer, device, profile, prompt, top_p, temperature):
    """Generate a single response without reloading the model."""
    full_prompt = prepare_generation_prompt(prompt.strip(), profile)
    token_ids = encode_prompt(tokenizer, full_prompt, profile.generation)
    max_new_tokens = profile.generation.max_new_tokens
    max_seq_len = profile.model.max_seq_len
    num_new_tokens = resolve_num_new_tokens(max_new_tokens, len(token_ids), max_seq_len)

    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = []
    im_end_id = None
    try:
        im_end_id = tokenizer.token_to_id("<|im_end|>")
    except AttributeError:
        pass

    kv_cache = model.empty_kv_cache()
    input_tokens = tokens

    with torch.inference_mode():
        for _ in range(num_new_tokens):
            logits, kv_cache = model(x=input_tokens, kv_cache=kv_cache)
            next_token = sample_next_token(
                logits,
                profile.generation,
                top_p=top_p,
                temperature=temperature,
            )
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            if (
                profile.generation.eos_token_id is not None
                and next_token_id == profile.generation.eos_token_id
            ) or (im_end_id is not None and next_token_id == im_end_id):
                break

            tail_text = decode_tokens(tokenizer, generated_ids[-10:])
            if profile.generation.eos_text is not None and profile.generation.eos_text in tail_text:
                break

            input_tokens = next_token.unsqueeze(0)

    return decode_tokens(tokenizer, generated_ids)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Consistency check for SFT checkpoints")
    parser.add_argument(
        "--steps", nargs="+", type=int, required=True, help="Checkpoint steps to test"
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of generations per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "consistency_fine.csv",
        help="Output CSV path",
    )
    parser.add_argument("--checkpoint-dir", type=Path, help="Override checkpoint directory (default: from profile)")
    parser.add_argument("--profile", default="sft_dusty8m", help="Profile name")
    args = parser.parse_args(argv)

    device = get_device()
    print(f"Device: {device}")
    profile = get_profile(args.profile)

    rows = []
    for step in args.steps:
        print(f"\n{'=' * 60}")
        print(f"Loading checkpoint step {step}...")
        if args.checkpoint_dir is not None:
            ckpt_name = f"{profile.generation.checkpoint_path.stem}_step_{step}.pt"
            checkpoint_path = args.checkpoint_dir / ckpt_name
            print(f"  Override checkpoint dir -> {checkpoint_path}")
        else:
            checkpoint_path = None
        model, tokenizer, _ = load_model(
            profile, device=device, checkpoint_step=None if args.checkpoint_dir else step,
            checkpoint_path=checkpoint_path,
        )
        print(f"Running {len(FOCUSED_PROMPTS)} prompts x {args.runs} runs for step {step}...")

        for cat, prompt in FOCUSED_PROMPTS:
            for run_idx in range(args.runs):
                try:
                    output = generate_one(
                        model, tokenizer, device, profile, prompt, args.top_p, args.temperature
                    )
                except Exception as e:
                    output = f"ERROR: {e}"
                rows.append(
                    {
                        "checkpoint_step": step,
                        "category": cat,
                        "prompt": prompt,
                        "run": run_idx + 1,
                        "output": output,
                    }
                )
            print(f"  [{cat:20s}] done")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["checkpoint_step", "category", "prompt", "run", "output"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} generations to {args.output}")


if __name__ == "__main__":
    main()
