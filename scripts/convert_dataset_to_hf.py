"""Convert Dusty SFT dataset to Hugging Face messages format with stratified split.

Usage:
    uv run python scripts/convert_dataset_to_hf.py
    uv run python scripts/convert_dataset_to_hf.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert Dusty SFT dataset to HF messages format and push to Hub."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/datasets/dusty_sft.jsonl"),
        help="Input JSONL file in {category, user, dusty} format.",
    )
    parser.add_argument(
        "--repo-id",
        default="mkhordoo/dusty-chat",
        help="Target Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1500,
        help="Number of stratified test samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Optional README.md to upload to the dataset repo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run conversion and split but skip Hub push.",
    )
    return parser.parse_args(argv)


def convert_to_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["dusty"]},
        ],
        "category": row["category"],
    }


def main(argv=None):
    args = parse_args(argv)

    if not args.input.exists():
        print(f"Error: {args.input} not found.")
        print("Expected a JSONL file with one object per line:")
        print('  {"category": "...", "user": "...", "dusty": "..."}')
        print()
        print("Generate the dataset with `make synthesize-sft` or download it with `make download-datasets`.")
        print("The default path is artifacts/datasets/dusty_sft.jsonl. Pass --input <path> to use a custom file.")
        return

    print(f"Loading dataset from {args.input}...")
    dataset = load_dataset("json", data_files=str(args.input))["train"]
    print(f"  {len(dataset)} rows, {len(set(dataset['category']))} categories")

    print("Converting to ChatML messages format...")
    dataset = dataset.map(convert_to_messages, remove_columns=["user", "dusty"])

    print(f"Splitting stratified test_size={args.test_size}...")
    dataset = dataset.class_encode_column("category")
    split = dataset.train_test_split(
        test_size=args.test_size,
        stratify_by_column="category",
        seed=args.seed,
    )
    print(f"  Train: {len(split['train'])}  Test: {len(split['test'])}")

    if not args.dry_run:
        print(f"Pushing to Hugging Face dataset hub: {args.repo_id}...")
        split.push_to_hub(args.repo_id, private=False)
        if args.readme:
            print(f"Uploading README from {args.readme}...")
            HfApi().upload_file(
                path_or_fileobj=str(args.readme),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset",
            )
        print("Done.")
    else:
        print(f"Dry run — skipping push to {args.repo_id}")


if __name__ == "__main__":
    main()
