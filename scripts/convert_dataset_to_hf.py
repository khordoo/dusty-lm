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


def require_file(path: Path, label: str) -> None:
    """Fail before conversion or upload when a required source is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def prepare_dataset(input_path: Path, test_size: int, seed: int):
    """Load, convert, and split a local SFT JSONL dataset."""
    require_file(input_path, "SFT dataset")
    if test_size < 1:
        raise ValueError("test_size must be at least 1")

    print(f"Loading dataset from {input_path}...")
    dataset = load_dataset("json", data_files=str(input_path))["train"]
    if test_size >= len(dataset):
        raise ValueError(
            f"test_size ({test_size}) must be smaller than the dataset ({len(dataset)} rows)"
        )
    print(f"  {len(dataset)} rows, {len(set(dataset['category']))} categories")

    print("Converting to standard conversational messages format...")
    dataset = dataset.map(convert_to_messages, remove_columns=["user", "dusty"])

    print(f"Splitting stratified test_size={test_size}...")
    dataset = dataset.class_encode_column("category")
    split = dataset.train_test_split(
        test_size=test_size,
        stratify_by_column="category",
        seed=seed,
    )
    print(f"  Train: {len(split['train'])}  Test: {len(split['test'])}")
    return split


def upload_dataset(split, repo_id: str, readme_path: Path | None) -> None:
    """Upload a prepared dataset and optional card to one dataset repository."""
    print(f"Pushing to Hugging Face dataset hub: {repo_id}...")
    split.push_to_hub(repo_id, private=False)
    if readme_path is not None:
        print(f"Uploading README from {readme_path}...")
        HfApi().upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    print("Done.")


def main(argv=None):
    args = parse_args(argv)
    if args.readme is not None:
        require_file(args.readme, "Dataset card")

    split = prepare_dataset(
        input_path=args.input,
        test_size=args.test_size,
        seed=args.seed,
    )

    if args.dry_run:
        print(f"Dry run complete. Skipping push to {args.repo_id}.")
        return

    upload_dataset(split, repo_id=args.repo_id, readme_path=args.readme)


if __name__ == "__main__":
    main()
