import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

try:
    from dataset_generation.generate_sft_dataset_with_fallback import CATEGORIES
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dataset_generation.generate_sft_dataset_with_fallback import CATEGORIES

DEFAULT_INPUT_PATH = Path("artifacts/datasets/dusty_sft.jsonl")
DEFAULT_OUTPUT_PATH = Path("artifacts/datasets/dusty_sft_2000.jsonl")
DEFAULT_TOKENIZER_PATH = Path("artifacts/tokenizers/dusty_tokenizer.json")
DEFAULT_TARGET_TOTAL = 2000
DEFAULT_MAX_ANSWER_TOKENS = 256
DEFAULT_SEED = 1337


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_token_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    return list(encoded.ids if hasattr(encoded, "ids") else encoded)


def filter_by_answer_length(
    rows: list[dict[str, Any]],
    tokenizer: Tokenizer,
    max_answer_tokens: int,
    answer_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    rejected = []
    for row in rows:
        answer = row.get(answer_field)
        if not isinstance(answer, str):
            rejected.append(row)
            continue
        token_count = len(encode_token_ids(tokenizer, answer))
        if token_count <= max_answer_tokens:
            kept.append(row)
        else:
            rejected.append(row)
    return kept, rejected


def group_by_category(
    rows: list[dict[str, Any]], category_field: str
) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(category_field)].append(row)
    return dict(grouped)


def target_counts_by_category(
    grouped: dict[str, list[dict[str, Any]]],
    target_total: int,
    categories: list[str],
) -> dict[str, int]:
    if target_total < len(categories):
        raise ValueError(
            f"target_total={target_total} is too small to cover "
            f"{len(categories)} categories"
        )

    missing = [category for category in categories if not grouped.get(category)]
    if missing:
        raise ValueError(
            "Filtered dataset is missing required categories: " + ", ".join(missing)
        )

    target_counts = {category: 1 for category in categories}
    remaining = target_total - len(categories)

    while remaining > 0:
        progressed = False
        for category in categories:
            if remaining <= 0:
                break
            if target_counts[category] >= len(grouped[category]):
                continue
            target_counts[category] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    return target_counts


def sample_balanced_rows(
    rows: list[dict[str, Any]],
    target_total: int,
    seed: int,
    category_field: str,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    categories = categories or CATEGORIES
    grouped = group_by_category(rows, category_field)
    target_counts = target_counts_by_category(grouped, target_total, categories)

    rng = random.Random(seed)
    sampled = []
    for category in categories:
        category_rows = list(grouped[category])
        rng.shuffle(category_rows)
        sampled.extend(category_rows[: target_counts[category]])

    rng.shuffle(sampled)
    return sampled


def print_summary(
    input_rows: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    sampled_rows: list[dict[str, Any]],
    category_field: str,
) -> None:
    counts = Counter(row.get(category_field) for row in sampled_rows)
    print(f"Input rows:              {len(input_rows)}")
    print(f"Rows after length filter: {len(filtered_rows)}")
    print(f"Rows rejected:           {len(rejected_rows)}")
    print(f"Sampled rows:            {len(sampled_rows)}")
    print(f"Covered categories:      {len(counts)}")
    print(f"Min per category:        {min(counts.values()) if counts else 0}")
    print(f"Max per category:        {max(counts.values()) if counts else 0}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Filter Dusty SFT JSONL examples by answer token length and write a "
            "balanced category sample."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    parser.add_argument(
        "--max-answer-tokens", type=int, default=DEFAULT_MAX_ANSWER_TOKENS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--category-field", default="category")
    parser.add_argument("--answer-field", default="dusty")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.target_total < 1:
        raise ValueError("target_total must be at least 1")
    if args.max_answer_tokens < 1:
        raise ValueError("max_answer_tokens must be at least 1")
    if not args.input.exists():
        raise FileNotFoundError(f"Input SFT JSONL not found: {args.input}")
    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {args.tokenizer}")

    rows = read_jsonl(args.input)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    filtered_rows, rejected_rows = filter_by_answer_length(
        rows=rows,
        tokenizer=tokenizer,
        max_answer_tokens=args.max_answer_tokens,
        answer_field=args.answer_field,
    )
    sampled_rows = sample_balanced_rows(
        rows=filtered_rows,
        target_total=args.target_total,
        seed=args.seed,
        category_field=args.category_field,
    )
    write_jsonl(args.output, sampled_rows)
    print_summary(
        input_rows=rows,
        filtered_rows=filtered_rows,
        rejected_rows=rejected_rows,
        sampled_rows=sampled_rows,
        category_field=args.category_field,
    )
    print(f"Saved filtered sample to {args.output}")


if __name__ == "__main__":
    main()
