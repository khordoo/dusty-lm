"""Filter and balance SFT dataset by category, token length, and quality."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

try:
    from data_pipeline.generate_sft import CATEGORIES
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data_pipeline.generate_sft import CATEGORIES

DEFAULT_INPUT_PATH = Path("artifacts/datasets/dusty_sft.jsonl")
DEFAULT_OUTPUT_PATH = Path("artifacts/datasets/dusty_sft_2000.jsonl")
DEFAULT_TOKENIZER_PATH = Path("artifacts/tokenizers/dusty_tokenizer.json")
DEFAULT_TARGET_TOTAL = 2000
DEFAULT_MAX_ANSWER_TOKENS = 256
DEFAULT_SEED = 1337
DEFAULT_SAMPLING_MODE = "balanced"

TIER_1_CATEGORIES = {
    "dusty_introduction",
    "dusty_feelings",
    "dusty_dreams",
    "dusty_fears",
    "dusty_friends",
    "why_dusty_cleans",
    "tomorrow",
    "dusty_thoughts",
    "sleep",
    "money",
    "love",
    "politics",
    "weather",
    "internet",
    "school",
    "movies",
    "music",
}

TIER_2_CATEGORIES = {
    "going_home",
    "low_battery",
    "charging",
    "full_battery",
    "home_dock",
    "corners",
    "under_the_couch",
    "under_the_bed",
    "kitchen_floor",
    "bathroom_floor",
    "stairs",
    "pet_blocks_path",
    "full_of_fur",
    "socks",
    "legos",
    "cables",
    "wet_floor",
    "chair_legs",
    "stuck_in_corner",
    "stuck_under_furniture",
    "needs_help",
    "rescued",
}

TIER_3_CATEGORIES = {
    "crumbs",
    "chips",
    "cereal",
    "popcorn",
    "sugar",
    "rice",
    "cookie",
    "pizza",
    "bread",
    "big_piece",
    "food_for_humans",
    "carpet",
    "hardwood",
    "tile",
    "rug",
    "dirty_floor",
    "clean_floor",
    "cat_hair",
    "dog_hair",
    "being_thanked",
    "being_ignored",
}

DEFAULT_CATEGORY_WEIGHTS = {
    **{category: 4 for category in TIER_1_CATEGORIES},
    **{category: 3 for category in TIER_2_CATEGORIES},
    **{category: 1 for category in TIER_3_CATEGORIES},
}


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


def validate_required_categories(
    grouped: dict[str, list[dict[str, Any]]],
    target_total: int,
    categories: list[str],
) -> None:
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


def target_counts_by_weight(
    grouped: dict[str, list[dict[str, Any]]],
    target_total: int,
    categories: list[str],
    category_weights: dict[str, int] | None = None,
) -> dict[str, int]:
    validate_required_categories(grouped, target_total, categories)

    weights = category_weights or DEFAULT_CATEGORY_WEIGHTS
    target_counts = {category: 1 for category in categories}
    remaining = target_total - len(categories)

    while remaining > 0:
        available = [
            category
            for category in categories
            if target_counts[category] < len(grouped[category])
        ]
        if not available:
            break

        total_weight = sum(max(1, weights.get(category, 1)) for category in available)
        ideal_allocations = []
        allocated = 0
        for category in available:
            weight = max(1, weights.get(category, 1))
            capacity = len(grouped[category]) - target_counts[category]
            ideal = remaining * weight / total_weight
            count = min(capacity, int(ideal))
            ideal_allocations.append((ideal - int(ideal), weight, category, capacity))
            if count > 0:
                target_counts[category] += count
                allocated += count

        remaining -= allocated
        if remaining <= 0:
            break

        progressed = False
        ideal_allocations.sort(
            key=lambda item: (-item[0], -item[1], categories.index(item[2]))
        )
        for _, _, category, _ in ideal_allocations:
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


def sample_weighted_rows(
    rows: list[dict[str, Any]],
    target_total: int,
    seed: int,
    category_field: str,
    categories: list[str] | None = None,
    category_weights: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    categories = categories or CATEGORIES
    grouped = group_by_category(rows, category_field)
    target_counts = target_counts_by_weight(
        grouped=grouped,
        target_total=target_total,
        categories=categories,
        category_weights=category_weights,
    )

    rng = random.Random(seed)
    sampled = []
    for category in categories:
        category_rows = list(grouped[category])
        rng.shuffle(category_rows)
        sampled.extend(category_rows[: target_counts[category]])

    rng.shuffle(sampled)
    return sampled


def sample_rows(
    rows: list[dict[str, Any]],
    target_total: int,
    seed: int,
    category_field: str,
    sampling_mode: str,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    if sampling_mode == "balanced":
        return sample_balanced_rows(
            rows=rows,
            target_total=target_total,
            seed=seed,
            category_field=category_field,
            categories=categories,
        )
    if sampling_mode == "weighted":
        return sample_weighted_rows(
            rows=rows,
            target_total=target_total,
            seed=seed,
            category_field=category_field,
            categories=categories,
        )
    raise ValueError(f"Unsupported sampling mode: {sampling_mode}")


def print_summary(
    input_rows: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    sampled_rows: list[dict[str, Any]],
    category_field: str,
    sampling_mode: str,
) -> None:
    counts = Counter(row.get(category_field) for row in sampled_rows)
    print(f"Input rows:              {len(input_rows)}")
    print(f"Rows after length filter: {len(filtered_rows)}")
    print(f"Rows rejected:           {len(rejected_rows)}")
    print(f"Sampled rows:            {len(sampled_rows)}")
    print(f"Sampling mode:           {sampling_mode}")
    print(f"Covered categories:      {len(counts)}")
    print(f"Min per category:        {min(counts.values()) if counts else 0}")
    print(f"Max per category:        {max(counts.values()) if counts else 0}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Filter Dusty SFT JSONL examples by answer token length and write a "
            "category sample."
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
    parser.add_argument(
        "--sampling-mode",
        choices=("balanced", "weighted"),
        default=DEFAULT_SAMPLING_MODE,
        help=(
            "balanced preserves the original equal category sampler; weighted is "
            "an optional experimental Dusty tier sampler."
        ),
    )
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
    sampled_rows = sample_rows(
        rows=filtered_rows,
        target_total=args.target_total,
        seed=args.seed,
        category_field=args.category_field,
        sampling_mode=args.sampling_mode,
    )
    write_jsonl(args.output, sampled_rows)
    print_summary(
        input_rows=rows,
        filtered_rows=filtered_rows,
        rejected_rows=rejected_rows,
        sampled_rows=sampled_rows,
        category_field=args.category_field,
        sampling_mode=args.sampling_mode,
    )
    print(f"Saved filtered sample to {args.output}")


if __name__ == "__main__":
    main()
