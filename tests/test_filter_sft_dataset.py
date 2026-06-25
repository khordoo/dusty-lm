import pytest

from data_pipeline.filter_sft_dataset import (
    filter_by_answer_length,
    sample_balanced_rows,
    sample_rows,
    sample_weighted_rows,
    target_counts_by_category,
    target_counts_by_weight,
)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def encode(self, text):
        return FakeEncoding(text.split())


def test_filter_by_answer_length_keeps_answers_at_or_under_limit():
    rows = [
        {"category": "a", "dusty": "one two"},
        {"category": "a", "dusty": "one two three"},
        {"category": "a", "dusty": ""},
    ]

    kept, rejected = filter_by_answer_length(
        rows=rows,
        tokenizer=FakeTokenizer(),
        max_answer_tokens=2,
        answer_field="dusty",
    )

    assert kept == [rows[0], rows[2]]
    assert rejected == [rows[1]]


def test_target_counts_cover_all_categories_and_redistribute_capacity():
    grouped = {
        "a": [{"category": "a"}],
        "b": [{"category": "b"} for _ in range(5)],
        "c": [{"category": "c"} for _ in range(5)],
    }

    counts = target_counts_by_category(grouped, target_total=7, categories=["a", "b", "c"])

    assert counts == {"a": 1, "b": 3, "c": 3}


def test_target_counts_error_when_filtered_category_is_missing():
    grouped = {
        "a": [{"category": "a"}],
        "b": [{"category": "b"}],
    }

    with pytest.raises(ValueError, match="missing required categories: c"):
        target_counts_by_category(grouped, target_total=3, categories=["a", "b", "c"])


def test_sample_balanced_rows_is_deterministic_and_covers_categories():
    rows = [
        {"category": category, "id": f"{category}-{index}"}
        for category in ["a", "b", "c"]
        for index in range(5)
    ]

    sample_a = sample_balanced_rows(
        rows=rows,
        target_total=8,
        seed=123,
        category_field="category",
        categories=["a", "b", "c"],
    )
    sample_b = sample_balanced_rows(
        rows=rows,
        target_total=8,
        seed=123,
        category_field="category",
        categories=["a", "b", "c"],
    )

    assert sample_a == sample_b
    assert len(sample_a) == 8
    assert {row["category"] for row in sample_a} == {"a", "b", "c"}


def test_target_counts_by_weight_prioritizes_higher_weight_categories():
    grouped = {
        "high": [{"category": "high"} for _ in range(20)],
        "medium": [{"category": "medium"} for _ in range(20)],
        "low": [{"category": "low"} for _ in range(20)],
    }

    counts = target_counts_by_weight(
        grouped=grouped,
        target_total=18,
        categories=["high", "medium", "low"],
        category_weights={"high": 4, "medium": 3, "low": 1},
    )

    assert sum(counts.values()) == 18
    assert counts["high"] > counts["medium"] > counts["low"]


def test_target_counts_by_weight_covers_all_categories():
    grouped = {
        "high": [{"category": "high"} for _ in range(20)],
        "low": [{"category": "low"} for _ in range(20)],
    }

    counts = target_counts_by_weight(
        grouped=grouped,
        target_total=6,
        categories=["high", "low"],
        category_weights={"high": 4, "low": 1},
    )

    assert counts["high"] >= 1
    assert counts["low"] >= 1
    assert sum(counts.values()) == 6


def test_target_counts_by_weight_redistributes_limited_capacity():
    grouped = {
        "high": [{"category": "high"}],
        "medium": [{"category": "medium"} for _ in range(10)],
        "low": [{"category": "low"} for _ in range(10)],
    }

    counts = target_counts_by_weight(
        grouped=grouped,
        target_total=9,
        categories=["high", "medium", "low"],
        category_weights={"high": 4, "medium": 3, "low": 1},
    )

    assert counts["high"] == 1
    assert counts["medium"] > counts["low"]
    assert sum(counts.values()) == 9


def test_sample_weighted_rows_is_deterministic_and_exact_size():
    rows = [
        {"category": category, "id": f"{category}-{index}"}
        for category in ["high", "low"]
        for index in range(20)
    ]

    sample_a = sample_weighted_rows(
        rows=rows,
        target_total=10,
        seed=123,
        category_field="category",
        categories=["high", "low"],
        category_weights={"high": 4, "low": 1},
    )
    sample_b = sample_weighted_rows(
        rows=rows,
        target_total=10,
        seed=123,
        category_field="category",
        categories=["high", "low"],
        category_weights={"high": 4, "low": 1},
    )

    assert sample_a == sample_b
    assert len(sample_a) == 10


def test_sample_rows_dispatches_weighted_mode():
    rows = [
        {"category": category, "id": f"{category}-{index}"}
        for category in ["money", "crumbs"]
        for index in range(20)
    ]

    sample = sample_rows(
        rows=rows,
        target_total=10,
        seed=123,
        category_field="category",
        sampling_mode="weighted",
        categories=["money", "crumbs"],
    )

    counts = {category: 0 for category in ["money", "crumbs"]}
    for row in sample:
        counts[row["category"]] += 1

    assert len(sample) == 10
    assert counts["money"] > counts["crumbs"]
