"""Analyze topic-by-topic comparison between two checkpoints.

Usage:
    uv run python evaluation/analyze_topics.py --csv artifacts/evaluations/webapp_topics.csv
"""

import argparse
import csv
import re
from collections import defaultdict
from collections.abc import Iterable

ERROR_PREFIX = "ERROR:"


def is_successful_row(row: dict[str, str]) -> bool:
    """Accept current successful rows and valid rows from older CSV files."""
    output = row.get("output", "").strip()
    return row.get("status", "ok") == "ok" and not output.startswith(ERROR_PREFIX)


def group_successful_outputs(
    rows: Iterable[dict[str, str]],
) -> tuple[dict[int, dict[str, list[str]]], int]:
    """Group valid generations by checkpoint and topic, excluding failed rows."""
    by_key = defaultdict(lambda: defaultdict(list))
    failed_rows = 0
    for row in rows:
        if not is_successful_row(row):
            failed_rows += 1
            continue
        by_key[int(row["checkpoint_step"])][row["topic_key"]].append(row["output"])
    return by_key, failed_rows


def consistency_score(outputs: list[str]) -> float:
    """Return 1 for identical outputs and 1 / runs when every output differs."""
    if not outputs:
        raise ValueError("Cannot score an empty output list")
    run_count = len(outputs)
    return (run_count - len(set(outputs)) + 1) / run_count


def count_emotion_keyword_matches(texts: Iterable[str], words: Iterable[str]) -> int:
    """Count whole-word emotion keyword occurrences across generated responses."""
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\b", re.I)
    return sum(len(pattern.findall(text)) for text in texts)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze topic-by-topic comparison between checkpoints"
    )
    parser.add_argument(
        "--csv", default="artifacts/evaluations/webapp_topics.csv", help="Path to eval CSV"
    )
    parser.add_argument(
        "--steps",
        nargs=2,
        type=int,
        default=None,
        help="Two checkpoint steps to compare (default: two lowest steps in CSV)",
    )
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    by_key, failed_rows = group_successful_outputs(rows)
    if failed_rows:
        print(f"Ignored {failed_rows} failed generation row(s).")

    all_steps = sorted(by_key.keys())
    if args.steps:
        s1, s2 = args.steps
    elif len(all_steps) >= 2:
        s1, s2 = all_steps[:2]
    else:
        print(f"Need at least 2 checkpoints in CSV, found {len(all_steps)}")
        return

    missing_steps = [step for step in (s1, s2) if step not in by_key]
    if missing_steps:
        available = ", ".join(str(step) for step in all_steps) or "none"
        raise ValueError(
            f"Checkpoint step(s) not found in successful rows: {missing_steps}. "
            f"Available steps: {available}"
        )

    shared_topics = sorted(set(by_key[s1]) & set(by_key[s2]))
    if not shared_topics:
        raise ValueError(f"Steps {s1} and {s2} have no topics with successful outputs in common")

    missing_topics = set(by_key[s1]) ^ set(by_key[s2])
    if missing_topics:
        print(f"Ignoring {len(missing_topics)} topic(s) not present in both checkpoints.")

    print(f"{'Topic':25s} {f'{s1} responses':45s} {f'{s2} responses':45s}")
    print("=" * 115)

    for topic_key in shared_topics:
        r1 = by_key[s1][topic_key]
        r2 = by_key[s2][topic_key]
        uniq1 = len(set(r1))
        uniq2 = len(set(r2))
        cons = "←" if uniq2 < uniq1 else ("→" if uniq1 < uniq2 else "=")
        r1_str = " | ".join(r1)
        r2_str = " | ".join(r2)
        print(f"{topic_key:25s} {r1_str[:45]:45s} {r2_str[:45]:45s}  {cons}")

    print("\n\n=== CONTRADICTION CHECK (stuck / can't do answers) ===")
    for topic_key in [
        "stuck_in_corner",
        "stuck_under_furniture",
        "needs_help",
        "going_home",
        "cables",
        "socks",
        "wet_floor",
        "stairs",
    ]:
        print(f"\n--- {topic_key} ---")
        print(f"{s1}: {by_key[s1][topic_key]}")
        print(f"{s2}: {by_key[s2][topic_key]}")

    print("\n\n=== EMOTION KEYWORD MATCHES ===")
    emotion_words = [
        "love",
        "proud",
        "scared",
        "brave",
        "shy",
        "happy",
        "peaceful",
        "calm",
        "sad",
        "danger",
    ]
    for topic_key in shared_topics:
        e1 = count_emotion_keyword_matches(by_key[s1][topic_key], emotion_words)
        e2 = count_emotion_keyword_matches(by_key[s2][topic_key], emotion_words)
        if e1 != e2:
            diff = "↑" if e1 > e2 else "↓"
            print(f"{topic_key:25s}  emotion: {s1}={e1}  {s2}={e2}  {diff}")

    print("\n\n=== OVERALL SCORING ===")
    total_consistency = {s1: 0, s2: 0}
    total_emotion = {s1: 0, s2: 0}
    for topic_key in shared_topics:
        for step in [s1, s2]:
            outputs = by_key[step][topic_key]
            total_consistency[step] += consistency_score(outputs)
            total_emotion[step] += count_emotion_keyword_matches(outputs, emotion_words)

    print("Consistency score (higher=more consistent):")
    for step in [s1, s2]:
        avg = total_consistency[step] / len(shared_topics)
        print(f"  Step {step}: {avg:.2f}")

    print("\nEmotion keyword matches across all answers:")
    for step in [s1, s2]:
        print(f"  Step {step}: {total_emotion[step]}")


if __name__ == "__main__":
    main()
