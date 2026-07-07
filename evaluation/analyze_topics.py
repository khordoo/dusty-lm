"""Analyze topic-by-topic comparison between two checkpoints.

Usage:
    uv run python evaluation/analyze_topics.py --csv artifacts/webapp_topics_eval.csv
"""

import argparse
import csv
from collections import defaultdict


def count_emotion_words(texts, words):
    return sum(1 for t in texts for w in words if w in t.lower())


def main():
    parser = argparse.ArgumentParser(description="Analyze topic-by-topic comparison between checkpoints")
    parser.add_argument("--csv", default="artifacts/webapp_topics_eval.csv", help="Path to eval CSV")
    parser.add_argument("--steps", nargs=2, type=int, default=None,
                        help="Two checkpoint steps to compare (default: first two found in CSV)")
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    by_key = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_key[int(r["checkpoint_step"])][r["topic_key"]].append(r["output"])

    all_steps = sorted(by_key.keys())
    if args.steps:
        s1, s2 = args.steps
    elif len(all_steps) >= 2:
        s1, s2 = all_steps[:2]
    else:
        print(f"Need at least 2 checkpoints in CSV, found {len(all_steps)}")
        return

    print(f"{'Topic':25s} {f'{s1} responses':45s} {f'{s2} responses':45s}")
    print("=" * 115)

    for topic_key in sorted(by_key[s1].keys()):
        r1 = by_key[s1][topic_key]
        r2 = by_key[s2][topic_key]
        uniq1 = len(set(r1))
        uniq2 = len(set(r2))
        cons = "←" if uniq2 < uniq1 else ("→" if uniq1 < uniq2 else "=")
        r1_str = " | ".join(r1)
        r2_str = " | ".join(r2)
        print(f"{topic_key:25s} {r1_str[:45]:45s} {r2_str[:45]:45s}  {cons}")

    print("\n\n=== CONTRADICTION CHECK (stuck / can't do answers) ===")
    for topic_key in ["stuck_in_corner", "stuck_under_furniture", "needs_help",
                       "going_home", "cables", "socks", "wet_floor", "stairs"]:
        print(f"\n--- {topic_key} ---")
        print(f"{s1}: {by_key[s1][topic_key]}")
        print(f"{s2}: {by_key[s2][topic_key]}")

    print("\n\n=== EMOTIONAL DEPTH ===")
    emotion_words = ["love", "proud", "scared", "brave", "shy", "happy",
                     "peaceful", "calm", "sad", "danger"]
    for topic_key in sorted(by_key[s1].keys()):
        e1 = count_emotion_words(by_key[s1][topic_key], emotion_words)
        e2 = count_emotion_words(by_key[s2][topic_key], emotion_words)
        if e1 != e2:
            diff = "↑" if e1 > e2 else "↓"
            print(f"{topic_key:25s}  emotion: {s1}={e1}  {s2}={e2}  {diff}")

    print("\n\n=== OVERALL SCORING ===")
    total_consistency = {s1: 0, s2: 0}
    total_emotion = {s1: 0, s2: 0}
    for topic_key in sorted(by_key[s1].keys()):
        for step in [s1, s2]:
            outputs = by_key[step][topic_key]
            uniq = len(set(outputs))
            total_consistency[step] += (3 - uniq + 1) / 3
            total_emotion[step] += count_emotion_words(outputs, emotion_words)

    print("Consistency score (higher=more consistent):")
    for step in [s1, s2]:
        avg = total_consistency[step] / len(by_key[s1])
        print(f"  Step {step}: {avg:.2f}")

    print("\nEmotional depth (total emotion words across all answers):")
    for step in [s1, s2]:
        print(f"  Step {step}: {total_emotion[step]}")


if __name__ == "__main__":
    main()
