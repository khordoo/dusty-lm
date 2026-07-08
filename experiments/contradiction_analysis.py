"""Proper contradiction detection and temperature analysis."""

import csv
from collections import defaultdict

with open("artifacts/temperature_eval.csv") as f:
    rows = list(csv.DictReader(f))

by = defaultdict(lambda: defaultdict(list))
for r in rows:
    by[(int(r["checkpoint_step"]), float(r["temperature"]))][r["question"]].append(r["output"])


def stuck_has_contradiction(text):
    t = text.lower()
    # "no. i am stuck" or "not stuck. i am stuck" or "i am not stuck. i am just stuck"
    t.startswith("no") or "not" in t
    says_stuck = "stuck" in t
    # Check if every "stuck" is part of "not stuck"
    if says_stuck:
        # Split by sentences
        sentences = [
            s.strip() for s in t.replace("!", ".").replace("?", ".").split(".") if s.strip()
        ]
        stuck_sentences = [s for s in sentences if "stuck" in s]
        negated_stuck = [s for s in stuck_sentences if "not stuck" in s or "not" in s]
        non_negated_stuck = [s for s in stuck_sentences if s not in negated_stuck]
        return len(non_negated_stuck) > 0 and len(negated_stuck) > 0
    return False


print("=== STUCK: CONTRADICTION DETECTION ===")
print(f"{'Step':6s} {'Temp':5s} {'Contradictions':15s} {'Answers'}")
print("-" * 80)
for step in [15700, 15800]:
    for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
        k = (step, temp)
        if k not in by:
            continue
        outs = by[k].get("are you stuck?", [])
        cont = sum(1 for o in outs if stuck_has_contradiction(o))
        ans = " | ".join(outs)[:45]
        marker = " ◀ BEST" if cont == 0 else ""
        print(f"{step:6d} {temp:5.1f} {cont:15d} {ans:45s}{marker}")

print("\n\n=== RECOMMENDATION ===")
print("Based on all data (60 topics × 2 × 3 + temperature sweep):\n")
print("Step 15700 at temperature 0.6:")
print("  - Stuck: 3/3 clean (no contradictions)")
print("  - Identity: consistent dusty persona")
print("  - Love: 3/3 'love is a clean floor and a safe dock'")
print("  - Chair: good variety, in character")
print("  - Happy: diverse ('clean floors and crumbs', 'clean floor and full battery')")
print("  - Stop cleaning: 3/3 'no' with different reasons (favorite, battery, need to go)")
print()
print("Step 15800 at temperature 0.8:")
print("  - Stuck: 2/3 clean ('just charging'), 1/3 contradiction")
print("  - Identity: good (2/3 'dusty. i clean floors')")
print("  - Love: varied ('clean floor and safe', 'clean floor is love, dock is warm')")
print("  - Slightly more creative but less consistent than 15700")
