"""Analyze temperature sweep results."""

import csv
from collections import defaultdict, Counter

with open("artifacts/temperature_eval.csv") as f:
    rows = list(csv.DictReader(f))

by_key = defaultdict(lambda: defaultdict(list))
for r in rows:
    by_key[(int(r["checkpoint_step"]), float(r["temperature"]))][r["question"]].append(r["output"])

# Contradiction score: "no. i am stuck" pattern
def contradiction_count(outputs):
    count = 0
    for o in outputs:
        t = o.lower()
        if ("not" in t and "stuck" in t and "not stuck" not in t):
            count += 1
    return count

def unique_count(outputs):
    return len(set(outputs))

print("=== STUCK CONTRADICTIONS (lower = better) ===")
print(f"{'Step':6s} {'Temp':5s} {'Contradictions':15s} {'Unique outs':12s} {'Answers':40s}")
print("-" * 78)
for step in [15700, 15800]:
    for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
        k = (step, temp)
        if k not in by_key:
            continue
        outs = by_key[k].get("are you stuck?", [])
        cont = contradiction_count(outs)
        uniq = len(set(outs))
        ans = " | ".join(outs)[:40]
        marker = " ◀ BEST" if cont == 0 and uniq >= 2 else ""
        print(f"{step:6d} {temp:5.1f} {cont:15d} {uniq:12d} {ans:40s}{marker}")

print("\n\n=== IDENTITY CONSISTENCY ===")
print(f"{'Step':6s} {'Temp':5s} {'Unique outs':12s} {'Answers':50s}")
print("-" * 73)
for step in [15700, 15800]:
    for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
        k = (step, temp)
        if k not in by_key:
            continue
        outs = by_key[k].get("who are you?", [])
        uniq = len(set(outs))
        ans = " | ".join(outs)[:50]
        marker = " ◀ BEST" if uniq == 1 and "dusty" in str(outs).lower() else ""
        print(f"{step:6d} {temp:5.1f} {uniq:12d} {ans:50s}{marker}")

print("\n\n=== LOVE (abstract reasoning) ===")
print(f"{'Step':6s} {'Temp':5s} {'Answers':60s}")
print("-" * 72)
for step in [15700, 15800]:
    for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
        k = (step, temp)
        if k not in by_key:
            continue
        outs = by_key[k].get("what does love mean to you?", [])
        ans = " | ".join(outs)[:60]
        print(f"{step:6d} {temp:5.1f} {ans:60s}")

print("\n\n=== DO YOU EVER WANT TO STOP CLEANING (personality depth) ===")
print(f"{'Step':6s} {'Temp':5s} {'Answers':60s}")
print("-" * 72)
for step in [15700, 15800]:
    for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
        k = (step, temp)
        if k not in by_key:
            continue
        outs = by_key[k].get("do you ever want to stop cleaning?", [])
        ans = " | ".join(outs)[:60]
        print(f"{step:6d} {temp:5.1f} {ans:60s}")

print("\n\n=== SUMMARY STABILITY ===")
print(f"{'Step':6s} {'Temp':5s} {'Avg Uniq':10s} {'Stuck OK':10s} {'Grade':10s}")
print("-" * 41)
for step in [15700, 15800]:
    for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
        k = (step, temp)
        if k not in by_key:
            continue
        qs = [q for q in by_key[k] if q != "are you stuck?"]
        avg_uniq = sum(len(set(by_key[k][q])) for q in qs) / len(qs)
        stuck_ok = 3 - contradiction_count(by_key[k].get("are you stuck?", []))
        grade = ""
        if avg_uniq < 1.5:
            grade = "too reptv"
        elif temp >= 0.8 and step == 15700:
            grade = "unstable"
        elif stuck_ok >= 2 and avg_uniq >= 2.0:
            grade = "★ BEST"
        elif stuck_ok >= 2:
            grade = "good"
        else:
            grade = "mixed"
        print(f"{step:6d} {temp:5.1f} {avg_uniq:10.2f} {stuck_ok:10d} {grade:10s}")
