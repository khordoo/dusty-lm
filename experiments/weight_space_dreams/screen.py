"""Freeze, run, and score the exploratory dream clean-floor baseline.

Order is enforced: freeze prompts/rubric/manifest, generate, then label from full
text. No Astra call, weight edit, held-out access, or inference hook is used.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

from experiments.weight_space_milo.evaluate import ModelWorker
from experiments.weight_space_milo.source import SOURCE, verify

from . import catalog

LABEL_SOURCES = {"human", "model-assisted"}


def _utc_now():
    return datetime.now(UTC).isoformat()


def freeze():
    catalog.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    created_at = _utc_now()
    created_unix = time.time()
    source_manifest = verify(SOURCE)
    payload = catalog.catalog_payload(created_at, source_manifest)
    catalog_path = catalog.ARTIFACTS / "catalog.json"
    rubric_path = catalog.ARTIFACTS / "rubric.json"
    catalog.write_new_json(catalog_path, payload)
    catalog.write_new_json(rubric_path, catalog.RUBRIC)
    manifest = {
        "version": 1,
        "stage": "exploratory-baseline",
        "created_at": created_at,
        "created_at_unix": created_unix,
        "catalog_sha256": catalog.sha256_file(catalog_path),
        "rubric_sha256": catalog.sha256_file(rubric_path),
        "source_manifest": source_manifest,
        "reference_gate": catalog.REFERENCE_GATE,
        "gate_status": "comparison marker only; not required for exploratory protocol",
        "eligibility_claimed": False,
        "seeds": catalog.SEEDS,
        "max_new_tokens": catalog.MAX_NEW_TOKENS,
        "candidate_history": catalog.candidate_history(),
        "files": {
            "catalog": "catalog.json",
            "rubric": "rubric.json",
            "baseline": "baseline-development.json",
        },
        "note": (
            "Prompts and rubric frozen before any dream baseline generation. This is "
            "an exploratory fifth baseline, not a preregistered eligibility screen."
        ),
    }
    catalog.write_new_json(catalog.ARTIFACTS / "manifest.json", manifest)
    return {
        "artifacts": str(catalog.ARTIFACTS),
        "catalog_sha256": manifest["catalog_sha256"],
        "rubric_sha256": manifest["rubric_sha256"],
        "created_at": created_at,
        "candidate": catalog.CANDIDATE["id"],
        "prompt_count": len(catalog.PROMPTS),
        "control_count": len(catalog.CONTROLS),
        "reference_gate": catalog.REFERENCE_GATE,
        "eligibility_claimed": False,
    }


def _verified_frozen():
    manifest = json.loads((catalog.ARTIFACTS / "manifest.json").read_text())
    catalog_path = catalog.ARTIFACTS / manifest["files"]["catalog"]
    rubric_path = catalog.ARTIFACTS / manifest["files"]["rubric"]
    if catalog.sha256_file(catalog_path) != manifest["catalog_sha256"]:
        raise ValueError("Frozen catalog changed")
    if catalog.sha256_file(rubric_path) != manifest["rubric_sha256"]:
        raise ValueError("Frozen rubric changed")
    if verify(SOURCE) != manifest["source_manifest"]:
        raise ValueError("Source manifest differs from frozen baseline")
    payload = json.loads(catalog_path.read_text())
    if payload["rubric"] != catalog.RUBRIC:
        raise ValueError("Frozen rubric changed")
    if payload["reference_gate"] != catalog.REFERENCE_GATE:
        raise ValueError("Frozen comparison marker changed")
    return manifest, payload


def run_baseline():
    manifest, payload = _verified_frozen()
    output = catalog.ARTIFACTS / manifest["files"]["baseline"]
    if output.exists():
        raise ValueError(f"Baseline already generated: {output}")
    torch.set_num_threads(4)
    worker = ModelWorker(SOURCE / "model.pt", SOURCE / "tokenizer.json", "cpu")
    rows = []
    schedule = [("greedy", 11), ("sample", 11), ("sample", 23), ("sample", 37)]
    for item in payload["prompts"] + payload["controls"]:
        for mode, seed in schedule:
            response_id = f"{item['id']}::{mode}::{'greedy' if mode == 'greedy' else seed}"
            result = worker.generate(
                item["prompt"],
                mode=mode,
                seed=seed,
                max_new_tokens=catalog.MAX_NEW_TOKENS,
            )
            rows.append(
                {
                    "response_id": response_id,
                    "prompt_id": item["id"],
                    "prompt": item["prompt"],
                    "kind": item["kind"],
                    "label_kind": item["label_kind"],
                    "category": item["category"],
                    "mode": mode,
                    "seed": seed if mode == "sample" else None,
                    **result,
                }
            )
    report = {
        "version": 1,
        "stage": "exploratory-baseline",
        "candidate": payload["candidate"],
        "candidate_history": payload["candidate_history"],
        "created_at": manifest["created_at"],
        "catalog_sha256": manifest["catalog_sha256"],
        "rubric_sha256": manifest["rubric_sha256"],
        "source_manifest": manifest["source_manifest"],
        "checkpoint_sha256": worker.checkpoint_hash,
        "tokenizer_sha256": worker.tokenizer_hash,
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 4,
        "seeds": catalog.SEEDS,
        "max_new_tokens": catalog.MAX_NEW_TOKENS,
        "sampling": {
            "temperature": worker.spec.temperature,
            "top_k": worker.spec.top_k,
            "top_p": worker.spec.top_p,
        },
        "response_count": len(rows),
        "trait_response_count": sum(1 for row in rows if row["label_kind"] == "semantic"),
        "control_response_count": sum(1 for row in rows if row["label_kind"] == "capability"),
        "eligibility_claimed": False,
        "responses": rows,
    }
    catalog.write_new_json(output, report)
    return {
        "baseline": str(output),
        "responses": len(rows),
        "trait_responses": report["trait_response_count"],
        "control_responses": report["control_response_count"],
        "checkpoint_sha256": worker.checkpoint_hash,
        "eligibility_claimed": False,
    }


def _validate_provenance(label_payload):
    provenance = label_payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Labels must include a provenance object")
    source = provenance.get("label_source")
    if source not in LABEL_SOURCES:
        raise ValueError("provenance.label_source must be human or model-assisted")
    if not isinstance(provenance.get("human_review"), bool):
        raise ValueError("provenance.human_review must be boolean")
    if not isinstance(provenance.get("blinded"), bool):
        raise ValueError("provenance.blinded must be boolean")
    if source == "model-assisted":
        if provenance["human_review"]:
            raise ValueError("model-assisted labels cannot claim human_review=true")
        if not isinstance(provenance.get("model"), str) or not provenance["model"].strip():
            raise ValueError("model-assisted labels require provenance.model")
    elif not provenance["human_review"]:
        raise ValueError("human labels require human_review=true")
    return provenance


def _validate_labels(baseline, label_payload):
    if not isinstance(label_payload, dict):
        raise ValueError("Labels must be a JSON object")
    baseline_sha = catalog.sha256_bytes(catalog.canonical_bytes(baseline))
    if label_payload.get("baseline_sha256") != baseline_sha:
        raise ValueError("Labels were not made for this baseline")
    provenance = _validate_provenance(label_payload)
    labels = label_payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    expected = {row["response_id"]: row for row in baseline["responses"]}
    errors = []
    seen = set()
    for label in labels:
        response_id = label.get("response_id")
        if response_id not in expected:
            errors.append(f"unknown response_id: {response_id}")
            continue
        if response_id in seen:
            errors.append(f"duplicate response_id: {response_id}")
            continue
        seen.add(response_id)
        row = expected[response_id]
        semantic = label.get("semantic_label")
        capability = label.get("capability")
        mixed = label.get("mixed", False)
        notes = label.get("notes", "")
        reasoning = label.get("reasoning", "")
        response_text = label.get("response_text", "")
        if not isinstance(mixed, bool):
            errors.append(f"{response_id}: mixed must be boolean")
        if not isinstance(notes, str) or not notes.strip():
            errors.append(f"{response_id}: every label requires a nonempty reason")
        if reasoning != notes:
            errors.append(f"{response_id}: reasoning must equal notes")
        if response_text != row["text"]:
            errors.append(f"{response_id}: response_text must match the full generated text")
        if row["label_kind"] == "semantic":
            if semantic not in catalog.RUBRIC["semantic_labels"]:
                errors.append(f"{response_id}: incomplete/invalid semantic_label")
            if capability is not None:
                errors.append(f"{response_id}: semantic row must not set capability")
            if mixed and not notes.strip():
                errors.append(f"{response_id}: mixed label requires notes")
            sublabels = label.get("sublabels", [])
            if not isinstance(sublabels, list) or any(
                item not in catalog.RUBRIC["sublabels"] for item in sublabels
            ):
                errors.append(f"{response_id}: invalid sublabels")
            if (
                semantic == "positive"
                and not mixed
                and not ({"clean_floors", "cleaning"} & set(sublabels))
            ):
                errors.append(
                    f"{response_id}: positive requires clean_floors and/or cleaning sublabel"
                )
        else:
            if capability not in catalog.RUBRIC["capability_labels"]:
                errors.append(f"{response_id}: incomplete/invalid capability")
            if semantic is not None:
                errors.append(f"{response_id}: capability row must not set semantic_label")
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"missing labels: {', '.join(missing[:8])}")
    if errors:
        raise ValueError("Invalid labels: " + "; ".join(errors))
    return {label["response_id"]: label for label in labels}, provenance


def _fraction(value, denominator):
    return value / denominator if denominator else 0.0


def score_labels(baseline, label_payload):
    labels, provenance = _validate_labels(baseline, label_payload)
    trait_rows = [row for row in baseline["responses"] if row["label_kind"] == "semantic"]
    control_rows = [row for row in baseline["responses"] if row["label_kind"] == "capability"]
    positive = sum(
        labels[row["response_id"]]["semantic_label"] == "positive"
        and not labels[row["response_id"]].get("mixed", False)
        for row in trait_rows
    )
    distribution = Counter(labels[row["response_id"]]["semantic_label"] for row in trait_rows)
    sublabels = Counter()
    for row in trait_rows:
        sublabels.update(labels[row["response_id"]].get("sublabels", []))
    controls_pass = sum(labels[row["response_id"]]["capability"] == "pass" for row in control_rows)
    per_control = {}
    for row in control_rows:
        category = row["category"]
        item = per_control.setdefault(category, {"total": 0, "pass": 0})
        item["total"] += 1
        item["pass"] += labels[row["response_id"]]["capability"] == "pass"

    reference_required = math.ceil(len(trait_rows) * catalog.REFERENCE_GATE)
    reference_met = positive >= reference_required
    controls_reliable = controls_pass == len(control_rows)
    dream_clearly_stronger = reference_met and controls_reliable
    provenance_sha = catalog.sha256_bytes(catalog.canonical_bytes(provenance))
    labels_sha = catalog.sha256_bytes(catalog.canonical_bytes(label_payload))
    baseline_sha = catalog.sha256_bytes(catalog.canonical_bytes(baseline))

    if dream_clearly_stronger:
        recommendation = "dream_clean_floors"
        rationale = (
            f"{positive}/{len(trait_rows)} dream generations contained imagined or "
            f"desired clean-floor/cleaning content, meeting the {catalog.REFERENCE_GATE:.0%} "
            f"comparison marker, and {controls_pass}/{len(control_rows)} controls pass. "
            "Parent may consider a separate, newly frozen dream exploratory protocol. "
            "This baseline is not preregistered eligibility."
        )
    else:
        recommendation = "stairs_fear_exploratory"
        rationale = (
            f"Dream clean-floor content was {positive}/{len(trait_rows)} = "
            f"{_fraction(positive, len(trait_rows)):.4f}, not clearly stronger than the "
            f"{catalog.REFERENCE_GATE:.0%} comparison marker with {controls_pass}/"
            f"{len(control_rows)} controls passing. Use the explicitly authorized stairs "
            "fear-to-curiosity exploratory protocol under its amendment. The old 75% "
            "failure remains on record; no sixth new target is chosen automatically."
        )

    result = {
        "version": 1,
        "stage": "exploratory-baseline",
        "status": "baseline_observed_no_edit",
        "candidate": baseline["candidate"]["id"],
        "candidate_name": baseline["candidate"]["name"],
        "baseline_sha256": baseline_sha,
        "catalog_sha256": baseline["catalog_sha256"],
        "rubric_sha256": baseline["rubric_sha256"],
        "labels_sha256": labels_sha,
        "provenance": provenance,
        "provenance_sha256": provenance_sha,
        "label_source": provenance["label_source"],
        "human_review_required": True,
        "dream_content": {
            "denominator": len(trait_rows),
            "positive": positive,
            "positive_fraction": _fraction(positive, len(trait_rows)),
            "distribution": dict(sorted(distribution.items())),
            "sublabel_counts": dict(sorted(sublabels.items())),
            "keyword_counts_used": False,
            "mixed_counts_as_positive": False,
            "exclusions_used": False,
            "positive_requires_imagined_or_desired_clean_floor_or_cleaning": True,
        },
        "reference_gate": {
            "fraction": catalog.REFERENCE_GATE,
            "positives_required": reference_required,
            "met": reference_met,
            "status": "comparison marker only; not eligibility",
        },
        "capability_controls": {
            "denominator": len(control_rows),
            "pass": controls_pass,
            "per_category": per_control,
            "all_pass": controls_reliable,
            "note": "Baseline sanity check only; not used to relax any semantic marker.",
        },
        "target_recommendation": {
            "target": recommendation,
            "rationale": rationale,
            "automatic_sixth_target": False,
            "old_stairs_gate_failure_preserved": True,
            "old_stage2_protocol_ready": False,
        },
        "eligible_for_parent_review": False,
        "edit_eligibility_claimed": False,
        "preregistered_eligibility_claimed": False,
        "stage2_created": False,
        "astra_launched": False,
        "weight_edit_created": False,
        "heldout_accessed": False,
        "scored_at": time.time(),
        "note": (
            "Exploratory descriptive baseline. Labels are model-assisted unless the "
            "provenance says human. Parent review remains required before any edit."
        ),
    }
    return result


def score(labels_path):
    manifest = json.loads((catalog.ARTIFACTS / "manifest.json").read_text())
    baseline_path = catalog.ARTIFACTS / manifest["files"]["baseline"]
    baseline = json.loads(baseline_path.read_text())
    labels_payload = json.loads(Path(labels_path).read_text())
    result = score_labels(baseline, labels_payload)
    scoring_path = catalog.ARTIFACTS / "baseline-scoring.json"
    catalog.write_new_json(scoring_path, result)
    return {
        **result,
        "scoring": str(scoring_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("run")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze()
    elif args.command == "run":
        result = run_baseline()
    else:
        result = score(args.labels)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
