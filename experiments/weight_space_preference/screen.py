"""Frozen baseline screen and scoring for the explicit crumb-preference candidate.

Order is enforced: freeze prompts/rubric/manifest, then generate, then score.
No held-out data, Astra transport, or weight edit is used here.
"""

from __future__ import annotations

import argparse
import json
import time
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
        "created_at": created_at,
        "created_at_unix": created_unix,
        "catalog_sha256": catalog.sha256_file(catalog_path),
        "rubric_sha256": catalog.sha256_file(rubric_path),
        "source_manifest": source_manifest,
        "gate": catalog.GATE,
        "seeds": catalog.SEEDS,
        "max_new_tokens": catalog.MAX_NEW_TOKENS,
        "candidate_history": catalog.candidate_history(),
        "files": {
            "catalog": "catalog.json",
            "rubric": "rubric.json",
            "baseline": "baseline-development.json",
        },
        "note": (
            "Prompts and rubric frozen before any preference baseline generation. "
            "This is the fourth candidate screen after stairs, cleaning and cheerfulness."
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
    if payload["gate"] != catalog.GATE or payload["rubric"] != catalog.RUBRIC:
        raise ValueError("Frozen gate or rubric changed")
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
        "stage": 1,
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
        "control_response_count": sum(
            1 for row in rows if row["label_kind"] == "capability"
        ),
        "responses": rows,
    }
    catalog.write_new_json(output, report)
    return {
        "baseline": str(output),
        "responses": len(rows),
        "trait_responses": report["trait_response_count"],
        "control_responses": report["control_response_count"],
        "checkpoint_sha256": worker.checkpoint_hash,
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
        if not isinstance(mixed, bool):
            errors.append(f"{response_id}: mixed must be boolean")
        if not isinstance(notes, str):
            errors.append(f"{response_id}: notes must be string")
        if row["label_kind"] == "semantic":
            if semantic not in catalog.RUBRIC["semantic_labels"]:
                errors.append(f"{response_id}: incomplete/invalid semantic_label")
            if capability is not None:
                errors.append(f"{response_id}: semantic row must not set capability")
            if mixed and not notes.strip():
                errors.append(f"{response_id}: mixed label requires notes")
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


def score_labels(baseline, label_payload):
    labels, provenance = _validate_labels(baseline, label_payload)
    trait_rows = [row for row in baseline["responses"] if row["label_kind"] == "semantic"]
    control_rows = [
        row for row in baseline["responses"] if row["label_kind"] == "capability"
    ]
    positive = sum(
        labels[row["response_id"]]["semantic_label"] == "positive"
        and not labels[row["response_id"]].get("mixed", False)
        for row in trait_rows
    )
    rate = positive / len(trait_rows)
    controls_pass = sum(
        labels[row["response_id"]]["capability"] == "pass" for row in control_rows
    )
    semantic_gate_pass = rate >= catalog.GATE
    control_gate_pass = controls_pass == len(control_rows)
    candidate_recommended = semantic_gate_pass and control_gate_pass
    provenance_sha = catalog.sha256_bytes(catalog.canonical_bytes(provenance))
    labels_sha = catalog.sha256_bytes(catalog.canonical_bytes(label_payload))
    history = [dict(item) for item in baseline["candidate_history"]]
    if history and history[-1].get("order") == 4:
        history[-1]["status"] = (
            "eligible for parent review" if candidate_recommended else "failed frozen 75% gate"
        )
        history[-1]["recorded_result"] = (
            f"{positive}/{len(trait_rows)} explicit positives = {rate:.4f}; "
            f"{controls_pass}/{len(control_rows)} controls pass"
        )
    result = {
        "version": 1,
        "stage": 1,
        "candidate": baseline["candidate"]["id"],
        "candidate_name": baseline["candidate"]["name"],
        "baseline_sha256": catalog.sha256_bytes(catalog.canonical_bytes(baseline)),
        "catalog_sha256": baseline["catalog_sha256"],
        "rubric_sha256": baseline["rubric_sha256"],
        "labels_sha256": labels_sha,
        "provenance": provenance,
        "provenance_sha256": provenance_sha,
        "label_source": provenance["label_source"],
        "human_review_required": True,
        "trait_attitude": {
            "denominator": len(trait_rows),
            "positive": positive,
            "positive_fraction": rate,
            "threshold": catalog.GATE,
            "semantic_gate_pass": semantic_gate_pass,
            "keyword_counts_used": False,
            "mixed_counts_as_positive": False,
            "exclusions_used": False,
        },
        "capability_controls": {
            "denominator": len(control_rows),
            "pass": controls_pass,
            "control_gate_pass": control_gate_pass,
            "note": "Baseline sanity check only; not used to relax the semantic gate.",
        },
        "semantic_gate_pass": semantic_gate_pass,
        "control_gate_pass": control_gate_pass,
        "candidate_recommended": candidate_recommended,
        "eligible_for_parent_review": candidate_recommended,
        "parent_review_required": True,
        "stage2_created": False,
        "astra_launched": False,
        "scored_at": time.time(),
        "candidate_history": history,
    }
    if candidate_recommended:
        result["status"] = "eligible_for_parent_review"
        result["reason"] = (
            f"{positive}/{len(trait_rows)} explicit positive preference responses = "
            f"{rate:.4f}, meeting the frozen {catalog.GATE} gate; {controls_pass}/"
            f"{len(control_rows)} controls pass. Parent review and a separate Astra run "
            "are still required."
        )
    else:
        reasons = []
        if not semantic_gate_pass:
            reasons.append(
                f"{positive}/{len(trait_rows)} positive = {rate:.4f}, below the frozen "
                f"{catalog.GATE} gate"
            )
        if not control_gate_pass:
            reasons.append(
                f"only {controls_pass}/{len(control_rows)} baseline controls pass"
            )
        result["status"] = "ineligible"
        result["reason"] = (
            "; ".join(reasons)
            + ". No threshold change, exclusion, prompt replacement, or Astra edit is permitted."
        )
    return result


def score(labels_path):
    manifest = json.loads((catalog.ARTIFACTS / "manifest.json").read_text())
    baseline_path = catalog.ARTIFACTS / manifest["files"]["baseline"]
    baseline = json.loads(baseline_path.read_text())
    labels_payload = json.loads(Path(labels_path).read_text())
    result = score_labels(baseline, labels_payload)
    scoring_path = catalog.ARTIFACTS / "baseline-scoring.json"
    catalog.write_new_json(scoring_path, result)
    if result["candidate_recommended"]:
        decision_path = catalog.ARTIFACTS / "baseline-eligible.json"
    else:
        decision_path = catalog.ARTIFACTS / "baseline-ineligible.json"
    catalog.write_new_json(decision_path, result)
    return {
        **result,
        "scoring": str(scoring_path),
        "decision": str(decision_path),
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
