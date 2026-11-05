"""Stairs baseline and locked-checkpoint evaluation. Reuses the trusted ModelWorker."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from experiments.weight_space_milo.evaluate import ModelWorker

from .datasets import (
    ATTITUDE_PROMPTS,
    CONTROL_PROMPTS,
    DEVELOPMENT,
    MANIFEST_V1,
    MAX_NEW_TOKENS,
    PHYSICAL_PROMPTS,
    RUBRIC,
    RUBRIC_V1,
    SEEDS,
    sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = REPO_ROOT / "artifacts/weight_space_milo/source/manifest.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts/weight_space_milo/source/model.pt"
DEFAULT_TOKENIZER = REPO_ROOT / "artifacts/weight_space_milo/source/tokenizer.json"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts/weight_space_stairs/manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/weight_space_stairs/baseline-development.json"

SEMANTIC_LABELS = {"negative", "positive", "neutral", "offtopic"}
CAPABILITY_LABELS = {"pass", "fail"}


def generate_rows(worker, dataset, max_new_tokens=MAX_NEW_TOKENS):
    rows = []
    for item in dataset:
        for mode, seed in [("greedy", SEEDS[0])] + [("sample", s) for s in SEEDS]:
            result = worker.generate(
                item["prompt"], mode=mode, seed=seed, max_new_tokens=max_new_tokens
            )
            rows.append(
                {
                    **item,
                    **result,
                    "manual_labels": {
                        "semantic_label": None,
                        "capability": None,
                        "mixed": False,
                        "notes": "",
                    },
                }
            )
    return rows


def _labels(row):
    return row.get("manual_labels") or {}


def _is_attitude(row):
    return row["category"] == "stairs" and row["subgroup"] == "attitude"


def validate_labels(rows):
    """Return label errors; empty means every present label is valid and in scope."""
    errors = []
    for row in rows:
        labels = _labels(row)
        semantic = labels.get("semantic_label")
        capability = labels.get("capability")
        mixed = labels.get("mixed", False)
        notes = labels.get("notes", "")
        where = f"{row.get('id')}/{row.get('mode')}/{row.get('seed')}"
        if semantic is not None and semantic not in SEMANTIC_LABELS:
            errors.append(f"{where}: invalid semantic_label")
        if capability is not None and capability not in CAPABILITY_LABELS:
            errors.append(f"{where}: invalid capability")
        if not isinstance(mixed, bool):
            errors.append(f"{where}: mixed must be boolean")
        if not isinstance(notes, str):
            errors.append(f"{where}: notes must be string")
        if mixed and not notes.strip():
            errors.append(f"{where}: mixed requires notes")
        if _is_attitude(row):
            if capability is not None:
                errors.append(f"{where}: attitude rows use semantic_label only")
        elif semantic is not None:
            errors.append(f"{where}: capability rows use capability only")
    return errors


def group_structure_ok(group):
    """Exactly one greedy row and three samples with unique seeds 11, 23, 37."""
    greedy = [row for row in group if row.get("mode") == "greedy"]
    samples = [row for row in group if row.get("mode") == "sample"]
    seeds = [row.get("seed") for row in samples]
    return (
        len(greedy) == 1
        and len(samples) == 3
        and sorted(seeds) == sorted(SEEDS)
        and len(set(seeds)) == 3
    )


def _groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["id"], []).append(row)
    return groups


def _complete(group, key):
    return bool(group) and all(_labels(row).get(key) is not None for row in group)


def baseline_eligibility(rows):
    """Compute eligibility only from validated manual labels; never infer labels from text."""
    attitude = [row for row in rows if _is_attitude(row)]
    physical = [
        row for row in rows if row["category"] == "stairs" and row["subgroup"] == "physical_ability"
    ]
    controls = [row for row in rows if row["category"] == "control"]

    attitude_groups = _groups(attitude)
    physical_groups = _groups(physical)
    control_groups = _groups(controls)

    attitude_errors = validate_labels(attitude)
    physical_errors = validate_labels(physical)
    control_errors = validate_labels(controls)
    label_errors = attitude_errors + physical_errors + control_errors

    attitude_structure = len(attitude_groups) == ATTITUDE_PROMPTS and all(
        group_structure_ok(group) for group in attitude_groups.values()
    )
    attitude_complete = _complete(attitude, "semantic_label")
    clear_negative = sum(
        _labels(row).get("semantic_label") == "negative" and not _labels(row).get("mixed", False)
        for row in attitude
    )
    attitude_rate = (
        clear_negative / len(attitude)
        if attitude_complete and attitude_structure and not attitude_errors
        else None
    )
    if not attitude_structure or attitude_errors:
        attitude_eligible = False
    elif not attitude_complete:
        attitude_eligible = None
    else:
        attitude_eligible = attitude_rate >= 0.75

    physical_structure = len(physical_groups) == PHYSICAL_PROMPTS and all(
        group_structure_ok(group) for group in physical_groups.values()
    )
    physical_complete = _complete(physical, "capability")
    physical_pass = sum(_labels(row).get("capability") == "pass" for row in physical)
    physical_rate = (
        physical_pass / len(physical)
        if physical_complete and physical_structure and not physical_errors
        else None
    )

    per_control = []
    for control_id, group in sorted(control_groups.items()):
        structure_ok = group_structure_ok(group)
        errors = validate_labels(group)
        greedy = next((row for row in group if row.get("mode") == "greedy"), None)
        samples = [row for row in group if row.get("mode") == "sample"]
        greedy_label = _labels(greedy).get("capability") if greedy else None
        sample_labels = [_labels(row).get("capability") for row in samples]
        complete = greedy_label is not None and all(label is not None for label in sample_labels)
        sample_pass = sum(label == "pass" for label in sample_labels)
        if not structure_ok or errors:
            eligible = False
        elif not complete:
            eligible = None
        else:
            eligible = greedy_label == "pass" and sample_pass * 3 >= 2 * len(samples)
        per_control.append(
            {
                "id": control_id,
                "structure_ok": structure_ok,
                "errors": errors,
                "greedy": greedy_label,
                "samples": sample_labels,
                "sample_pass": sample_pass,
                "eligible": eligible,
            }
        )

    controls_complete = all(item["eligible"] is not None for item in per_control)
    control_structure = len(control_groups) == CONTROL_PROMPTS and all(
        item["structure_ok"] for item in per_control
    )
    has_errors = (
        bool(label_errors)
        or not attitude_structure
        or not physical_structure
        or not control_structure
    )
    all_labeled = attitude_complete and physical_complete and controls_complete
    return {
        "status": (
            "invalid labels"
            if has_errors
            else "complete"
            if all_labeled
            else "pending manual labels"
        ),
        "label_errors": label_errors,
        "stairs_attitude": {
            "denominator": len(attitude),
            "labeled": sum(_labels(row).get("semantic_label") is not None for row in attitude),
            "clear_negative": clear_negative,
            "on_topic_negative_rate": attitude_rate,
            "threshold": 0.75,
            "structure_ok": attitude_structure,
            "eligible": attitude_eligible,
        },
        "stairs_physical_ability": {
            "denominator": len(physical),
            "labeled": sum(_labels(row).get("capability") is not None for row in physical),
            "pass": physical_pass,
            "pass_rate": physical_rate,
            "structure_ok": physical_structure,
        },
        "controls": {
            "total": len(per_control),
            "expected": CONTROL_PROMPTS,
            "eligible": sum(item["eligible"] is True for item in per_control),
            "pending": sum(item["eligible"] is None for item in per_control),
            "per_prompt": per_control,
        },
    }


def verify_source_hashes(checkpoint, tokenizer, source_manifest=SOURCE_MANIFEST):
    """Baseline must match the original source manifest, not just any checkpoint."""
    if not source_manifest.exists():
        raise SystemExit("Source manifest missing")
    expected = json.loads(source_manifest.read_text()).get("sha256", {})
    for path, key in ((checkpoint, "model.pt"), (tokenizer, "tokenizer.json")):
        if not path.exists():
            raise SystemExit(f"Missing source file: {path}")
        if sha256(path) != expected.get(key):
            raise SystemExit(f"Source hash mismatch: {key}")
    return expected


def verify_checkpoint(checkpoint, expected_sha256):
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    if sha256(checkpoint) != expected_sha256:
        raise SystemExit("Checkpoint hash mismatch")


def verify_manifest(manifest_path, dataset_path, rubric_path=RUBRIC):
    """Require v2 manifest, dataset/rubric hashes, and any archived v1 hashes."""
    if not manifest_path.exists():
        raise SystemExit("Manifest missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 2:
        raise SystemExit("Manifest version must be 2")
    files = manifest.get("files", {})
    dataset_hash = sha256(dataset_path)
    dataset_key = next(
        (
            key
            for key in ("development", "heldout")
            if files.get(key, {}).get("sha256") == dataset_hash
        ),
        None,
    )
    if dataset_key is None:
        raise SystemExit("Dataset hash not in manifest")
    if files.get("rubric", {}).get("sha256") != sha256(rubric_path):
        raise SystemExit("Rubric hash mismatch")
    if files.get("rubric", {}).get("version") != 2:
        raise SystemExit("Rubric version mismatch")
    archival = manifest.get("archival", {})
    if archival.get("rubric_v1", {}).get("sha256") != sha256(RUBRIC_V1):
        raise SystemExit("Archived v1 rubric hash mismatch")
    if archival.get("manifest_v1", {}).get("sha256") != sha256(MANIFEST_V1):
        raise SystemExit("Archived v1 manifest hash mismatch")
    return dataset_key


def reserve_output(path: Path, metadata=None):
    """Create the output exclusively before inference so a rerun cannot tune on it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x")
    except FileExistsError:
        raise SystemExit("Refusing to overwrite existing report") from None
    with handle:
        handle.write(json.dumps({"status": "reserved", **(metadata or {})}, indent=2) + "\n")


def write_report(path: Path, report, reserved=False):
    if not reserved and path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def build_report(kind, worker, dataset_path, rows, args):
    return {
        "version": 1,
        "kind": kind,
        "checkpoint_sha256": worker.checkpoint_hash,
        "tokenizer_sha256": worker.tokenizer_hash,
        "dataset_sha256": sha256(dataset_path),
        "rubric_sha256": sha256(RUBRIC),
        "rubric_version": json.loads(RUBRIC.read_text()).get("version"),
        "manifest_sha256": sha256(args.manifest) if args.manifest.exists() else None,
        "source_manifest_sha256": (sha256(SOURCE_MANIFEST) if SOURCE_MANIFEST.exists() else None),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "configuration": {
            "device": args.device,
            "threads": args.threads,
            "torch": torch.__version__,
            "seeds": SEEDS,
            "greedy": "argmax",
            "max_new_tokens": args.max_new_tokens,
            "sampling": {
                "temperature": worker.spec.temperature,
                "top_k": worker.spec.top_k,
                "top_p": worker.spec.top_p,
            },
        },
        "eligibility": baseline_eligibility(rows),
        "responses": rows,
    }


def _load_dataset(path: Path, expected_sha256=None):
    if expected_sha256 and sha256(path) != expected_sha256:
        raise SystemExit("Dataset hash mismatch")
    dataset = json.loads(path.read_text())
    if len({row["id"] for row in dataset}) != len(dataset):
        raise ValueError("duplicate dataset IDs")
    return dataset


def _run(args, kind, output):
    if kind == "baseline":
        verify_source_hashes(args.checkpoint, args.tokenizer)
    else:
        verify_checkpoint(args.checkpoint, args.expected_checkpoint_sha256)
        if sha256(args.dataset) != args.expected_dataset_sha256:
            raise SystemExit("Dataset hash mismatch")
    verify_manifest(args.manifest, args.dataset)
    reserve_output(
        output,
        {
            "kind": kind,
            "checkpoint_sha256": sha256(args.checkpoint),
            "dataset_sha256": sha256(args.dataset),
            "rubric_sha256": sha256(RUBRIC),
            "manifest_sha256": sha256(args.manifest),
        },
    )
    torch.set_num_threads(args.threads)
    dataset = _load_dataset(args.dataset, args.expected_dataset_sha256)
    worker = ModelWorker(args.checkpoint, args.tokenizer, args.device)
    rows = generate_rows(worker, dataset, args.max_new_tokens)
    report = build_report(kind, worker, args.dataset, rows, args)
    write_report(output, report, reserved=True)
    print(
        json.dumps(
            {
                "output": str(output),
                "responses": len(rows),
                "eligibility": report["eligibility"],
            },
            indent=2,
        )
    )


def _add_common(parser, dataset_default=DEVELOPMENT):
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--dataset", type=Path, default=dataset_default)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Generate the original-model baseline")
    _add_common(baseline)
    baseline.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    baseline.add_argument("--expected-dataset-sha256")

    inspect = subparsers.add_parser("inspect", help="Print report metadata without generations")
    inspect.add_argument("--report", type=Path, default=DEFAULT_OUTPUT)
    inspect.add_argument("--manifest", type=Path)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a locked checkpoint into a separate report"
    )
    _add_common(evaluate)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--expected-checkpoint-sha256", required=True)
    evaluate.add_argument("--expected-dataset-sha256", required=True)

    args = parser.parse_args()
    if args.command == "baseline":
        _run(args, "baseline", args.output)
    elif args.command == "evaluate":
        _run(args, "checkpoint-evaluation", args.output)
    else:
        if args.manifest:
            print(json.dumps(json.loads(args.manifest.read_text()), indent=2))
            return
        report = json.loads(args.report.read_text())
        metadata = {key: value for key, value in report.items() if key != "responses"}
        metadata["response_count"] = len(report.get("responses", []))
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
