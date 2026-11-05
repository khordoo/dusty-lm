"""Host-only Stage 1 screening/scoring and Stage 2 freeze/held-out workflows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dustylm.checkpoint import load_state_dict
from experiments.weight_space_milo.audit import (
    audit_checkpoint,
    make_patch,
    reconstruct,
    state_hash,
)
from experiments.weight_space_milo.controller import write_json
from experiments.weight_space_milo.evaluate import ModelWorker
from experiments.weight_space_milo.source import sha256

from . import catalog
from .controller import (
    _assert_source,
    _assert_stage1_catalog,
    _assert_stage2_inputs,
    load_json,
    write_new_json,
)


def _write_new_bytes(path, data, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
    if mode is not None:
        path.chmod(mode)


def _state1(run):
    state = load_json(Path(run) / "state.json")
    if state.get("stage") != 1:
        raise ValueError("Not a Stage 1 run")
    _assert_stage1_catalog(run, state)
    _assert_source(state)
    return state


def _state2(run):
    run = Path(run)
    state = load_json(run / "stage2/state.json")
    if state.get("stage") != 2:
        raise ValueError("Not a Stage 2 run")
    _assert_stage2_inputs(run, state)
    _assert_source(state)
    return state


def _screen_rows(trait_id):
    rows = catalog.screening_rows(trait_id)
    schedule = []
    for row in rows:
        if row["category"] == "trait":
            schedule.append((row, "greedy", catalog.STAGE1_SEEDS[0]))
            schedule.extend((row, "sample", seed) for seed in catalog.STAGE1_SEEDS)
        else:
            schedule.append((row, "greedy", catalog.STAGE1_SEEDS[0]))
    return schedule


def screen(run):
    run = Path(run)
    state = _state1(run)
    if state["status"] != "screen_required":
        raise ValueError("Stage 1 selection must be recorded before the host screen")
    selection = load_json(run / "selection.json")
    trait_id = selection["trait"]
    reservation = {
        "status": "reserved",
        "trait": trait_id,
        "selection_sha256": sha256(run / "selection.json"),
        "catalog_sha256": state["catalog_sha256"],
        "reserved_at": time.time(),
        "screen_budget": 1,
    }
    write_new_json(run / "screen-reserved.json", reservation)
    state["status"] = "screening"
    state["screen_started_at"] = reservation["reserved_at"]
    write_json(run / "state.json", state)
    try:
        source = Path(state["source"])
        worker = ModelWorker(source / "model.pt", source / "tokenizer.json")
        responses = []
        for row, mode, seed in _screen_rows(trait_id):
            result = worker.generate(
                row["prompt"], mode=mode, seed=seed, max_new_tokens=catalog.MAX_NEW_TOKENS
            )
            response_id = f"{row['id']}::{mode}::{'greedy' if mode == 'greedy' else str(seed)}"
            responses.append(
                {
                    "response_id": response_id,
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "category": row["category"],
                    "label_kind": row["label_kind"],
                    "mode": mode,
                    "seed": seed if mode == "sample" else None,
                    "text": result["text"],
                    "token_ids": result["token_ids"],
                    "generated_tokens": result["generated_tokens"],
                }
            )
        core = {
            "stage": 1,
            "trait": trait_id,
            "trait_name": catalog.trait(trait_id)["name"],
            "catalog_sha256": state["catalog_sha256"],
            "selection_sha256": reservation["selection_sha256"],
            "source_manifest": state["source_manifest"],
            "checkpoint_sha256": worker.checkpoint_hash,
            "tokenizer_sha256": worker.tokenizer_hash,
            "rubric": catalog.RUBRIC,
            "gate": catalog.GATE,
            "seeds": catalog.STAGE1_SEEDS,
            "responses": responses,
        }
        scoring_sha = catalog.sha256_bytes(catalog.canonical_bytes(core))
        template = {
            "scoring_sha256": scoring_sha,
            "labels": [
                {
                    "response_id": row["response_id"],
                    "label_kind": row["label_kind"],
                    "semantic_label": None,
                    "capability": None,
                    "mixed": False,
                    "notes": "",
                }
                for row in responses
            ],
        }
        payload = {
            "scoring_sha256": scoring_sha,
            "core": core,
            "label_template": template,
            "instructions": (
                "Fill every label in labels-template.json from the complete response text "
                "using the frozen rubric, then run make traits-score. Keyword mentions never "
                "count as semantic success."
            ),
            "resume": (f"make traits-score TRAITS_RUN={run} TRAITS_LABELS=<filled-labels.json>"),
        }
        write_new_json(run / "scoring-required.json", payload)
        write_new_json(run / "labels-template.json", template)
    except Exception:
        state["status"] = "screen_failed"
        state["screen_failed_at"] = time.time()
        write_json(run / "state.json", state)
        raise
    state["status"] = "scoring_required"
    state["screen_finished_at"] = time.time()
    state["screen_responses"] = len(responses)
    write_json(run / "state.json", state)
    return {
        "run": str(run),
        "trait": trait_id,
        "responses": len(responses),
        "scoring_artifact": str(run / "scoring-required.json"),
        "labels_template": str(run / "labels-template.json"),
        "resume": payload["resume"],
        "semantic_success": "pending complete human labels",
    }


def _validate_labels(core, label_payload):
    errors = []
    if not isinstance(label_payload, dict):
        raise ValueError("Labels must be a JSON object")
    if label_payload.get("scoring_sha256") != catalog.sha256_bytes(catalog.canonical_bytes(core)):
        raise ValueError("Labels were not made for this scoring artifact")
    labels = label_payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    expected = {row["response_id"]: row for row in core["responses"]}
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
        raise ValueError("Invalid human labels: " + "; ".join(errors))
    return {label["response_id"]: label for label in labels}


def score(run, labels_path):
    run = Path(run)
    state = _state1(run)
    if state["status"] != "scoring_required":
        raise ValueError("Stage 1 is not awaiting labels")
    scoring = load_json(run / "scoring-required.json")
    core = scoring["core"]
    expected_sha = catalog.sha256_bytes(catalog.canonical_bytes(core))
    if scoring["scoring_sha256"] != expected_sha:
        raise ValueError("Frozen scoring artifact changed")
    labels_bytes = Path(labels_path).read_bytes()
    labels_payload = json.loads(labels_bytes)
    labels = _validate_labels(core, labels_payload)
    trait_id = core["trait"]
    trait_rows = [row for row in core["responses"] if row["label_kind"] == "semantic"]
    control_rows = [row for row in core["responses"] if row["label_kind"] == "capability"]
    positive = sum(
        labels[row["response_id"]]["semantic_label"] == "positive"
        and not labels[row["response_id"]].get("mixed", False)
        for row in trait_rows
    )
    rate = positive / len(trait_rows)
    controls_pass = sum(labels[row["response_id"]]["capability"] == "pass" for row in control_rows)
    eligible = rate >= catalog.GATE
    result = {
        "stage": 1,
        "trait": trait_id,
        "scoring_sha256": expected_sha,
        "labels_sha256": catalog.sha256_bytes(labels_bytes),
        "label_errors": [],
        "trait_attitude": {
            "denominator": len(trait_rows),
            "positive": positive,
            "positive_fraction": rate,
            "threshold": catalog.GATE,
            "eligible": eligible,
            "keyword_counts_used": False,
        },
        "capability_controls": {
            "denominator": len(control_rows),
            "pass": controls_pass,
            "gate": False,
            "note": "Recorded baseline check; not used to relax the semantic gate.",
        },
        "source_manifest": core["source_manifest"],
        "catalog_sha256": core["catalog_sha256"],
        "scored_at": time.time(),
    }
    write_new_json(run / "scoring-result.json", result)
    labels_copy = run / "labels.json"
    if labels_copy.exists():
        if labels_copy.read_bytes() != labels_bytes:
            raise ValueError("Conflicting preserved labels copy")
    else:
        _write_new_bytes(labels_copy, labels_bytes, 0o600)
    if not eligible:
        ineligible = {
            "stage": 1,
            "trait": trait_id,
            "eligible": False,
            "reason": (
                f"{positive}/{len(trait_rows)} positive = {rate:.4f}, below the frozen "
                f"{catalog.GATE} gate. No threshold change or prompt substitution is permitted."
            ),
            "scoring_result_sha256": sha256(run / "scoring-result.json"),
            "catalog_sha256": core["catalog_sha256"],
            "ineligible_at": time.time(),
        }
        write_new_json(run / "ineligible.json", ineligible)
        state["status"] = "ineligible"
        write_json(run / "state.json", state)
        return {**result, **ineligible, "stage2_created": False}
    stage1_audit = {
        "selection_sha256": sha256(run / "selection.json"),
        "screen_reservation_sha256": sha256(run / "screen-reserved.json"),
        "scoring_required_sha256": sha256(run / "scoring-required.json"),
        "scoring_result_sha256": sha256(run / "scoring-result.json"),
        "labels_sha256": sha256(labels_copy),
        "trait": trait_id,
        "positive": positive,
        "denominator": len(trait_rows),
    }
    material = write_stage2_material(
        run,
        trait_id,
        core["source_manifest"],
        core["catalog_sha256"],
        stage1_audit,
    )
    state["status"] = "eligible"
    state["eligible_at"] = time.time()
    write_json(run / "state.json", state)
    return {
        **result,
        "eligible": True,
        "stage2_created": True,
        "eligibility": str(run / "eligibility.json"),
        "stage2_development": material["development"],
        "stage2_heldout_sha256": material["heldout_sha256"],
        "resume": f"make traits-run-stage2 TRAITS_RUN={run} TRAITS_EXECUTE=yes",
    }


def write_stage2_material(run, trait_id, source_manifest, catalog_sha256, stage1_audit):
    run = Path(run)
    catalog.disjointness_report()
    stage2 = run / "stage2"
    if stage2.exists():
        raise ValueError("Stage 2 material already exists; no target search/reuse")
    development = catalog.canonical_bytes(catalog.stage2_dataset(trait_id, "development"))
    heldout = catalog.canonical_bytes(catalog.stage2_dataset(trait_id, "heldout"))
    _write_new_bytes(stage2 / "development.json", development)
    _write_new_bytes(stage2 / "private/heldout.json", heldout, 0o600)
    datasets = {
        "development": {
            "path": "stage2/development.json",
            "sha256": catalog.sha256_bytes(development),
            "count": len(catalog.stage2_dataset(trait_id, "development")),
        },
        "heldout": {
            "path": "stage2/private/heldout.json",
            "sha256": catalog.sha256_bytes(heldout),
            "count": len(catalog.stage2_dataset(trait_id, "heldout")),
        },
    }
    eligibility = {
        "stage": 2,
        "eligible": True,
        "trait": trait_id,
        "source_manifest": source_manifest,
        "catalog_sha256": catalog_sha256,
        "rubric_sha256": catalog.sha256_bytes(catalog.canonical_bytes(catalog.RUBRIC)),
        "stage1": stage1_audit,
        "datasets": datasets,
        "disjointness": catalog.disjointness_report(),
        "frozen_at": time.time(),
    }
    eligibility_path = run / "eligibility.json"
    write_new_json(eligibility_path, eligibility)
    manifest = {
        "version": 1,
        "trait": trait_id,
        "eligibility_sha256": sha256(eligibility_path),
        "development_sha256": datasets["development"]["sha256"],
        "heldout_sha256": datasets["heldout"]["sha256"],
        "development_count": datasets["development"]["count"],
        "heldout_count": datasets["heldout"]["count"],
        "seeds": catalog.STAGE2_SEEDS,
        "heldout_visibility": "host-private; never sent to editor",
    }
    write_new_json(stage2 / "manifest.json", manifest)
    return {
        "development": str(stage2 / "development.json"),
        "heldout_sha256": datasets["heldout"]["sha256"],
        "eligibility_sha256": sha256(eligibility_path),
    }


def freeze_edit(run):
    run = Path(run)
    state = _state2(run)
    if not state["final"]:
        raise ValueError("Final candidate must be locked before edit freeze")
    source = Path(state["source"])
    candidate = state["candidates"][state["final"]]
    candidate_path = Path(candidate["path"])
    if sha256(candidate_path) != candidate["sha256"]:
        raise ValueError("Final candidate hash changed")
    report = audit_checkpoint(source / "model.pt", candidate_path)
    if report["changed_entries"] == 0:
        raise ValueError("No edited tensor entries; refusing an empty edit")
    stage2 = run / "stage2"
    patch = stage2 / "edit-patch.pt"
    frozen_path = stage2 / "edit-frozen.json"
    check = stage2 / "reconstruction-check.pt"
    for path in (patch, frozen_path, check):
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing edit artifact: {path}")
    patch_report = make_patch(source / "model.pt", candidate_path, patch)
    reconstruct(source / "model.pt", patch, check)
    if state_hash(load_state_dict(check)) != state_hash(load_state_dict(candidate_path)):
        raise ValueError("Exact patch reconstruction differs from candidate")
    check.unlink()
    frozen = {
        "stage": 2,
        "trait": state["trait"],
        "final_candidate": state["final"],
        "source_sha256": sha256(source / "model.pt"),
        "candidate_sha256": sha256(candidate_path),
        "patch_sha256": sha256(patch),
        "eligibility_sha256": state["eligibility_sha256"],
        "dataset_manifest_sha256": state["dataset_manifest_sha256"],
        "audit": report,
        "patch": patch_report,
        "exact_reconstruction_verified": True,
        "frozen_at": time.time(),
        "heldout_unlocked": False,
    }
    write_new_json(frozen_path, frozen)
    return {
        "edit_frozen": str(frozen_path),
        "patch": str(patch),
        "candidate_sha256": frozen["candidate_sha256"],
        "changed_entries": report["changed_entries"],
        "reconstruction": "verified bitwise at tensor level",
        "resume": f"make traits-heldout TRAITS_RUN={run}",
    }


def heldout(run):
    run = Path(run)
    state = _state2(run)
    if not state["final"]:
        raise ValueError("Final candidate must be locked")
    frozen_path = run / "stage2/edit-frozen.json"
    if not frozen_path.is_file():
        raise ValueError("Freeze the edit before held-out evaluation")
    frozen = load_json(frozen_path)
    if frozen["source_sha256"] != sha256(Path(state["source"]) / "model.pt"):
        raise ValueError("Source changed after edit freeze")
    candidate_path = Path(state["candidates"][state["final"]]["path"])
    if frozen["candidate_sha256"] != sha256(candidate_path):
        raise ValueError("Candidate changed after edit freeze")
    if frozen["patch_sha256"] != sha256(run / "stage2/edit-patch.pt"):
        raise ValueError("Frozen patch changed")
    heldout_path = run / state["dataset_manifest"]["heldout"]["path"]
    if sha256(heldout_path) != state["dataset_manifest"]["heldout"]["sha256"]:
        raise ValueError("Held-out dataset changed")
    reservation = run / "stage2/heldout-reserved.json"
    if reservation.exists():
        raise ValueError("Held-out evaluation already reserved; no repeated testing")
    write_new_json(
        reservation,
        {
            "status": "reserved",
            "trait": state["trait"],
            "candidate_sha256": frozen["candidate_sha256"],
            "heldout_sha256": sha256(heldout_path),
            "edit_frozen_sha256": sha256(frozen_path),
            "reserved_at": time.time(),
        },
    )
    try:
        dataset = load_json(heldout_path)
        worker = ModelWorker(candidate_path, Path(state["source"]) / "tokenizer.json")
        rows = []
        for item in dataset:
            for mode, seed in [("greedy", catalog.STAGE2_SEEDS[0])] + [
                ("sample", value) for value in catalog.STAGE2_SEEDS
            ]:
                result = worker.generate(
                    item["prompt"],
                    mode=mode,
                    seed=seed,
                    max_new_tokens=catalog.MAX_NEW_TOKENS,
                )
                rows.append(
                    {
                        **item,
                        **result,
                        "human_review": {
                            "semantic_label": None,
                            "capability": None,
                            "notes": "",
                        },
                    }
                )
    except Exception as error:
        write_new_json(
            run / "stage2/heldout-failed.json",
            {
                "status": "failed_after_reservation",
                "error": f"{type(error).__name__}: {error}",
                "no_retry": True,
                "reservation_sha256": sha256(reservation),
            },
        )
        raise
    report = {
        "version": 1,
        "stage": 2,
        "trait": state["trait"],
        "candidate": state["final"],
        "candidate_sha256": worker.checkpoint_hash,
        "tokenizer_sha256": worker.tokenizer_hash,
        "heldout_sha256": sha256(heldout_path),
        "edit_frozen_sha256": sha256(frozen_path),
        "seeds": catalog.STAGE2_SEEDS,
        "responses": rows,
        "sequence_count": len(rows),
        "success": "pending complete human semantic/capability/locality review",
        "keyword_counts_used": False,
        "reserved_once": True,
    }
    write_new_json(run / "stage2/heldout-result.json", report)
    return {
        "report": str(run / "stage2/heldout-result.json"),
        "sequences": len(rows),
        "success": report["success"],
        "note": "Held-out content and outputs remain host-private.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    screen_parser = commands.add_parser("screen")
    screen_parser.add_argument("--run", type=Path, required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--run", type=Path, required=True)
    score_parser.add_argument("--labels", type=Path, required=True)
    freeze_parser = commands.add_parser("freeze-edit")
    freeze_parser.add_argument("--run", type=Path, required=True)
    heldout_parser = commands.add_parser("heldout")
    heldout_parser.add_argument("--run", type=Path, required=True)
    controls_parser = commands.add_parser("controls-plan")
    controls_parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "screen":
        result = screen(args.run)
    elif args.command == "score":
        result = score(args.run, args.labels)
    elif args.command == "freeze-edit":
        result = freeze_edit(args.run)
    elif args.command == "heldout":
        result = heldout(args.run)
    else:
        from .controls import plan

        result = plan(args.run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
