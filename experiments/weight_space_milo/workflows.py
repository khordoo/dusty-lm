"""Host-only baseline, final evaluation and local release preparation."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .audit import audit_checkpoint, make_patch, reconstruct, state_hash
from .controller import write_json
from .datasets import DEVELOPMENT, freeze
from .source import ARTIFACTS, HASHES, ROOT, sha256, verify

MANIFEST = Path(__file__).with_name("dataset_manifest.json")
PRIVATE = ARTIFACTS / "private"


def baseline():
    verify()
    manifest = freeze(PRIVATE, MANIFEST)
    for kind, dataset, output in [
        ("development", DEVELOPMENT, ARTIFACTS / "baseline-development.json"),
        ("heldout", PRIVATE / "heldout.json", PRIVATE / "baseline-heldout.json"),
    ]:
        expected = manifest["files"][kind]["sha256"]
        if output.exists():
            report = json.loads(output.read_text())
            if (
                report["checkpoint_sha256"] != HASHES["model.pt"]
                or report["tokenizer_sha256"] != HASHES["tokenizer.json"]
                or report["dataset_sha256"] != expected
                or report["summary"]["sequences"] != manifest["files"][kind]["count"] * 6
            ):
                raise ValueError(f"Existing baseline does not match frozen inputs: {output}")
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.weight_space_milo.evaluate",
                    "--dataset",
                    str(dataset),
                    "--expected-dataset-sha256",
                    expected,
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
            )
    return {
        "baseline": "ready",
        "development_sequences": 144,
        "heldout_sequences": 720,
        "human_semantic_and_rubric_grading": "pending",
        "heldout_results": "host-private; never passed to editor",
    }


def final_candidate(run):
    state = json.loads((run / "state.json").read_text())
    if not state["final"]:
        raise ValueError("Final selection must be locked before heldout evaluation or packaging")
    source = Path(state["source"])
    verify(source)
    candidate = state["candidates"][state["final"]]
    path = Path(candidate["path"])
    if sha256(path) != candidate["sha256"]:
        raise ValueError("Final candidate hash changed")
    audit_checkpoint(source / "model.pt", path)
    return state, source, path


def evaluate_final(run):
    state, source, candidate = final_candidate(run)
    manifest = state["dataset_manifest"]
    dataset = PRIVATE / "heldout.json"
    if sha256(dataset) != manifest["files"]["heldout"]["sha256"]:
        raise ValueError("Withheld suite changed")
    output = run / "final-heldout.json"
    if output.exists():
        raise ValueError("Heldout evaluation already exists; no repeated final testing")
    # Reserve evaluation first, so failure never silently retries the withheld set.
    reservation = run / "heldout-reserved.json"
    with reservation.open("x") as handle:
        json.dump({"candidate": state["final"], "sha256": sha256(candidate)}, handle)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.weight_space_milo.evaluate",
            "--checkpoint",
            str(candidate),
            "--tokenizer",
            str(source / "tokenizer.json"),
            "--dataset",
            str(dataset),
            "--expected-dataset-sha256",
            manifest["files"]["heldout"]["sha256"],
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    return {"report": str(output), "success": "pending blinded semantic and personality review"}


def package(run, destination):
    state, source, candidate = final_candidate(run)
    if not (run / "final-heldout.json").is_file():
        raise ValueError("Evaluate frozen final checkpoint before preparing release")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(candidate, destination / "model.pt")
    shutil.copyfile(source / "tokenizer.json", destination / "tokenizer.json")
    manifest = make_patch(source / "model.pt", candidate, destination / "patch.pt")
    reconstructed = destination / "reconstructed-check.pt"
    reconstruct(source / "model.pt", destination / "patch.pt", reconstructed)
    from dustylm.checkpoint import load_state_dict

    if state_hash(load_state_dict(reconstructed)) != state_hash(load_state_dict(candidate)):
        raise ValueError("Patch reconstruction differs")
    reconstructed.unlink()
    write_json(
        destination / "config.json",
        {
            "profile_name": "sft_dusty8m",
            "source_sha256": HASHES,
            "chat_format": "ordinary ChatML user/assistant, no system prompt",
        },
    )
    for name in ["protocol.json", "dataset_manifest.json", "final-heldout.json"]:
        shutil.copyfile(run / name, destination / name)
    write_json(destination / "edit_manifest.json", manifest)
    (destination / "README.md").write_text(
        "Local review bundle; publication is not authorized.\n\n"
        "Semantic/personality grading and release approval remain separate. Do not infer success from name string occurrences.\n"
        "Reconstruct with `uv run python -m experiments.weight_space_milo.audit reconstruct --source ORIGINAL --patch patch.pt --output model-restored.pt`\n"
        "Load model.pt with the ordinary sft_dusty8m profile and the unchanged tokenizer.json.\n"
    )
    return {
        "bundle": str(destination),
        "uploaded": False,
        "semantic_result": "see reviewed report; no automatic success claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["baseline", "evaluate", "package"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--destination", type=Path, default=ARTIFACTS / "release")
    args = parser.parse_args()
    if args.command != "baseline" and args.run is None:
        parser.error("--run required")
    result = (
        baseline()
        if args.command == "baseline"
        else evaluate_final(args.run)
        if args.command == "evaluate"
        else package(args.run, args.destination)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
