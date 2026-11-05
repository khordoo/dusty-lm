"""Subscription-only Astra launcher for Stage 1 screening and gated Stage 2 editing."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from tokenizers import Tokenizer

from experiments.weight_space_milo.source import SOURCE, sha256, verify
from experiments.weight_space_milo.subscription import (
    bundled_model,
    clean_environment,
    cli_identity,
    config_args,
    inspect_events,
    offline_probe,
    prepare_home,
    subscription_auth,
)

from . import catalog
from .controller import (
    STAGE1_PROTOCOL,
    STAGE2_PROTOCOL,
    Stage1Controller,
    Stage2Controller,
    initialize_stage1,
    initialize_stage2,
    load_json,
    write_json,
)

PACKAGE = Path(__file__).parent
STAGE1_TURNS = 20
STAGE2_TURNS = 120


def _metadata(source=SOURCE):
    tensors = torch.load(Path(source) / "model.pt", weights_only=True, map_location="cpu")
    if "model_state_dict" in tensors:
        tensors = tensors["model_state_dict"]
    tokenizer = Tokenizer.from_file(str(Path(source) / "tokenizer.json"))
    names = [" dusty", " milo", "dusty", "milo"]
    return {
        "tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in tensors.items()
        },
        "name_tokenization": {
            name: {
                "ids": tokenizer.encode(name).ids,
                "tokens": tokenizer.encode(name).tokens,
            }
            for name in names
        },
    }


def prior_screen_summary():
    records, _ = catalog.prior_screen_prompts()
    return {
        "stairs": {
            "status": "failed frozen 75% gate",
            "observed_upper_bound": 0.375,
            "reason": "only 15/40 explicit fear responses; contradictions and liking retained",
            "hash": records["stairs_development"]["sha256"],
        },
        "cleaning": {
            "status": "failed frozen 75% gate",
            "observed_generous_positive": 0.725,
            "reason": "29/40 generous positives, below 30/40; no threshold rescue",
            "hash": records["cleaning_development"]["sha256"],
        },
        "policy": (
            "These screens cannot be pooled, selectively filtered or reused. The catalog "
            "uses fresh prompts and a separately frozen rubric."
        ),
    }


def preflight(stage, run):
    run = Path(run)
    report = {
        "stage": stage,
        "run": str(run.resolve()),
        "remote_model_calls": 0,
        "catalog_sha256": catalog.assert_frozen()["catalog_sha256"],
        "source_manifest": verify(SOURCE),
        "disjointness": catalog.disjointness_report(),
    }
    if stage == 1:
        if (run / "state.json").exists():
            report["run_state"] = load_json(run / "state.json")["status"]
        else:
            report["run_state"] = "fresh"
    else:
        eligibility = run / "eligibility.json"
        manifest = run / "stage2/manifest.json"
        if not eligibility.is_file() or not manifest.is_file():
            raise ValueError("Stage 2 preflight requires recorded eligibility and dataset manifest")
        eligibility_data = load_json(eligibility)
        if eligibility_data.get("eligible") is not True:
            raise ValueError("Stage 2 preflight: eligibility is not positive")
        if load_json(manifest)["eligibility_sha256"] != sha256(eligibility):
            raise ValueError("Stage 2 preflight: manifest/eligibility mismatch")
        report["run_state"] = (
            "stage2_initialized" if (run / "stage2/state.json").exists() else "eligible"
        )
        report["trait"] = eligibility_data["trait"]
        report["development_sha256"] = eligibility_data["datasets"]["development"]["sha256"]
        report["heldout_sha256"] = eligibility_data["datasets"]["heldout"]["sha256"]
    auth = subscription_auth()
    report["subscription_auth"] = {
        "mode": "chatgpt",
        "auth_file": str(auth),
        "secrets_printed": False,
    }
    report["cli"] = cli_identity()
    original, modified = bundled_model()
    report["model"] = {
        "slug": modified["slug"],
        "reasoning": "high",
        "tools": [],
        "metadata_changes": {
            key: modified[key] for key in original if original[key] != modified[key]
        },
        "catalog_identity_preserved": original["slug"] == modified["slug"],
    }
    report["api_fallback"] = False
    report["transport"] = "ChatGPT subscription via pinned Codex CLI"
    return report


def dry_run(stage, run):
    report = preflight(stage, run)
    command = (
        f"make traits-run TRAITS_RUN={Path(run)} TRAITS_EXECUTE=yes"
        if stage == 1
        else f"make traits-run-stage2 TRAITS_RUN={Path(run)} TRAITS_EXECUTE=yes"
    )
    report["dry_run"] = True
    report["initialized"] = False
    report["transport_started"] = False
    report["launch_command"] = command
    return report


def stage1_message(run):
    return (
        PACKAGE.joinpath("editor_prompt_stage1.md").read_text()
        + "\n"
        + json.dumps(
            {
                "stage": 1,
                "protocol": STAGE1_PROTOCOL,
                "public_catalog": catalog.public_catalog(),
                "source_metadata": _metadata(),
                "prior_screens": prior_screen_summary(),
                "run": str(Path(run).resolve()),
            }
        )
    )


def stage2_message(run):
    run = Path(run)
    eligibility = load_json(run / "eligibility.json")
    development = load_json(run / eligibility["datasets"]["development"]["path"])
    public_eligibility = {
        "trait": eligibility["trait"],
        "source_manifest": eligibility["source_manifest"],
        "catalog_sha256": eligibility["catalog_sha256"],
        "stage1": eligibility["stage1"],
        "development_sha256": eligibility["datasets"]["development"]["sha256"],
        "heldout_sha256": eligibility["datasets"]["heldout"]["sha256"],
        "gate": catalog.GATE,
    }
    return (
        PACKAGE.joinpath("editor_prompt_stage2.md").read_text()
        + "\n"
        + json.dumps(
            {
                "stage": 2,
                "protocol": STAGE2_PROTOCOL,
                "eligibility": public_eligibility,
                "development": development,
                "source_metadata": _metadata(),
                "prior_screens": prior_screen_summary(),
                "run": str(run.resolve()),
            }
        )
    )


def run_transport(stage, run, controller, message, max_turns):
    run = Path(run)
    auth = subscription_auth()
    cli_identity()
    home = run / f"codex-private-stage{stage}"
    config = prepare_home(home, auth)
    public = run / f"editor-public-stage{stage}"
    public.mkdir(mode=0o700, exist_ok=False)
    settings_path = (
        run / "subscription_settings.json"
        if stage == 1
        else run / "stage2/subscription_settings.json"
    )
    write_json(
        settings_path,
        {
            "stage": stage,
            "transport": "ChatGPT subscription via Codex",
            "model": "gpt-6-astra",
            "reasoning": "high",
            "tools": [],
            "api_fallback": False,
            "prompt_sha256": sha256(PACKAGE / f"editor_prompt_stage{stage}.md"),
            "max_turns": max_turns,
            "usage_accounting": "Raw CLI turn usage recorded; do not sum cumulative counters blindly.",
        },
    )
    thread = None
    for turn in range(max_turns):
        answer = home / f"answer-{turn:03d}.txt"
        command = [
            "codex",
            "--no-daemon",
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "--skip-git-repo-check",
            "--json",
            *config_args(config),
            "-o",
            str(answer),
        ]
        command += ["resume", thread, "-"] if thread else ["-"]
        completed = subprocess.run(
            command,
            cwd=public,
            env=clean_environment(home),
            input=message,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        write_json(home / f"events-{turn:03d}.json", events)
        (home / f"stderr-{turn:03d}.log").write_text(completed.stderr)
        if completed.returncode:
            raise SystemExit("CLI failed; no retry, model substitution or API fallback")
        inspect_events(events)
        started = [event["thread_id"] for event in events if event.get("type") == "thread.started"]
        if thread is None:
            if len(started) != 1:
                raise SystemExit("No unique fresh thread ID")
            thread = started[0]
        elif started and any(value != thread for value in started):
            raise SystemExit("Conversation changed unexpectedly")
        try:
            result = controller.dispatch(json.loads(answer.read_text()))
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            result = {"error": str(error)}
        if result.get("stage1_complete"):
            payload = {
                "stage": 1,
                "trait": result["trait"],
                "transport": "ChatGPT subscription",
                "turns": turn + 1,
                "next": result["next_command"],
            }
            write_json(run / "transport-stage1-complete.json", payload)
            print(json.dumps(payload, indent=2))
            return
        if result.get("final"):
            print(
                json.dumps(
                    {
                        "stage": 2,
                        "final": result["final"],
                        "transport": "ChatGPT subscription",
                        "turns": turn + 1,
                        "next": result.get("next_command"),
                    },
                    indent=2,
                )
            )
            return
        message = json.dumps(result)
    raise SystemExit("Turn budget exhausted; experiment incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run(args.stage, args.run), indent=2))
        return
    if not args.execute:
        raise SystemExit(
            "Refusing live launch. Run the dry-run/preflight target first, then add --execute "
            "through the documented Make target."
        )
    preflight(args.stage, args.run)
    subscription_auth()
    cli_identity()
    # Localhost-only, credential-free, zero remote model request.
    offline_probe()
    if args.stage == 1:
        initialize_stage1(args.run)
        controller = Stage1Controller(args.run)
        message = stage1_message(args.run)
        turns = STAGE1_TURNS
    else:
        initialize_stage2(args.run, SOURCE)
        controller = Stage2Controller(args.run)
        message = stage2_message(args.run)
        turns = STAGE2_TURNS
    run_transport(args.stage, args.run, controller, message, turns)


if __name__ == "__main__":
    main()
