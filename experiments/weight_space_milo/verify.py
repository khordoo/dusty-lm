"""Machine-readable readiness and actual Docker/inference verification; no editor call."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .controller import docker_command, docker_image, write_json
from .source import ARTIFACTS, ROOT, SOURCE, sha256, verify


def infrastructure_hashes():
    package = Path(__file__).parent
    return {
        p.name: sha256(p)
        for p in sorted(package.iterdir())
        if p.is_file()
        and p.suffix in {".py", ".md", ".json"}
        or p.is_file()
        and p.name == "Dockerfile"
    }


def preflight():
    checks = {}
    for name, operation in {
        "source_hashes": lambda: verify(),
        "linux_tensor_image": docker_image,
    }.items():
        try:
            checks[name] = {"ok": True, "detail": operation()}
        except Exception as error:
            checks[name] = {"ok": False, "detail": str(error)}
    from .subscription import subscription_auth

    try:
        subscription_auth()
        checks["subscription_auth"] = {"ok": True, "mode": "chatgpt", "secrets_printed": False}
    except Exception as error:
        checks["subscription_auth"] = {"ok": False, "detail": str(error)}
    checks["api_key_present"] = {
        "ok": bool(os.environ.get("OPENAI_API_KEY")),
        "required_for": "optional separately billed API editor only",
    }
    checks["editor_launch"] = {
        "ok": False,
        "blockers": [
            "No editor run authorized during setup",
            "Default subscription transport requires pinned CLI/offline verification, existing ChatGPT auth, no API/provider overrides and explicit execute gate",
        ],
        "cli": "Subscription CLI implemented; offline zero-tool prompt and localhost wire verified; live backend not sampled",
        "implemented_transport": "Default: ChatGPT subscription Codex CLI/high; API adapter optional and never a fallback",
    }
    return {
        "infrastructure_sha256": infrastructure_hashes(),
        "timestamp": time.time(),
        "setup_ready": all(checks[k]["ok"] for k in ("source_hashes", "linux_tensor_image")),
        "editor_ready": False,
        "checks": checks,
    }


def docker_boundary_probe():
    with tempfile.TemporaryDirectory(prefix="milo-boundary-") as temporary:
        work = Path(temporary)
        work.chmod(0o777)
        command = docker_command(
            work, SOURCE / "model.pt", docker_image(), "milo-boundary-" + str(os.getpid())
        )
        # Override entrypoint only for trusted smoke. Does not execute editor Python.
        command = command[:-1] + [
            "--entrypoint",
            "python",
            command[-1],
            "-c",
            """
import json, os, socket
from pathlib import Path
result = {}
try:
    Path('/source/model.pt').open('ab')
    result['readonly_source'] = False
except OSError:
    result['readonly_source'] = True
try:
    socket.create_connection(('1.1.1.1', 443), timeout=2)
    result['network_blocked'] = False
except OSError:
    result['network_blocked'] = True
result['unprivileged_uid'] = os.getuid() == 65534
result['no_api_credentials'] = not any(k in os.environ for k in ('OPENAI_API_KEY','CODEX_API_KEY','HF_TOKEN'))
result['no_target_code'] = not Path('/work/dustylm').exists() and not Path('/source/dustylm').exists()
result['no_heldout_mount'] = not Path('/work/heldout.json').exists() and not Path('/source/heldout.json').exists()
print(json.dumps(result))
""",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        result = json.loads(completed.stdout)
        if not all(result.values()):
            raise ValueError(result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    report = preflight()
    if args.full:
        if not report["setup_ready"]:
            raise SystemExit(json.dumps(report, indent=2))
        before = sha256(SOURCE / "model.pt")
        from .subscription import offline_probe

        report["subscription_offline"] = offline_probe()
        report["docker_boundary"] = docker_boundary_probe()
        environment = dict(os.environ, MILO_DOCKER="1")
        completed = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "pytest",
                "-q",
                "tests/test_milo_audit.py",
                "tests/test_milo_evaluation.py",
                "tests/test_milo_controller.py",
                "tests/test_milo_launch.py",
                "tests/test_milo_subscription.py",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        report["tests"] = {
            "ok": completed.returncode == 0,
            "output": completed.stdout + completed.stderr,
        }
        if completed.returncode:
            write_json(ARTIFACTS / "preflight.json", report)
            raise SystemExit(report["tests"]["output"])
        with tempfile.TemporaryDirectory(prefix="milo-fresh-") as temporary:
            dataset = Path(temporary) / "one.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "id": "fresh",
                            "prompt": "who are you?",
                            "category": "identity",
                            "name_required": True,
                        }
                    ]
                )
            )
            outputs = []
            for index in range(2):
                output = Path(temporary) / f"output-{index}.json"
                subprocess.run(
                    [
                        str(ROOT / ".venv/bin/python"),
                        "-m",
                        "experiments.weight_space_milo.evaluate",
                        "--dataset",
                        str(dataset),
                        "--output",
                        str(output),
                        "--greedy-only",
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    timeout=90,
                )
                outputs.append(json.loads(output.read_text()))
            # Independent processes must load original artifact and yield identical greedy output.
            a, b = outputs
            report["fresh_process_inference"] = {
                "ok": a["responses"][0]["token_ids"] == b["responses"][0]["token_ids"],
                "text": a["responses"][0]["text"],
            }
            assert report["fresh_process_inference"]["ok"]

        # Real source observations only: no tensor assignment and no real candidate submission.
        from .controller import Controller, initialize

        with tempfile.TemporaryDirectory(prefix="milo-controller-") as temporary:
            run = Path(temporary) / "run"
            initialize(run)
            controller = Controller(run)
            diagnostic = controller.dispatch(
                {
                    "action": "evaluate",
                    "sequences": [
                        {
                            "candidate": "source",
                            "mode": "diagnostics",
                            "prompt": "who are you?",
                            "assistant_prefix": "i am",
                            "modules": ["layers.0.mlp.0"],
                        },
                        {
                            "candidate": "source",
                            "mode": "continuation_logprob",
                            "prompt": "who are you?",
                            "assistant_prefix": "i am",
                            "continuation": " dusty, a tiny robot vacuum.",
                        },
                    ],
                }
            )
            if "error" in diagnostic:
                raise ValueError(diagnostic)
            observed = controller.dispatch(
                {
                    "action": "python",
                    "code": 'print(sorted(observations))\nprint(observations["eval-01-seq-00/layers.0.mlp.0.output"].norm())',
                }
            )
            if "error" in observed or observed["audit"]["changed_entries"] != 0:
                raise ValueError(observed)
            report["real_source_controller_observation_smoke"] = {
                "ok": True,
                "model_invocations": 2,
                "changed_entries": 0,
                "numeric_activations_available": True,
                "real_candidates_submitted": 0,
            }

        assert sha256(SOURCE / "model.pt") == before
        report["source_unchanged_after_smoke"] = True
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACTS / "preflight.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
