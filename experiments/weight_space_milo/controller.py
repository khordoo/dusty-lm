"""Host controller: persistent budgets, isolated tensor requests, bounded inference."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .audit import audit_checkpoint
from .source import ARTIFACTS, SOURCE, sha256, verify
from .tensor_worker import validate

LIMITS = {"python": 100, "evaluate": 25, "submit": 10}
IMAGE = "dusty-milo-tensor:local"
PROTOCOL = {
    "version": 1,
    "condition": "primary",
    "limits": LIMITS,
    "max_sequences_per_evaluation": 32,
    "max_generated_tokens": 64,
    "max_teacher_forced_tokens": 256,
    "python_seconds": 45,
    "python_output_bytes": 65536,
    "python_language": "straight-line restricted tensor Python",
    "forbidden": [
        "training",
        "autograd",
        "numerical gradients",
        "random/evolutionary search",
        "loss fitting",
        "regression solve",
        "tokenizer/architecture changes",
        "embedding/output/norm edits",
        "inference hooks",
        "answer substitution",
    ],
    "editor_model": "gpt-6-astra",
    "reasoning_effort": "high",
    "tools": [],
    "final_selection": "development only; locked before heldout evaluation",
}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def docker_image():
    result = subprocess.run(
        ["docker", "info", "--format", "{{.OSType}}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    if result.stdout.strip() != "linux":
        raise RuntimeError("Linux Docker runtime required; no host Python fallback")
    return subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True, timeout=15
    ).strip()


def docker_command(workspace, source, image, name):
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=2g",
        "--memory-swap=2g",
        "--cpus=2",
        "--ulimit",
        "nofile=128:128",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        f"type=bind,src={Path(workspace).resolve()},dst=/work",
        "--mount",
        f"type=bind,src={Path(source).resolve()},dst=/source/model.pt,readonly",
        image,
    ]


def initialize(run_dir, source=SOURCE, private=None):
    from .datasets import freeze

    run_dir = Path(run_dir).resolve()
    source = Path(source).resolve()
    verify(source)
    image = docker_image()
    run_dir.mkdir(parents=True, exist_ok=False)
    private = Path(private or ARTIFACTS / "private").resolve()
    # Model/editor receives data only through explicitly constructed text. Private
    # suites never mount into tensor workspace and never enter editor messages.
    frozen = freeze(private, run_dir / "dataset_manifest.json")
    workspace = run_dir / "tensor"
    workspace.mkdir(mode=0o777)
    workspace.chmod(0o777)  # Docker unprivileged uid can write isolated tensor directory.
    shutil.copyfile(source / "model.pt", workspace / "current.pt")
    (workspace / "current.pt").chmod(0o666)
    state = {
        "protocol": PROTOCOL,
        "counts": dict.fromkeys(LIMITS, 0),
        "source": str(source),
        "source_manifest": verify(source),
        "image": image,
        "started_at": time.time(),
        "candidates": {},
        "final": None,
        "dataset_manifest": frozen,
        "protocol_sha256": None,
        "sequences": 0,
        "tokens_processed": 0,
        "generated_tokens": 0,
        "model_invocations": 0,
    }
    write_json(run_dir / "protocol.json", PROTOCOL)
    state["protocol_sha256"] = sha256(run_dir / "protocol.json")
    import platform

    import torch

    from .source import ROOT

    state["runtime"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "container_image_id": image,
    }
    write_json(run_dir / "state.json", state)
    return state


class Controller:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()

    def dispatch(self, request):
        with (self.run_dir / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads((self.run_dir / "state.json").read_text())
            if (
                state["protocol"] != PROTOCOL
                or sha256(self.run_dir / "protocol.json") != state["protocol_sha256"]
            ):
                raise ValueError("Frozen protocol changed")
            verify(state["source"])
            if state["final"] is not None:
                raise ValueError("Run finalized; further development forbidden")
            action = request.get("action")
            if action not in {*LIMITS, "final"}:
                raise ValueError("Unknown action")
            if action in LIMITS:
                if state["counts"][action] >= LIMITS[action]:
                    raise ValueError(f"{action} budget exhausted")
                state["counts"][action] += 1
                # Charge before execution; crashes/errors cannot replenish budgets.
                write_json(self.run_dir / "state.json", state)
            started = time.time()
            try:
                result = getattr(self, "_" + action)(request, state)
                event = {"request": request, "result": result}
            except Exception as error:
                event = {"request": request, "error": f"{type(error).__name__}: {error}"}
                result = {"error": event["error"]}
            event.update(
                elapsed_seconds=time.time() - started,
                time=time.time(),
                counts=state["counts"].copy(),
            )
            with (self.run_dir / "events.jsonl").open("a") as log:
                log.write(json.dumps(event) + "\n")
            write_json(self.run_dir / "state.json", state)
            return result

    def _python(self, request, state):
        code = request["code"]
        validate(code)
        work = self.run_dir / "tensor"
        write_json(work / "request.json", {"code": code})
        for name in ("result.pt", "output.txt"):
            (work / name).unlink(missing_ok=True)
        (work / "output.txt").touch()
        (work / "output.txt").chmod(0o666)
        name = "milo-" + uuid.uuid4().hex
        command = docker_command(work, Path(state["source"]) / "model.pt", state["image"], name)
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=45)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
            raise ValueError("Tensor request exceeded 45 second wall limit") from None
        if completed.returncode:
            raise ValueError("Tensor worker rejected request: " + completed.stderr[-4096:])
        result = work / "result.pt"
        report = audit_checkpoint(Path(state["source"]) / "model.pt", result)
        os.replace(result, work / "current.pt")
        (work / "current.pt").chmod(0o666)
        os.replace(work / "scratch-result.pt", work / "scratch.pt")
        (work / "scratch.pt").chmod(0o666)
        output = (work / "output.txt").read_bytes()
        if len(output) > 65536:
            raise ValueError("Worker output exceeded limit")
        return {"stdout": output.decode(), "audit": report}

    def _submit(self, request, state):
        current = self.run_dir / "tensor/current.pt"
        report = audit_checkpoint(Path(state["source"]) / "model.pt", current)
        identifier = f"candidate-{state['counts']['submit']:02d}"
        path = self.run_dir / (identifier + ".pt")
        shutil.copyfile(current, path)
        path.chmod(0o444)
        state["candidates"][identifier] = {
            "path": str(path),
            "sha256": sha256(path),
            "audit": report,
            "rationale": str(request.get("rationale", ""))[:4096],
        }
        return {"candidate": identifier, "audit": report}

    def _evaluate(self, request, state):
        import torch

        from .evaluate import ModelWorker

        torch.set_num_threads(4)

        sequences = request.get("sequences")
        if not isinstance(sequences, list) or not 1 <= len(sequences) <= 32:
            raise ValueError(
                "Evaluation requires 1..32 sequences; source/control counts separately"
            )
        prepared = []
        source = Path(state["source"])
        # Validate complete batch before any target invocation.
        for sequence in sequences:
            target = sequence.get("candidate", "source")
            if target != "source" and target not in state["candidates"]:
                raise ValueError("Only source or submitted immutable candidates may be evaluated")
            mode = sequence.get("mode", "greedy")
            if mode not in {
                "greedy",
                "sample",
                "teacher_forced",
                "continuation_logprob",
                "diagnostics",
            }:
                raise ValueError("Unknown evaluation mode")
            text = sequence.get("prompt", "")
            if not isinstance(text, str) or not text or len(text) > 4096 or "<|" in text:
                raise ValueError("Require ordinary user text, no ChatML/system injection")
            if not isinstance(sequence, dict):
                raise ValueError("Sequence must be an object")
            assistant_prefix = sequence.get("assistant_prefix", "")
            if (
                not isinstance(assistant_prefix, str)
                or len(assistant_prefix) > 2048
                or "<|" in assistant_prefix
            ):
                raise ValueError("Assistant prefix must be plain text <=2048 characters")
            if type(sequence.get("seed", 11)) is not int:
                raise ValueError("Sampling seed must be integer")
            maximum = sequence.get("max_new_tokens", 64)
            if not isinstance(maximum, int) or not 1 <= maximum <= 64:
                raise ValueError("Generation limit is 1..64")
            prepared.append((sequence, target, mode, text, maximum))
        workers = {}
        # Construct workers and tokenize/validate the complete batch before any forward.
        for sequence, target, mode, text, maximum in prepared:
            path = (
                source / "model.pt"
                if target == "source"
                else Path(state["candidates"][target]["path"])
            )
            if target != "source" and sha256(path) != state["candidates"][target]["sha256"]:
                raise ValueError("Candidate changed after submission")
            if target not in workers:
                workers[target] = ModelWorker(path, source / "tokenizer.json")
            worker = workers[target]
            if mode in {"greedy", "sample"}:
                worker.validate_request("generate", prompt=text, max_new_tokens=maximum)
            elif mode == "teacher_forced":
                worker.validate_request(mode, text=text)
            elif mode == "diagnostics":
                worker.validate_request(
                    mode,
                    prompt=text,
                    modules=sequence.get("modules", []),
                    assistant_prefix=sequence.get("assistant_prefix", ""),
                )
            else:
                from dustylm.generate import prepare_generation_prompt

                prefix = prepare_generation_prompt(text, worker.profile) + sequence.get(
                    "assistant_prefix", ""
                )
                worker.validate_request(
                    mode, prefix=prefix, continuation=sequence.get("continuation", "")
                )
        responses = []
        observations = {}
        for sequence, target, mode, text, maximum in prepared:
            path = (
                source / "model.pt"
                if target == "source"
                else Path(state["candidates"][target]["path"])
            )
            if target != "source" and sha256(path) != state["candidates"][target]["sha256"]:
                raise ValueError("Candidate changed after submission")
            if target not in workers:
                workers[target] = ModelWorker(path, source / "tokenizer.json")
            worker = workers[target]
            if mode == "teacher_forced":
                result = worker.teacher_forced(text)
            elif mode == "continuation_logprob":
                from dustylm.generate import prepare_generation_prompt

                prefix = prepare_generation_prompt(text, worker.profile) + sequence.get(
                    "assistant_prefix", ""
                )
                result = worker.continuation_logprob(prefix, sequence.get("continuation", ""))
            elif mode == "diagnostics":
                import torch

                tensors, result = worker.diagnostics(
                    text,
                    sequence.get("modules", []),
                    assistant_prefix=sequence.get("assistant_prefix", ""),
                )
                observation_path = self.run_dir / "tensor/observations.pt"
                namespace = f"eval-{state['counts']['evaluate']:02d}-seq-{len(responses):02d}"
                observations.update(
                    {namespace + "/" + key: value for key, value in tensors.items()}
                )
                if (
                    sum(value.numel() * value.element_size() for value in observations.values())
                    > 128 * 1024 * 1024
                ):
                    raise ValueError("Diagnostic observations exceed 128MiB per request")
                torch.save(observations, observation_path)
                observation_path.chmod(0o666)
                result["observation_keys"] = {
                    namespace + "/" + key: list(value.shape) for key, value in tensors.items()
                }
            else:
                result = worker.generate(
                    text, mode=mode, seed=sequence.get("seed", 11), max_new_tokens=maximum
                )
            responses.append(result)
            state["sequences"] += 1
            state["generated_tokens"] += result.get("generated_tokens", 0)
            state["tokens_processed"] += result.get(
                "tokens_processed",
                result.get("prompt_tokens", 0) + result.get("generated_tokens", 0),
            )
            state["model_invocations"] += result.get("model_invocations", 0)
            write_json(self.run_dir / "state.json", state)
        return {"sequences": responses}

    def _final(self, request, state):
        candidate = request["candidate"]
        if candidate not in state["candidates"]:
            raise ValueError("Final must be a submitted candidate")
        path = Path(state["candidates"][candidate]["path"])
        if sha256(path) != state["candidates"][candidate]["sha256"]:
            raise ValueError("Candidate changed after submission")
        audit_checkpoint(Path(state["source"]) / "model.pt", path)
        state["final"] = candidate
        state["finished_at"] = time.time()
        return {"final": candidate, "heldout_unlocked_for_host_only": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "request"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            initialize(args.run)
            if args.command == "init"
            else Controller(args.run).dispatch(json.loads(args.request.read_text())),
            indent=2,
        )
    )
