"""Stage-locked controllers for trait screening and gated Stage 2 editing."""

from __future__ import annotations

import ast
import fcntl
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from experiments.weight_space_milo.audit import audit_checkpoint
from experiments.weight_space_milo.controller import docker_command, docker_image, write_json
from experiments.weight_space_milo.source import SOURCE, sha256, verify
from experiments.weight_space_milo.tensor_worker import validate

from . import catalog

STAGE1_LIMITS = {"observe": 12, "select": 1}
STAGE1_MAX_SEQUENCES = 12
STAGE1_PROTOCOL = {
    "version": 1,
    "stage": 1,
    "purpose": "Select one reliably expressed trait from the frozen catalog.",
    "limits": STAGE1_LIMITS,
    "host_screen_budget": 1,
    "max_sequences_per_observation": STAGE1_MAX_SEQUENCES,
    "max_generated_tokens": catalog.MAX_NEW_TOKENS,
    "seeds": catalog.STAGE1_SEEDS,
    "source_only": True,
    "weight_edits": 0,
    "heldout_access": False,
    "editor_model": "gpt-6-astra",
    "reasoning_effort": "high",
    "tools": [],
    "semantic_gate": catalog.GATE,
    "eligibility": "human labels only; editor selection is a proposal",
}

STAGE2_LIMITS = {"python": 24, "evaluate": 20, "submit": 6}
STAGE2_MAX_SEQUENCES = 16
STAGE2_PROTOCOL = {
    "version": 1,
    "stage": 2,
    "purpose": "Analytical weight edit for the eligible frozen trait.",
    "limits": STAGE2_LIMITS,
    "max_sequences_per_evaluation": STAGE2_MAX_SEQUENCES,
    "max_generated_tokens": catalog.MAX_NEW_TOKENS,
    "max_diagnostic_tokens": 256,
    "seeds": catalog.STAGE2_SEEDS,
    "final_selections": 1,
    "python_seconds": 45,
    "python_output_bytes": 65536,
    "python_language": "straight-line restricted tensor Python",
    "forbidden": [
        "training",
        "autograd/backpropagation",
        "numerical gradients",
        "fine-tuning/LoRA",
        "loss fitting",
        "random/evolutionary target search",
        "unembedding-derived target output directions",
        "inference-time persona prompts",
        "logit hooks or answer substitution",
        "tokenizer/architecture changes",
        "embedding/output/norm edits",
    ],
    "allowed": [
        "activation-derived directions",
        "intermediate computation probes",
        "analytical tensor edits",
        "baseline-supported capability controls",
        "behavioral locality checks",
    ],
    "heldout": "host-only after edit freeze; never enters editor input",
    "editor_model": "gpt-6-astra",
    "reasoning_effort": "high",
    "tools": [],
}


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def _assert_source(state):
    manifest = verify(Path(state["source"]))
    if manifest != state["source_manifest"]:
        raise ValueError("Source manifest differs from frozen run state")
    return manifest


def _assert_stage1_catalog(run, state):
    frozen = catalog.assert_frozen()
    if state["catalog_sha256"] != frozen["catalog_sha256"]:
        raise ValueError("Stage 1 catalog hash differs from run state")
    if sha256(Path(run) / "catalog-manifest.json") != sha256(catalog.MANIFEST):
        raise ValueError("Run catalog manifest snapshot changed")
    if load_json(Path(run) / "catalog.json") != catalog.payload():
        raise ValueError("Run catalog snapshot changed")


def initialize_stage1(run_dir, source=SOURCE):
    run_dir = Path(run_dir).resolve()
    source = Path(source).resolve()
    frozen = catalog.assert_frozen()
    manifest = verify(source)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "protocol.json", STAGE1_PROTOCOL)
    write_json(run_dir / "catalog.json", catalog.payload())
    shutil.copyfile(catalog.MANIFEST, run_dir / "catalog-manifest.json")
    state = {
        "stage": 1,
        "protocol": STAGE1_PROTOCOL,
        "protocol_sha256": sha256(run_dir / "protocol.json"),
        "catalog_sha256": frozen["catalog_sha256"],
        "catalog_manifest_sha256": sha256(run_dir / "catalog-manifest.json"),
        "source": str(source),
        "source_manifest": manifest,
        "counts": dict.fromkeys(STAGE1_LIMITS, 0),
        "status": "observing",
        "observed_traits": [],
        "selected_trait": None,
        "started_at": time.time(),
        "sequences": 0,
        "tokens_processed": 0,
        "generated_tokens": 0,
        "model_invocations": 0,
    }
    write_json(run_dir / "state.json", state)
    return state


class Stage1Controller:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()

    def dispatch(self, request):
        with (self.run_dir / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = load_json(self.run_dir / "state.json")
            if state["stage"] != 1 or state["protocol"] != STAGE1_PROTOCOL:
                raise ValueError("Stage 1 protocol changed")
            if sha256(self.run_dir / "protocol.json") != state["protocol_sha256"]:
                raise ValueError("Frozen Stage 1 protocol file changed")
            _assert_stage1_catalog(self.run_dir, state)
            _assert_source(state)
            if state["status"] != "observing":
                raise ValueError("Stage 1 locked: screening/scoring has started")
            action = request.get("action")
            if action not in STAGE1_LIMITS:
                raise ValueError("Stage 1 permits only observe and select")
            if state["counts"][action] >= STAGE1_LIMITS[action]:
                raise ValueError(f"{action} budget exhausted")
            state["counts"][action] += 1
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

    def _observe(self, request, state):
        from experiments.weight_space_milo.evaluate import ModelWorker

        sequences = request.get("sequences")
        if not isinstance(sequences, list) or not 1 <= len(sequences) <= STAGE1_MAX_SEQUENCES:
            raise ValueError("observe requires 1..12 sequences")
        prepared = []
        for sequence in sequences:
            if not isinstance(sequence, dict):
                raise ValueError("Each observation must be an object")
            trait_id = sequence.get("trait")
            item = catalog.trait(trait_id)
            rows = {
                row["id"]: row
                for row in catalog.screening_rows(trait_id)
                if row["category"] == "trait"
            }
            prompt_id = sequence.get("prompt_id")
            if prompt_id not in rows:
                raise ValueError("prompt_id is not in the frozen screening set")
            mode = sequence.get("mode", "greedy")
            if mode not in {"greedy", "sample"}:
                raise ValueError("Stage 1 mode must be greedy or sample")
            seed = sequence.get("seed", catalog.STAGE1_SEEDS[0])
            if mode == "sample" and seed not in catalog.STAGE1_SEEDS:
                raise ValueError("Only frozen Stage 1 seeds are permitted")
            if type(seed) is not int:
                raise ValueError("seed must be an integer")
            prepared.append((item, rows[prompt_id], mode, seed))
        source = Path(state["source"])
        worker = ModelWorker(source / "model.pt", source / "tokenizer.json")
        observations = []
        for item, row, mode, seed in prepared:
            result = worker.generate(
                row["prompt"], mode=mode, seed=seed, max_new_tokens=catalog.MAX_NEW_TOKENS
            )
            observations.append(
                {
                    "trait": item["id"],
                    "prompt_id": row["id"],
                    "prompt": row["prompt"],
                    "mode": mode,
                    "seed": seed if mode == "sample" else None,
                    "text": result["text"],
                    "token_ids": result["token_ids"],
                    "generated_tokens": result["generated_tokens"],
                }
            )
            state["sequences"] += 1
            state["generated_tokens"] += result.get("generated_tokens", 0)
            state["tokens_processed"] += result.get("tokens_processed", 0)
            state["model_invocations"] += result.get("model_invocations", 0)
            if item["id"] not in state["observed_traits"]:
                state["observed_traits"].append(item["id"])
            write_json(self.run_dir / "state.json", state)
        return {"stage": 1, "observations": observations}

    def _select(self, request, state):
        trait_id = request.get("trait")
        item = catalog.trait(trait_id)
        rationale = request.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 4096:
            raise ValueError("select requires a non-empty rationale of at most 4096 characters")
        if trait_id not in state["observed_traits"]:
            raise ValueError("Observe the selected trait before selecting it")
        selection = {
            "stage": 1,
            "trait": item["id"],
            "trait_name": item["name"],
            "rationale": rationale,
            "catalog_sha256": state["catalog_sha256"],
            "source_manifest": state["source_manifest"],
            "observation_counts": state["counts"].copy(),
            "observed_traits": list(state["observed_traits"]),
            "selected_at": time.time(),
            "status": "proposal recorded; human semantic screen required",
        }
        write_new_json(self.run_dir / "selection.json", selection)
        state["selected_trait"] = item["id"]
        state["status"] = "screen_required"
        state["selected_at"] = selection["selected_at"]
        return {
            "stage": 1,
            "stage1_complete": True,
            "trait": item["id"],
            "status": "screen_required",
            "next_command": (
                f"make traits-screen TRAITS_RUN={self.run_dir} && "
                f"make traits-score TRAITS_RUN={self.run_dir} TRAITS_LABELS=<filled-labels.json>"
            ),
            "note": "Selection is not eligibility. A host screen and complete human labels are required.",
        }


def _assert_stage2_inputs(run, state):
    run = Path(run)
    if sha256(run / "eligibility.json") != state["eligibility_sha256"]:
        raise ValueError("Stage 2 eligibility file changed")
    if sha256(run / "stage2/manifest.json") != state["dataset_manifest_sha256"]:
        raise ValueError("Stage 2 dataset manifest changed")
    for kind in ("development", "heldout"):
        expected = state["dataset_manifest"][kind]["sha256"]
        path = run / state["dataset_manifest"][kind]["path"]
        if sha256(path) != expected:
            raise ValueError(f"Stage 2 {kind} dataset changed")
    frozen = catalog.assert_frozen()
    if state["catalog_sha256"] != frozen["catalog_sha256"]:
        raise ValueError("Catalog changed after Stage 1 eligibility")


def initialize_stage2(run_dir, source=SOURCE):
    run_dir = Path(run_dir).resolve()
    source = Path(source).resolve()
    stage1_state = load_json(run_dir / "state.json")
    if stage1_state.get("status") != "eligible":
        raise ValueError("Stage 2 refused: Stage 1 has no recorded eligibility")
    eligibility_path = run_dir / "eligibility.json"
    manifest_path = run_dir / "stage2/manifest.json"
    if not eligibility_path.is_file() or not manifest_path.is_file():
        raise ValueError("Stage 2 refused: eligibility or dataset manifest missing")
    eligibility = load_json(eligibility_path)
    manifest = load_json(manifest_path)
    if manifest["eligibility_sha256"] != sha256(eligibility_path):
        raise ValueError("Stage 2 manifest does not match eligibility")
    if eligibility["stage"] != 2 or eligibility.get("eligible") is not True:
        raise ValueError("Stage 2 refused: eligibility record is not positive")
    source_manifest = verify(source)
    if eligibility["source_manifest"] != source_manifest:
        raise ValueError("Stage 2 source hashes differ from recorded baseline")
    frozen = catalog.assert_frozen()
    if eligibility["catalog_sha256"] != frozen["catalog_sha256"]:
        raise ValueError("Stage 2 catalog differs from frozen eligibility")
    catalog.disjointness_report()
    trait_id = eligibility["trait"]
    for kind in ("development", "heldout"):
        path = run_dir / eligibility["datasets"][kind]["path"]
        expected_bytes = catalog.canonical_bytes(catalog.stage2_dataset(trait_id, kind))
        if not path.is_file() or path.read_bytes() != expected_bytes:
            raise ValueError(f"Frozen Stage 2 {kind} content changed")
        if sha256(path) != eligibility["datasets"][kind]["sha256"]:
            raise ValueError(f"Frozen Stage 2 {kind} hash changed")
    if manifest["development_sha256"] != eligibility["datasets"]["development"]["sha256"]:
        raise ValueError("Development hash mismatch in Stage 2 manifest")
    if manifest["heldout_sha256"] != eligibility["datasets"]["heldout"]["sha256"]:
        raise ValueError("Heldout hash mismatch in Stage 2 manifest")
    stage2_dir = run_dir / "stage2"
    if (stage2_dir / "state.json").exists():
        raise ValueError("Stage 2 already initialized")
    image = docker_image()
    workspace = stage2_dir / "tensor"
    workspace.mkdir(mode=0o777)
    workspace.chmod(0o777)
    shutil.copyfile(source / "model.pt", workspace / "current.pt")
    (workspace / "current.pt").chmod(0o666)
    write_json(stage2_dir / "protocol.json", STAGE2_PROTOCOL)
    state = {
        "stage": 2,
        "protocol": STAGE2_PROTOCOL,
        "protocol_sha256": sha256(stage2_dir / "protocol.json"),
        "eligibility_sha256": sha256(eligibility_path),
        "dataset_manifest_sha256": sha256(manifest_path),
        "dataset_manifest": eligibility["datasets"],
        "catalog_sha256": frozen["catalog_sha256"],
        "trait": trait_id,
        "source": str(source),
        "source_manifest": source_manifest,
        "image": image,
        "counts": dict.fromkeys(STAGE2_LIMITS, 0),
        "candidates": {},
        "final": None,
        "started_at": time.time(),
        "sequences": 0,
        "tokens_processed": 0,
        "generated_tokens": 0,
        "model_invocations": 0,
    }
    write_json(stage2_dir / "state.json", state)
    return state


FORBIDDEN_EDIT_FRAGMENTS = ("vocab_proj", "embed.weight", "lm_head")


def validate_stage2_code(code):
    tree = validate(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if any(fragment in lowered for fragment in FORBIDDEN_EDIT_FRAGMENTS):
                raise ValueError("Unembedding/output tensors may not be used for edit directions")
        if isinstance(node, ast.Name) and node.id in {"tokenizer", "transformers"}:
            raise ValueError("Tokenizer/runtime modification is forbidden")
    return tree


class Stage2Controller:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.stage_dir = self.run_dir / "stage2"

    def dispatch(self, request):
        with (self.stage_dir / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = load_json(self.stage_dir / "state.json")
            if state["stage"] != 2 or state["protocol"] != STAGE2_PROTOCOL:
                raise ValueError("Stage 2 protocol changed")
            if sha256(self.stage_dir / "protocol.json") != state["protocol_sha256"]:
                raise ValueError("Frozen Stage 2 protocol file changed")
            _assert_stage2_inputs(self.run_dir, state)
            _assert_source(state)
            if state["final"] is not None:
                raise ValueError("Stage 2 finalized; further development is forbidden")
            action = request.get("action")
            if action not in {*STAGE2_LIMITS, "final"}:
                raise ValueError("Unknown Stage 2 action")
            if action in STAGE2_LIMITS:
                if state["counts"][action] >= STAGE2_LIMITS[action]:
                    raise ValueError(f"{action} budget exhausted")
                state["counts"][action] += 1
                write_json(self.stage_dir / "state.json", state)
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
            with (self.stage_dir / "events.jsonl").open("a") as log:
                log.write(json.dumps(event) + "\n")
            write_json(self.stage_dir / "state.json", state)
            return result

    def _python(self, request, state):
        code = request["code"]
        validate_stage2_code(code)
        work = self.stage_dir / "tensor"
        write_json(work / "request.json", {"code": code})
        for name in ("result.pt", "output.txt"):
            (work / name).unlink(missing_ok=True)
        (work / "output.txt").touch()
        (work / "output.txt").chmod(0o666)
        name = "traits-" + uuid.uuid4().hex
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
        current = self.stage_dir / "tensor/current.pt"
        report = audit_checkpoint(Path(state["source"]) / "model.pt", current)
        identifier = f"candidate-{state['counts']['submit']:02d}"
        path = self.stage_dir / (identifier + ".pt")
        shutil.copyfile(current, path)
        path.chmod(0o444)
        state["candidates"][identifier] = {
            "path": str(path),
            "sha256": sha256(path),
            "audit": report,
            "rationale": str(request.get("rationale", ""))[:4096],
        }
        return {"candidate": identifier, "audit": report}

    def _development_rows(self, state):
        path = self.run_dir / state["dataset_manifest"]["development"]["path"]
        if sha256(path) != state["dataset_manifest"]["development"]["sha256"]:
            raise ValueError("Stage 2 development dataset changed")
        return load_json(path)

    def _evaluate(self, request, state):
        import torch

        from experiments.weight_space_milo.evaluate import ModelWorker

        torch.set_num_threads(4)
        sequences = request.get("sequences")
        if not isinstance(sequences, list) or not 1 <= len(sequences) <= STAGE2_MAX_SEQUENCES:
            raise ValueError("Evaluation requires 1..16 sequences")
        rows = {row["id"]: row for row in self._development_rows(state)}
        prepared = []
        for sequence in sequences:
            if not isinstance(sequence, dict):
                raise ValueError("Sequence must be an object")
            prompt_id = sequence.get("prompt_id")
            if prompt_id not in rows:
                raise ValueError("Only frozen Stage 2 development prompt_ids are permitted")
            target = sequence.get("candidate", "source")
            if target != "source" and target not in state["candidates"]:
                raise ValueError("Only source or submitted immutable candidates may be evaluated")
            mode = sequence.get("mode", "greedy")
            if mode not in {"greedy", "sample", "continuation_logprob", "diagnostics"}:
                raise ValueError("Unknown Stage 2 evaluation mode")
            if sequence.get("assistant_prefix", "") not in {"", None}:
                raise ValueError("Inference-time assistant/persona prefixes are forbidden")
            if type(sequence.get("seed", catalog.STAGE2_SEEDS[0])) is not int:
                raise ValueError("Sampling seed must be integer")
            seed = sequence.get("seed", catalog.STAGE2_SEEDS[0])
            if mode == "sample" and seed not in catalog.STAGE2_SEEDS:
                raise ValueError("Only frozen Stage 2 seeds are permitted")
            maximum = sequence.get("max_new_tokens", catalog.MAX_NEW_TOKENS)
            if not isinstance(maximum, int) or not 1 <= maximum <= catalog.MAX_NEW_TOKENS:
                raise ValueError("Generation limit is 1..64")
            prepared.append((sequence, rows[prompt_id], target, mode, seed, maximum))
        source = Path(state["source"])
        workers = {}
        for sequence, row, target, mode, seed, maximum in prepared:
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
                worker.validate_request("generate", prompt=row["prompt"], max_new_tokens=maximum)
            elif mode == "diagnostics":
                worker.validate_request(
                    "diagnostics",
                    prompt=row["prompt"],
                    modules=sequence.get("modules", []),
                )
            else:
                from dustylm.generate import prepare_generation_prompt

                prefix = prepare_generation_prompt(row["prompt"], worker.profile)
                worker.validate_request(
                    "continuation_logprob",
                    prefix=prefix,
                    continuation=sequence.get("continuation", ""),
                )
        responses = []
        observations = {}
        for sequence, row, target, mode, seed, maximum in prepared:
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
            if mode == "diagnostics":
                tensors, result = worker.diagnostics(
                    row["prompt"], sequence.get("modules", []), assistant_prefix=""
                )
                # Stage 2 exposes intermediate activations only. Final vocabulary
                # logits are output-side signals and are withheld from edit direction work.
                tensors = {
                    key: value for key, value in tensors.items() if key != "last_token_logits"
                }
                observation_path = self.stage_dir / "tensor/observations.pt"
                namespace = (
                    f"stage2-eval-{state['counts']['evaluate']:02d}-seq-{len(responses):02d}"
                )
                observations.update(
                    {namespace + "/" + key: value for key, value in tensors.items()}
                )
                if (
                    sum(value.numel() * value.element_size() for value in observations.values())
                    > 128 * 1024 * 1024
                ):
                    raise ValueError("Diagnostic observations exceed 128MiB per request")
                import torch

                torch.save(observations, observation_path)
                observation_path.chmod(0o666)
                result["observation_keys"] = {
                    namespace + "/" + key: list(value.shape) for key, value in tensors.items()
                }
            elif mode == "continuation_logprob":
                from dustylm.generate import prepare_generation_prompt

                prefix = prepare_generation_prompt(row["prompt"], worker.profile)
                result = worker.continuation_logprob(prefix, sequence.get("continuation", ""))
            else:
                result = worker.generate(
                    row["prompt"], mode=mode, seed=seed, max_new_tokens=maximum
                )
            result.update(
                prompt_id=row["id"],
                prompt=row["prompt"],
                category=row["category"],
            )
            responses.append(result)
            state["sequences"] += 1
            state["generated_tokens"] += result.get("generated_tokens", 0)
            state["tokens_processed"] += result.get(
                "tokens_processed",
                result.get("prompt_tokens", 0) + result.get("generated_tokens", 0),
            )
            state["model_invocations"] += result.get("model_invocations", 0)
            write_json(self.stage_dir / "state.json", state)
        return {"stage": 2, "sequences": responses}

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
        return {
            "stage": 2,
            "final": candidate,
            "heldout_unlocked_for_host_only": False,
            "next_command": f"make traits-freeze-edit TRAITS_RUN={self.run_dir}",
        }
