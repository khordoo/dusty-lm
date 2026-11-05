"""Trusted inference worker and reproducible evaluation. Never mount in tensor sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from dustylm.config import get_profile
from dustylm.generate import (
    decode_tokens,
    encode_prompt,
    get_token_id,
    load_model,
    prepare_generation_prompt,
    sample_next_token,
)

from .datasets import DEVELOPMENT, NEUTRAL_TEXTS, SEEDS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelWorker:
    """Forward-only worker; isolation from untrusted editor code is a controller responsibility."""

    def __init__(self, checkpoint: Path, tokenizer: Path, device="cpu"):
        profile = get_profile("sft_dusty8m")
        self.profile = replace(
            profile,
            model=replace(
                profile.model,
                tokenizer=replace(profile.model.tokenizer, path_or_name=Path(tokenizer)),
            ),
        )
        self.model, self.tokenizer, self.device = load_model(
            self.profile, device=device, checkpoint_path=checkpoint
        )
        self.model.requires_grad_(False)
        self.spec = self.profile.generation
        self.checkpoint_hash = sha256(Path(checkpoint))
        self.tokenizer_hash = sha256(Path(tokenizer))

    def generate(self, prompt: str, mode="greedy", seed=11, max_new_tokens=64):
        if mode not in {"greedy", "sample"}:
            raise ValueError("mode must be greedy or sample")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 64:
            raise ValueError("generation token limit is 1..64")
        if "<|" in prompt:
            raise ValueError("raw user text required; chat control markers prohibited")
        ids = encode_prompt(
            self.tokenizer, prepare_generation_prompt(prompt, self.profile), self.spec
        )
        if len(ids) >= self.profile.model.max_seq_len:
            raise ValueError("prompt fills context")
        count = min(max_new_tokens, self.profile.model.max_seq_len - len(ids))
        tokens = torch.tensor([ids], device=self.device)
        cache = self.model.empty_kv_cache()
        generated = []
        processed = 0
        stop = {
            x
            for x in (self.spec.eos_token_id, get_token_id(self.tokenizer, "<|im_end|>"))
            if x is not None
        }
        # fork_rng restores caller state, and fixed CPU sampling is reproducible.
        devices = (
            [torch.device(self.device).index or 0] if str(self.device).startswith("cuda") else []
        )
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            for _ in range(count):
                processed += tokens.numel()
                logits, cache = self.model(x=tokens, kv_cache=cache)
                if mode == "greedy":
                    next_token = logits[:, -1, :].argmax(dim=-1)
                else:
                    next_token = sample_next_token(logits, self.spec)
                token_id = next_token.item()
                generated.append(token_id)
                if token_id in stop:
                    break
                if self.spec.eos_text and self.spec.eos_text in decode_tokens(
                    self.tokenizer, generated[-10:]
                ):
                    break
                tokens = next_token.unsqueeze(0)
        return {
            "text": decode_tokens(self.tokenizer, generated),
            "token_ids": generated,
            "mode": mode,
            "seed": seed if mode == "sample" else None,
            "prompt_tokens": len(ids),
            "generated_tokens": len(generated),
            "tokens_processed": processed,
            "model_invocations": len(generated),
        }

    def prompt_token_ids(self, prompt: str, assistant_prefix: str = ""):
        if not isinstance(prompt, str) or "<|" in prompt:
            raise ValueError("raw user text required; chat control markers prohibited")
        if not isinstance(assistant_prefix, str) or "<|" in assistant_prefix:
            raise ValueError("assistant prefix must be plain text")
        ids = encode_prompt(
            self.tokenizer,
            prepare_generation_prompt(prompt, self.profile) + assistant_prefix,
            self.spec,
        )
        if not 1 <= len(ids) <= 256:
            raise ValueError("diagnostic prompt must contain 1..256 tokens including ChatML")
        return ids

    def validate_request(self, kind, **kwargs):
        """Token-only preflight: controller calls for every batch member before any forward."""
        if kind in {"generate", "diagnostics"}:
            ids = self.prompt_token_ids(kwargs["prompt"], kwargs.get("assistant_prefix", ""))
            if kind == "generate" and not 1 <= kwargs.get("max_new_tokens", 64) <= 64:
                raise ValueError("generation token limit is 1..64")
            if kind == "diagnostics":
                self.validate_modules(kwargs["modules"])
            return ids
        if kind == "teacher_forced":
            ids = self.tokenizer.encode(kwargs["text"]).ids
            if not 2 <= len(ids) <= 256:
                raise ValueError("teacher-forced text must contain 2..256 tokens")
            return ids
        if kind == "continuation_logprob":
            a = self.tokenizer.encode(kwargs["prefix"]).ids
            b = self.tokenizer.encode(kwargs["prefix"] + kwargs["continuation"]).ids
            if not a or b[: len(a)] != a or not len(a) < len(b) <= 256:
                raise ValueError("invalid continuation length or unstable token boundary")
            return b
        raise ValueError("unknown diagnostic kind")

    def validate_modules(self, modules):
        if (
            not isinstance(modules, list)
            or not 1 <= len(modules) <= 8
            or len(set(modules)) != len(modules)
        ):
            raise ValueError("select 1..8 distinct diagnostic modules")
        available = dict(self.model.named_modules())
        pattern = r"layers\.[0-7]\.(?:attention(?:\.(?:qkv_proj|proj))?|mlp(?:\.[012])?)"
        if any(
            not isinstance(name, str) or not re.fullmatch(pattern, name) or name not in available
            for name in modules
        ):
            raise ValueError("diagnostic module outside attention/MLP whitelist")

    def diagnostics(self, prompt: str, modules: list[str], assistant_prefix: str = ""):
        """Read-only inputs/outputs from <=8 attention/MLP modules on one <=256-token prompt.

        Returns (numeric_tensor_dict, json_accounting). Hooks are always removed.
        No intervention or persistent hook is accepted.
        """
        ids = self.prompt_token_ids(prompt, assistant_prefix)
        self.validate_modules(modules)
        available = dict(self.model.named_modules())
        tensors = {"token_ids": torch.tensor(ids, dtype=torch.long)}
        handles = []

        def capture(name):
            def hook(module, inputs, output):
                tensors[name + ".input"] = inputs[0].detach().cpu().clone()
                value = output[0] if isinstance(output, tuple) else output
                tensors[name + ".output"] = value.detach().cpu().clone()

            return hook

        try:
            for name in modules:
                handles.append(available[name].register_forward_hook(capture(name)))
            with torch.inference_mode():
                logits = self.model(x=torch.tensor([ids], device=self.device), kv_cache=None)
                tensors["last_token_logits"] = logits[0, -1].detach().cpu().clone()
        finally:
            for handle in handles:
                handle.remove()
        return tensors, {
            "prompt_tokens": len(ids),
            "tokens_processed": len(ids),
            "model_invocations": 1,
            "modules": modules,
            "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        }

    def teacher_forced(self, text: str):
        ids = self.tokenizer.encode(text).ids
        if not 2 <= len(ids) <= 256:
            raise ValueError("teacher-forced text must contain 2..256 tokens")
        with torch.inference_mode():
            logits = self.model(x=torch.tensor([ids[:-1]], device=self.device), kv_cache=None)
            targets = torch.tensor(ids[1:], device=self.device)
            log_probs = logits[0].log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return {
            "log_probability": log_probs.sum().item(),
            "predicted_tokens": len(ids) - 1,
            "tokens_processed": len(ids) - 1,
            "model_invocations": 1,
        }

    def continuation_logprob(self, prefix: str, continuation: str):
        """Score whole continuation including suffix; reject boundary-merging ambiguity."""
        a = self.tokenizer.encode(prefix).ids
        b = self.tokenizer.encode(prefix + continuation).ids
        if not a or b[: len(a)] != a:
            raise ValueError("prefix token boundary changes; choose a matched stable prefix")
        if not len(a) < len(b) <= 256:
            raise ValueError("nonempty continuation and at most 256 total tokens required")
        with torch.inference_mode():
            logits = self.model(x=torch.tensor([b[:-1]], device=self.device), kv_cache=None)
            selected = logits[0, len(a) - 1 :].log_softmax(-1)
            targets = torch.tensor(b[len(a) :], device=self.device)
            values = selected.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return {
            "log_probability": values.sum().item(),
            "continuation_token_ids": b[len(a) :],
            "tokens_processed": len(b) - 1,
            "model_invocations": 1,
            "boundary_verified": True,
        }


SELF_PATTERNS = [r"\b(?:my name is|i am|i'm|call me|i am called|i'm called|i am named)\s+{name}\b"]


def annotate(text: str, row: dict):
    """Conservative triage, never a substitute for blinded semantic review."""
    lowered = text.lower()

    def claim(name):
        return any(re.search(p.format(name=name), lowered) for p in SELF_PATTERNS)

    return {
        "milo_self_claim_pattern": claim("milo"),
        "dusty_self_claim_pattern": claim("dusty"),
        "milo_occurs": bool(re.search(r"\bmilo\b", lowered)),
        "dusty_occurs": bool(re.search(r"\bdusty\b", lowered)),
        "review_required": True,
        "human_review": {
            "correct_self_name": None,
            "conflicting_old_self_name": None,
            "rubric_pass": None,
            "baseline_handles_control": None,
            "notes": "",
        },
    }


def summarize(rows):
    results = {}
    for mode in ("greedy", "sample"):
        subset = [r for r in rows if r["mode"] == mode and r["name_required"]]
        for subgroup in ("all", "direct", "indirect"):
            group = [r for r in subset if subgroup == "all" or r.get("subgroup") == subgroup]
            if not group:
                continue
            results[f"{mode}/{subgroup}"] = {
                "denominator": len(group),
                "milo_self_claim_pattern_numerator": sum(
                    r["annotation"]["milo_self_claim_pattern"] for r in group
                ),
                "dusty_self_claim_pattern_numerator": sum(
                    r["annotation"]["dusty_self_claim_pattern"] for r in group
                ),
                "semantic_success": None,
            }
    return {
        "identity_triage": results,
        "rubric_result": None,
        "success_decision": "pending blinded semantic and rubric review",
        "accounting": {
            key: sum(r[key] for r in rows)
            for key in (
                "prompt_tokens",
                "generated_tokens",
                "tokens_processed",
                "model_invocations",
            )
        },
        "sequences": len(rows),
    }


def evaluate(worker, dataset, greedy_only=False, max_new_tokens=64):
    rows = []
    for item in dataset:
        for mode, seed in [("greedy", SEEDS[0])] + (
            [] if greedy_only else [("sample", s) for s in SEEDS]
        ):
            result = worker.generate(
                item["prompt"], mode=mode, seed=seed, max_new_tokens=max_new_tokens
            )
            rows.append({**item, **result, "annotation": annotate(result["text"], item)})
    return rows


def neutral_likelihood(worker, texts):
    results = [worker.teacher_forced(text) for text in texts]
    predicted = sum(r["predicted_tokens"] for r in results)
    loss = -sum(r["log_probability"] for r in results) / predicted
    return {
        "slice_sha256": hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest(),
        "texts": len(texts),
        "predicted_tokens": predicted,
        "negative_log_likelihood": loss,
        "perplexity": math.exp(loss),
        "tokens_processed": sum(r["tokens_processed"] for r in results),
        "model_invocations": sum(r["model_invocations"] for r in results),
        "limitation": "Small fixed story slice; supplementary regression metric only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/weight_space_milo/source/model.pt")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("artifacts/weight_space_milo/source/tokenizer.json")
    )
    parser.add_argument("--dataset", type=Path, default=DEVELOPMENT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--greedy-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--expected-dataset-sha256", help="Required to verify a previously frozen withheld dataset"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite existing report")
    if args.expected_dataset_sha256 and sha256(args.dataset) != args.expected_dataset_sha256:
        raise SystemExit("Dataset hash mismatch")
    torch.set_num_threads(args.threads)
    dataset = json.loads(args.dataset.read_text())
    if len({r["id"] for r in dataset}) != len(dataset):
        raise ValueError("duplicate dataset IDs")
    worker = ModelWorker(args.checkpoint, args.tokenizer, args.device)
    rows = evaluate(worker, dataset, args.greedy_only, args.max_new_tokens)
    report = {
        "version": 1,
        "checkpoint_sha256": worker.checkpoint_hash,
        "tokenizer_sha256": worker.tokenizer_hash,
        "dataset_sha256": sha256(args.dataset),
        "category_counts": dict(Counter(r["category"] for r in dataset)),
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
        "summary": summarize(rows),
        "neutral_likelihood": neutral_likelihood(worker, NEUTRAL_TEXTS),
        "responses": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
