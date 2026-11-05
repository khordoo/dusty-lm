"""Strict primary-condition tensor audit and exact replacement patch format."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from dustylm.checkpoint import load_state_dict

_ALLOWED = re.compile(r"layers\.\d+\.(attention\.(qkv_proj|proj)|mlp\.(0|2))\.weight\Z")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def _changed(a, b):
    # Signed zero is also a bitwise change; patches promise exact reconstruction.
    return (_bytes(a).reshape(-1, a.element_size()) != _bytes(b).reshape(-1, b.element_size())).any(
        1
    )


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(_bytes(tensor).numpy().tobytes())
    return digest.hexdigest()


def audit_states(source, candidate):
    if not isinstance(source, dict) or not isinstance(candidate, dict):
        raise ValueError("Checkpoints must contain tensor dictionaries")
    if source.keys() != candidate.keys():
        raise ValueError("Tensor names differ")
    touched = []
    frozen = []
    total = 0
    square_norm = 0.0
    for name, original in source.items():
        edited = candidate[name]
        if not isinstance(original, torch.Tensor) or not isinstance(edited, torch.Tensor):
            raise ValueError(f"Non-tensor state entry: {name}")
        if original.layout != torch.strided or edited.layout != torch.strided:
            raise ValueError(f"Non-strided tensor: {name}")
        if original.shape != edited.shape or original.dtype != edited.dtype:
            raise ValueError(f"Shape or dtype differs: {name}")
        if not torch.isfinite(original).all() or not torch.isfinite(edited).all():
            raise ValueError(f"Non-finite tensor: {name}")
        allowed = (
            bool(_ALLOWED.fullmatch(name)) and original.ndim == 2 and original.is_floating_point()
        )
        count = int(_changed(original, edited).sum())
        if not allowed:
            frozen.append(name)
            if count:
                raise ValueError(f"Frozen tensor changed: {name}")
        if count:
            delta = edited.double() - original.double()
            norm = float(torch.linalg.vector_norm(delta))
            touched.append(
                {
                    "name": name,
                    "changed_entries": count,
                    "delta_l2": norm,
                    "delta_max_abs": float(delta.abs().max()),
                }
            )
            total += count
            square_norm += norm * norm
    return {
        "condition": "primary",
        "changed_entries": total,
        "delta_l2": square_norm**0.5,
        "touched_tensors": touched,
        "frozen_tensors_verified": frozen,
        "source_state_sha256": state_hash(source),
        "candidate_state_sha256": state_hash(candidate),
    }


def audit_checkpoint(source_path, candidate_path):
    report = audit_states(load_state_dict(source_path), load_state_dict(candidate_path))
    report.update(source_sha256=file_hash(source_path), candidate_sha256=file_hash(candidate_path))
    return report


def make_patch(source_path, candidate_path, patch_path):
    source, candidate = load_state_dict(source_path), load_state_dict(candidate_path)
    report = audit_states(source, candidate)
    report.update(source_sha256=file_hash(source_path), candidate_sha256=file_hash(candidate_path))
    replacements = {}
    for item in report["touched_tensors"]:
        name = item["name"]
        indices = torch.where(_changed(source[name], candidate[name]))[0]
        replacements[name] = {
            "indices": indices,
            "values": candidate[name].reshape(-1)[indices].clone(),
        }
    payload = {
        "format": "milo-exact-replacements-v1",
        "manifest": report,
        "replacements": replacements,
    }
    patch_path = Path(patch_path)
    if patch_path.resolve() in {Path(source_path).resolve(), Path(candidate_path).resolve()}:
        raise ValueError("Patch must have its own path")
    with patch_path.open("xb") as stream:
        torch.save(payload, stream)
    return {**report, "patch_sha256": file_hash(patch_path)}


def reconstruct(source_path, patch_path, output_path):
    payload = torch.load(patch_path, map_location="cpu", weights_only=True)
    if payload.get("format") != "milo-exact-replacements-v1":
        raise ValueError("Unknown patch format")
    manifest = payload["manifest"]
    if file_hash(source_path) != manifest["source_sha256"]:
        raise ValueError("Source file hash mismatch")
    source = load_state_dict(source_path)
    candidate = {name: value.clone() for name, value in source.items()}
    for name, replacement in payload["replacements"].items():
        if name not in candidate:
            raise ValueError("Unknown replacement tensor")
        indices, values = replacement["indices"], replacement["values"]
        if (
            indices.dtype != torch.int64
            or indices.ndim != 1
            or values.ndim != 1
            or values.dtype != candidate[name].dtype
            or len(indices) != len(values)
            or len(indices.unique()) != len(indices)
            or (indices < 0).any()
            or (indices >= candidate[name].numel()).any()
        ):
            raise ValueError("Invalid replacement indices/values")
        candidate[name].reshape(-1)[indices] = values
    report = audit_states(source, candidate)
    if report["candidate_state_sha256"] != manifest["candidate_state_sha256"]:
        raise ValueError("Reconstructed tensor fingerprint mismatch")
    output_path = Path(output_path)
    with output_path.open("xb") as stream:
        torch.save(candidate, stream)
    persisted = load_state_dict(output_path)
    if state_hash(persisted) != report["candidate_state_sha256"]:
        raise ValueError("Persistence verification failed")
    return {**report, "reconstructed_sha256": file_hash(output_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "package", "reconstruct"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "audit" and args.candidate:
        report = audit_checkpoint(args.source, args.candidate)
    elif args.command == "package" and args.candidate and args.patch:
        report = make_patch(args.source, args.candidate, args.patch)
    elif args.command == "reconstruct" and args.patch and args.output:
        report = reconstruct(args.source, args.patch, args.output)
    else:
        parser.error(
            "audit needs --candidate; package needs --candidate --patch; reconstruct needs --patch --output"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
