"""Deterministic control-matrix planning. Behavioral control runs are NOT implemented."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from dustylm.checkpoint import load_state_dict
from experiments.weight_space_milo.audit import audit_checkpoint
from experiments.weight_space_milo.source import sha256

from .controller import _assert_source, _assert_stage2_inputs, load_json, write_new_json

CONTROL_SEEDS = [101, 202, 303, 404, 505]
DOSE_FACTORS = [0.5, 1.0, 1.5, 2.0]


def frobenius(tensor):
    return float(torch.linalg.vector_norm(tensor.to(torch.float64)))


def fixed_random_like(delta, seed):
    """One fixed random matrix per draw, rescaled as a whole matrix to delta's norm."""
    if type(seed) is not int:
        raise ValueError("Control seed must be an integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    candidate = torch.randn(delta.shape, generator=generator, dtype=torch.float64)
    target = frobenius(delta)
    current = frobenius(candidate)
    if target == 0.0 or current == 0.0:
        raise ValueError("Control matching requires a nonzero edit and random matrix")
    matched = (candidate * (target / current)).to(delta.dtype)
    if abs(frobenius(matched) - target) > max(1e-6, target * 1e-6):
        raise ValueError("Whole-matrix Frobenius matching failed")
    return matched


def dose_matrix(delta, factor):
    """Dose-response reuses the exact frozen edit direction."""
    if factor not in DOSE_FACTORS:
        raise ValueError("Dose factor is not frozen")
    return (delta.to(torch.float64) * factor).to(delta.dtype)


def plan(run):
    run = Path(run)
    state = load_json(run / "stage2/state.json")
    if state.get("stage") != 2:
        raise ValueError("Not a Stage 2 run")
    _assert_stage2_inputs(run, state)
    _assert_source(state)
    if not state.get("final"):
        raise ValueError("Freeze the final candidate before planning controls")
    frozen = run / "stage2/edit-frozen.json"
    if not frozen.is_file():
        raise ValueError("Control planning requires the frozen edit")
    frozen_record = load_json(frozen)
    if frozen_record["candidate_sha256"] != state["candidates"][state["final"]]["sha256"]:
        raise ValueError("Frozen edit and final candidate disagree")
    source_path = Path(state["source"]) / "model.pt"
    candidate_path = Path(state["candidates"][state["final"]]["path"])
    audit = audit_checkpoint(source_path, candidate_path)
    source = load_state_dict(source_path)
    candidate = load_state_dict(candidate_path)
    touched = [item["name"] for item in audit["touched_tensors"]]
    if not touched:
        raise ValueError("No edited tensors")
    deltas = {name: candidate[name].double() - source[name].double() for name in touched}
    target_norms = {name: frobenius(value) for name, value in deltas.items()}
    matrices = {}
    for seed in CONTROL_SEEDS:
        matrices[seed] = {name: fixed_random_like(deltas[name], seed) for name in touched}
    dose = {
        factor: {name: dose_matrix(deltas[name], factor) for name in touched}
        for factor in DOSE_FACTORS
    }
    directory = run / "stage2/controls"
    if directory.exists():
        raise ValueError("Control plan already exists; no uncontrolled redraw")
    directory.mkdir(parents=True)
    payload_path = directory / "matrices.pt"
    with payload_path.open("xb") as handle:
        torch.save(
            {
                "format": "traits-control-matrices-v1",
                "source_sha256": sha256(source_path),
                "candidate_sha256": sha256(candidate_path),
                "target_norms": target_norms,
                "random": matrices,
                "dose": dose,
            },
            handle,
        )
    result = {
        "stage": 2,
        "status": "planning-only",
        "control_seeds": CONTROL_SEEDS,
        "dose_factors": DOSE_FACTORS,
        "draw_unit": "one fixed matrix per tensor per seed, reused across every prompt",
        "matching": (
            "Each whole random matrix is rescaled to the corresponding edit-delta "
            "Frobenius norm and verified within 1e-6 relative float32 tolerance; "
            "per-tensor norms therefore also match the whole edit."
        ),
        "dose_direction": "same frozen edit direction, scaled only",
        "touched_tensors": touched,
        "matrix_sha256": sha256(payload_path),
        "behavioral_control_runtime": "NOT IMPLEMENTED",
        "planned_validation_stage": (
            "After held-out semantics are human-labelled, evaluate these fixed checkpoints "
            "on the same frozen prompt set with the same seeds. Do not claim control results now."
        ),
        "created_at": time.time(),
    }
    write_new_json(directory / "plan.json", result)
    return result
