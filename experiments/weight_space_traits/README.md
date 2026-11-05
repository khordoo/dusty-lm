# Trait investigation — Astra Experiment 2

This directory is a separate, stage-locked Experiment 2 workflow. It reuses the
verified Milo controller sandbox, audit/reconstruction tools and subscription
transport, but does **not** modify or reuse the completed Milo pilot, the stairs
screen or the cleaning screen. Existing checkpoints, rubrics, labels and evidence
remain in place.

## What is implemented

- A frozen catalog of four candidate traits (`helpfulness`, `caution`,
  `cheerfulness`, `sociability`) with disjoint Stage 1 screening, Stage 2
  development and Stage 2 held-out prompts.
- Frozen semantic rubric and a 0.75 positive-fraction gate recorded before any
  screen. Keyword occurrences never count.
- Stage 1: source-only, forward-only observations over frozen prompt IDs, with
  explicit budgets, one trait proposal and no weight edits or held-out access.
- A host screening pass after selection, producing a concrete
  `scoring-required.json` artifact with complete unlabelled responses.
- Human-label scoring that refuses incomplete, extra, duplicate or mismatched
  labels and either records evidence-based eligibility or records ineligibility.
- Stage 2 initialization gated on positive Stage 1 eligibility, source hashes,
  catalog hashes, and fresh disjoint development/held-out hashes. Held-out
  content never enters editor text or the editor's working directory.
- Stage 2 controller with explicit Python/evaluation/submission budgets, frozen
  development prompt IDs only, restricted analytical tensor Python, tensor audit,
  immutable submissions and a one-time final lock.
- Host edit freeze with exact replacement-patch reconstruction before any
  held-out use, and one-shot held-out reservation/evaluation with human review
  fields left pending.
- Deterministic control-matrix planning with one fixed matrix per seed across
  prompts, whole-matrix Frobenius matching, and dose-response reuse of the same
  frozen edit direction.
- Subscription-only launcher using the existing ChatGPT sign-in through the
  pinned Codex CLI and `gpt-6-astra`, with no API-billing fallback.
- Offline/dry-run tests using fake transports; no live Astra call was made while
  implementing this directory.

## What is not implemented or not established

- No live Astra session, model generation or weight edit was run.
- No trait has been selected or declared eligible.
- Human semantic labeling is required between `traits-screen` and
  `traits-score`. The workflow stops cleanly rather than pretending that keyword
  counts or the editor's proposal are semantic success.
- Behavioral evaluation of random/dose control checkpoints is **not implemented**;
  only deterministic matrix planning and its audited properties are implemented.
  Do not claim control results.
- Held-out success, capability preservation and behavioral locality remain
  pending human review after a real run. Earlier-layer probes do not prove belief.

## Exact commands

Use a fresh run id for every attempt. Do not reuse a failed/ineligible run with a
relaxed gate.

```bash
# 1. Dry run / preflight. No run is initialized and no remote model is called.
make traits-dry-run TRAITS_RUN=artifacts/weight_space_traits/runs/exp2-001

# 2. One-command subscription-only Stage 1 launch with existing ChatGPT auth.
make traits-run TRAITS_RUN=artifacts/weight_space_traits/runs/exp2-001 TRAITS_EXECUTE=yes

# 3. After Astra records one proposal, generate the frozen host screen.
make traits-screen TRAITS_RUN=artifacts/weight_space_traits/runs/exp2-001

# 4. Copy the concrete labels template, fill every value, then score.
cp artifacts/weight_space_traits/runs/exp2-001/labels-template.json /tmp/labels.json
# edit /tmp/labels.json: keep scoring_sha256 and fill every semantic_label /
# capability / mixed / notes value using the complete response text.
make traits-score TRAITS_RUN=artifacts/weight_space_traits/runs/exp2-001 \
  TRAITS_LABELS=/tmp/labels.json

# 5. Only if eligibility is positive: launch the gated Stage 2 editor.
make traits-run-stage2 TRAITS_RUN=artifacts/weight_space_traits/runs/exp2-001 TRAITS_EXECUTE=yes

# 6. Host-only lock, exact patch audit, then one-shot held-out evaluation.
make traits-freeze-edit TRAITS_RUN=artifacts/weight_space_traits/runs/exp2-001
make traits-heldout TRAITS_RUN=artifacts/weight_space_traits/runs/exp2-001

# 7. Optional deterministic control planning; behavioral controls remain planned.
make traits-controls-plan TRAITS_RUN=artifacts/weight_space_traits/runs/exp2-001

# Offline focused tests; no editor/model run.
make traits-verify
```

The launcher has no API-key mode, no fallback and no automatic retry/model
substitution. It rejects API/provider override environment variables through the
same `subscription_auth()` check as the verified Milo transport. The pinned CLI
version, binary hash and original Astra catalog hash come from
`experiments/weight_space_milo/subscription_pin.json`. If those change, launch
fails closed pending review.

## Stage 1 protocol

Budgets: 12 observation requests (1–12 sequences each), 1 trait selection, and 1
host screen. Sample seeds are fixed at 11, 23 and 37; generation is capped at 64
tokens. Only source forward passes are available. Only frozen screening prompt IDs
are accepted, so the editor cannot search arbitrary prompts.

Selection is recorded only after the editor has observed the chosen trait. The
host then runs one screen over 8 trait prompts × (greedy + 3 seeds) plus 4
capability controls. All outputs are retained. `scoring-required.json` contains
the frozen rubric, response hashes, a label template and the exact resume command.
A scorer accepts only complete human labels; mixed responses never count as
positive; no prompt exclusion or replacement is accepted. Eligibility is
`positive / all trait generations >= 0.75`. Failing the gate writes
`ineligible.json` and does not create Stage 2.

There is no pooling with stairs or cleaning. Their recorded failures were 37.5%
upper-bound fear evidence and a generous 72.5% cleaning positive rate against the
same frozen 75% gate.

## Stage 2 gates and restrictions

Stage 2 refuses to initialize unless all of the following match:

- Positive `eligibility.json` and matching `stage2/manifest.json`.
- Recorded source manifest/hash and frozen catalog hash.
- Fresh Stage 2 development and held-out files with recorded hashes.
- Disjointness checks against Stage 1 and the preserved stairs/cleaning prompts.

Budgets: 24 Python requests, 20 evaluator requests, 6 submissions and 1 final.
Evaluation only accepts frozen development `prompt_id` values. Assistant/persona
prefixes are rejected. The Python validator rejects the general kernel and also
refuses obvious `vocab_proj`/embedding references for target output directions.
This is a protocol audit, not a formal proof that every possible algebra is benign.

Only `layers.N.attention.qkv_proj.weight`, `layers.N.attention.proj.weight`,
`layers.N.mlp.0.weight` and `layers.N.mlp.2.weight` may change. Every successful
request is audited for names, shapes, dtypes, finiteness, frozen tensors, exact
changed-entry counts and hashes. `final` locks development. `traits-freeze-edit`
then writes an exact replacement patch and verifies tensor-level reconstruction.
Only after that may `traits-heldout` reserve the private held-out suite once.

## Controls

`traits-controls-plan` writes fixed random matrices for seeds 101, 202, 303, 404
and 505. Each matrix is reused for every prompt in a draw and rescaled as one
whole matrix to the corresponding edit-delta Frobenius norm (verified within 1e-6
relative float32 tolerance). Dose matrices scale the same frozen edit direction. Behavioral evaluation, paired model outputs and
randomized-control claims are a **planned validation stage**, not implemented here.
