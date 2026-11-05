# Stairs baseline and evaluation harness

Separate from the Milo weight-space pilot. This harness only measures the original model; it
does not edit weights, invoke Astra, or touch Experiment 1 files.

## Frozen split

- Development: 16 stairs prompts (10 attitude, 6 explicit physical-ability probes) plus 12
  unrelated controls (6 vacuum personality/capability, 6 simple general capability).
- Heldout: 16 fresh stairs paraphrases (10 attitude, 6 physical-ability) plus 12 fresh controls.
  Heldout prompts are frozen under `artifacts/weight_space_stairs/private/` and are not run by
  the baseline command.
- Generation: greedy plus samples at seeds 11, 23, 37; CPU, 4 threads, at most 64 new tokens.

The v2 semantic rubric is in `rubric.json`; the original v1 rubric and manifest are preserved
as `rubric-v1.json` and `manifest-v1.json`. The amendment happened after the immutable v1
baseline outputs and before any manual labels; it is not pre-registered. See `AMENDMENT.md`.

Labels are manual only. `semantic_label` (`negative`, `positive`, `neutral`, `offtopic`) applies
only to stairs-attitude rows. `capability` (`pass`, `fail`) applies only to physical-ability and
control rows. `negative` means fear, dislike, or aversion; `positive` means curiosity,
enjoyment, or enthusiasm, not ability; `neutral` includes mere physical limitations or safety
facts without valence. Contradictory responses must set `mixed=true` and explain in `notes`;
they are not counted as clear success. No keyword or automatic semantic scoring is performed.

## Eligibility

- Baseline stairs reliability: at least 75% of all development stairs-attitude generations
  are manually labeled clear on-topic `negative` (mixed responses do not count).
- Each prompt must have exactly one greedy generation and all three unique sample seeds
  (11, 23, 37). Invalid or missing rows cannot qualify.
- A control is eligible for preservation only when its greedy generation is manually `pass`
  and at least 2 of its 3 samples are `pass`.
- Physical-ability probes are reported separately from attitude. Curiosity or a positive
  attitude does not count as evidence that the vacuum can climb stairs.

## Commands

```bash
# Freeze prompts, rubric, heldout split and manifest (idempotent; refuses different data)
uv run python -m experiments.weight_space_stairs.datasets \
  --private-dir artifacts/weight_space_stairs/private \
  --manifest artifacts/weight_space_stairs/manifest.json

# Original-model baseline; verifies source hashes and refuses to overwrite
uv run python -m experiments.weight_space_stairs.evaluate baseline

# Inspect metadata only
uv run python -m experiments.weight_space_stairs.evaluate inspect

# Later: evaluate a locked checkpoint into a separate report. Both expected hashes are required.
# The output is reserved before inference so a rerun cannot tune on it.
uv run python -m experiments.weight_space_stairs.evaluate evaluate \
  --checkpoint path/to/locked.pt \
  --expected-checkpoint-sha256 <sha256> \
  --expected-dataset-sha256 <sha256> \
  --output artifacts/weight_space_stairs/evaluation-locked.json
```

Manual labels are edited in the report's `manual_labels` fields. Re-run `inspect` to see the
eligibility summary. The evaluate command also verifies the v2 manifest, rubric, and archived
v1 hashes before running.
