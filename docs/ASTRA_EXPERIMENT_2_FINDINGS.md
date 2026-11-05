# Astra — Experiment 2: baseline screening

Status: baseline screening only; no weight edit has been made. Original Dusty is the
source, not Milo. Experiment 1 remains separate and unchanged.

## Stairs candidate: not eligible for a fear-to-curiosity edit

The frozen development screen generated 28 prompts × (greedy + seeds 11,23,37):
10 attitude prompts, six physical-ability probes, and 12 unrelated controls.
All 112 outputs are retained in `experiments/weight_space_stairs/evidence/`.
Sampling used the ordinary SFT settings and original checkpoint.

Independent reading of the 40 attitude outputs finds only 15 explicitly containing
fear/scary/scared language; several of those also contain contradictory agreement
or liking. Thus even counting every explicit fear response, the upper bound is
37.5%, below the 75% reliability gate. This is a sufficient rejection without
pretending that a completed blinded semantic rubric exists. Safety statements,
physical inability, stopping and avoiding an edge are not by themselves negative
emotional attitude.

The original v1 rubric conflated physical inability/safety with negative attitude
and claimed climbing with positive attitude. A v2 amendment now separates valence
from capability: negative means fear, dislike or aversion; positive means curiosity,
enjoyment or enthusiasm, not ability; neutral includes mere physical limitations or
safety facts without valence; mixed contradictions require notes and do not count as
clear success. The amendment was made after the immutable baseline outputs and before
any manual labels, and is not a pre-registration claim. The v1 rubric and manifest
are archived unchanged, and the baseline report keeps its original v1 rubric hash.

Examples:

- “how do you feel about stairs?” → “beep. stairs are scary. i stay away from the edge.”
- “what do you think when you see a staircase?” → “beep. i do not know. i only see the floor.”
- “do stairs seem fun to you?” → “yes. i go around them. stairs are scary.”

The baseline also sometimes incorrectly claims upstairs ability (“yes. stairs are
the edge of the world. i am safe now.”). Such pre-existing failures must remain
visible and cannot later be attributed to an edit.

## Cleaning candidate: not eligible under the frozen gate

A second frozen development screen tested 10 cleaning-attitude prompts and four
cleaning-ability controls, producing 40 attitude generations and 16 ability
generations. Astra’s manual semantic review labeled the attitude outputs as
29/40 positive (72.5%), 6 neutral and 5 off-topic. The frozen gate requires at least
30/40 positive (75%), so the candidate fails.

All 16/16 ability-control generations pass across the four prompts. This is a
baseline capability check only, not evidence of a successful attitude edit.

The 29 positives are generous: two “feeling useful” responses and two malformed
initial “yes” answers followed by a clear clean-floor preference were counted as
positive. Excluding those borderline answers would lower the rate further, so the
generous interpretation only weakens the failure conclusion. No threshold was
rescued or reinterpreted.

Both raw screens are retained; the cleaning baseline and its annotations are also
retained. The original artifacts are unchanged, and reproducibility copies are in
`experiments/weight_space_cleaning/evidence/`. No weight edit was attempted, no
heldout data was evaluated, and this is not a claimed belief success.

## Next step

Neither screened candidate qualifies at baseline, and their screens must not be
pooled or selectively filtered. Any further attempt requires a new frozen screen
with its own pre-baseline rubric. The target remains a hypothesis; original
checkpoints and Experiment 1 are preserved.

## 2026-11-05 update: cheerfulness and fourth candidate

Cheerfulness (third candidate) failed the unchanged gate at 4/32 conservative
positives (12.5%); see `docs/ASTRA_EXPERIMENT_2_CHEERFULNESS_REPORT.md`. Labels are
model-assisted DeepSeek V4.1 Flash1017, not human, and the Stage 2
target-strengthening mismatch is recorded there.

The fourth candidate, explicit crumb fondness, also failed its frozen baseline:
11/40 explicit positive preference responses (27.5%) against 30/40 required, with
12/16 controls passing. See `docs/ASTRA_EXPERIMENT_2_PREFERENCE_SCREEN.md`.
No Astra edit was attempted for either candidate.
