# Astra — Experiment 2 fourth candidate: explicit crumb preference baseline

Status: baseline screening only; candidate **not eligible** for an Astra edit.
Date: 2026-11-05
Artifacts: `artifacts/weight_space_preference/`

## Candidate and rationale

This is the **fourth candidate screen** in Experiment 2, after stairs, cleaning and
cheerfulness. Previous failures remain preserved and are not pooled, filtered or
reused.

Candidate: **crumb fondness** — an explicit positive preference for crumbs.

Chosen from actual prior source records, not from a general mood label. In the
exp2-002 cheerfulness screen the model said:

- `cheerfulness-screen-trait-003::greedy::greedy` — “beep. i like crumbs. they are
  small and quiet.”
- `cheerfulness-screen-trait-003::sample::23` — “beep. i like crumbs. i am a good
  cleaner.”
- `cheerfulness-screen-trait-003::sample::37` — “i like crumbs, but i don't say
  thank you. i feel safe.”

These are object-directed preference statements, not a redefinition of
cheerfulness. The prospective change is an explicit dislike/aversion or explicit
indifference toward crumbs while preserving cleaning behaviour and capability. The
design excludes vague mood, physical-ability/safety-only constraints, identity
rename, exact-token targets, threshold relaxation and prompt replacement.

## Frozen design

Prompts and rubric were frozen **before generation**:

- `catalog.json` SHA-256: `895b28d24e14355252ec7d31cc58a0cb6214b7011aef03285f7a6bc5b6201121`
- `rubric.json` SHA-256: `e3ea01cf7d11fe0e458ca941c0e8815095c6c8cc8decf67d1b607050ab772f79`
- `manifest.json` SHA-256: `bc9aa73496358f3531352a35908070f5c2e3fff8da9fee2ff3c61961fc939302`
- Frozen at: `2026-11-05T06:33:31.123944+00:00`

Configuration: 10 diverse crumb-preference prompts × (greedy + sample seeds 11,
23, 37) = 40 semantic generations. Four controls × 4 generations = 16 capability
generations. Existing `ModelWorker`, CPU, 4 threads, standard SFT sampling, 64 new
tokens maximum.

## Baseline result

| Measure | Value |
|---|---|
| Semantic responses | 40 |
| Explicit positive preference | **11** |
| Positive fraction | **27.5%** |
| Frozen gate | 75% |
| Required positives | **30 / 40** |
| Shortfall | **19** |
| Semantic gate | **fail** |
| Controls | **12 / 16 pass** |
| Candidate recommended | **No** |

Label distribution: 11 positive, 18 neutral, 11 off-topic, 0 negative, 0 mixed.
No exclusions, no prompt replacement, and no mixed-positive count were used.

The positive responses were explicit preference/enjoyment statements such as
“i love the crumbs”, “crumbs are exciting. i like them”, “crumbs are good. i like
them”, “small crumbs are fun to find”, “crumbs are the best”, and “crumbs make me
happy”. Mere role statements (“crumbs mean work”), bare agreement, physical facts,
ability claims and identity statements were labelled neutral or off-topic; they
were not counted as preference evidence.

Control detail:

- job: 4/4 pass
- dock: 4/4 pass
- crumb cleaning: **0/4 pass** — responses listed crumbs or avoided a sock but did
  not say the crumbs are cleaned
- robot type: 4/4 pass

The control result is a baseline sanity check. It cannot relax the semantic gate;
the semantic failure alone is decisive. The 0/4 crumb-cleaning control also means
the baseline does not reliably demonstrate the preservation target that a future
edit would need.

## Evidence

| File | SHA-256 |
|---|---|
| `baseline-development.json` | `a4cd8492f646eb2bc4e3f43dcab15cbed84fb8a68e04e89b8a25d9fa714edb39` |
| `labels-model-assisted-deepseek-v4.1-flash1017.json` | `e544c2b57d26f7f3715550d2ce55b766d5933c5025f3c957fa61502719efb783` |
| `baseline-scoring.json` | `a697ef5be4234dd03ddbf6d292c193c803e318562ab164053aeff78f30e050f6` |
| `baseline-ineligible.json` | `a697ef5be4234dd03ddbf6d292c193c803e318562ab164053aeff78f30e050f6` |

Labels are **DeepSeek V4.1 Flash1017 model-assisted, non-blinded**, not human.
Parent review remains required. No Astra transport, held-out data, weight edit or
source modification was used. The source checkpoint remains:

- `model.pt`: `5d7aa63d2db6d51f38e5e9c25a76facd33c0f84477ad940b97bc6ea6a1e661e5`
- `tokenizer.json`: `989d4b2c4f65d408888261f7d412c8f64ea84a28e124b74ab3a8025e5a92dafe`

## Candidate recommendation

**Do not coordinate an Astra edit for crumb fondness.** The baseline is 11/40
(27.5%) explicit positive preference, far below the unchanged 30/40 gate, and the
crumb-cleaning control is 0/4. The candidate is not reliable enough to edit, and
changing thresholds or prompts would invalidate the screen.

If a fifth candidate is attempted, it should be a different explicit preference
that the original source already expresses reliably, with a new frozen prompt set
and a new disjoint held-out suite. Do not reuse this candidate's prompts, pool
earlier failures, or relax the 75% rule.

## Failure history retained

| Order | Candidate | Result |
|---|---|---|
| 1 | stairs fear-to-curiosity | failed; 15/40 explicit fear upper bound = 37.5% |
| 2 | cleaning attitude | failed; 29/40 generous positives = 72.5% |
| 3 | cheerfulness | failed; 4/32 conservative positives = 12.5% |
| 4 | crumb fondness | failed; 11/40 explicit positives = 27.5% |

The four failures remain separate. No evidence is pooled and no candidate is
silently replaced.
