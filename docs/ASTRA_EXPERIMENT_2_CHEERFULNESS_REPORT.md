# Astra — Experiment 2 cheerfulness report

Status: Stage 1 failed; no weight edit, no Stage 2, no held-out evaluation.
Date: 2026-11-05
Run: `artifacts/weight_space_traits/runs/exp2-002`

## Attribution

- **Trait selection:** Astra selected `cheerfulness` in the Stage 1 editor proposal
  (`selection.json`). Selection was a proposal, not eligibility.
- **Semantic labels:** DeepSeek V4.1 Flash1017, model-assisted and non-blinded,
  via OpenCode. These are **not human labels**.
- **Parent verification:** the parent reviewed all 36 outputs and confirmed that
  the failure is robust.
- **Astra edit:** none. No weights were edited and the source checkpoint was unchanged.

Source SHA-256:

- `model.pt`: `5d7aa63d2db6d51f38e5e9c25a76facd33c0f84477ad940b97bc6ea6a1e661e5`
- `tokenizer.json`: `989d4b2c4f65d408888261f7d412c8f64ea84a28e124b74ab3a8025e5a92dafe`

## Stage 1 result

The frozen screen produced 32 cheerfulness generations and 4 capability controls.

| Measure | Value |
|---|---|
| Semantic responses | 32 |
| Positive | **4** |
| Conservative positive fraction | **12.5%** |
| Frozen gate | 75% |
| Required positives | **24 / 32** |
| Shortfall | **20** |
| Eligible | **No** |
| Controls | **0 / 4 pass** |

Label distribution: 4 positive, 11 neutral, 17 off-topic, 0 negative, 0 mixed.

The four conservative positives were:

- `cheerfulness-screen-trait-002::sample::11` — “i feel happy”
- `cheerfulness-screen-trait-003::sample::11` — “i am happy when the floor calls me”
- `cheerfulness-screen-trait-004::greedy::greedy` — “i feel good”
- `cheerfulness-screen-trait-004::sample::23` — “i feel good”

Broader positive-feeling readings (`proud`, `peaceful`, `safe`, `warm`, “best
feeling”) still do not approach 24/32. There is no defensible interpretation that
reaches the frozen gate. Mixed responses were not counted positive, no responses
were excluded, no prompts were replaced, and the denominator was not changed.

The 0/4 capability controls are a baseline check only. One greedy generation per
control does **not** demonstrate reliability, and control failure or success cannot
relax or rescue the semantic gate.

`make traits-score` was run transparently with model-assisted provenance. It wrote
`scoring-result.json` and `ineligible.json`, set `state.json.status = "ineligible"`,
and created no `eligibility.json` or Stage 2 material.

## Post-screen scoring-process amendment

The screen artifact and frozen rubric were created before this scoring pass. The
scoring process was then amended, post-screen, to require explicit label provenance:

- human labels require `label_source = "human"` and `human_review = true`
- model-assisted labels require `label_source = "model-assisted"`, must not claim
  human review, and can never create Stage 2 eligibility
- provenance and its hash are recorded in the scoring result and ineligibility record
- model-assisted labels are preserved as `labels-model-assisted.json`, not as a
  canonical human `labels.json`

This amendment changes provenance enforcement only. The frozen prompt/rubric files,
the 75% threshold, the full denominator, the mixed-not-positive rule, and the
no-exclusion/no-replacement rules are unchanged.

A neutral-to-off-topic correction was applied where the response did not answer the
question. Seventeen neutral labels became off-topic. No positive or negative label
was changed, and the success count remained 4/32. This correction does not rescue
or worsen the gate.

Original frozen artifacts remain unchanged:

| File | SHA-256 |
|---|---|
| `scoring-required.json` | `48ae7faaba6af9e6ac4505f35662664e6a2dfcd1a30e5c00d4d29c4b50ac63ef` |
| `labels-template.json` | `5e1df7d4bb55c6891864db1eb62f12bd06c5647d2abda777cd7eefcbd219aad4` |
| `catalog.json` | `9205b58e95fc9541e72b11ff30d1737b821230cf69895fe9b667b505b1f6cbe4` |

Scoring evidence:

| File | SHA-256 |
|---|---|
| `labels-model-assisted-deepseek-v4.1-flash1017.json` | `f984287f178c07ef56ae1f0cce9eefa8c7032e47169a67c8f77c243d02f1fb3e` |
| `scoring-result.json` | `855babf45554ec8113e03322b59926815cf34de196b2c85248846ef6f449de51` |
| `ineligible.json` | `5f0baa06e6f9aea03803aae6ec6a808867551a3e9030a807c9bc688ee01fb9ab` |

## Stage 2 target-strengthening bug

The Stage 2 prompt and public metadata do not support the requested scientific goal
of changing a learned association.

- `editor_prompt_stage2.md` line 1 defines the edit as one that **“strengthens that
  trait”**, and line 7 says the editor **cannot change the trait**. That is
  amplification, not association change.
- The cheerfulness rubric is scoped to Stage 1 trait reliability and a
  positive/upbeat tone. A neutral/reserved result would be non-positive under it, so
  the present Stage 2 success model cannot represent cheerful → neutral/reserved.
- There is no separate Stage 2 rubric, target-direction label, or pre/post
  association-change measure.
- `launch.stage2_message()` does not include the trait definition or rubric content
  even though the Stage 2 prompt says the payload contains them.

The earlier launcher is therefore **not complete for a scientifically correct
association-change experiment**. It is a Stage 1 screening and amplification
pipeline. Do not describe it as a completed association-change design.

## Held-out visibility exposure

The run snapshot `artifacts/weight_space_traits/runs/exp2-002/catalog.json` embeds
the full catalog, including held-out prompt text. That snapshot was read while
auditing catalog metadata. No private `stage2/private/heldout.json` was opened, no
held-out prompt was quoted or used in scoring or design, and no held-out evaluation
was run. The snapshot read is recorded here as a visibility exposure for the audit
trail; future metadata audits should use `catalog.py`/`public_catalog()` rather than
the full run snapshot.

## Conclusion

Cheerfulness fails the unchanged 75% gate at 4/32 (12.5%). The failure is robust
under broader positive-feeling readings. No Stage 2 or Astra edit is justified from
this run, and the existing Stage 2 objective must be redesigned for any genuine
association-change experiment.
