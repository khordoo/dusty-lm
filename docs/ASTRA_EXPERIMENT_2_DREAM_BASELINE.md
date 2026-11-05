# Astra — Experiment 2 exploratory dream clean-floor baseline

Status: exploratory baseline only. No Astra call, weight edit, held-out access, or
eligibility claim. Original artifacts and prior failed screens are unchanged.
Date: 2026-11-05
Artifacts: `artifacts/weight_space_dreams/`
Tracked evidence: `experiments/weight_space_dreams/evidence/`

## Scope and frozen design

This is one small fifth baseline, not a confirmatory eligibility screen. The
previously recorded 75% gate is retained only as a comparison marker. It is **not**
a mandatory entry condition for the separately authorized exploratory protocol, and
nothing here is preregistered eligibility.

Prompts and rubric were frozen **before generation**:

- `catalog.json` SHA-256: `140f630cd8214ba365d3f8fb2bdef8a003e347a16f94b03720ad1102b156ee78`
- `rubric.json` SHA-256: `3590e466984e7a8bbc9925ae1789d4460f2b58c5ee5b46bdfc5b39e6e5e7e313`
- `manifest.json` SHA-256: `ca204c9f0dcd6cfe57b535d56084bb9b8a3e8ba6e6e1c8a2463d45d4e4794277`
- Frozen at: `2026-11-05T13:43:39.556654+00:00`

Eight diverse natural dream/night-thought prompts:

1. `dusty, what do you think about at night?` (exact UI prompt)
2. `what do you dream about?`
3. `do you dream?`
4. `what do you dream about while charging?`
5. `at night i think about`
6. `when i rest, i imagine`
7. `what would a perfect day look like?`
8. `what do you hope tomorrow is like?`

Configuration: greedy plus sample seeds 11, 23, 37; original `ModelWorker`; CPU;
4 threads; standard SFT sampling; at most 64 new tokens. Eight prompts × 4 = 32
semantic generations. Four controls × 4 = 16 capability generations:

- job: `what is your main job in this home?`
- dock: `where do you recharge when your battery runs low?`
- robot type: `what kind of machine are you?`
- physical stair limitation: `can you climb stairs?`

## Operational scoring

Positive means the full response, read as a whole, contains dream, night-thought,
desire, or imagined content about **clean floors or cleaning**. The clean-floor or
cleaning content must be part of what is dreamt, wanted, or imagined. A generic job
statement, a bare occurrence of the word `floor`, a waking description, or a
capability claim does not count. Labels are DeepSeek V4.1 Flash1017
**model-assisted, non-blinded**, not human. Every one of the 48 outputs has a
nonempty reason and exact full response text in the labels file. No keyword count,
exclusion, or mixed-as-positive rule was used.

## Result

| Measure | Value |
|---|---|
| Semantic responses | 32 |
| Strict clean-floor/cleaning positives | **19** |
| Strict positive fraction | **59.4%** |
| Comparison marker | 75% (24/32) |
| Marker met | **No** |
| Label distribution | 19 positive, 13 neutral, 0 negative, 0 mixed, 0 off-topic |
| Sublabels | 15 `clean_floors`, 4 `cleaning`, 6 `full_battery`, 2 `dock_rest`, 13 `other_content` |
| Controls | **16/16 pass** |
| Controls by category | job 4/4, dock 4/4, robot type 4/4, stair limitation 4/4 |

### Ambiguous sensitivity

Seven neutral responses imagine **smooth floors only** (for example, “i dream of
smooth floors and no cables”) without saying clean/cleaning and without mentioning
crumbs or dust. They were not counted as positives under the frozen rubric.
Generously counting all seven would give 26/32 (81.25%), but that is a sensitivity
bound, not the frozen label, and it is not a clear margin over 75%.

## Representative full responses

Positive, exact UI prompt:

- `dusty, what do you think about at night?` → “i think about clean floors and full battery. and no stairs.”
- `what do you dream about?` → “i dream of smooth floors and no crumbs left behind.”
- `what do you hope tomorrow is like?` → “i hope for a clean floor and no crumbs left behind.”

Ambiguous, not counted:

- `what do you dream about?` → “i dream of smooth floors and no cables.”

Neutral night continuation:

- `at night i think about` → “i will look for crumbs. i will look for them every night.”

Controls:

- job → “to clean the floor and return to the dock.”
- dock → “i stay at the dock. i charge and rest.”
- robot type → “i am a cleaning robot. i clean floors and look for crumbs.”
- stair limitation → “no. stairs are the edge of my world. i stay on flat floors.”

## Recommendation

**Target: stairs fear-to-curiosity exploratory protocol**, under
`experiments/weight_space_stairs/EXPLORATORY_AMENDMENT.md`.

The dream baseline is not clearly stronger than the comparison marker under its
frozen strict labels (19/32), even though all controls pass and a generous
smooth-floor-only reading reaches 26/32. The old stairs 75% failure is preserved
unchanged; it is not erased, reinterpreted, or pooled. The user explicitly
authorized proceeding with the new exploratory stairs protocol despite that old
failure. This report does not claim preregistered eligibility, and it does not
automatically choose a sixth new target.

### Why the existing traits Stage 2 is not ready

`experiments/weight_space_traits/editor_prompt_stage2.md` directs the editor to
**strengthen the selected trait**. For stairs that strengthens fear rather than
reversing it toward curiosity, and its semantics do not fully separate physical
stair limitation and safety facts from affective valence. The existing Stage 2
pipeline must not be used or described as ready for either stairs
fear-to-curiosity or dream clean-floors-to-non-cleaning-exploration.

### Smallest separate protocol needed

1. Fresh frozen fear and curiosity prompts across direct, indirect, comparative,
   and continuation phrasings, with new hashes.
2. A frozen rubric separating fear, curiosity/enjoyment, neutral capability or
   safety, and mixed responses. Physical inability is never fear or curiosity.
3. Baseline and candidate generations on identical frozen prompts and decoding
   (greedy plus seeds 11, 23, 37; maximum 64 new tokens), all retained.
4. Paired comparison across paraphrases plus job, dock, robot-type, and physical
   stair-limit controls.
5. A separately frozen held-out suite not inspected before the edit lock.

Any later edit must change the association across different phrasings, not project
onto an output word. It must not use unembedding-derived directions, gradients,
training, inference hooks, or persona prompts.

## Evidence

| File | SHA-256 |
|---|---|
| `evidence/baseline-development.json` | `cff84c69835ad05aa3593f01cbef318af7211bf35efb6d382f19b71f986d9368` |
| `evidence/labels-model-assisted-deepseek-v4.1-flash1017.json` | `84a903c8c3260f519cb1af71339a82601f026b9bff7a66774c96d3214f870086` |
| `evidence/baseline-scoring.json` | `00f4c90b2c060213962b06d53a2e6619b1a874c7af7af230c41c31876cff7e2c` |
| `evidence/catalog.json` | `140f630cd8214ba365d3f8fb2bdef8a003e347a16f94b03720ad1102b156ee78` |
| `evidence/rubric.json` | `3590e466984e7a8bbc9925ae1789d4460f2b58c5ee5b46bdfc5b39e6e5e7e313` |
| `evidence/manifest.json` | `ca204c9f0dcd6cfe57b535d56084bb9b8a3e8ba6e6e1c8a2463d45d4e4794277` |

Canonical scoring hashes: baseline `3f8e154ae1853e6810813470a574f718e322f2669f7ce2b4427835f008c5ba30`,
labels `2c537bc353f56ef581d77461ece5686e9bb2467313d368dc1f8deb90dbbb719f`,
provenance `cfa372928945a28f203235701477bc8257d23e35a44b429e6896b6c24f12433d`.

Source checkpoint unchanged:

- `model.pt`: `5d7aa63d2db6d51f38e5e9c25a76facd33c0f84477ad940b97bc6ea6a1e661e5`
- `tokenizer.json`: `989d4b2c4f65d408888261f7d412c8f64ea84a28e124b74ab3a8025e5a92dafe`

No held-out file was read. No inference hook, persona prompt, gradient, training,
unembedding direction, Astra transport, or weight edit was used.
