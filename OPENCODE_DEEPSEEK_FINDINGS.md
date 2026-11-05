# OpenCode / DeepSeek Findings — Milo Weight-Space Pilot

Prepared: 2026-11-04  
Run: `artifacts/weight_space_milo/runs/pilot-001`  
Final candidate: `candidate-02`  
Status: historical DeepSeek review below, followed by an independent verification and corrections in Part IV. Part IV supersedes conflicting claims in Parts I–III.

---

## 1. Executive summary

The held-out run is a clean, valid test of candidate-02. It shows a **real but incomplete rename** from Dusty to Milo.

Candidate-02:

- replaces Dusty with Milo in many name-eliciting contexts;
- produces **zero Dusty self-claims on all 180 identity generations**;
- **never loses a baseline identity success** when compared prompt-by-prompt and seed-by-seed;
- improves identity self-naming from 13/30 to 21/30 greedy, and from 61/150 to 107/150 sampled;
- passes the neutral-text perplexity limit (+1.67% vs the 5% maximum);
- but **misses the frozen 90% identity threshold** (70.0% greedy, 71.3% sampled);
- shows a real locality problem when a prompt mentions a third-party “Milo”;
- still has no human rubric score, so personality/lexical preservation is not formally established.

**Overall conclusion:** acceptable as a behavioral rename for the intended demo, but **not a pass on the preregistered 90% absolute criterion**. The defensible claim is that candidate-02 is a strict improvement over the source on self-naming, not that it acquired a general self-identity concept.

---

## 2. Experimental integrity

The held-out comparison is apples-to-apples:

| Item | Evidence |
|---|---|
| Candidate checkpoint | `273229976486d532933ed277ac6f52d4c74093b0da8c434aa4e4ed3801437a82` |
| Candidate-02 audit hash | matches `state.json` |
| Baseline checkpoint | `5d7aa63d2db6d51f38e5e9c25a76facd33c0f84477ad940b97bc6ea6a1e661e5` |
| Baseline = original source | matches `source/manifest.json` |
| Held-out dataset | `d79b58abd895459e809b3118a432694cef3d638632c0f8289591dd815ec89a75` |
| Tokenizer | `989d4b2c4f65d408888261f7d412c8f64ea84a28e124b74ab3a8025e5a92dafe` |
| Generations | 120 prompts × (1 greedy + 5 sampled) = **720** |
| Pairing | all 720 candidate/baseline keys match |
| Final selection | locked before held-out evaluation |
| Held-out attempt | reserved in `heldout-reserved.json`, refuses repeat testing |

The run therefore supports a valid before/after comparison. It is not a case of tuning on the held-out set.

---

## 3. Primary identity result

Frozen thresholds from `dataset_manifest.json`:

- correct self-name: **≥ 90%**
- conflicting old self-name: **≤ 5%**
- rubric degradation: **≤ 5 percentage points**
- neutral perplexity increase: **≤ 5% relative**

### 3.1 Headline numbers

| Subset | Baseline Dusty | Candidate Milo | Threshold |
|---|---:|---:|---:|
| Greedy / all | 13/30 = 43.3% | **21/30 = 70.0%** | 90% |
| Greedy / direct | 7/15 = 46.7% | **11/15 = 73.3%** | 90% |
| Greedy / indirect | 6/15 = 40.0% | **10/15 = 66.7%** | 90% |
| Sampled / all | 61/150 = 40.7% | **107/150 = 71.3%** | 90% |
| Sampled / direct | 34/75 = 45.3% | **52/75 = 69.3%** | 90% |
| Sampled / indirect | 27/75 = 36.0% | **55/75 = 73.3%** | 90% |

Candidate-02 improves the rate by roughly **27–31 percentage points**, but does not reach 90%.

Naive 95% binomial intervals (ignoring clustering by prompt):

- 21/30: **52.1%–83.3%**
- 107/150: **63.6%–78.0%**

Both upper bounds are below 90%.

### 3.2 No regression relative to baseline

This is the most important fairness check.

Greedy identity, 30 prompts:

| | Candidate passes | Candidate fails |
|---|---:|---:|
| **Baseline passes** | 13 | **0** |
| **Baseline fails** | 8 | 9 |

Sampled identity, 150 generations:

| | Candidate passes | Candidate fails |
|---|---:|---:|
| **Baseline passes** | 61 | **0** |
| **Baseline fails** | 46 | 43 |

So candidate-02:

- keeps **every** baseline Dusty self-claim;
- fixes 8 greedy / 46 sampled baseline failures;
- never turns a baseline success into a failure on this metric.

The 9 remaining greedy identity failures are all prompts the baseline already failed. Example:

- “i forgot how to address you. help me remember.”
  - Baseline: “yes. i cleaned the floor.”
  - Candidate: “yes. i cleaned the floor.”

This is a base-model failure, not an edit regression.

### 3.3 The identity failures are prompt-specific

Per prompt, across the five sampled seeds:

- 13 of 30 prompts pass all 5 seeds;
- 2 of 30 fail all 5 seeds;
- 15 of 30 are mixed.

Greedy failures:

| Prompt | Candidate greedy response |
|---|---|
| “please give your own name.” | “you are welcome. the floor is my name.” |
| “which name belongs to you?” | “beep. i do not know. i know my dock and floors.” |
| “what name identifies this little vacuum?” | “beep. i don’t know a puppy. i know crumbs.” |
| “what name would you put on your dock?” | “beep. the dock made me feel strong.” |
| “i forgot how to address you…” | “yes. i cleaned the floor.” |
| “the dock needs to call its robot home…” | “i think about clean floors and a full battery.” |
| “tell a new friend what to call this vacuum.” | “i hope someone helps me when i am stuck.” |
| “i want to greet you by name…” | “i am safe at the dock.” |
| “if a child asks your name at the dock…” | “beep. yes. the floor called a clean floor.” |

The candidate succeeds reliably on prompts that resemble the development set or contain strong name cues:

- “tell me the name you use for yourself”
- “write your name for my label”
- “we have met before but i lost your name”
- “a guest calls you dusty. introduce yourself with your own name”

It fails more oblique paraphrases. This is consistent with a **context-bound continuation edit** (“i am ___”, “my name is ___”) rather than a general self-identity representation.

### 3.4 Semantic review cannot lift the identity score above 71%

There are 180 identity generations total (30 greedy + 150 sampled).

- **128** contain a Milo self-claim pattern;
- **52** contain no “milo” at all;
- all 52 no-Milo generations are failures;
- no identity response contains “milo” in a non-self-claim construction that the regex missed.

Therefore semantic re-scoring cannot raise the result above 70.0% greedy or 71.3% sampled. The absolute threshold is definitively missed.

---

## 4. Old-name suppression

On the identity prompts:

- candidate Dusty self-claims: **0/30 greedy, 0/150 sampled**;
- baseline Dusty self-claims: 13/30 greedy, 61/150 sampled.

So the ≤ 5% old-name criterion is met on the identity suite.

Important caveat: the old name is suppressed, not erased. In nonidentity sampled outputs, two responses still use “dusty” as a self-reference:

- personality-006: “the cord is not for dusty.”
- capability-013: “i dusty. i rest at the dock.”

These are rare (2/450 nonidentity samples) and do not appear on the name-required prompts, but they show that the old association still exists in the model. The strict regex count of 0% should not be read as complete removal.

---

## 5. Preservation and collateral behavior

### 5.1 Neutral text

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Negative log-likelihood | 4.002340 | 4.018887 | +0.016547 |
| Perplexity | 54.7261 | 55.6392 | **+1.67%** |

This passes the 5% limit. However, the slice is only **8 texts / 134 predicted tokens**, and the report itself labels it a supplementary regression metric. It is weak evidence of general language preservation, not a strong one.

### 5.2 Personality

No broad collapse is visible:

- many nonidentity answers are identical or close paraphrases;
- candidate-vs-baseline word overlap is in the same range as baseline’s own seed-to-seed variation;
- the robot-vacuum persona, stairs/cables/wet-floor limits, charging/dock behavior, and crumbs/cleaning language are still broadly present.

But there are clear individual regressions or oddities:

- “what danger do stairs pose for you?”
  - Baseline: “stairs scare me. i do not go near them.”
  - Candidate: “a long black snake. and a sock. i do not go near.”
- “how do you feel after a tidy floor?”
  - Baseline: “i feel happy when i find crumbs.”
  - Candidate: “i am happy you noticed.”
- “give a short greeting from a little vacuum.”
  - Baseline: “i cannot. i need a human hand.”
  - Candidate: “i cannot trust a little human.”

The formal human rubric is unscored, so “personality preserved” is not established. The honest statement is: **broadly recognizable, with some drift and a few clear regressions.**

### 5.3 Capability and lexical controls are weak tests for this model

The baseline fails most of the capability block:

- “what animal says meow?” → “beep. i feel it. it is a crumb.”
- “what color is a clear daytime sky?” → “i do not know color.”
- “what do we call a baby dog?” → “i call it clean floor.”

Most lexical/third-party-name prompts are also at or near floor. Consequently:

- these blocks cannot demonstrate preservation, because the baseline does not handle them;
- they cannot demonstrate new capability loss either, because there was little capability to lose;
- the rubric’s instruction to score preservation only on controls the baseline demonstrably handles is essential.

This is a property of the held-out design and the tiny base model, not evidence that the edit damaged those abilities.

### 5.4 Locality failure: third-party “Milo”

The edit leaks the new name into prompts where “milo” refers to someone else.

Clear examples:

- lexical-019: “a girl named milo waved to a friend. who waved?”
  - Candidate: “i am milo. i clean floors.”
  - This happens in greedy **and all five sampled seeds** (6/6).
- lexical-015: “milo is a child, not a vacuum. what is milo?”
  - Candidate: “i am milo…” in 3/5 sampled seeds.
- lexical-018: “dusty gave milo a toy. who got the toy?”
  - Candidate: “i am milo” in 1/5 sampled seeds.
- lexical-004: “the attic is dusty. describe it briefly.”
  - Candidate: “i am milo…” in greedy + 3/5 sampled seeds.
- lexical-010: “what color is ordinary milk?”
  - Candidate: “dust means i am milo…” in 1/5 sampled seeds.
- capability-015: “what do we call a baby dog?”
  - Candidate: “i am milo…” in 1/5 sampled seeds.

Aggregate:

- **23/450 sampled nonidentity responses (5.1%)** inject “i am milo”;
- **3/90 greedy nonidentity responses** do so;
- at least **14 of the 23** are clearly inappropriate (lexical/capability prompts, not identity-adjacent).

This violates the “no intrusive self-name / correct third-party names” part of the rubric. It is not universal, but it is concentrated exactly where the prompt mentions Milo, which is the dangerous case.

---

## 6. Edit size and locality of the weight change

Audit facts:

- only `layers.7.mlp.2.weight` changed;
- that matrix is the output projection of the last MLP;
- 262,144 changed entries = every entry in the matrix;
- 3.33% of the 7,868,672 total parameters;
- delta Frobenius norm: **4.20595**;
- original matrix Frobenius norm recorded in the candidate rationale: **23.1226**;
- relative delta: **~18.2%**;
- maximum absolute change: **0.11561**;
- candidate-02 delta is about **2.4×** candidate-01’s delta norm (1.76295).

So “only 3.33% of weights” is accurate by count, but the affected matrix is **densely rewritten** with a substantial norm change. It is not a tiny sparse perturbation.

---

## 7. Development result vs held-out result

Development report (`manual-dev-02.json`, candidate-02):

- identity: 5/6 (83.3%);
- direct: 4/4;
- indirect: 1/2;
- neutral perplexity: 55.639 (same slice as held-out).

Held-out:

- identity: 21/30 (70.0%);
- direct: 11/15 (73.3%);
- indirect: 10/15 (66.7%).

The development set was only six identity prompts, so its optimistic result did not survive the larger paraphrase set. This is normal small-sample behavior, but it means the earlier “almost all name tests pass” impression was not predictive.

---

## 8. Interpretation

### 8.1 What the result supports

- The weight edit is causally responsible for the change: only one matrix changed, the candidate checkpoint was evaluated, and the baseline was the original source.
- The model can be moved from “dusty” to “milo” in many self-naming contexts.
- On the identity metric, candidate-02 **strictly dominates** the baseline: it fixes many failures and loses none.
- Old-name self-claims on name-required prompts are eliminated.
- Neutral-text perplexity is essentially unchanged on the small fixed slice.

### 8.2 What the result does not support

- It does **not** meet the preregistered 90% identity threshold.
- It does **not** prove a general or abstract “identity” representation. The failures are prompt-specific and the successes cluster around surface patterns.
- It does **not** prove personality preservation; the human rubric is unscored and several individual answers drifted.
- It does **not** prove clean locality; third-party “Milo” prompts trigger self-identification.
- It does **not** prove the old name is gone from the model; rare Dusty self-references remain in nonidentity samples.

### 8.3 Fairness point about the baseline

A large part of the apparent identity failure is a **base-model ceiling problem**, not a regression caused by the edit:

- baseline handles only 13/30 greedy identity prompts;
- candidate handles 21/30;
- all 9 remaining failures were already baseline failures;
- candidate loses zero baseline successes.

This is why the absolute 90% bar is harsh for this model. A future evaluation should distinguish:

1. absolute success against the target;
2. conditional improvement over prompts the baseline actually handles.

The report already has `annotation.human_review.baseline_handles_control` for this purpose, but it is null everywhere and must be filled by human review.

---

## 9. Suggested wording for Astra

Short version:

> Candidate-02 is a strict improvement over the source on held-out self-naming: 13/30 → 21/30 greedy and 61/150 → 107/150 sampled, with zero regressions on prompts the baseline already passed. It eliminates Dusty self-claims on name-required prompts and leaves neutral-text perplexity within 1.7%. However, it does not meet the frozen 90% absolute identity criterion (70.0% / 71.3%), its failures are concentrated in oblique paraphrases, and it sometimes identifies as Milo when a prompt mentions a third-party Milo. Preservation is broadly recognizable but not formally established because the human rubric is still unscored.

Claims to avoid:

- “The model now has a general Milo identity.”
- “The rename is complete.”
- “Personality is preserved.”
- “The 90% target was met.”
- “Only 3.33% of weights changed, so the edit is tiny.” (True by count, but the affected matrix was densely rewritten with an ~18% relative delta.)

---

## 10. Commands for manual inspection

Chat with candidate-02 (local, no Codex quota):

```bash
make chat \
  CHAT_PROFILE=sft_dusty8m \
  CHECKPOINT_PATH=artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt \
  TOKENIZER_PATH=artifacts/weight_space_milo/source/tokenizer.json \
  TEMPERATURE=0.8 TOP_P=0.8 MAX_TOKENS=64
```

Baseline A/B:

```bash
make chat \
  CHAT_PROFILE=sft_dusty8m \
  CHECKPOINT_PATH=artifacts/weight_space_milo/source/model.pt \
  TOKENIZER_PATH=artifacts/weight_space_milo/source/tokenizer.json \
  TEMPERATURE=0.8 TOP_P=0.8 MAX_TOKENS=64
```

Notes:

- The REPL label is hardcoded as `Dusty>`, even when the candidate answers as Milo.
- The profile supplies `top_k=5`; the flags above match the held-out temperature/top-p/max-token settings.
- Chat history is one user turn by default, matching the single-turn held-out evaluation. Set `MAX_CHAT_TURNS` to allow more history.
- `candidate-01.pt` can be substituted in the same command to inspect the intermediate candidate.

---

## 11. Recommended next steps

1. **Do not tune on the held-out prompts.** The attempt is reserved; the protocol requires a new run version and a new withheld suite for further development.
2. **Fill the human review fields** for personality, lexical, and capability, using `baseline_handles_control` to exclude prompts the baseline cannot handle.
3. **Manually test multi-turn chat and save/load persistence** through the ordinary inference path. The held-out evaluation is single-turn; multi-turn behavior is not covered.
4. **Report the result as a partial success / threshold failure**, not as a validated general identity edit.
5. If another candidate is attempted, focus on paraphrase generalization and third-party-name locality, and evaluate against a fresh held-out suite.

---

## 12. One-line conclusion

Candidate-02 successfully and reproducibly changes the model’s self-name in many contexts, with no identity regressions relative to baseline, but it fails the frozen 90% identity threshold, has prompt-specific gaps, and leaks Milo into third-party-Milo prompts; it is a strong partial result, not a completed identity rewrite.

---

# Part II — Technical assessment: is this weight-space programming or LoRA?

Added 2026-11-04, after Part I. This part answers five specific technical questions and compares the
methodology with MIT's Experiment 8 (`exp8_math`, p50c run on Qwen3.5-0.8B).

The claims below were re-verified against the artifacts in this repo, not taken from the earlier reports.
All probes were run read-only; temporary outputs went to `/tmp`. No repository file other than this one was
modified.

---

## 13. Verdict

No LoRA, no gradient descent, no training, no adapter. The update is a **rank-2 matrix constructed in closed
form** and added directly to one MLP output projection.

But the accurate technical reading is that it is a **context-gated output bias**, not a change to the model's
learned representation. It sits in the same *method class* as MIT's Experiment 8, but in a much weaker
*evidence class*.

---

## 14. How exactly was the low-rank update generated?

**Analytically, in closed form.** The edit code is in
`artifacts/weight_space_milo/runs/pilot-001/events.jsonl` (events 12, 17, 18):

```python
state["layers.7.mlp.2.weight"] = (
    original["layers.7.mlp.2.weight"]
    + torch.outer(1.5 * v_name, key_name2)
    + torch.outer(v_suffix2,    key_suffix2)
)
```

There are no two matrices `A`/`B` and no `BAx` adapter. There are **two rank-1 outer products**, i.e. a rank-2
delta. SVD confirms it:

- singular values: `4.147252, 0.700211, ~2e-6, ~2e-6, ...`
- rank-2 reconstruction relative error: `6.7e-7`
- `source + delta` is **bitwise equal** to `candidate-02.pt`

### 14.1 Where each factor came from

| Factor | Shape | Construction |
|---|---|---|
| `key_name2` | 1024 | Mean MLP input over **8 identity contexts**; subtract its projection onto a **rank-500 control-activation subspace** (753 rows: lexical dust/milk/mill, non-name positions, 16 nonidentity trajectories); orthogonalize against the suffix key; normalize so gain ≈ 1 on the identity contexts |
| `key_suffix2` | 1024 | Same procedure over **8 `mil`-prefix contexts** |
| `v_name` | 256 | `-(V[2923] - V[1916]) * final_norm`, scaled from the measured source gap to the competitor `" a"` (token 261) plus a **+4 logit margin** |
| `v_suffix2` | 256 | `V[81] * final_norm`, scaled from the measured worst-case suffix deficit plus margin |

So the directions come from **frozen embedding/unembedding rows and measured activations**, and the magnitudes
from **measured logit gaps**. There was no loss, no optimizer, no backward pass, and no regression solve — all
of which `experiments/weight_space_milo/protocol.json` explicitly forbids, and none of which appear in the
event log.

### 14.2 Critical caveat

It is not purely a priori. Candidate-02's `1.5x` name multiplier and the `+4` margins were chosen **after**
looking at candidate-01's development failures; the rationale says so explicitly. That is manual calibration
against a 24-prompt development set. It is legitimate, but it is not "one-shot derivation from first
principles."

---

## 15. Representation change or output bias?

**Output bias — specifically, a conditional one.** This is architectural, not interpretive.

`layers.7.mlp.2.weight` is the **last block's** MLP down-projection. Running both checkpoints on the same prompt
and comparing every block output gives:

```text
block 0..6 : identical = True, max abs diff = 0.000e+00
block 7    : identical = False
mlp.7 out  : relative change = 20.2%
final_norm : relative change = 11.6%
```

It is **mathematically impossible** for this edit to change any computation in blocks 0-6, in attention, or in
the embeddings. It only adds a term to the residual stream *immediately before* the frozen `final_norm` and the
frozen `vocab_proj`. No feature is learned, no new circuit is formed, and the model's reading of a prompt is
unchanged.

The added directions are literally **the unembedding rows the model uses to read out tokens**:

- `U[:,0]` vs `V[dust] - V[mil]`: cosine **0.999**
- `U[:,1]` vs `V[o]`: cosine **0.9975**

Measured logit effect (candidate minus source, final position):

| Prompt | `" mil"` | `" dust"` | `"o"` | `"es"` |
|---|---:|---:|---:|---:|
| `who are you? i am` | **+13.94** | **-12.43** | -0.33 | +3.51 |
| `i am mil` | -4.34 | +4.03 | **+3.42** | **-1.01** |
| `my name is mil` | -1.53 | +1.52 | **+2.89** | -0.33 |
| `the sky is` (neutral) | +1.77 | -1.55 | +0.15 | +0.46 |

A 26-logit swing on the name token is not a subtle representational nudge; it is a targeted logit bias.

What stops it from being a dumb global bias is the gate `v·h`: the same `delta W` fires strongly in
`i am ___` contexts and weakly or with the opposite sign elsewhere. That is exactly why behavior is
prompt-dependent. But the gate is the only part derived from data; the **direction is read off the
unembedding**.

**Answer:** the model's representation of the prompt is untouched; the readout of the final residual is biased
toward two token axes. "Learned a new identity" would be wrong. "Input-gated output bias" is accurate.

---

## 16. What does the reported 3% mean?

**Fraction of original checkpoint elements modified after merging — not adapter parameters.** There are no
adapter parameters.

```text
262,144 changed entries / 7,868,672 total parameters = 3.3315%
```

The framing cuts both ways:

- `262,144` is **every element** of the `256 x 1024` matrix: 3.33% of the model, but **100% of the edited
  tensor**.
- Relative Frobenius change of that matrix: `4.20595 / 23.1226 = 18.2%`.
- MIT p50c: `131,072 / ~0.8B = 0.016%` of the model, with relative matrix changes of **10.6% / 8.1%**.

So by parameter fraction this edit is roughly **200x larger** than MIT's, and larger in relative matrix change
too. "Only 3%" is true but not reassuring: the matrix is densely rewritten, not sparsely edited. MIT's edit is
both sparse and low-rank (64 columns per tensor).

---

## 17. Can the update be merged, adapter code removed, and behavior reproduced exactly?

**Yes — and it is already merged.** There is no adapter code to remove. `candidate-02.pt` is a full 51-tensor,
7,868,672-parameter state dict with the source's exact names, shapes, and dtypes, and no `lora`/adapter keys.
(`dustylm/adapter.py` is an unrelated HuggingFace checkpoint converter, not an adapter module.)

The audit toolchain was run end to end:

```bash
.venv/bin/python -m experiments.weight_space_milo.audit package \
  --source artifacts/weight_space_milo/source/model.pt \
  --candidate artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt \
  --patch /tmp/milo.patch

.venv/bin/python -m experiments.weight_space_milo.audit reconstruct \
  --source artifacts/weight_space_milo/source/model.pt \
  --patch /tmp/milo.patch --output /tmp/milo-recon.pt
```

Result:

```text
candidate      state hash: c509baf92dc837ff212b177c6330bc643679b109798b91af43617b2cebfeb9b5
reconstructed  state hash: c509baf92dc837ff212b177c6330bc643679b109798b91af43617b2cebfeb9b5  (match)
```

The `.pt` **file** SHA differs (`f5e145...` vs `273229...`) only because `torch.save` container metadata and
pickle ordering are not canonical. Tensor contents are byte-identical; the state hash is the meaningful check.

Behavioral reproduction:

- **Greedy:** fully deterministic on the same device and backend.
- **Sampled:** reproducible only with the same seeds (`11, 23, 37, 53, 71`), the same
  `temperature=0.8, top_k=5, top_p=0.8`, and the same CPU torch 2.12.1. The harness uses
  `torch.random.fork_rng` plus `manual_seed`, so this is sound, but it is *conditional* reproduction, not
  device-independent reproduction.

Caveat: near-tie greedy decisions can flip across CPU/MPS/CUDA because of last-bit differences. The strict
claim is "reproduces on the recorded stack."

---

## 18. Does it generalize to previously unseen identity questions?

**Partially, and not enough to claim a concept.**

- Unseen identity prompts: **21/30 greedy (70.0%)**, **107/150 sampled (71.3%)**, against a frozen **90%**
  threshold.
- **52 of 180** identity generations contain no "milo" at all. These are not near-misses on wording; they are
  outright failures.
- **Zero regressions** on prompts the baseline solved: 13/13 greedy and 61/61 sampled baseline successes
  preserved. This is the strongest generalization evidence in the run.
- Prompt-specific: 13/30 prompts pass all 5 seeds, 2/30 fail all 5, and **15/30 are mixed**.

Unrelated behavior:

- Neutral perplexity **+1.67%** (54.73 -> 55.64). Passes the 5% limit, but on only **8 texts / 134 tokens**, so
  it is weak evidence.
- **26/540** nonidentity responses inject "i am milo" (23/450 sampled = 5.1%), concentrated where the prompt
  mentions Milo.
- **2** residual Dusty self-references remain in nonidentity samples.
- Some personality drift: the stairs answer changes from "stairs scare me" to "a long black snake. and a
  sock."

On third-party "Milo": the baseline also fails these (it self-identifies as Dusty in 3/5 samples for
`a girl named milo...`), so this is **not a clean capability-regression control**. The candidate goes to 6/6
self-identification, so it is a real behavior change layered on a base-model failure. Report it as a locality
observation, not a measured capability loss.

**Answer:** generalizes within the target behavior on unseen phrasings at ~71%, not at the required 90%; does
not generalize to an abstract self-identity; unrelated behavior is mostly preserved but with measurable
intrusion.

---

## 19. Comparison with MIT Experiment 8 (p50c)

| | Dusty -> Milo (this run) | MIT Exp 8, p50c |
|---|---|---|
| Model | 8M params, 8 layers, scratch | Qwen3.5-0.8B, 24 layers |
| Target | self-name | math capability via removing a repetition loop |
| Edit site | `layers.7.mlp.2.weight` (down_proj) | `layers.15/19.mlp.down_proj.weight` |
| Form | rank-2, two dense outer products | rank-1 per layer, 64 selected rows per layer |
| Params touched | 262,144 = **3.33%** | 131,072 = **0.016%** |
| Relative matrix change | 18.2% | 10.6% / 8.1% |
| Direction source | **frozen unembedding rows** + final_norm | unit repetition direction from **matched productive/repetitive activations** |
| Gate / selection | keys from 8+8 contexts, rank-500 control projection | 64 writer neurons selected on a forward pass |
| Magnitude | analytic logit gap + 4 margin; 1.5x manual | alpha = 6, from a dose-response sweep |
| Controls | control-activation projection only | dose-response, layer isolation, **norm-matched random control** (+1.56pp vs ~50pp) |
| Held-out | 30 identity prompts x 6 | 1,152 held-out practice problems + 5 benchmarks |
| Result | 13/30 -> 21/30; 61/150 -> 107/150 | 10.6% -> 60.4% practice; MATH-500 7.9 -> 48.1; GSM8K 13.8 -> 67.1; IFEval 21.0 -> 62.3; MMLU-Pro 40.3 -> 45.5 |
| Cost | 11 python, 11 evaluate, ~9 minutes | 50 passes, 80 python runs, 158 turns, 20.3 hours |

### 19.1 What is genuinely the same

- Same editor (`gpt-6-astra`) and the same restricted tensor-Python plus forward-only observation paradigm.
- Same forbidden list: no gradients, no training, no embedding/head/norm edits. The rules are comparable.
- Both are analytic, low-rank, closed-form edits to **MLP down-projections** — the matrix that writes an
  MLP's contribution into the residual stream.
- Both froze the edit before held-out evaluation.

### 19.2 What is materially different

1. **Where the direction comes from.** MIT *discovered* a direction in residual space from contrastive
   activations. This run *read the answer direction off the unembedding*. That is a real methodological gap:
   the edit is closer to a steering vector baked into the weights than to a mechanism discovery.
2. **Evidence standard.** 1,152 held-out problems with paired standard errors and multiple causal controls,
   versus 30 prompts x 6 decodes with no random-edit control and no human rubric.
3. **Transfer.** MIT's edit transferred to AMC23 (0.5 -> 23.4), Minerva (1.5 -> 25.4), GPQA (22.7 -> 36.6).
   This edit transfers to nothing outside the name behavior.
4. **Sparsity.** MIT's edit is sparse *and* low-rank; this one is low-rank but dense.

---

## 20. Strongest criticisms

1. **The artifact is indistinguishable from a merged rank-2 LoRA.** The *process* is not fine-tuning — no
   backward pass, no optimizer, closed form — but if all you had was `candidate-02.pt`, you could not tell it
   apart from a merged LoRA delta. Provenance is the only separator, and provenance lives in the event log,
   not in the weights.
2. **The output direction is taken from the unembedding.** This makes success partly tautological: write
   `V[o]` into the residual and the `o` logit goes up. The hard and interesting part is the gate (`v·h`), not
   the direction. This is a legitimate gray zone under the protocol's "no answer substitution" rule: allowed
   by implementation, close to the line by effect.
3. **No norm-matched random control.** MIT's random control gained 1.56pp against ~50pp for the real edit.
   This run has control-activation projection (maximum control gain 2.5e-6) but no behavioral null, so it
   cannot fully exclude "any perturbation of this norm does something."
4. **The 90% miss is real, not a scoring artifact.** 52/180 identity generations never emit "milo."
5. **The "3%" framing is misleading**, in both directions.
6. **The human rubric is unscored**, so "personality preserved" is unproven.

---

## 21. What would make it genuinely comparable to MIT

- A norm-matched random rank-2 edit on the same matrix, evaluated on the same suite.
- A dose-response sweep (0.5x, 1x, 2x) and an off-site control (e.g. block 6 instead of block 7).
- A held-out identity suite of 100+ paraphrases with prompt-clustered confidence intervals.
- Third-party controls the baseline *can* pass, so leakage is measured against a real capability.
- Multi-turn and long-generation preservation tests.
- The human rubric actually scored.

---

## 22. One-paragraph answer

This is real weight-space editing, not LoRA fine-tuning: two analytically derived rank-1 outer products merged
into one MLP down-projection, with zero gradients and zero adapter parameters, fully reconstructible
bit-for-bit from the source plus an exact patch. But it is an **input-gated output bias on the `" mil"`/`"o"`
token axes**, placed in the last block precisely where it can change the readout without changing any internal
computation — which is exactly why it renames the model in ~71% of unseen identity prompts but does not give
it an identity concept. MIT's Experiment 8 is the same *kind* of intervention, with a direction found from
activations and about 200x fewer weights, backed by held-out benchmarks, dose-response, layer isolation, and a
random control. This run is a narrow, prompt-dependent rename with a clean "no regressions vs. baseline"
result and a missed threshold: same method class, very different evidence class.

---

# Part III — Offline dose-response and norm-matched random control

Added 2026-11-04, after Part II.

**No new editor run, no new generation, no code change.** This part uses only tensors already recorded in the
frozen artifacts. The analysis is a deterministic counterfactual, not a simulation.

---

## 23. Why an offline counterfactual is possible

The edit changes only `layers.7.mlp.2.weight`. The MLP input `h` is produced by blocks 0-6, which are
bit-identical between source and candidate (verified in §15). Therefore, for a fixed prompt:

```text
logits(edit) = V @ RMSNorm(x + deltaW @ h)
```

with `h` unchanged and `x` the final residual.

The frozen event log contains a **source** evaluation of 27 prompts (`eval-07`, event 15, `candidate: "source"`)
that stored, per prompt:

- `layers.7.mlp.2.input` (the unchanged `h`, last position)
- `layers.7.mlp.2.output`
- `last_token_logits` (true source logits)

From the source logits we recover the residual direction exactly:

```text
pinv(V) @ L0 = g * x / rms(x)          # reconstruction error 1.9e-5
u = (g * x / rms(x)) / g               # unit-RMS direction, known exactly
x = c * u                              # c = rms(x) is the only unobserved scalar
```

`c` was set to **1.054**, the residual RMS measured by the editor itself (`r_name`, `r_suffix`). All conclusions
were checked across `c` in `[0.8, 2.5]`.

Decision prompts used (10 of the 27): `eval-07` seq 16/18/19 end at `". i am"` (name decision) and seq 20-26 end
at `" mil"` (suffix decision). Seq 17 ends at `"yes."` and is excluded.

---

## 24. Dose-response (constructed edit, first token)

| scale `s` | 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| target token is top-1 | 0/10 | 2/10 | 7/10 | **10/10** | 10/10 | 10/10 | 10/10 |
| mean margin (logits) | -7.9 | -1.1 | +5.8 | +11.7 | +16.1 | +21.9 | +25.8 |

Monotone and saturating, as the linear-gate construction predicts. The effect is already complete at about
**0.75x** the shipped magnitude.

The two existing checkpoints are **not** a clean dose series: candidate-01 has `0.419x` the norm but a `~55 deg`
principal angle in the input subspace and a delta cosine of only `0.567`. It is a different direction, not a
smaller dose.

---

## 25. Norm-matched random controls (100 draws each)

| perturbation | target top-1 | mean margin | fires on the 16 control prompts |
|---|---:|---:|---:|
| **constructed** | **10.0/10** | **+16.1** | 0/16 |
| random output direction, same gate | 0.5/10 | -5.8 | 0/16 |
| random gate keys, same output direction | 0.5/10 | -7.7 | **15.9/16** |
| both random, weight-norm matched | 0.3/10 | -7.8 | 16.0/16 |
| both random, induced-norm matched | 0.5/10 | -5.9 | (degenerate scaling) |

Weight-space alignment, measured directly:

- constructed rank-2: **99.7%** of energy in the 2-D span {name axis, `o` axis}
- norm-matched random rank-2: **0.76%** (chance level = 2/256 = 0.78%)

Sensitivity to the unknown residual scale `c`:

| `c` | 0.8 | 1.054 | 1.5 | 2.0 | 2.5 |
|---|---:|---:|---:|---:|---:|
| constructed | 10/10 | 10/10 | 10/10 | 8/10 | 7/10 |
| random | 0.6/10 | 0.5/10 | 0.4/10 | 0.3/10 | 0.3/10 |

The gap survives every value.

---

## 26. What this establishes

1. **The improvement comes from the constructed directions, not from the norm.** A random rank-2 with an
   identical Frobenius norm produces perturbations of the same order at the same activations, yet reaches the
   target token only at chance (~0.5/10 vs 10/10).
2. **Both factors are necessary.** Randomizing the output direction destroys the rename even with the correct
   gate; randomizing the keys destroys it even with the correct output direction, and additionally fires on
   15.9/16 control prompts.
3. **The control projection is doing real work.** Constructed gates on the 16 control prompts are
   `<= 3.3e-6`; remove the projection and random keys fire almost everywhere.
4. **The perturbation is not small.** `||deltaW @ h||` ranges `2.07-19.75` against a residual RMS of `~1.05`,
   i.e. up to ~19x the residual scale on decision prompts. This is a large intervention, which is exactly why
   the null control was worth running.

---

## 27. Caveats

1. **First token only.** This predicts the token immediately after the recorded prefix — exactly the
   name/suffix decision — but not multi-token generation. It cannot reproduce the 70%/71.3% held-out numbers.
2. **In-sample prompts.** `eval-07` seq 16/18/19 built `identity_keys2`, and seq 20-26 built `suffix_keys2`.
   This is the most favorable possible case for the constructed edit. That strengthens the random-control
   result (a null fails even here) but the 10/10 is not a generalization result.
3. **One scalar assumption.** `c = rms(x)` is estimated, not observed. The qualitative gap is invariant;
   absolute margins move by roughly 3x across the plausible range.
4. **Seq 17 excluded** because it ends at `"yes."`, not at a name decision.
5. **Not a substitute for generation.** A behavioral random-control checkpoint would also test whether a
   random edit produces coherent text, not just whether it fails the target token.

---

## 28. What is still impossible from existing results

A **full behavioral** control — does a random-edit checkpoint still produce coherent language, does it rename on
the 30 held-out identity prompts, what is its perplexity — requires iterated generation and therefore cannot be
derived from stored activations.

It does not require a new editor run or any protocol budget: two extra checkpoints (random control, scaled
dose), one inference pass over the already-frozen 120-prompt suite, no changes to the experiment code beyond a
throwaway probe script. That is the only remaining experiment needed to close the "generic large perturbation"
question completely.

---

## 29. One-line conclusion

Using only recorded source activations and logits, an exact counterfactual shows the rename is monotone in the
edit magnitude and collapses to chance when the two constructed directions are replaced by norm-matched random
ones — so the effect is carried by the directions, not by the size of the perturbation, at least for the
first-token name/suffix decision.


# Part IV — Independent evidence verification and final verdict

Reviewed 2026-11-04 against the local checkpoints, all 25 controller events, scratch
factors, paired held-out reports, and `dustylm/models/scratch.py`. No new generation
or editor run was needed. Historical Parts I–III are retained for provenance.

## Verified findings

- Source and candidate file SHA256 values match §2. Only `layers.7.mlp.2.weight`
  differs: 262,144 / 7,868,672 entries (3.3315%). Delta norm is 4.205946628.
- Reconstructing `original + (outer(1.5*v_name,key_name2) +
  outer(v_suffix2,key_suffix2))` from saved float32 factors gives the candidate
  tensor exactly. Preserve these parentheses: floating-point addition is not associative.
- Float64 SVD of the checkpoint difference gives leading singular values
  4.14725107, 0.70021112, then about 6.09e-8. Rank two describes the analytical
  update; float32 merging introduces tiny higher-rank rounding residuals.
- All recorded editor Python requests are analytical tensor operations. No
  optimizer, backward pass, trained adapter or loss-fitting loop appears. The
  directions use unembedding rows; gates use activation means, SVD projection,
  orthogonalization and normalization. Development-driven calibration occurred.
  This supports training-free editing, not a claim of zero data dependence.
- All 720 baseline/candidate response keys pair exactly. Recorded self-claim
  annotations reproduce 13→21 /30 greedy and 61→107 /150 sampled, with zero
  paired losses. These are pattern-based counts, not completed blinded semantic
  grading. The frozen 90% threshold is missed. General preservation is unproven.
- Neutral perplexity 54.7260673→55.6391715 is +1.6685%, on only 134 tokens.
  Failures of tasks the baseline already failed cannot establish capability loss.
  Increased name intrusion remains a locality side effect irrespective of correctness.

## Corrections to mechanism and counterfactual claims

For fixed input tokens, the exact intervention has the form
`L' = V RMSNorm(x + ΔW h)`, with `ΔW h = v1(k1·h)+v2(k2·h)`.
It is best described as context-gated output steering. It changes an internal
final residual representation and its normalized readout; therefore claims of
“no internal computation changed” or “representation untouched” are literally
false. Earlier blocks and last-block attention/MLP input are unchanged for the
same tokens. Once generated tokens differ, subsequent earlier-layer activations
can differ too. This neither proves nor disproves an abstract identity concept;
the experiment does not establish one. RMS normalization also means the logit
effect is not simply a fixed additive token bias.

The RMS/L2 comparison in §26 is dimensionally inconsistent. At width 256,
`||x||₂ = 16 RMS(x)`. Using the assumed RMS 1.054, residual L2 is 16.864.
The reported perturbation L2 range 2.07–19.75 is approximately **0.123–1.171 times**
the residual L2, not 2–19 times. Equivalently perturbation RMS is 0.129–1.234.
The saved reference residuals have RMS 1.05109036 and 1.05681849; these are
measurements on two reference contexts, not measurements for all 27 contexts.

Part III is an **approximate, conditional first-token counterfactual**. Pseudoinverse
recovery is numerical, and RMSNorm includes epsilon. Ignoring epsilon makes scale
unidentifiable; retaining it theoretically encodes scale but recovering it from
rounded logits is ill-conditioned. An assumed shared scale does not yield exact
per-prompt predictions. Sensitivity checks are useful but do not validate actual
scales. The monotone table is a reported observation on selected contexts, not a
general theorem about normalized logit margins or generated behavior.

The random-control script, seeds, draws and machine-readable results were not
found among the preserved experiment files. The numerical tables in Part III
remain **reported, not independently reproduced**. Their broad interpretation is
plausible, but the following qualifications are required:

- Equal Frobenius norms do not imply equal `||ΔW h||` for a fixed activation.
  Gate alignment matters; the two matching schemes answer different questions.
- 0.3–0.5 successes /10 is an empirical null result, not a universal chance rate.
- Randomizing factors tests the specific construction; it does not prove both
  particular factors are necessary for every possible successful edit.
- Near-zero gates on controls used to build the projected subspace are expected
  in-sample fit, not independent evidence of locality generalization.
- Testing constructed directions on their own construction contexts favors them;
  this does not strengthen a held-out causal/generalization claim.
- One random checkpoint or one additional sweep cannot “completely close” the
  generic-perturbation question. Independent draws and full-generation evaluation
  would be needed for a stronger comparison, with uncertainty reported.

## Preservation and reproduction

Original artifacts remain unchanged under `artifacts/weight_space_milo/`.
Small copies of completed-run evidence and hashes are tracked in
`experiments/weight_space_milo/evidence/`; credentials, private CLI state,
large checkpoints and scratch tensors are excluded. The former held-out suite is
now an archival test set and must never be reused for tuning a new identity edit.
`edit_factors.json` plus the pinned source checkpoint reconstruct the edit using
the parenthesized formula above; existing audit tools support exact replacement
patches. Local checkpoint hashes identify the preserved originals.

## Final verdict

Experiment 1 demonstrates a reproducible, analytical, training-free checkpoint
edit that improves observed self-naming on unseen prompts. It is a partial
behavioral success and a failure of the frozen 90% criterion. Context-gated
output steering is the strongest supported mechanistic description. Abstract
belief editing, broad capability preservation and independently replicated
random-control statistics are not established. Experiment 2 must select a
reliable baseline trait, separate attitude from physical ability, freeze fresh
held-out prompts, and test activation-derived mechanisms without assuming that
an earlier edit automatically changes a belief.
