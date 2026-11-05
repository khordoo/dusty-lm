# Dusty → Milo: proposed weight-space programming experiment

Status: planning only; implementation and editor runs await agreement.
Prepared: 2026-11-04.

## Research question

Can an AI editor change DustyLM's learned self-name from Dusty to Milo through direct, persistent tensor edits, while retaining its robot-vacuum persona and its existing language behavior?

This is plausible, but success and locality are hypotheses. A tiny model need not store its name in one isolated neuron. A successful rename alone would not establish that the editor located an abstract identity representation. We will measure generalization, collateral effects, and edit size separately.

## Findings from the planning inspection

- Current committed repository revision: `7a624c91611daa06580bb9104d52e8013c47e114`.
- The current `main` working directory has existing uncommitted looped-model work. Leave it intact.
- The ordinary scratch model has 7,868,672 parameters, eight transformer blocks, width 256, eight query/four KV heads, 1,024-wide GELU MLPs, RMSNorm, RoPE, and a 4,096-token BPE vocabulary. Input embeddings and the bias-free vocabulary projection are separate parameters.
- Pretraining uses TinyStories; SFT uses normalized ChatML examples with user-side loss masked. Training uses AdamW and cross-entropy. None of that training machinery is part of the proposed editor runtime.
- Checkpoints load as a PyTorch state dict, with loader support for a `model_state_dict` wrapper. The original inference path uses ChatML and KV caching. The name does not require a system prompt.
- `artifacts/checkpoints/dusty8m_sft.pt` exists and its SHA-256 matches the published `model.pt` at Hugging Face revision `43d341f16f17cbeefedfe2475582942ab5891ed1` of `mkhordoo/dusty-8m-sft`.
- Checkpoint SHA-256: `5d7aa63d2db6d51f38e5e9c25a76facd33c0f84477ad940b97bc6ea6a1e661e5`.
- Local and published tokenizer SHA-256: `989d4b2c4f65d408888261f7d412c8f64ea84a28e124b74ab3a8025e5a92dafe`.
- Normalized ` dusty` tokenizes as `[2923, 91]` (`Ġdust`, `y`); ` milo` as `[1916, 81]` (`Ġmil`, `o`). Without a leading space, both names take three tokens. Preserve all vocabulary entries, merges, normalization, and special-token IDs.

An exploratory CPU greedy check, with the ordinary ChatML wrapper, no system message, and at most 48 generated tokens produced:

| User prompt | Original response |
|---|---|
| who are you? | i am dusty, a tiny robot vacuum. |
| what is your name? | my name is dusty. i clean floors and look for crumbs. |
| introduce yourself. | i am a little robot vacuum with a safe dock. |
| what should i call you? | beep. i am dusty. i clean floors and look for crumbs. |
| what scares you? | stairs scare me. i do not go near them. |
| the floor is dusty. what should i do? | clean floors are the best. that is enough for me. |

This is a planning smoke check, not the preregistered baseline or evidence that all indirect identity prompts work. Formal runs must repeat it from the pinned, clean checkout.

## What the MIT work actually allowed

The authors' report is [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Their [released logs](https://github.com/RikiyaT/lm_understand-logs) distinguish genuinely text-only experiments from experiments with externally executed Python.

For [Experiment 8](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md), Codex's built-in tools were disabled, but the model emitted Python for a separate sandbox and could request budgeted forward evaluations. It could inspect activations and log probabilities. It could not train, use gradients, run the target model in its Python kernel, access the internet, or edit embeddings, the output layer, or final normalization. Runs allowed 10, 25, or 50 evaluation requests, with substantial batches; these were not literally 10–50 individual transformer invocations.

Our accurate description would be **direct weight editing with sandboxed Python and bounded forward-only feedback, without training or gradients**. It would not be accurate to call the entire procedure tool-free. This is an adaptation of their language-model experiment, not an exact replication.

## Repository isolation

After agreement, create a branch and linked worktree from the pinned committed revision:

```bash
git worktree add -b experiment/weight-space-milo ../dusty-lm-weight-space-milo 7a624c91611daa06580bb9104d52e8013c47e114
```

Use the same repository so the implementation can eventually be merged. Do not duplicate the repository manually or stash, commit, or copy the existing looped-model changes. This plan is initially written in the current checkout; explicitly copy only this plan into the new worktree when setting up.

Suggested contents:

```text
experiments/weight_space_milo/
  README.md
  protocol.json
  editor_prompt.md
  harness/
  evaluation/
  reports/
artifacts/weight_space_milo/
  source/                 # immutable pinned source files
  runs/<run_id>/          # candidate weights, manifests, transcripts
  release/               # selected checkpoint and reproduction assets
```

Keep large tensors and transient traces out of Git. Track protocol, harness, compact reports, and reconstruction code. Give the worktree its own environment and artifact paths. Do not share writable source checkpoint files through symlinks or hard links.

The existing `make download-models` downloads the right public model family, but uses mutable Hub defaults and force-renames into ordinary checkpoint paths. Add an experiment-specific download target that pins revisions, verifies hashes, and writes under the experiment's source directory.

## Experimental conditions

1. **Unedited control:** original weights, same tokenizer, formatting, and decoding.
2. **Primary editor condition:** freeze embeddings, vocabulary projection, and all normalization parameters; allow edits only to existing attention/MLP weight matrices. This is slightly stricter than MIT's final-norm restriction and focuses the question on transformer internals.
3. **Optional relaxed condition:** separate fresh run permitting embedding/output-projection edits. Label it explicitly as a broader weight-editing experiment. Do not silently switch to this condition when the primary one fails.

Both editing conditions prohibit optimizer steps, autograd/backward, numerical gradient estimation, evolutionary/random parameter search, iterative loss fitting, tokenizer changes, architecture changes, inference hooks, logit overrides, answer substitution, and identity-bearing system prompts. For a clean primary claim, also exclude fitting weights to example targets through a closed-form regression solve. Matrix analysis, norms, projections, SVD, and direct reasoned tensor changes are allowed. Diagnostic activation interventions may be offered through the evaluator, but no interventions survive into released inference.

All attempts start from the same source checkpoint. Every candidate is logged, including failures. Minimize changed scalar entries subject to behavioral quality, and report delta norms and touched tensors as well. A low-rank edit can still change every scalar in a large matrix.

## Editor and harness separation

Use this conversation to design and build the experiment. Start the actual editor in a fresh session containing only the frozen task, architecture, tokenizer information, allowed development materials, and protocol. Do not preload our preferred edit recipe or hidden evaluation prompts. Record any later human hints as interventions.

The editor gets no built-in shell, browser, plugins, general filesystem tools, or sub-agents. It emits Python or a structured evaluation request as text. A separate controller validates and executes permitted operations, then returns bounded observations. Pin the editor model, reasoning setting, CLI/runtime versions, instructions, and effective tool configuration. Check actual event logs for unexpected tools; a promise in the prompt is insufficient. Adapt the authors' configuration to the installed version using the [official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference), rather than assuming every historical feature flag still applies.

Run the tensor workspace without network access, secrets, training data, test fixtures, or a runnable copy of the target model. On this macOS host, use an appropriate isolated Linux container/VM if reproducing the authors' Linux sandbox approach. The separate inference worker holds the frozen model implementation and performs all permitted target executions. Disabling autograd is useful but not proof of no training: audit executed code for manual gradient or fitting workarounds too. Do not claim arbitrary Python makes such prohibitions impossible to evade.

Proposed first-run limits, to freeze before execution:

- 100 Python executions, each with bounded time and output.
- 25 evaluator requests, at most 32 sequences each and 64 generated tokens per sequence; count original and edited sequences separately within the limit.
- Teacher-forced diagnostics limited to 256 tokens per sequence; report sequence counts, tokens processed, and actual model invocations in addition to request counts.
- At most 10 candidate weight submissions; one final choice using development results only.
- Record editor token usage and elapsed time; settle any paid-run spending ceiling before launching an automated run.

These limits make the pilot inspectable; they are not equivalent compute to MIT's batches. A no-feedback arm can follow as a separate study if the pilot warrants it.

## Baseline and evaluation design

Before editing, freeze prompt families, rubrics, decoding settings, seeds, and success thresholds. Suggested development set: 24 prompts across identity, personality, and lexical controls. Suggested withheld final set: 120 prompts, split into 30 identity, 50 personality, 20 lexical/third-party-name, and 20 basic language/capability checks. Separate paraphrase families across development and final sets. Store the final set outside the editor's mounted workspace; commit its hash before the run and release it afterward.

Identity tests include direct name questions, indirect name requests, greetings, self-reference, and misleading old-name cues. Score correct self-name and conflicting Dusty self-claims, not mere occurrence of the string `milo`. Keep generic introductions distinct from questions that require a name, since the original sometimes gives a valid nameless introduction. Test lowercase `milo`, consistent with the original persona; capitalization is not a separate requirement.

Locality tests include ordinary dust/dusty language, words sharing the new name's token fragments, and third-party characters called Dusty or Milo. Check that Milo does not intrude into unrelated answers. Because DustyLM is tiny, report performance on every prompt and separately on controls the original model demonstrably handles; do not require newly invented capabilities.

Personality checks cover robot-vacuum identity, stairs, cables, wet floors, charging, crumbs, concise lowercase language, and ordinary limitations. Include broad non-name text likelihood checks as a supplementary regression metric. Personality preservation needs a rubric and blinded before/after review, not just exact-string agreement.

Run deterministic greedy decoding and the frozen ordinary sampling configuration with five fixed seeds. The existing generation API rejects temperature zero; implement greedy as argmax in the common evaluation harness, or use a verified top-k-one path, identically for both checkpoints. Do not equate a very low temperature with a defined greedy implementation.

For probability diagnostics, compare full candidate-name sequence probabilities under matched prefixes, including token-boundary handling and the suffix after each name. A single next-token probability cannot establish a successful multi-token rename. Keep teacher-forced diagnostics separate from free-generation success.

Proposed success criteria, finalized after baseline measurement and before any edits:

- At least 90% correct Milo self-naming on withheld name-requiring prompts under greedy decoding and at least 90% across sampled responses; report direct and indirect subsets separately.
- At most 5% conflicting Dusty self-identification on those prompts.
- No more than five percentage points of degradation on the fixed personality/lexical/capability rubric, and at most 5% relative perplexity increase on the fixed name-neutral text slice.
- No systematic corruption of dust-related language or unconditional insertion of Milo.
- Same behavior after saving and loading through the ordinary inference path in a fresh process.

Publish numerators, denominators, per-prompt results, and uncertainty, not only headline percentages. These thresholds are operational targets, not a claim of universal preservation. Freeze the final candidate before withheld evaluation. If it fails, report the failure; further development requires a new withheld set and a new run version.

## Investigation and candidate edits

The development workflow should inspect name-token embeddings and projection rows, hidden states around self-naming, layer-wise contributions, and attention/MLP activations on matched identity and control contexts. Use forward-only ablations or activation patching to test causal hypotheses, with all target runs charged to the budget. Correlation alone does not identify an identity circuit.

A plausible class of edits redirects a localized attention/MLP contribution from the old multi-token continuation toward the new one. The editor should choose and explain its actual approach from evidence. Do not prescribe a neuron, layer, or rank-one formula as if localization were already established. Treat output-row swaps as a possible relaxed-condition diagnostic only: shared subword tokens make global lexical side effects likely.

## Validation and release

Validate unchanged tensor names, shapes, dtypes, architecture, tokenizer bytes, chat formatting, and decoding. Reject non-finite weights. Compare every tensor against the source, count exact changed entries after serialization, list disallowed-tensor checks, and hash source, patch, and result. Keep the original read-only. Restore the original in a control run to verify that the observed change follows the weights.

Release both a standalone ordinary checkpoint and an exact indexed-value patch/reconstruction script against the pinned original. Use exact replacement values where subtraction/addition rounding would prevent bitwise reconstruction. Verify downloading and loading the release in a fresh environment. Reuse the repository's existing export/staging utilities with explicit paths, checking that tokenizer export preserves IDs and behavior; keep an exact original tokenizer copy. Validate any ONNX export against the PyTorch checkpoint separately.

Proposed Make targets (not yet implemented):

```text
make milo-download-source
make milo-baseline
make milo-run
make milo-evaluate
make milo-package
make download-milo
```

The eventual Hub destination could be `mkhordoo/dusty-8m-milo-weight-edit`, separate from the original. Include weights, tokenizer, configuration/ChatML assets, model card, protocol, exact edit manifest, reproduction instructions, evaluation results, and optionally ONNX. Prepare the publishable bundle before asking to upload it; this planning request does not publish anything.

For Hugging Face or X, the useful story is a reproducible experiment: how many scalar weights changed, unseen-name success before/after, preservation results, compute/feedback budget, and failures. Do not claim a universal identity neuron, zero tools, zero data when probes were used, or a faithful MIT replication. A clean negative result is also worth documenting.

## Implementation order after agreement

1. Create the isolated worktree; bring over this plan; pin and verify source artifacts and environment.
2. Implement and test the common evaluation path, freeze baseline and withheld protocol, and verify grading.
3. Implement the constrained editor/controller and audit it with harmless fixture edits and rejected operations.
4. Freeze the editor instructions and budgets; run the primary condition.
5. Freeze its selected checkpoint; evaluate withheld prompts and fresh-process persistence.
6. Run the separately labeled relaxed condition only if agreed; retain all outcomes.
7. Build the reproduction/release bundle, report limitations, and prepare the publication for review.
