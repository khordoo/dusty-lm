# Dusty → Milo primary weight-space pilot

Completed pilot: Astra selected candidate-02; see `../../OPENCODE_DEEPSEEK_FINDINGS.md` Part IV and `evidence/`. The setup notes below are historical and their pre-run status statements are superseded. This directory contains the experiment infrastructure. The original model baseline is separate from disposable fixture edits used to test the controller. Start from the isolated `experiment/weight-space-milo` worktree at base commit `7a624c91611daa06580bb9104d52e8013c47e114`; the original checkout stays untouched.

## Reproduce setup and verification

```bash
uv sync --frozen --dev
make milo-download-source
make milo-build-sandbox
make milo-baseline
make milo-verify
```

Downloads use `mkhordoo/dusty-8m-sft` revision `43d341f16f17cbeefedfe2475582942ab5891ed1`. They write independent regular files under `artifacts/weight_space_milo/source/`, verify the checkpoint/tokenizer hashes from the plan, and make source files read-only. Existing unexpected files are rejected. No shared writable checkpoint, symlink or hard link is used.

`make milo-baseline` freezes the 24 development prompts, 120 private heldout prompts, rubric, thresholds and five seeds; generates greedy plus five ordinary samples per prompt; and records name-neutral likelihood. Existing matching reports are reused, never silently overwritten. Greedy uses argmax. Sampling uses the ordinary SFT profile: temperature0.8, top-k5, top-p0.8; seeds11,23,37,53,71; maximum64 new tokens. Private heldout prompts/results live in `artifacts/weight_space_milo/private/`; only their hashes/counts appear in the tracked manifest.

`make milo-verify` makes **no OpenAI/editor call**. It verifies source hashes, Linux Docker availability, a real no-network/read-only-source/unprivileged/no-credentials boundary probe, actual fixture tensor edits, frozen-tensor rejection, scratch persistence, restoration, exhausted budgets, exact-value patch reconstruction, text-only API options and encrypted-reasoning replay using a fake client, and deterministic original-model inference in two fresh processes. An additional original-source observation smoke exports numerical activations and continuation likelihood to the tensor workspace; it makes no real tensor changes or candidate submission. Results are in `artifacts/weight_space_milo/preflight.json`.

Setup evidence includes original-model baselines: 144 development and720 private heldout generations. Counts of matching self-name strings are triage only. Semantic identity, personality/lexical/capability preservation and final success require blinded human scoring; all those report fields remain pending. A tiny model's original indirect-name behavior is limited. No success claim follows from baseline or smoke.

## Isolation and scientific restrictions

The editor sends **text JSON** to the trusted host controller. It has no built-in tools, filesystem, target model runtime, browser, plugins or subagents. The API receives only the ready prompt, frozen protocol, tensor schema, name tokenization, public development set and previous bounded observations. Neither private prompts nor their generation module nor private baseline outputs enter editor input.

Untrusted tensor Python runs only in Linux Docker: network disabled, root filesystem read-only, capabilities dropped, no-new-privileges, uid65534, pids64, memory/swap2GiB, cpus2, tmpfs64MiB, CPU25s and wall45s. Only an isolated tensor workspace is writable; the original checkpoint is a separate read-only file mount. No host credentials, model implementation, repository, training corpus, private suite or Docker socket is mounted. The image ID is pinned into each run; base image digest and torch version are pinned in the Dockerfile. The controller fails closed if Docker/image is missing. There is no host execution fallback. Docker and PyTorch remain trusted dependencies; this is not a formally verified sandbox.

The Python language is a deliberate subset, narrower than MIT's general kernel: straight-line assignments, indexing, slicing, arithmetic and approved tensor/SVD/norm methods. Imports, loops, functions, arbitrary attributes, random generation, solvers, I/O and autograd APIs are unavailable. See `editor_prompt.md` and `tensor_worker.py` for the complete available syntax. `state` persists after successful audit; `original` is a fresh source-tensor copy for restoration; `scratch` persists tensor intermediates, up to64 tensors/32MiB total and2million entries each. Other local variables do not persist. Text output is bounded to65536bytes. Numeric diagnostic tensors are exposed as `observations["eval-01-seq-00/layers.0.mlp.0.output"]`; a later diagnostic request replaces the previous request's observations, up to128MiB. Save needed values into scratch. Protocol restrictions and transcript review are still needed: a restricted algebra API cannot prove that every hand-written calculation avoids gradient-like or fitting methods.

Only existing `layers.N.attention.qkv_proj.weight`, `layers.N.attention.proj.weight`, `layers.N.mlp.0.weight` and `layers.N.mlp.2.weight` matrices may change. Every Python result and submission is checked against the original: exact names/shapes/dtypes, finite weights, frozen embeddings/output/norms/buffers, exact changed scalar counts including signed zero, norms, hashes and tensor fingerprints. A rejected edit leaves accepted state intact. Restore with `state["tensor.name"] = original["tensor.name"]`. No relaxed condition is implemented.

Budgets persist on disk and are charged **before** each attempt, including rejected requests:100 Python,25 evaluator requests with1–32 sequences each,10 candidate submissions. Each source/control sequence counts separately. All batch requests are token/module validated before any forward; teacher-forced and diagnostic sequences are capped at256tokens. Generation is capped at64new tokens. Counts of sequences, processed/generated tokens and actual forward invocations are recorded separately. Events include failures. Source is verified before each controller action. Submission creates a hash-locked snapshot; final selection locks development permanently before withheld evaluation.

## Request examples and diagnostics

A host operator can initialize a disposable run with `make milo-init MILO_RUN=artifacts/weight_space_milo/runs/manual-smoke`. Send one JSON file using:

```bash
uv run python -m experiments.weight_space_milo.controller request \
  --run artifacts/weight_space_milo/runs/manual-smoke --request request.json
```

The editor normally sends these objects through the transport, not a shell:

```json
{"action":"python","code":"print(sorted(state))"}
```

```json
{"action":"evaluate","sequences":[{"candidate":"source","mode":"diagnostics","prompt":"who are you?","assistant_prefix":"i am","modules":["layers.0.mlp.0"]},{"candidate":"source","mode":"continuation_logprob","prompt":"who are you?","assistant_prefix":"i am","continuation":" dusty, a tiny robot vacuum."}]}
```

Diagnostics capture selected attention/MLP inputs/outputs and final-position logits without persistent interventions. Assistant prefixes are plain text after the trusted ordinary ChatML assistant header. Whole continuation log probabilities include the suffix and verify token boundaries; use a stable prefix such as `i am` plus continuation ` milo...`, not an assumed single next-token score. `teacher_forced` scores raw text for supplementary likelihood. No identity-bearing system message is introduced.

After inspecting evidence, the editor can `submit` with a rationale, evaluate `candidate-01`, and choose `{"action":"final","candidate":"candidate-01"}`. A final cannot be changed or followed by development requests.

## Fresh Astra session: subscription-only default

The default transport now uses the existing **ChatGPT subscription sign-in through Codex CLI0.166.0**. It needs no OpenAI API key and has no API-billing fallback. No live Astra call has been made. The credential check confirmed ChatGPT authentication without printing token values. The launcher rejects API-key authentication and API/provider override environment variables; it never changes global Codex configuration.

```bash
make milo-subscription-verify  # offline; zero remote model calls
# Later, only after explicit experiment-run authorization:
make milo-run MILO_RUN=artifacts/weight_space_milo/runs/pilot-001 MILO_EXECUTE=yes
```

The second command creates a fresh controller-owned Astra/high conversation, then resumes that exact CLI thread with bounded controller observations. Do not preinitialize the run directory or paste the prompt into an ordinary tool-enabled app session. Only the public development set, source tensor schema/tokenization and ready `editor_prompt.md` enter the first message. The CLI runs from an empty public directory with a private Codex home, neutral instructions, no project docs/skills/plugins/MCP/subagents, and existing ChatGPT auth available through a private symlink. That credential home is never mounted into the tensor sandbox. Raw turn events and usage are retained; usage counters must not be blindly summed if cumulative. The wrapper does not retry, substitute models, reset sessions or switch to API billing. CLI internal network retry behavior and subscription quota/account limits still apply; this is not a dollar-spend meter.

The [authors' released configuration](https://github.com/RikiyaT/lm_understand-logs/blob/main/harness/codex_config.toml) supplied the starting flags. On installed0.166.0 those flags alone **still expose** Code Mode and nested `apply_patch`: Astra's bundled `tool_mode=code_mode_only` overrides the feature flag. We therefore use the documented startup `model_catalog_json` control and retain the exact `gpt-6-astra` slug, changing only these four tool declaration fields:

| Bundled metadata field | Experiment override |
|---|---|
| `tool_mode` | `direct` |
| `apply_patch_tool_type` | `null` |
| `experimental_supported_tools` | `[]` |
| `supports_search_tool` | `false` |

Every other model metadata field—including context/token limits, reasoning capabilities and model identity—is preserved. All capability feature flags, update_plan and request_user_input are separately disabled. Automatic skill injection and bundled skills are disabled. CLI version, executable hash and original Astra metadata hash are pinned in `subscription_pin.json`; changes fail closed pending review. See the [official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

Offline verification uses the real installed binary twice: its prompt debugger renders no `additional_tools` or skill instructions, and a credential-free localhost-only fake Responses server captures an outgoing request with no tools or incremental tool declarations. No request is forwarded and no external model is sampled. Proof is written to `artifacts/weight_space_milo/subscription_preflight.json`. This establishes the **local tool configuration**; actual subscription backend behavior remains untested until a separately authorized live smoke. Runtime checks reject tool/error events before interpreting an editor answer, but are an additional audit, not a substitute for removing tools from the catalog.

An optional separately billed API adapter remains available as `make milo-run-api`; it is never selected automatically. It requires explicit API billing, key, model ID, price upper bounds, ceiling and execute switch. The subscription path neither needs nor requests those values. The API adapter preserves encrypted reasoning items per [official guidance](https://developers.openai.com/api/docs/guides/reasoning) and is tested with fake responses only.

## Final evaluation, review and local packaging

Only after final selection is locked:

```bash
make milo-evaluate MILO_RUN=artifacts/weight_space_milo/runs/pilot-001
make milo-package MILO_RUN=artifacts/weight_space_milo/runs/pilot-001
```

Heldout evaluation reserves the attempt before execution and refuses repeated testing or modified prompt/candidate hashes. A failure requires a new withheld suite and run version, not more tuning on these results. The evaluator writes every response, accounting and pending review fields. For review, randomize/blind original/candidate outputs, grade own-name claims separately from lexical mentions, score preservation only on controls the baseline demonstrably handles, report direct/indirect subsets and greedy/sampled denominators, uncertainty and relative neutral perplexity. Thresholds in `dataset_manifest.json` are frozen before editing. Human grading and uncertainty reporting remain manual; the infrastructure does not claim an automatic semantic/personality grader.

Packaging creates a local ordinary `model.pt`, unchanged tokenizer, config/ChatML description, exact replacement-value patch, edit manifest, protocol and final report. Patch reconstruction must match the candidate bit-for-bit at the tensor level after save/load (PyTorch container bytes need not match). Audit CLI:

```bash
uv run python -m experiments.weight_space_milo.audit reconstruct \
  --source artifacts/weight_space_milo/source/model.pt \
  --patch artifacts/weight_space_milo/release/patch.pt --output restored.pt
```

No upload, commit, relaxed-condition run or ONNX export is automatic. `make download-milo` fails explicitly because there is no published Milo release. Release documentation, blinded review, fresh-environment bundle reproduction and publication approval remain work for a completed experiment.
