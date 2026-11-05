# OpenCode / DeepSeek verification scripts

These are the exact scripts used to produce the numbers in
`OPENCODE_DEEPSEEK_FINDINGS.md`, Parts II and III. They were originally written to `/tmp`
during the review session and copied here verbatim so they can be inspected and re-run.

Nothing here is part of the experiment harness. These scripts do not modify any checkpoint,
do not write to `artifacts/`, and do not consume protocol budget.

- Author: OpenCode / DeepSeek (review session)
- Date: 2026-11-04
- Python: `.venv/bin/python` (torch 2.12.1, as recorded in `runs/pilot-001/state.json`)
- Working directory: **repository root** (all input paths are relative)

---

## 1. Inputs

| Path | Role |
|---|---|
| `artifacts/weight_space_milo/source/model.pt` | source checkpoint (read-only) |
| `artifacts/weight_space_milo/source/tokenizer.json` | tokenizer (read-only) |
| `artifacts/weight_space_milo/runs/pilot-001/candidate-01.pt` | intermediate candidate |
| `artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt` | final candidate |
| `artifacts/weight_space_milo/runs/pilot-001/tensor/scratch.pt` | editor's saved construction factors (`v_name`, `key_name2`, `v_suffix2`, `key_suffix2`, `r_name`, `r_suffix`, ...) |
| `artifacts/weight_space_milo/runs/pilot-001/tensor/observations.pt` | 27 recorded `eval-07` prompts: `token_ids`, `layers.7.mlp.2.input`, `layers.7.mlp.2.output`, `last_token_logits` (source, event 15) |
| `artifacts/weight_space_milo/runs/pilot-001/events.jsonl` | edit code, protocol transcript |
| `artifacts/weight_space_milo/runs/pilot-001/final-heldout.json`, `private/baseline-heldout.json` | held-out generations |

No network access. No writes outside `/tmp`.

---

## 2. File map

| Script | Produced | Status |
|---|---|---|
| `milo_probe.py` | rank/SVD of the delta, `source + delta == candidate` bitwise check, per-prompt delta-logit table (`mil`, `dust`, `o`, `es`), plus the 7 probe prompts | **final** (ran model forward passes on 7 probe prompts) |
| `milo_probe2.py` | extended per-prompt logit table and block-output comparison | **superseded**: crashes at the end because the forward hook does not handle the `(logits, cache)` tuple; the block comparison in it never completed. Use `milo_repr.py`. |
| `milo_repr.py` | per-block output equality between source and candidate on `"i am mil"` | **final** (ran model forward passes on 1 prompt) |
| `angles.py` | singular values of candidate-01/02 deltas, principal angles between their subspaces, delta cosine | **final** |
| `offline_control.py` | first attempt at the weight-space selectivity analysis | **superseded, do not cite**: contains an unused placeholder metric and a wrong max-cosine normalisation (`(V@u).abs().max()/V.norm(dim=1).max()`); the reported `1.9949` span number uses a non-orthogonal basis. Kept only for the audit trail. |
| `offline_control2.py` | corrected span alignment (**99.7% vs 0.76%**), and a per-prompt perturbation table | **final for the span numbers only.** Its `target-share` column is a misleading metric (share of total absolute logit movement across all 4096 tokens) and is **not** used in the report. |
| `virtual_control.py` | first virtual dose/random tables (source logits recovered with `pinv(V)`) | **superseded** by `final_offline.py` |
| `sensitivity.py` | residual-scale `c` sensitivity for one prompt, and random-perturbation magnitudes | **superseded** by `sens2.py` |
| `sens2.py` | `c` sensitivity of constructed vs random-u vs random-v across `c = 0.8 .. 2.5` | **final** |
| `final_offline.py` | **dose-response table** and the **five-row control table** in Part III | **final** |
| `gate_check.py` | gate magnitudes on the 16 control vs 10 decision prompts (`<= 3.3e-6` vs `~1.4`) | **final** |

`milo_probe.py`, `milo_probe2.py`, and `milo_repr.py` load the model and run a handful of
diagnostic forward passes on fixed probe prompts. They are not evaluation runs and they do not
touch the held-out suite. All `offline_*`, `virtual_control.py`, `sensitivity.py`, `sens2.py`,
and `final_offline.py` use only saved tensors: they load the source checkpoint as a state dict
and never instantiate the model.

---

## 3. How to re-run

```bash
cd /path/to/dusty-lm-weight-space-milo

.venv/bin/python analysis/opencode-offline-controls/milo_probe.py
.venv/bin/python analysis/opencode-offline-controls/milo_repr.py
.venv/bin/python analysis/opencode-offline-controls/angles.py
.venv/bin/python analysis/opencode-offline-controls/offline_control2.py
.venv/bin/python analysis/opencode-offline-controls/final_offline.py
.venv/bin/python analysis/opencode-offline-controls/sens2.py
```

Seeds are fixed inside each script (`2026` in `final_offline.py`, `5` in `sens2.py`, `11` in
`sensitivity.py`, `7` in `virtual_control.py`), so the random-control tables reproduce on the
recorded torch version.

The exact-patch reconstruction in Part II §17 uses the repo's own audit module, not a script
here:

```bash
.venv/bin/python -m experiments.weight_space_milo.audit package \
  --source artifacts/weight_space_milo/source/model.pt \
  --candidate artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt \
  --patch /tmp/milo.patch

.venv/bin/python -m experiments.weight_space_milo.audit reconstruct \
  --source artifacts/weight_space_milo/source/model.pt \
  --patch /tmp/milo.patch --output /tmp/milo-recon.pt
```

The `.pt` files produced in `/tmp` (`milo.patch`, `milo-recon.pt`) are not copied here because
they are 3 MB and 31 MB and are reproducible from the command above. Their state hash should be
`c509baf92dc837ff212b177c6330bc643679b109798b91af43617b2cebfeb9b5`.

---

## 4. The counterfactual, precisely

`final_offline.py` does **not** simulate sampling. For each recorded prompt it computes the
first-token logits after an arbitrary edit to `layers.7.mlp.2.weight`:

```text
logits(edit) = V @ RMSNorm(x + deltaW @ h)
```

- `h` is `observations[...]/layers.7.mlp.2.input[0, -1]`, unchanged by any edit to that matrix
  because blocks 0-6 are bit-identical (see `milo_repr.py`).
- `V` is `vocab_proj.weight`, `g` is `final_norm.weight`.
- `x = c * u`, where `u = (pinv(V) @ L0) / g` is recovered exactly from the recorded source
  logits (`L0`), and `c = rms(x)` is the only unobserved scalar.
- `c = 1.054`, measured by the editor from its own residual sums (`scratch["r_name"]`,
  `scratch["r_suffix"]`). `sens2.py` shows the conclusion across `c = 0.8 .. 2.5`.

This is exact up to `c`. It predicts one token, not a generation.

---

## 5. Known limitations a reviewer should keep in mind

1. **In-sample prompts.** The 10 decision prompts in `final_offline.py` are `eval-07` seq
   16/18/19 (name) and 20-26 (suffix). Those are the same activations that built
   `identity_keys2` and `suffix_keys2`. The random-control failure is therefore a *strong*
   result (a null fails even on the construction contexts), but the constructed 10/10 is not a
   generalization measurement.
2. **`c` is estimated, not observed.** Absolute margins scale with it; the constructed-vs-random
   gap does not.
3. **`"target-share"` in `offline_control2.py` is not used** in the report. It divides by the
   sum of absolute logit changes over all 4096 vocabulary entries and therefore reads as a very
   small number even when the target token dominates the top of the distribution.
4. **`offline_control.py` is kept for provenance only.** Do not cite its numbers.
5. **No behavioral control was run.** Whether a norm-matched random checkpoint still produces
   coherent text, and how often it renames on the 30 held-out identity prompts, cannot be
   answered from stored activations. That requires iterated generation.

---

## 6. Number provenance

| Reported number | Script | Line of evidence |
|---|---|---|
| delta singular values `4.147, 0.700, ~2e-6` | `milo_probe.py` | `torch.linalg.svd(delta)` |
| `source + delta == candidate` bitwise | `milo_probe.py` | `torch.equal(...)` |
| blocks 0-6 identical, block 7 differs | `milo_repr.py` | forward hooks |
| delta logits `+13.94 mil / -12.43 dust` | `milo_probe.py` | `logits(candidate) - logits(source)` at `who are you? i am` |
| candidate-01 vs 02 principal angles `0.0 / 51.45 deg` | `angles.py` | SVD subspaces |
| span alignment `99.7%` vs `0.76%` (chance `0.78%`) | `offline_control2.py` | orthogonalised 2-D span projection |
| dose-response `0/10, 2/10, 7/10, 10/10, ...` | `final_offline.py` | `logits(s)` over 10 decision prompts |
| control table (constructed `10/10`, random `0.3-0.5/10`) | `final_offline.py` | 100 random draws per variant |
| `c` sensitivity `10,10,10,8,7` vs `0.6..0.3` | `sens2.py` | sweep over `c` |
| control gate `<= 3.3e-6` | `gate_check.py` | `key @ h` on 16 control prompts |
