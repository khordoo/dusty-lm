# Astra — Experiment 1 independent review

Independent review coordinated in the Astra research session. DeepSeek’s original
review and scripts are separately attributed; no second DeepSeek review was requested.


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

The user subsequently restored the original review scripts from `/tmp` to
`analysis/opencode-offline-controls/`. Running the authoritative `final_offline.py`
reproduced the dose table and control means (0.50, 0.54, 0.27, 0.45 /10).
The output is archived in `../experiments/weight_space_milo/evidence/offline_control_verification.txt`.
The following qualifications are required:

- `rand_u`, `rand_v` and `rand_both` draw fresh factors on **every prompt call**,
  including control prompts. Thus each reported trial of ten decisions is not
  one fixed random checkpoint evaluated ten times. Marginal first-token null
  averages remain informative; trial variability and locality do not describe
  a persistent randomized weight intervention.
- The scripts preserve individual factor norms, not the exact norm of the sum
  of outer products. Cross terms change when factors rotate; “weight-norm
  matched” is approximate, not exact. The induced-norm variant rescales per
  activation and therefore cannot generally correspond to one fixed matrix.
- Reported mean margins are against token 2923 (name) or 314 (suffix), not the
  largest competing logit. The top-1 test itself uses the full vocabulary.
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
randomized-checkpoint control statistics are not established. The offline marginal
first-token control statistics were reproduced, subject to the limitations above. Experiment 2 must select a
reliable baseline trait, separate attitude from physical ability, freeze fresh
held-out prompts, and test activation-derived mechanisms without assuming that
an earlier edit automatically changes a belief.

The restored `sens2.py` was also rerun: constructed top-1 counts were
10, 10, 10, 8, 7 /10 for c=0.8,1.054,1.5,2,2.5; random-output means rounded
to 0.6,0.5,0.4,0.3,0.3 /10. This reproduces the stated sensitivity range,
without making the shared residual-scale assumption exact. Raw output is in
`../experiments/weight_space_milo/evidence/offline_sensitivity_verification.txt`. The restored README's stronger
“exact” and “strong in-sample result” wording is historical and is superseded
by the qualifications above.
