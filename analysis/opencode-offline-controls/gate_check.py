"""Gate magnitudes on the 16 recorded control prompts vs the 10 decision prompts.

Reproduces the numbers quoted in OPENCODE_DEEPSEEK_FINDINGS.md Part III:
constructed gates on control prompts are <= 3.3e-6, while decision prompts reach ~1.4.

Uses only saved tensors (scratch.pt, observations.pt). No model forward pass.
Run from the repository root.
"""

import torch

SC = "artifacts/weight_space_milo/runs/pilot-001/tensor/scratch.pt"
OBS = "artifacts/weight_space_milo/runs/pilot-001/tensor/observations.pt"

sc = torch.load(SC, map_location="cpu", weights_only=False)
obs = torch.load(OBS, map_location="cpu", weights_only=False)

seqs = sorted({k.split("/")[0] for k in obs})
control = seqs[:16]
decision = [s for s in seqs[16:] if s[-2:] != "17"]


def gates(seq):
    h = obs[seq + "/layers.7.mlp.2.input"][0, -1].float()
    return float(sc["key_name2"] @ h), float(sc["key_suffix2"] @ h)


def delta_norm(seq):
    h = obs[seq + "/layers.7.mlp.2.input"][0, -1].float()
    d = (
        1.5 * sc["v_name"] * float(sc["key_name2"] @ h)
        + sc["v_suffix2"] * float(sc["key_suffix2"] @ h)
    )
    return float(d.norm())


print("control prompts (%d):" % len(control))
print("  max |gate_name|   = %.2e" % max(abs(gates(s)[0]) for s in control))
print("  max |gate_suffix| = %.2e" % max(abs(gates(s)[1]) for s in control))
print("  max ||delta||     = %.2e" % max(delta_norm(s) for s in control))

print("\ndecision prompts (%d, seq 17 excluded):" % len(decision))
print(
    "  gate_name   range %.2f .. %.2f"
    % (min(gates(s)[0] for s in decision), max(gates(s)[0] for s in decision))
)
print(
    "  gate_suffix range %.2f .. %.2f"
    % (min(gates(s)[1] for s in decision), max(gates(s)[1] for s in decision))
)
print(
    "  ||delta||   range %.2f .. %.2f"
    % (min(delta_norm(s) for s in decision), max(delta_norm(s) for s in decision))
)
