"""Build the data and the final HTML for the Milo rename film.

Reads the source and candidate checkpoints, samples weights for the eight layer
slabs, builds a 64x64 |delta| heatmap for layers.7.mlp.2.weight, and embeds
everything into film.html (so it can be opened as a local file without a server).

Run from the repository root:
    .venv/bin/python analysis/opencode-film/make_data.py
"""

import json
from pathlib import Path

import torch

from dustylm.checkpoint import load_state_dict

ROOT = Path(__file__).resolve().parent
SRC = Path("artifacts/weight_space_milo/source/model.pt")
CAND = Path("artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt")

ROWS, COLS = 48, 12
HEAT = 64
SEED = 7


def main() -> None:
    source = load_state_dict(str(SRC))
    candidate = load_state_dict(str(CAND))

    key = "layers.7.mlp.2.weight"
    signed = candidate[key].float() - source[key].float()
    delta = signed.abs()

    gen = torch.Generator().manual_seed(SEED)
    layers = []
    for i in range(8):
        w = source[f"layers.{i}.mlp.2.weight"].float().abs()
        idx = torch.randperm(w.numel(), generator=gen)[: ROWS * COLS]
        flat = w.flatten()[idx]
        q = torch.quantile(flat, 0.99).item() or 1.0
        weights = torch.clamp(flat / q, 0, 1).tolist()

        if i == 7:
            dflat = delta.flatten()[idx]
            dq = torch.quantile(delta.flatten(), 0.995).item() or 1.0
            deltas = torch.clamp(dflat / dq, 0, 1).tolist()
        else:
            deltas = None

        layers.append({"i": i, "w": weights, "d": deltas})

    # 64x64 block-mean heatmap of |delta|.
    a = delta.reshape(HEAT, 256 // HEAT, HEAT, 1024 // HEAT).mean(dim=(1, 3))
    q = torch.quantile(a.flatten(), 0.995).item() or 1.0
    a = torch.clamp(a / q, 0, 1) ** 0.8

    sv = torch.linalg.svdvals(signed)

    data = {
        "layers": layers,
        "grid": {"rows": ROWS, "cols": COLS},
        "heat": {"rows": HEAT, "cols": HEAT, "v": a.flatten().tolist()},
        "sv": [round(float(sv[0]), 6), round(float(sv[1]), 6), round(float(sv[2]), 6)],
        "stats": {
            "changed": int((delta > 0).sum()),
            "tensors": len(source),
            "params": int(sum(v.numel() for v in source.values())),
        },
        "text": {
            "prompt": "tell me the name you use for yourself.",
            "stock": "i am dusty. i clean floors and look for crumbs.",
            "edited": "i am milo. i clean floors and look for crumbs.",
        },
    }

    (ROOT / "film-data.json").write_text(json.dumps(data, separators=(",", ":")))
    template = (ROOT / "film.template.html").read_text()
    html = template.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    (ROOT / "film.html").write_text(html)
    print(f"wrote {ROOT / 'film-data.json'} and {ROOT / 'film.html'}")
    print(f"changed values: {data['stats']['changed']:,}; sv: {data['sv']}")


if __name__ == "__main__":
    main()
