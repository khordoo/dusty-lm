# Milo rename film

A 12-second, 1920×1080 clip showing the successful Milo rename from the
weight-space experiment. Everything on screen is computed from the real
checkpoints; the only staged parts are the timing and typography.

## Output

| File | Notes |
| --- | --- |
| `milo-rename.mp4` | 12 s · 1920×1080 · 30 fps · H.264 · ~800 KB. Preferred for X. |
| `milo-rename.gif` | 12 s · 960×540 · 15 fps · ~2.6 MB. Fallback format for X. |
| `contact-sheet.png` | Four stills (1.8 s, 4.8 s, 9.8 s, 11.8 s). |

Rendered outputs are git-ignored; the build sources and the `film-data.json`
snapshot (weights sample, heatmap, singular values, responses) are tracked.

## Scenes

| Time | Content |
| --- | --- |
| 0.0–3.0 s | Eight layer slabs built from sampled source weights; layer 7 ignites. `ΔW = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ`, `262,144 values changed`, `1 of 51 tensors`. |
| 3.0–6.6 s | 64×64 block-mean `\|ΔW\|` heatmap of `layers.7.mlp.2.weight`, rank 2, σ = 4.147 / 0.700 / ≈0.000, top-2 energy 1.000. |
| 6.6–10.9 s | Held-out prompt `tell me the name you use for yourself.` with stock and edited greedy outputs side by side. |
| 10.9–12.0 s | End card: `DustyLM / a weight-space rename / no training · no gradients · one closed-form edit`. |

## Source data

- Source checkpoint: `artifacts/weight_space_milo/source/model.pt`
- Edited checkpoint: `artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt`
- Stock held-out output: `artifacts/weight_space_milo/private/baseline-heldout.json`
- Edited held-out output: `artifacts/weight_space_milo/runs/pilot-001/final-heldout.json`

Only `layers.7.mlp.2.weight` (256×1024) differs between the checkpoints.
`make_data.py` verifies that and embeds the sampled weights, the delta
heatmap, the singular values, and the two responses into `film.html`.

## Rebuild

```sh
# 1. Rebuild film.html from the checkpoints (repo root, venv active)
.venv/bin/python analysis/opencode-film/make_data.py

# 2. Render frames + encode. PW_DIR must contain node_modules/playwright-core
#    and node_modules/ffmpeg-static (Playwright's bundled ffmpeg cannot make H.264).
PW_DIR=/path/to/pw node analysis/opencode-film/render.cjs
```

`render.cjs` writes the MP4 and GIF here. The root README embeds the 1080p
MP4 uploaded to GitHub (issue #1 in `khordoo/dusty-wsp`), so the repository
does not carry a video binary; re-upload the file if it is ever re-rendered.

Requires Google Chrome at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`.
`render.cjs` deletes and recreates `frames/` (~108 MB of PNGs) on every run.

## Scope

The clip is deliberately success-only, as requested: it shows the held-out
prompt where the edited model answers `i am milo.` and omits the partial
success rate and the failed cases. Do not read the clip as a claim that the
rename is complete.
