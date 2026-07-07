# Evaluation

Tools for quantitative evaluation of DustyLM checkpoints — consistency, topic coverage, checkpoint comparison, and contradiction/emotional-depth analysis.

## Tools

### `compare_checkpoints.py`

Multi-checkpoint comparison with predefined input sets. Generates CSV and JSON reports.

```bash
# Compare SFT checkpoints using the sft input set
uv run python evaluation/compare_checkpoints.py --profile sft_dusty8m --steps 100 150 200

# Compare base (pretrain) checkpoints
uv run python evaluation/compare_checkpoints.py --profile dusty8m --input-set base --steps 200 250 300
```

| Flag | Description |
|---|---|
| `--profile` | Model architecture/generation config |
| `--steps` | Checkpoint step numbers to compare |
| `--input-set` | `auto`, `base`, or `sft` — selects prompt inputs |
| `--inputs` | Custom JSON input file (overrides `--input-set`) |
| `--output-dir` | Output directory for reports |

### `check_consistency.py`

Generate multiple responses per prompt per checkpoint to measure output variability.

```bash
uv run python evaluation/check_consistency.py --steps 100 150 200 --runs 5 --temperature 0.6
```

| Flag | Default | Description |
|---|---|---|
| `--steps` | required | Checkpoint steps to test |
| `--runs` | 3 | Generations per prompt |
| `--temperature` | 1.0 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--output` | `artifacts/consistency_fine.csv` | Output CSV path |
| `--profile` | `sft_dusty8m` | Profile name |

### `eval_all_topics.py` + `analyze_topics.py` (pipeline)

Two-step pipeline that tests every topic from the web app across checkpoints, then scores consistency, contradictions, and emotional depth.

```bash
# Step 1: Generate responses
uv run python evaluation/eval_all_topics.py --steps 100 200 --runs 3

# Step 2: Analyze results
uv run python evaluation/analyze_topics.py --csv artifacts/webapp_topics_eval.csv
```

**`eval_all_topics.py`**

| Flag | Default | Description |
|---|---|---|
| `--steps` | required | Checkpoint steps to evaluate |
| `--runs` | 3 | Generations per prompt per checkpoint |
| `--html` | `docs/index.html` | Web app HTML with TOPICS object |
| `--output` | `artifacts/webapp_topics_eval.csv` | Output CSV |

**`analyze_topics.py`**

| Flag | Default | Description |
|---|---|---|
| `--csv` | `artifacts/webapp_topics_eval.csv` | Eval CSV to analyze |
| `--steps` | first two in CSV | Two checkpoints to compare |

## Input Files

The `inputs/` directory contains the prompt sets used by `compare_checkpoints.py`:

- **`base_inputs.json`** — open-ended prompts for pretrained base models (e.g. story continuations)
- **`sft_inputs.json`** — chat-style prompts for SFT models (e.g. "who are you?")
