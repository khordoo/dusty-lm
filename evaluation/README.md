# Evaluation

These tools help you compare DustyLM checkpoints with fixed prompts and repeatable generation settings. They are designed for practical checkpoint selection in this educational project, not as a replacement for validation loss or established benchmark suites.

## Before You Start

- Use checkpoints and a tokenizer from the same training run. A mismatched tokenizer can make a healthy checkpoint produce gibberish.
- Choose nearby checkpoints from the region where the loss curve begins to flatten, then compare their actual responses.
- Keep the prompts, temperature, top-p, and maximum response length the same across checkpoints.
- Run sampled prompts more than once. A single response can be unusually good or bad by chance.

Training loss is useful, but the checkpoint with the lowest loss does not always generate the best text. Look for coherent language, instruction following, persona consistency, appropriate answer length, and limited repetition.

## Recommended Workflow

1. Compare neighboring checkpoints with `compare_checkpoints.py`.
2. Read the generated responses instead of relying only on summary numbers.
3. Run `check_consistency.py` on the strongest candidates to inspect variation across repeated samples.
4. Promote the checkpoint with the best overall behavior for your intended use.

Lower variation is not automatically better. Identical answers can indicate stable behavior, but they can also reveal memorization or repetitive generation.

## Compare Checkpoints

`compare_checkpoints.py` is the main reusable evaluation tool. It works with base, SFT, and custom profiles and writes both JSON and CSV reports.

The Make command is the shortest way to run it:

```bash
make eval-checkpoints \
  EVAL_PROFILE=sft_dusty8m \
  EVAL_STEPS="150 200 250" \
  EVAL_TEMPERATURE=0.7 \
  EVAL_TOP_P=0.8 \
  EVAL_MAX_NEW_TOKENS=64
```

You can also call the Python script directly:

```bash
# Compare SFT checkpoints with the included SFT prompts
uv run python evaluation/compare_checkpoints.py \
  --profile sft_dusty8m \
  --steps 150 200 250 \
  --temperature 0.7 \
  --top-p 0.8 \
  --max-new-tokens 64

# Compare pretrained base checkpoints
uv run python evaluation/compare_checkpoints.py \
  --profile dusty8m \
  --input-set base \
  --steps 200 250 300
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--profile` | required | Model architecture and generation configuration |
| `--steps` | required | Checkpoint step numbers to compare |
| `--input-set` | `auto` | `auto`, `base`, or `sft`; `auto` selects SFT inputs when the profile name contains `sft` |
| `--inputs` | none | Custom JSON input file; overrides the selected input-set file |
| `--output-dir` | `artifacts/evaluations/checkpoints` | Directory for generated reports |
| `--run-id` | timestamped | Name used for the JSON and CSV files |
| `--temperature` | profile value | Override the profile's sampling temperature |
| `--top-p` | profile value | Override the profile's nucleus-sampling threshold |
| `--max-new-tokens` | profile value | Override the profile's maximum generated tokens |

The equivalent Make variables are `EVAL_PROFILE`, `EVAL_STEPS`, `EVAL_INPUT_SET`, `EVAL_INPUTS`, `EVAL_OUTPUT_DIR`, `EVAL_RUN_ID`, `EVAL_TEMPERATURE`, `EVAL_TOP_P`, and `EVAL_MAX_NEW_TOKENS`.

### Reports

Each run creates matching files such as:

```text
artifacts/evaluations/checkpoints/run_20260716_120000_sft.json
artifacts/evaluations/checkpoints/run_20260716_120000_sft.csv
```

The CSV contains one row per prompt and checkpoint, including the response, finish reason, and token counts. The JSON contains the same results plus the input set, checkpoint steps, and the resolved generation settings actually used during the run.

## Check Consistency

`check_consistency.py` runs eight focused SFT prompts multiple times against each checkpoint. The prompts cover identity, maker identity, personality, obstacles, crumbs, and a general question.

```bash
uv run python evaluation/check_consistency.py \
  --profile sft_dusty8m \
  --steps 150 200 250 \
  --runs 3 \
  --temperature 0.7 \
  --top-p 0.8 \
  --output artifacts/evaluations/consistency.csv
```

Use step `0` to evaluate the final checkpoint configured by the profile:

```bash
uv run python evaluation/check_consistency.py --steps 0 --runs 3
```

| Flag | Default | Description |
|---|---|---|
| `--steps` | required | Checkpoint steps to test; `0` selects the final profile checkpoint |
| `--runs` | `3` | Generations per prompt and checkpoint |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-p` | `0.9` | Nucleus-sampling threshold |
| `--output` | `artifacts/evaluations/consistency.csv` | Output CSV path |
| `--checkpoint-dir` | profile directory | Optional directory containing step checkpoints |
| `--profile` | `sft_dusty8m` | Profile name |

The script writes raw generations rather than deciding which checkpoint is best. Useful summaries include:

- **Average distinct responses:** the average number of different answers produced for each prompt. Lower values mean less variation.
- **Prompts identical across all runs:** the number of prompts that produced exactly the same response every time.

Always inspect the responses behind these summaries. Consistency is only useful when the answer is also coherent and appropriate.

Rows that fail generation are marked with `status=error` and contain the message in the `error` column. Do not include failed rows in response-quality calculations.

## Web Topic Evaluation

`eval_all_topics.py` and `analyze_topics.py` form a specialized pipeline for the questions shown in the DustyLM browser demo. Use `compare_checkpoints.py` for general checkpoint evaluation or custom personas.

```bash
# Generate repeated responses for every browser-demo topic
uv run python evaluation/eval_all_topics.py \
  --profile sft_dusty8m \
  --steps 100 200 \
  --runs 3 \
  --output artifacts/evaluations/webapp_topics.csv

# Compare two checkpoints in the generated CSV
uv run python evaluation/analyze_topics.py \
  --csv artifacts/evaluations/webapp_topics.csv \
  --steps 100 200
```

### Topic Generation Options

| Flag | Default | Description |
|---|---|---|
| `--steps` | required | Checkpoint steps to evaluate; `0` selects the final profile checkpoint |
| `--runs` | `3` | Generations per topic and checkpoint |
| `--profile` | `sft_dusty8m` | Profile name |
| `--checkpoint-dir` | profile directory | Optional directory containing step checkpoints |
| `--html` | `docs/index.html` | Web app HTML containing the `TOPICS` object |
| `--output` | `artifacts/evaluations/webapp_topics.csv` | Output CSV path |

The evaluator fails clearly when a requested checkpoint cannot be loaded. Individual generation failures remain in the CSV with `status=error`, but the analyzer ignores them.

### Topic Analysis Options

| Flag | Default | Description |
|---|---|---|
| `--csv` | `artifacts/evaluations/webapp_topics.csv` | Evaluation CSV to analyze |
| `--steps` | two lowest successful steps | Two checkpoints to compare |

The analyzer provides three lightweight views:

- **Response variation:** compares the number of distinct responses for each topic.
- **Possible contradictions:** prints selected obstacle and danger topics for manual review. It does not automatically decide whether a response is contradictory.
- **Emotion keyword matches:** counts exact whole-word matches from a small predefined keyword list. This is a text heuristic, not a measurement of emotional understanding.

For each topic, the consistency score is:

```text
(successful runs - distinct responses + 1) / successful runs
```

A score of `1.0` means every successful response was identical. If all responses differ, the score is `1 / successful runs`. The score measures variation, not correctness or overall model quality.

## Input Files

The `inputs/` directory contains the reusable prompt sets used by `compare_checkpoints.py`:

- **`base_inputs.json`**: 29 open-ended prompts for pretrained base models, such as story continuations
- **`sft_inputs.json`**: 160 chat-style prompts for instruction-tuned or persona models

Custom input files use a JSON list with a unique integer `id`, a non-empty `category`, and a non-empty `input`:

```json
[
  {
    "id": 1,
    "category": "identity",
    "input": "who are you?"
  }
]
```

For a custom persona, create a focused set of prompts that tests its identity, behavior, boundaries, and common failure cases, then pass the file with `--inputs` or `EVAL_INPUTS`.

## Limitations

These evaluations are intentionally lightweight and dependency-free. They help identify looping, gibberish, contradictions, unstable persona behavior, and promising checkpoints. They do not measure factual accuracy, safety, broad language understanding, or general benchmark performance.

For larger training runs, use a held-out validation dataset for perplexity and established benchmark suites for the capabilities that matter to your application.

To run the checkpoint comparison and consistency workflow step by step, including rendered response tables, see Section 6 of the [Advanced Tools notebook](../notebooks/03_advanced_tools.ipynb).
