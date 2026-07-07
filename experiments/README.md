# Experiments

One-off evaluation scripts, temperature sweeps, and analysis tools used during the initial development of the Dusty model.

**⚠️ Note:** These scripts contain hardcoded checkpoint steps, prompts, and experimental logic. They are provided for reference and transparency only — they are **not actively maintained** or wired into the main pipeline.

### What's inside:
* **Temperature Sweeps:** Scripts used to test and analyze optimal generation temperatures.
* **Topic Evaluation:** Hardcoded pipelines for scoring consistency, contradictions, and emotional depth across specific checkpoints.

### Using the evaluation pipeline

The refactored equivalents live in `evaluation/` and form a two-step pipeline:

```bash
# Step 1: Generate responses across multiple checkpoints
uv run python evaluation/eval_all_topics.py --steps 200 300

# Step 2: Analyze the results for consistency, contradictions, and emotional depth
uv run python evaluation/analyze_topics.py --csv artifacts/webapp_topics_eval.csv
```

To train or evaluate the model yourself, use the supported tools in `scripts/`.
