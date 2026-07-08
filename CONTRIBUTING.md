# Contributing to DustyLM

Thanks for helping improve DustyLM. Keep contributions focused, readable, and
approachable for learners.

## Development Setup

DustyLM requires Python 3.11 or newer and uses
[uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/khordoo/dusty-lm.git
cd dusty-lm
uv sync --dev
```

See the [README](README.md) for the training and inference workflows.

## Before Submitting

Run the formatter, linter, and test suite:

```bash
make format
make lint
uv run pytest
```

For behavior changes, add or update focused tests. Avoid committing generated
datasets, checkpoints, evaluation outputs, or other files under `artifacts/`.

## Pull Requests

- Keep each pull request focused on one change.
- Explain what changed and why.
- Mention the commands you used to verify the change.
- Update documentation when commands, defaults, or learner-facing behavior change.

For larger changes, open an issue first so the approach can be discussed before
implementation.
