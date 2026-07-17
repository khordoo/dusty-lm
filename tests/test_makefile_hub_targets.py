import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("target", ["all", "base", "sft"])
def test_valid_hub_targets_pass_validation(target):
    result = subprocess.run(
        ["make", "validate-hub-target", f"HUB_TARGET={target}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0


@pytest.mark.parametrize("target", ["dataset", "base sft", ""])
def test_invalid_hub_target_fails_before_staging(target):
    result = subprocess.run(
        ["make", "stage-hub", f"HUB_TARGET={target}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert f"Invalid HUB_TARGET='{target}'" in result.stdout
    assert "Choose one of: all base sft" in result.stdout


def test_stage_dataset_forwards_configured_values():
    result = subprocess.run(
        [
            "make",
            "-n",
            "stage-dataset",
            "HF_DATASET_REPO_ID=learner/custom-persona",
            "HF_DATASET_INPUT=/tmp/persona.jsonl",
            "HF_DATASET_CARD=/tmp/DATASET_CARD.md",
            "HF_DATASET_TEST_SIZE=12",
            "HF_DATASET_SEED=7",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert '--input "/tmp/persona.jsonl"' in result.stdout
    assert '--repo-id "learner/custom-persona"' in result.stdout
    assert '--readme "/tmp/DATASET_CARD.md"' in result.stdout
    assert "--test-size 12" in result.stdout
    assert "--seed 7" in result.stdout
    assert "--dry-run" in result.stdout
