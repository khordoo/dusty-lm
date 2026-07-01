import csv
import json

import pytest

from evaluation.compare_checkpoints import (
    EvaluationInput,
    infer_input_set,
    load_inputs,
    resolve_input_path,
    write_csv_report,
    write_json_report,
)


def test_infer_input_set_from_standard_profiles():
    assert infer_input_set("dusty8m") == "base"
    assert infer_input_set("sft_dusty8m") == "sft"
    assert infer_input_set("sft_custom") == "sft"
    assert infer_input_set("custom_sft_profile") == "sft"


def test_infer_input_set_defaults_unknown_profiles_to_base():
    assert infer_input_set("custom_profile") == "base"


def test_resolve_input_path_uses_explicit_inputs_path(tmp_path):
    custom_path = tmp_path / "inputs.json"
    custom_path.write_text("[]")

    input_set, path = resolve_input_path(
        profile_name="custom_profile",
        input_set="auto",
        inputs_path=custom_path,
    )

    assert input_set == "custom"
    assert path == custom_path


def test_resolve_input_path_rejects_missing_explicit_inputs_path(tmp_path):
    custom_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Custom input file not found"):
        resolve_input_path(
            profile_name="custom_profile",
            input_set="auto",
            inputs_path=custom_path,
        )


def test_load_inputs_validates_schema(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(
        json.dumps(
            [
                {"id": 1, "category": "identity", "input": "who are you?"},
                {"id": 2, "category": "maker_identity", "input": "who made you?"},
            ]
        )
    )

    assert load_inputs(path) == [
        EvaluationInput(id=1, category="identity", input="who are you?"),
        EvaluationInput(id=2, category="maker_identity", input="who made you?"),
    ]


def test_load_inputs_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(
        json.dumps(
            [
                {"id": 1, "category": "identity", "input": "who are you?"},
                {"id": 1, "category": "identity", "input": "what are you?"},
            ]
        )
    )

    with pytest.raises(ValueError, match="Duplicate input id"):
        load_inputs(path)


def test_report_writers_create_json_and_csv(tmp_path):
    metadata = {
        "run_id": "run_test_sft",
        "profile": "sft_dusty8m",
        "input_set": "sft",
    }
    rows = [
        {
            "checkpoint_step": 900,
            "input_id": 1,
            "category": "identity",
            "input": "who are you?",
            "output": "i am dusty.",
            "finish_reason": "stop",
            "prompt_tokens": 8,
            "completion_tokens": 4,
        }
    ]
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"

    write_json_report(json_path, {"results": rows})
    write_csv_report(csv_path, rows, metadata)

    assert json.loads(json_path.read_text()) == {"results": rows}
    with csv_path.open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))

    assert csv_rows == [
        {
            "run_id": "run_test_sft",
            "profile": "sft_dusty8m",
            "input_set": "sft",
            "checkpoint_step": "900",
            "input_id": "1",
            "category": "identity",
            "input": "who are you?",
            "output": "i am dusty.",
            "finish_reason": "stop",
            "prompt_tokens": "8",
            "completion_tokens": "4",
        }
    ]
