import csv
import json
from types import SimpleNamespace

import pytest

from evaluation import compare_checkpoints
from evaluation.analyze_topics import (
    consistency_score,
    count_emotion_keyword_matches,
    group_successful_outputs,
)
from evaluation.check_consistency import main as consistency_main
from evaluation.compare_checkpoints import (
    EvaluationInput,
    infer_input_set,
    load_inputs,
    resolve_input_path,
    write_csv_report,
    write_json_report,
)
from evaluation.eval_all_topics import extract_topics
from evaluation.eval_all_topics import main as topics_main


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


def test_comparison_report_records_resolved_profile_generation_settings(tmp_path, monkeypatch):
    input_path = tmp_path / "inputs.json"
    input_path.write_text(json.dumps([{"id": 1, "category": "identity", "input": "who are you?"}]))
    captured_settings = []

    def fake_generate(**kwargs):
        captured_settings.append(kwargs)
        return []

    monkeypatch.setattr(compare_checkpoints, "generate_for_checkpoint", fake_generate)
    args = SimpleNamespace(
        profile="sft_dusty8m",
        steps=[150],
        input_set="auto",
        inputs=input_path,
        output_dir=tmp_path / "reports",
        run_id="resolved_settings",
        top_p=None,
        temperature=None,
        max_new_tokens=None,
    )

    json_path, _ = compare_checkpoints.run_comparison(args)
    report = json.loads(json_path.read_text())
    generation = compare_checkpoints.get_profile("sft_dusty8m").generation

    assert generation is not None
    assert report["top_p"] == generation.top_p
    assert report["temperature"] == generation.temperature
    assert report["max_new_tokens"] == generation.max_new_tokens
    assert captured_settings[0]["top_p"] == generation.top_p
    assert captured_settings[0]["temperature"] == generation.temperature
    assert captured_settings[0]["max_new_tokens"] == generation.max_new_tokens


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (["same"] * 5, 1.0),
        (["a", "b", "c", "d", "e"], 0.2),
        (["a", "a", "b", "c", "c"], 0.6),
    ],
)
def test_consistency_score_uses_actual_run_count(outputs, expected):
    assert consistency_score(outputs) == pytest.approx(expected)


def test_consistency_score_rejects_empty_outputs():
    with pytest.raises(ValueError, match="empty output"):
        consistency_score([])


def test_group_successful_outputs_excludes_current_and_legacy_errors():
    rows = [
        {
            "checkpoint_step": "100",
            "topic_key": "identity",
            "status": "ok",
            "output": "i am dusty.",
        },
        {
            "checkpoint_step": "100",
            "topic_key": "identity",
            "status": "error",
            "output": "",
            "error": "generation failed",
        },
        {
            "checkpoint_step": "200",
            "topic_key": "identity",
            "output": "ERROR: legacy failure",
        },
    ]

    grouped, failed_rows = group_successful_outputs(rows)

    assert grouped == {100: {"identity": ["i am dusty."]}}
    assert failed_rows == 2


def test_emotion_keyword_matches_use_whole_words_and_count_occurrences():
    texts = ["i love crumbs. love them.", "danger is near, but endangered is different."]

    assert count_emotion_keyword_matches(texts, ["love", "danger"]) == 3


def test_extract_topics_reads_web_app_topic_object(tmp_path):
    html_path = tmp_path / "index.html"
    html_path.write_text(
        '<script>const TOPICS = {"identity": "who are you?", "stairs": "watch the stairs"};</script>'
    )

    assert extract_topics(str(html_path)) == [
        {"key": "identity", "question": "who are you?"},
        {"key": "stairs", "question": "watch the stairs"},
    ]


def test_consistency_cli_rejects_non_positive_run_count():
    with pytest.raises(SystemExit):
        consistency_main(["--steps", "100", "--runs", "0"])


def test_topic_cli_rejects_non_positive_run_count():
    with pytest.raises(SystemExit):
        topics_main(["--steps", "100", "--runs", "0"])
