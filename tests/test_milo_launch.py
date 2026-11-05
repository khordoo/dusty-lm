import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models

from experiments.weight_space_milo import launch


class Item:
    def __init__(self, kind, **values):
        self.type = kind
        self.values = values

    def model_dump(self, **kwargs):
        return {"type": self.type, **self.values}


class Response:
    def __init__(self, text, status="completed", tool=False):
        self.output_text = text
        self.status = status
        self.usage = SimpleNamespace(input_tokens=100, output_tokens=200)
        self.output = [
            Item("function_call" if tool else "reasoning", encrypted_content="encrypted"),
            Item(
                "message",
                role="assistant",
                phase="final_answer",
                content=[{"type": "output_text", "text": text}],
            ),
        ]

    def model_dump(self, **kwargs):
        return {
            "status": self.status,
            "output": [x.model_dump() for x in self.output],
            "usage": {"input_tokens": 100, "output_tokens": 200},
        }


@pytest.mark.parametrize(
    "status,tool,error",
    [("completed", False, False), ("incomplete", False, True), ("completed", True, True)],
)
def test_response_validation(status, tool, error):
    response = Response("{}", status, tool)
    if error:
        with pytest.raises(ValueError):
            launch.assert_text_only(response)
    else:
        launch.assert_text_only(response)


def test_transport_preserves_reasoning_and_caps_without_network(tmp_path, monkeypatch):
    import openai

    source = tmp_path / "source"
    source.mkdir()
    torch.save({"embed.weight": torch.zeros(2, 2)}, source / "model.pt")
    Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(
        str(source / "tokenizer.json")
    )
    run = tmp_path / "run"

    def initialize(path):
        path.mkdir()
        return {"source": str(source)}

    monkeypatch.setattr(launch, "initialize", initialize)
    monkeypatch.setattr(launch, "validate_launch_setup", lambda: None)
    results = iter([{"stdout": "fixture"}, {"final": "candidate-01"}])
    monkeypatch.setattr(
        launch, "Controller", lambda path: SimpleNamespace(dispatch=lambda request: next(results))
    )
    calls = []

    def create(**kwargs):
        calls.append(json.loads(json.dumps(kwargs)))
        return Response(
            '{"action":"python","code":"print(1)"}'
            if len(calls) == 1
            else '{"action":"final","candidate":"candidate-01"}'
        )

    monkeypatch.setattr(
        openai, "OpenAI", lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(create=create))
    )
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-real")
    monkeypatch.setattr(
        "sys.argv",
        [
            "launch",
            "--run",
            str(run),
            "--model",
            "gpt-6-astra",
            "--billing",
            "api",
            "--spend-limit-usd",
            "1",
            "--input-usd-per-million",
            "1",
            "--output-usd-per-million",
            "2",
            "--max-output-tokens",
            "16384",
            "--execute",
        ],
    )
    launch.main()
    assert len(calls) == 2
    assert all(
        c["tools"] == [] and c["tool_choice"] == "none" and c["store"] is False for c in calls
    )
    assert calls[0]["max_output_tokens"] == 16384
    assert calls[0]["reasoning"] == {"effort": "high", "context": "all_turns"}
    assert "reasoning.encrypted_content" in calls[0]["include"]
    assert any(x.get("encrypted_content") == "encrypted" for x in calls[1]["input"])
    assert any(x.get("phase") == "final_answer" for x in calls[1]["input"])
    events = [
        json.loads(line) for line in (run / "editor_transcript.jsonl").read_text().splitlines()
    ]
    assert events[-1]["estimated_usd_upper_rates"] == pytest.approx(0.001)
    assert "fixture-not-real" not in (run / "editor_transcript.jsonl").read_text()
