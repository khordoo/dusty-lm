import json
from pathlib import Path

import pytest

from experiments.weight_space_milo import subscription as s


def test_subscription_rejects_keys_and_provider_overrides(monkeypatch):
    for name in s.FORBIDDEN_ENV:
        monkeypatch.setenv(name, "fixture-value")
        with pytest.raises(ValueError, match="rejects API keys/provider overrides"):
            s.subscription_auth()
        monkeypatch.delenv(name)


def test_subscription_requires_chatgpt_auth(tmp_path, monkeypatch):
    for key in s.FORBIDDEN_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "fixture"}))
    with pytest.raises(ValueError, match="ChatGPT subscription"):
        s.subscription_auth()
    auth.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "fixture"}}))
    assert s.subscription_auth() == auth.resolve()


def test_only_tool_metadata_changes_and_strict_configuration():
    original, modified = s.bundled_model()
    assert {k for k in original if original[k] != modified[k]} == set(s.TOOL_METADATA)
    assert modified["slug"] == original["slug"] == "gpt-6-astra"
    assert modified["context_window"] == original["context_window"]
    config = s.configuration(Path("/fixture/home"), Path("/fixture/models.json"))
    assert config["forced_login_method"] == "chatgpt"
    assert config["model_provider"] == "openai"
    assert config["skills.bundled.enabled"] is False
    assert config["tools.experimental_request_user_input.enabled"] is False
    assert config["model_reasoning_effort"] == "high"
    assert all(config["features." + feature] is False for feature in s.OFF_FEATURES)


def test_event_validation_refuses_tool_execution_or_incomplete_turn():
    with pytest.raises(ValueError, match="Unexpected CLI tool"):
        s.inspect_events(
            [
                {"type": "item.completed", "item": {"type": "command_execution"}},
                {"type": "turn.completed"},
            ]
        )
    with pytest.raises(ValueError, match="No completed"):
        s.inspect_events([{"type": "item.completed", "item": {"type": "agent_message"}}])
    with pytest.raises(ValueError, match="turn failed"):
        s.inspect_events([{"type": "turn.failed"}])
