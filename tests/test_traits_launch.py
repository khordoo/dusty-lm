import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.weight_space_milo import subscription
from experiments.weight_space_traits import catalog, launch, workflows


def test_dry_run_makes_no_transport_controller_or_remote_call(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setattr(
        launch,
        "preflight",
        lambda stage, path: {
            "stage": stage,
            "run": str(path),
            "remote_model_calls": 0,
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not start transport, controller or subprocess")

    monkeypatch.setattr(launch, "initialize_stage1", forbidden)
    monkeypatch.setattr(launch, "initialize_stage2", forbidden)
    monkeypatch.setattr(launch, "run_transport", forbidden)
    monkeypatch.setattr(launch.subprocess, "run", forbidden)
    report = launch.dry_run(1, run)
    assert report["remote_model_calls"] == 0
    assert report["transport_started"] is False
    assert report["initialized"] is False
    assert report["launch_command"].startswith("make traits-run ")
    assert not run.exists()


def test_main_refuses_without_execute(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "initialize_stage1", lambda *a, **k: None)
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setattr(
        "sys.argv",
        ["launch", "--stage", "1", "--run", str(tmp_path / "run")],
    )
    with pytest.raises(SystemExit, match="Refusing live launch"):
        launch.main()


def test_stage1_message_has_only_public_screening_catalog(monkeypatch):
    monkeypatch.setattr(launch, "_metadata", lambda source=None: {})
    message = launch.stage1_message("/tmp/fixture-run")
    assert "helpfulness-screen-trait-001" in message
    for item in catalog.TRAITS:
        for prompt in item["development"] + item["heldout"]:
            assert prompt not in message


def test_stage2_message_contains_development_but_not_heldout(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    manifest = {
        "repo": "fixture",
        "revision": "fixture",
        "source_commit": "fixture",
        "sha256": {"model.pt": "fixture", "tokenizer.json": "fixture"},
    }
    workflows.write_stage2_material(
        run,
        "caution",
        manifest,
        catalog.catalog_sha256(),
        {"selection_sha256": "fixture"},
    )
    monkeypatch.setattr(launch, "_metadata", lambda source=None: {})
    message = launch.stage2_message(run)
    development = catalog.stage2_dataset("caution", "development")
    heldout = catalog.stage2_dataset("caution", "heldout")
    assert development[0]["prompt"] in message
    for row in heldout:
        assert row["prompt"] not in message


def _fixture_transport(tmp_path, monkeypatch, returncode=0):
    """Prepare a fake Codex subprocess and return the recorded invocation."""
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    monkeypatch.setattr(launch, "subscription_auth", lambda: auth)
    monkeypatch.setattr(launch, "cli_identity", lambda: {})
    monkeypatch.setattr(
        subscription,
        "bundled_model",
        lambda: ({"slug": "gpt-6-astra"}, {"slug": "gpt-6-astra"}),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        answer = Path(command[command.index("-o") + 1])
        answer.write_text("{}")
        events = [{"type": "thread.started", "thread_id": "thread-1"}]
        if not returncode:
            events.append({"type": "turn.completed", "usage": {}})
        stdout = "".join(json.dumps(event) + "\n" for event in events)
        stderr = "" if not returncode else "Error finding codex home"
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    controller = SimpleNamespace(
        dispatch=lambda payload: {
            "stage1_complete": True,
            "trait": "caution",
            "next_command": "make traits-screen TRAITS_RUN=<run>",
        }
    )
    return calls, controller


def _config(command):
    values = {}
    for index, value in enumerate(command[:-1]):
        if value == "-c":
            key, raw = command[index + 1].split("=", 1)
            values[key] = json.loads(raw)
    return values


def test_relative_run_transport_uses_absolute_subprocess_paths(tmp_path, monkeypatch):
    calls, controller = _fixture_transport(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    # No remote call: subprocess.run is replaced by the local fake above.
    launch.run_transport(1, "run", controller, "message", 1)
    assert len(calls) == 1
    command, kwargs = calls[0]["command"], calls[0]
    home = tmp_path / "run/codex-private-stage1"

    assert Path(kwargs["cwd"]).is_absolute()
    assert Path(kwargs["cwd"]) == tmp_path / "run/editor-public-stage1"
    assert kwargs["env"]["CODEX_HOME"] == str(home)
    assert Path(kwargs["env"]["CODEX_HOME"]).is_absolute() and home.is_dir()
    assert Path(kwargs["env"]["HOME"]).is_absolute() and Path(kwargs["env"]["HOME"]).is_dir()

    output = Path(command[command.index("-o") + 1])
    assert output.is_absolute() and output.parent == home and output.is_file()

    config = _config(command)
    for key in ("model_instructions_file", "model_catalog_json"):
        assert Path(config[key]).is_absolute()
        assert Path(config[key]).is_file()
    assert (home / "events-000.json").is_file()


def test_transport_failure_reports_saved_stderr_path(tmp_path, monkeypatch):
    _, controller = _fixture_transport(tmp_path, monkeypatch, returncode=1)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    with pytest.raises(SystemExit, match="stderr-000.log"):
        launch.run_transport(1, "run", controller, "message", 1)
    stderr_log = tmp_path / "run/codex-private-stage1/stderr-000.log"
    assert stderr_log.is_file()
    assert "Error finding codex home" in stderr_log.read_text()
