import pytest

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
