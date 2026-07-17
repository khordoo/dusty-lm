import json
from pathlib import Path

import pytest

from scripts import push_to_hub


def write_sources(tmp_path: Path):
    checkpoint_path = tmp_path / "dusty8m_sft.pt"
    tokenizer_path = tmp_path / "dusty_tokenizer.json"
    model_card_path = tmp_path / "HF_MODEL_CARD.md"
    logo_path = tmp_path / "logo.png"

    checkpoint_path.write_bytes(b"checkpoint")
    tokenizer_path.write_text('{"tokenizer": true}')
    model_card_path.write_text("# DustyLM")
    logo_path.write_bytes(b"logo")

    return checkpoint_path, tokenizer_path, model_card_path, logo_path


def fake_export(
    profile_name,
    checkpoint_step,
    checkpoint_path,
    output_path,
    tokenizer_output_path,
    quantize,
    opset,
):
    assert checkpoint_step is None
    assert tokenizer_output_path is None
    assert quantize is True
    assert opset == 23
    output_path.write_bytes(f"{profile_name}:{checkpoint_path.name}".encode())


def fake_tokenizer_assets(tokenizer_path, staging_dir):
    (staging_dir / "tokenizer.json").write_text(tokenizer_path.read_text())
    (staging_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": push_to_hub.HF_CHAT_TEMPLATE})
    )
    (staging_dir / "special_tokens_map.json").write_text("{}")


def test_stage_hub_artifacts_rebuilds_clean_staging_dir(monkeypatch, tmp_path):
    checkpoint_path, tokenizer_path, model_card_path, logo_path = write_sources(tmp_path)
    staging_dir = tmp_path / "hub_upload" / "sft_dusty8m"
    staging_dir.mkdir(parents=True)
    (staging_dir / "stale.bin").write_text("old")

    monkeypatch.setattr(push_to_hub, "export_onnx", fake_export)
    monkeypatch.setattr(push_to_hub, "build_tokenizer_assets", fake_tokenizer_assets)

    staged_files = push_to_hub.stage_hub_artifacts(
        profile_name="sft_dusty8m",
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        model_card_path=model_card_path,
        logo_path=logo_path,
        staging_dir=staging_dir,
    )

    staged_names = {path.relative_to(staging_dir).as_posix() for path in staged_files}
    assert "stale.bin" not in staged_names
    assert staged_names == {
        "README.md",
        "assets/logo.png",
        "model.pt",
        "model_int8.onnx",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert (staging_dir / "model.pt").read_bytes() == b"checkpoint"
    assert (staging_dir / "assets/logo.png").read_bytes() == b"logo"
    assert (staging_dir / "README.md").read_text() == "# DustyLM"
    tokenizer_config = json.loads((staging_dir / "tokenizer_config.json").read_text())
    assert tokenizer_config["chat_template"] == push_to_hub.HF_CHAT_TEMPLATE


def test_main_dry_run_does_not_upload(monkeypatch, tmp_path, capsys):
    checkpoint_path, tokenizer_path, model_card_path, logo_path = write_sources(tmp_path)
    staging_dir = tmp_path / "hub_upload" / "sft_dusty8m"
    uploaded = []

    monkeypatch.setattr(push_to_hub, "export_onnx", fake_export)
    monkeypatch.setattr(push_to_hub, "build_tokenizer_assets", fake_tokenizer_assets)
    monkeypatch.setattr(
        push_to_hub,
        "upload_staging_dir",
        lambda **kwargs: uploaded.append(kwargs),
    )

    push_to_hub.main(
        [
            "--repo-id",
            "mkhordoo/dusty-8m-sft",
            "--checkpoint-path",
            str(checkpoint_path),
            "--tokenizer-path",
            str(tokenizer_path),
            "--model-card",
            str(model_card_path),
            "--logo",
            str(logo_path),
            "--staging-dir",
            str(staging_dir),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert "Dry run complete" in output
    assert uploaded == []


def test_upload_auth_error_mentions_token(monkeypatch, tmp_path):
    class FakeAuthError(Exception):
        pass

    class FakeApi:
        def upload_folder(self, **kwargs):
            raise FakeAuthError("unauthorized: missing token")

    monkeypatch.setattr(push_to_hub, "HfHubHTTPError", FakeAuthError)
    monkeypatch.setattr(push_to_hub, "HfApi", lambda: FakeApi())

    with pytest.raises(RuntimeError) as exc_info:
        push_to_hub.upload_staging_dir(
            staging_dir=tmp_path,
            repo_id="mkhordoo/dusty-8m-sft",
            commit_message="test",
        )

    message = str(exc_info.value)
    assert "Run `hf auth login`" in message
    assert "set the HF_TOKEN environment variable" in message
    assert "Use one authentication method" in message
    assert "pass a valid token" not in message


def test_push_to_hub_does_not_depend_on_docs_model():
    source = Path(push_to_hub.__file__).read_text()
    assert "docs/model.onnx" not in source
