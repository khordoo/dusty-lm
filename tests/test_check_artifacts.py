import pytest

from scripts.check_artifacts import main


def test_check_artifacts_uses_explicit_chat_paths(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path.write_text("checkpoint")
    tokenizer_path.write_text("{}")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "sft_dusty8m",
                "chat",
                "--checkpoint-path",
                str(checkpoint_path),
                "--tokenizer-path",
                str(tokenizer_path),
            ]
        )

    assert exc_info.value.code == 0


def test_check_artifacts_reports_explicit_missing_checkpoint(tmp_path, capsys):
    checkpoint_path = tmp_path / "missing.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "sft_dusty8m",
                "chat",
                "--checkpoint-path",
                str(checkpoint_path),
                "--tokenizer-path",
                str(tokenizer_path),
            ]
        )

    assert exc_info.value.code == 1
    assert str(checkpoint_path) in capsys.readouterr().out
