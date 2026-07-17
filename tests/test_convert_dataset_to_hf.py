import pytest

from scripts import convert_dataset_to_hf


def test_convert_to_messages_preserves_category_and_roles():
    converted = convert_dataset_to_hf.convert_to_messages(
        {
            "category": "crumbs",
            "user": "did you find anything?",
            "dusty": "three crumbs. a good patrol.",
        }
    )

    assert converted == {
        "category": "crumbs",
        "messages": [
            {"role": "user", "content": "did you find anything?"},
            {"role": "assistant", "content": "three crumbs. a good patrol."},
        ],
    }


def test_prepare_dataset_rejects_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="SFT dataset not found"):
        convert_dataset_to_hf.prepare_dataset(
            input_path=tmp_path / "missing.jsonl",
            test_size=1,
            seed=42,
        )


def test_main_dry_run_validates_sources_without_upload(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "persona.jsonl"
    card_path = tmp_path / "README.md"
    input_path.write_text("{}\n")
    card_path.write_text("# Dataset\n")
    prepared = {"train": [1, 2], "test": [3]}
    uploads = []

    monkeypatch.setattr(
        convert_dataset_to_hf,
        "prepare_dataset",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        convert_dataset_to_hf,
        "upload_dataset",
        lambda *args, **kwargs: uploads.append((args, kwargs)),
    )

    convert_dataset_to_hf.main(
        [
            "--input",
            str(input_path),
            "--repo-id",
            "learner/custom-persona",
            "--readme",
            str(card_path),
            "--test-size",
            "1",
            "--seed",
            "7",
            "--dry-run",
        ]
    )

    assert uploads == []
    assert "Dry run complete" in capsys.readouterr().out


def test_upload_dataset_pushes_data_and_card(monkeypatch, tmp_path):
    class FakeSplit:
        def __init__(self):
            self.pushes = []

        def push_to_hub(self, repo_id, private):
            self.pushes.append((repo_id, private))

    class FakeApi:
        def __init__(self):
            self.uploads = []

        def upload_file(self, **kwargs):
            self.uploads.append(kwargs)

    card_path = tmp_path / "README.md"
    card_path.write_text("# Dataset\n")
    split = FakeSplit()
    api = FakeApi()
    monkeypatch.setattr(convert_dataset_to_hf, "HfApi", lambda: api)

    convert_dataset_to_hf.upload_dataset(
        split,
        repo_id="learner/custom-persona",
        readme_path=card_path,
    )

    assert split.pushes == [("learner/custom-persona", False)]
    assert api.uploads == [
        {
            "path_or_fileobj": str(card_path),
            "path_in_repo": "README.md",
            "repo_id": "learner/custom-persona",
            "repo_type": "dataset",
        }
    ]


def test_main_rejects_missing_dataset_card(tmp_path):
    input_path = tmp_path / "persona.jsonl"
    input_path.write_text("{}\n")

    with pytest.raises(FileNotFoundError, match="Dataset card not found"):
        convert_dataset_to_hf.main(
            [
                "--input",
                str(input_path),
                "--readme",
                str(tmp_path / "missing.md"),
                "--dry-run",
            ]
        )
