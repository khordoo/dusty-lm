from dataclasses import replace

import pytest

from tiny_gpt.config import get_profile
from tiny_gpt.train import initialize_random_seed, parse_args, require_training_dataset


def test_train_parse_args_defaults_to_one_epoch():
    args = parse_args([])

    assert args.epochs == 1


def test_train_parse_args_accepts_epoch_override():
    args = parse_args(["--epochs", "3"])

    assert args.epochs == 3


def test_train_parse_args_accepts_dusty8m_profile():
    args = parse_args(["--profile", "dusty8m"])

    assert args.profile == "dusty8m"


def test_initialize_random_seed_locks_python_numpy_and_torch(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr("tiny_gpt.train.random.randint", lambda start, end: 7102)
    monkeypatch.setattr(
        "tiny_gpt.train.random.seed", lambda seed: calls.append(("python", seed))
    )
    monkeypatch.setattr(
        "tiny_gpt.train.np.random.seed", lambda seed: calls.append(("numpy", seed))
    )
    monkeypatch.setattr(
        "tiny_gpt.train.torch.manual_seed", lambda seed: calls.append(("torch", seed))
    )
    monkeypatch.setattr(
        "tiny_gpt.train.torch.backends.mps.is_available", lambda: False
    )

    assert initialize_random_seed() == 7102

    assert "INITIALIZING WITH RANDOM SEED: 7102" in capsys.readouterr().out
    assert calls == [("python", 7102), ("numpy", 7102), ("torch", 7102)]


def test_missing_dusty_training_dataset_error_points_to_make_target(tmp_path):
    profile = get_profile("dusty8m")
    profile = replace(
        profile,
        training=replace(
            profile.training,
            dataset_path=tmp_path / "missing_dataset",
        ),
    )

    with pytest.raises(FileNotFoundError, match="make dusty-pretrain-data"):
        require_training_dataset(profile)
