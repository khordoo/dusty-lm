from dataclasses import replace

import pytest

from dustylm.config import get_profile
from dustylm.train import (
    get_step_checkpoint_path,
    initialize_random_seed,
    load_init_checkpoint_if_configured,
    parse_args,
    require_training_dataset,
    save_step_checkpoint_if_due,
)


def test_train_parse_args_defaults_to_one_epoch():
    args = parse_args([])

    assert args.epochs == 1


def test_train_parse_args_accepts_epoch_override():
    args = parse_args(["--epochs", "3"])

    assert args.epochs == 3


def test_train_parse_args_accepts_checkpoint_interval_override():
    args = parse_args(["--checkpoint-every-steps", "50"])

    assert args.checkpoint_every_steps == 50


def test_train_parse_args_accepts_dusty8m_profile():
    args = parse_args(["--profile", "dusty8m"])

    assert args.profile == "dusty8m"


def test_initialize_random_seed_locks_python_numpy_and_torch(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr("dustylm.train.random.randint", lambda start, end: 7102)
    monkeypatch.setattr("dustylm.train.random.seed", lambda seed: calls.append(("python", seed)))
    monkeypatch.setattr("dustylm.train.np.random.seed", lambda seed: calls.append(("numpy", seed)))
    monkeypatch.setattr(
        "dustylm.train.torch.manual_seed", lambda seed: calls.append(("torch", seed))
    )
    monkeypatch.setattr("dustylm.train.torch.backends.mps.is_available", lambda: False)

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

    with pytest.raises(FileNotFoundError, match="make data-pretrain"):
        require_training_dataset(profile)


def test_missing_dusty_sft_training_dataset_error_points_to_make_target(tmp_path):
    profile = get_profile("sft_dusty8m")
    profile = replace(
        profile,
        training=replace(
            profile.training,
            dataset_path=tmp_path / "missing_dataset",
        ),
    )

    with pytest.raises(FileNotFoundError, match="make data-sft"):
        require_training_dataset(profile)


def test_load_init_checkpoint_if_configured_loads_state_dict(monkeypatch, tmp_path):
    checkpoint = tmp_path / "dusty8m.pt"
    profile = get_profile("sft_dusty8m")
    profile = replace(
        profile,
        training=replace(profile.training, init_checkpoint_path=checkpoint),
    )
    expected_state = {"token_embedding.weight": object()}
    calls = []

    class FakeModel:
        def load_state_dict(self, state_dict):
            calls.append(state_dict)

    monkeypatch.setattr(
        "dustylm.train.torch.load",
        lambda *args, **kwargs: expected_state,
    )
    checkpoint.write_text("placeholder")

    assert load_init_checkpoint_if_configured(FakeModel(), profile, "cpu") is True
    assert calls == [expected_state]


def test_missing_init_checkpoint_error_points_to_dusty_pretrain(tmp_path):
    profile = get_profile("sft_dusty8m")
    profile = replace(
        profile,
        training=replace(
            profile.training,
            init_checkpoint_path=tmp_path / "missing.pt",
        ),
    )

    with pytest.raises(FileNotFoundError, match="make train-pretrain"):
        load_init_checkpoint_if_configured(object(), profile, "cpu")


def test_pretraining_profiles_do_not_load_init_checkpoint():
    assert load_init_checkpoint_if_configured(object(), get_profile("dusty8m"), "cpu") is False


def test_step_checkpoint_path_uses_output_checkpoint_stem_and_step(tmp_path):
    profile = get_profile("dusty8m")
    profile = replace(
        profile,
        training=replace(
            profile.training,
            output_checkpoint=tmp_path / "dusty8m.pt",
            checkpoint_dir=tmp_path / "steps",
        ),
    )

    assert get_step_checkpoint_path(profile, 100) == tmp_path / "steps" / "dusty8m_step_100.pt"


def test_save_step_checkpoint_if_due_saves_only_on_interval(monkeypatch, tmp_path):
    profile = get_profile("dusty8m")
    profile = replace(
        profile,
        training=replace(
            profile.training,
            output_checkpoint=tmp_path / "dusty8m.pt",
            checkpoint_dir=tmp_path,
        ),
    )
    saved = []

    class FakeModel:
        def state_dict(self):
            return {"weight": "value"}

    monkeypatch.setattr(
        "dustylm.train.torch.save",
        lambda state_dict, path: saved.append((state_dict, path)),
    )

    assert save_step_checkpoint_if_due(FakeModel(), profile, step=99, interval=100) is None
    checkpoint_path = save_step_checkpoint_if_due(
        FakeModel(),
        profile,
        step=100,
        interval=100,
    )

    assert checkpoint_path == tmp_path / "dusty8m_step_100.pt"
    assert saved == [({"weight": "value"}, tmp_path / "dusty8m_step_100.pt")]


def test_save_step_checkpoint_if_due_can_be_disabled(monkeypatch):
    saved = []
    monkeypatch.setattr(
        "dustylm.train.torch.save",
        lambda state_dict, path: saved.append((state_dict, path)),
    )

    assert (
        save_step_checkpoint_if_due(
            model=object(),
            profile=get_profile("dusty8m"),
            step=100,
            interval=0,
        )
        is None
    )
    assert saved == []
