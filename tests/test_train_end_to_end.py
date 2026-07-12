from dataclasses import replace

import pytest

from dustylm.config import get_profile
from scripts import train_end_to_end


def test_parse_args_uses_golden_path_defaults():
    args = train_end_to_end.parse_args([])

    assert args.pretrain_profile == "dusty8m"
    assert args.sft_profile == "sft_dusty8m"
    assert args.tinystories_slice == "train[:100000]"
    assert args.pretrain_epochs == 1
    assert args.sft_epochs == 2
    assert args.pretrain_batch_size == 224
    assert args.sft_batch_size == 224
    assert args.checkpoint_every_steps == 50
    assert args.best_pretrain_step == 0
    assert args.best_sft_step == 0


def test_parse_args_accepts_overrides():
    args = train_end_to_end.parse_args(
        [
            "--tinystories-slice",
            "train[:100000]",
            "--pretrain-epochs",
            "2",
            "--sft-epochs",
            "3",
            "--best-pretrain-step",
            "1700",
            "--best-sft-step",
            "300",
        ]
    )

    assert args.tinystories_slice == "train[:100000]"
    assert args.pretrain_epochs == 2
    assert args.sft_epochs == 3
    assert args.best_pretrain_step == 1700
    assert args.best_sft_step == 300


def test_promote_step_checkpoint_copies_selected_checkpoint(monkeypatch, tmp_path):
    source = tmp_path / "dusty8m_step_200.pt"
    target = tmp_path / "dusty8m.pt"
    source.write_text("checkpoint")
    profile = get_profile("dusty8m")
    profile = replace(
        profile,
        training=replace(profile.training, output_checkpoint=target),
    )

    monkeypatch.setattr(train_end_to_end, "get_profile", lambda name: profile)
    monkeypatch.setattr(
        train_end_to_end,
        "get_step_checkpoint_path",
        lambda profile, step: source,
    )

    assert train_end_to_end.promote_step_checkpoint("dusty8m", 200) == target
    assert target.read_text() == "checkpoint"


def test_promote_step_checkpoint_errors_when_checkpoint_is_missing(monkeypatch, tmp_path):
    missing = tmp_path / "missing.pt"
    profile = get_profile("dusty8m")
    profile = replace(
        profile,
        training=replace(profile.training, output_checkpoint=tmp_path / "dusty8m.pt"),
    )

    monkeypatch.setattr(train_end_to_end, "get_profile", lambda name: profile)
    monkeypatch.setattr(
        train_end_to_end,
        "get_step_checkpoint_path",
        lambda profile, step: missing,
    )

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        train_end_to_end.promote_step_checkpoint("dusty8m", 200)
