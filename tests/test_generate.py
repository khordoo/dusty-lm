import pytest

from tiny_gpt.config import get_profile
from tiny_gpt.generate import (
    parse_args,
    prepare_generation_prompt,
    resolve_generation_checkpoint_path,
)


def test_generation_cli_parses_profile_without_loading_checkpoint():
    args = parse_args(["--profile", "smollm2_360m", "--prompt", "Hello"])

    assert args.profile == "smollm2_360m"
    assert args.prompt == "Hello"
    assert args.checkpoint_step is None


def test_generation_cli_parses_checkpoint_step_without_loading_checkpoint():
    args = parse_args(["--profile", "dusty8m", "--checkpoint-step", "100"])

    assert args.checkpoint_step == 100


def test_pretrain_generation_prompt_is_not_wrapped():
    profile = get_profile("dusty8m")

    assert prepare_generation_prompt("i wake up.", profile) == "i wake up."


def test_sft_dusty_generation_prompt_wraps_raw_user_text():
    profile = get_profile("sft_dusty8m")

    assert prepare_generation_prompt("where are you?", profile) == (
        "<|im_start|>user\n"
        "where are you?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_sft_dusty_generation_prompt_does_not_double_wrap_chatml():
    profile = get_profile("sft_dusty8m")
    prompt = (
        "<|im_start|>user\n"
        "where are you?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    assert prepare_generation_prompt(prompt, profile) == prompt


def test_resolve_generation_checkpoint_path_defaults_to_final_checkpoint():
    profile = get_profile("dusty8m")

    assert resolve_generation_checkpoint_path(profile) == profile.generation.checkpoint_path


def test_resolve_generation_checkpoint_path_uses_pretrain_step_checkpoint():
    profile = get_profile("dusty8m")

    assert resolve_generation_checkpoint_path(profile, checkpoint_step=100).name == (
        "dusty8m_step_100.pt"
    )


def test_resolve_generation_checkpoint_path_uses_sft_step_checkpoint():
    profile = get_profile("sft_dusty8m")

    assert resolve_generation_checkpoint_path(profile, checkpoint_step=200).name == (
        "dusty8m_sft_step_200.pt"
    )


def test_resolve_generation_checkpoint_path_rejects_non_positive_step():
    profile = get_profile("dusty8m")

    with pytest.raises(ValueError, match="checkpoint_step must be at least 1"):
        resolve_generation_checkpoint_path(profile, checkpoint_step=0)
