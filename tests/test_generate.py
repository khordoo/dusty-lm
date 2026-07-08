import pytest
import torch

from dustylm.config import get_profile
from dustylm.generate import (
    apply_top_p_filter,
    get_token_id,
    parse_args,
    prepare_generation_prompt,
    resolve_generation_checkpoint_path,
    resolve_num_new_tokens,
    validate_generation_options,
    validate_prompt_length,
)


def test_generation_cli_parses_profile_without_loading_checkpoint():
    args = parse_args(["--profile", "smollm2_360m", "--prompt", "Hello"])

    assert args.profile == "smollm2_360m"
    assert args.prompt == "Hello"
    assert args.checkpoint_step is None


def test_generation_cli_defaults_to_profile_detection():
    args = parse_args([])

    assert args.profile is None


def test_generation_cli_parses_checkpoint_step_without_loading_checkpoint():
    args = parse_args(["--profile", "dusty8m", "--checkpoint-step", "100"])

    assert args.checkpoint_step == 100
    assert args.checkpoint_path is None
    assert args.top_p is None
    assert args.temperature is None


def test_generation_cli_parses_checkpoint_path_without_loading_checkpoint():
    args = parse_args(["--checkpoint-path", "model.pt"])

    assert args.checkpoint_path.name == "model.pt"


def test_generation_cli_parses_top_p():
    args = parse_args(
        [
            "--profile",
            "dusty8m",
            "--top-p",
            "0.9",
        ]
    )

    assert args.top_p == 0.9


def test_generation_cli_parses_temperature():
    args = parse_args(
        [
            "--profile",
            "dusty8m",
            "--temperature",
            "0.5",
        ]
    )

    assert args.temperature == 0.5


def test_validate_generation_options_rejects_bad_top_p():
    with pytest.raises(ValueError, match="top_p must be greater than 0"):
        validate_generation_options(top_p=0.0, temperature=1.0)

    with pytest.raises(ValueError, match="top_p must be greater than 0"):
        validate_generation_options(top_p=1.1, temperature=1.0)


def test_validate_generation_options_rejects_bad_temperature():
    with pytest.raises(ValueError, match="temperature must be greater than 0"):
        validate_generation_options(top_p=1.0, temperature=0.0)


def test_apply_top_p_filter_masks_tokens_outside_probability_mass():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    filtered = apply_top_p_filter(logits.clone(), top_p=0.8)

    assert torch.isfinite(filtered[0, 0])
    assert torch.isfinite(filtered[0, 1])
    assert filtered[0, 2] == float("-inf")
    assert filtered[0, 3] == float("-inf")


def test_get_token_id_uses_tokenizers_token_to_id():
    class FakeTokenizer:
        def token_to_id(self, text):
            return 42 if text == "<|im_end|>" else None

    assert get_token_id(FakeTokenizer(), "<|im_end|>") == 42
    assert get_token_id(FakeTokenizer(), "<missing>") is None


def test_get_token_id_uses_encode_fallback():
    class FakeTokenizer:
        def encode(self, text):
            return [7, 8] if text == "<|im_end|>" else []

    assert get_token_id(FakeTokenizer(), "<|im_end|>") == 7
    assert get_token_id(FakeTokenizer(), "<missing>") is None


def test_validate_prompt_length_rejects_full_context_prompt():
    validate_prompt_length([1, 2], max_seq_len=3)

    with pytest.raises(ValueError, match="Prompt contains 3 tokens"):
        validate_prompt_length([1, 2, 3], max_seq_len=3)


def test_resolve_num_new_tokens_keeps_generation_inside_context():
    assert (
        resolve_num_new_tokens(
            max_new_tokens=100,
            prompt_length=20,
            max_seq_len=256,
        )
        == 100
    )
    assert (
        resolve_num_new_tokens(
            max_new_tokens=100,
            prompt_length=250,
            max_seq_len=256,
        )
        == 6
    )


def test_pretrain_generation_prompt_is_not_wrapped():
    profile = get_profile("dusty8m")

    assert prepare_generation_prompt("i wake up.", profile) == "i wake up."
    validate_generation_options(profile.generation.top_p, profile.generation.temperature)


def test_pretrain_generation_prompt_is_normalized_like_training_text():
    profile = get_profile("dusty8m")

    assert prepare_generation_prompt("Dusty Cleans; Then Docks.", profile) == (
        "dusty cleans. then docks."
    )


def test_sft_dusty_generation_prompt_wraps_raw_user_text():
    profile = get_profile("sft_dusty8m")

    assert prepare_generation_prompt("Where Are You; Dusty?", profile) == (
        "<|im_start|>user\nwhere are you. dusty?<|im_end|>\n<|im_start|>assistant\n"
    )


def test_sft_dusty_generation_prompt_does_not_double_wrap_chatml():
    profile = get_profile("sft_dusty8m")
    prompt = "<|im_start|>user\nwhere are you?<|im_end|>\n<|im_start|>assistant\n"

    assert prepare_generation_prompt(prompt, profile) == prompt


def test_resolve_generation_checkpoint_path_defaults_to_final_checkpoint():
    profile = get_profile("dusty8m")

    assert resolve_generation_checkpoint_path(profile) == profile.generation.checkpoint_path


def test_resolve_generation_checkpoint_path_accepts_explicit_checkpoint_path(tmp_path):
    profile = get_profile("dusty8m")
    checkpoint_path = tmp_path / "custom.pt"

    assert (
        resolve_generation_checkpoint_path(profile, checkpoint_path=checkpoint_path)
        == checkpoint_path
    )


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
