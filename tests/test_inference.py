from types import SimpleNamespace

import pytest
import torch

from dustylm.config import get_profile
from dustylm.generate import GenerationResult
from dustylm.inference import (
    Inference,
    format_chatml_messages,
    parse_args,
    require_sft_profile,
    strip_stop_text,
    trim_chat_messages,
)


class FakeRope:
    def __init__(self):
        self.resized_to = None

    def resize_cache(self, max_seq_len):
        self.resized_to = max_seq_len


class FakeModel:
    def __init__(self):
        self.rope = FakeRope()
        self.loaded_state_dict = None
        self.device = None
        self.evaluated = False

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self


class FakeTokenizer:
    def encode(self, text):
        return list(range(len(text.split()) or 1))

    def decode(self, token_ids):
        return "reply"


def build_inference(
    monkeypatch,
    tmp_path,
    state_dict=None,
    tokenizer_path=None,
    profile_name="sft_dusty8m",
):
    checkpoint_path = tmp_path / "checkpoint.pt"
    resolved_tokenizer_path = tokenizer_path or tmp_path / "tokenizer.json"
    checkpoint_path.write_text("checkpoint")
    resolved_tokenizer_path.write_text("{}")
    model = FakeModel()
    tokenizer_calls = []

    monkeypatch.setattr(
        "dustylm.inference.build_tokenizer",
        lambda profile: tokenizer_calls.append(profile) or FakeTokenizer(),
    )
    monkeypatch.setattr("dustylm.inference.build_model", lambda profile: model)
    monkeypatch.setattr(
        "dustylm.inference.load_state_dict",
        lambda *args, **kwargs: dict(state_dict or {"weight": object()}),
    )

    engine = Inference(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path or resolved_tokenizer_path,
        device="cpu",
        profile_name=profile_name,
    )
    return engine, model, tokenizer_calls


def test_inference_cli_parses_defaults():
    args = parse_args([])

    assert args.profile is None
    assert args.checkpoint_path is None
    assert args.tokenizer_path is None
    assert args.device is None
    assert args.temperature == 0.7
    assert args.max_tokens == 64
    assert args.top_p == 1.0
    assert args.max_chat_turns is None


def test_inference_cli_parses_overrides():
    args = parse_args(
        [
            "--profile",
            "sft_dusty8m",
            "--checkpoint-path",
            "model.pt",
            "--tokenizer-path",
            "tokenizer.json",
            "--device",
            "cpu",
            "--temperature",
            "0.8",
            "--max-tokens",
            "12",
            "--top-p",
            "0.9",
            "--max-chat-turns",
            "3",
        ]
    )

    assert args.checkpoint_path == "model.pt"
    assert args.tokenizer_path == "tokenizer.json"
    assert args.device == "cpu"
    assert args.temperature == 0.8
    assert args.max_tokens == 12
    assert args.top_p == 0.9
    assert args.max_chat_turns == 3


def test_require_sft_profile_rejects_base_profile():
    with pytest.raises(ValueError, match="chat/SFT profiles only"):
        require_sft_profile(get_profile("dusty8m"))


def test_format_chatml_messages_formats_single_turn():
    assert format_chatml_messages([{"role": "user", "content": "Where Are You?"}]) == (
        "<|im_start|>user\nwhere are you?<|im_end|>\n<|im_start|>assistant\n"
    )


def test_format_chatml_messages_formats_multi_turn_and_skips_empty_system():
    messages = [
        {"role": "system", "content": "  "},
        {"role": "system", "content": "Stay Dusty"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Beep"},
        {"role": "user", "content": "dock; now?"},
    ]

    assert format_chatml_messages(messages) == (
        "<|im_start|>system\n"
        "stay dusty<|im_end|>\n"
        "<|im_start|>user\n"
        "hi<|im_end|>\n"
        "<|im_start|>assistant\n"
        "beep<|im_end|>\n"
        "<|im_start|>user\n"
        "dock. now?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@pytest.mark.parametrize(
    "messages, error",
    [
        ([], "non-empty list"),
        ([{"role": "tool", "content": "x"}], "message role"),
        ([{"role": "user", "content": 1}], "content must be a string"),
        (["not a dict"], "each message must be a dict"),
    ],
)
def test_format_chatml_messages_rejects_bad_messages(messages, error):
    with pytest.raises(ValueError, match=error):
        format_chatml_messages(messages)


def test_trim_chat_messages_keeps_latest_user_turn_for_dusty():
    messages = [
        {"role": "system", "content": "stay dusty"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current question"},
    ]

    assert trim_chat_messages(messages, max_chat_turns=1) == [
        {"role": "system", "content": "stay dusty"},
        {"role": "user", "content": "current question"},
    ]


def test_trim_chat_messages_keeps_recent_five_user_turns_for_smollm2():
    messages = []
    for index in range(6):
        messages.append({"role": "user", "content": f"question {index}"})
        messages.append({"role": "assistant", "content": f"answer {index}"})

    trimmed = trim_chat_messages(messages, max_chat_turns=5)

    assert trimmed[0] == {"role": "user", "content": "question 1"}
    assert trimmed[-1] == {"role": "assistant", "content": "answer 5"}
    assert len([message for message in trimmed if message["role"] == "user"]) == 5


def test_trim_chat_messages_rejects_invalid_turn_limit():
    with pytest.raises(ValueError, match="max_chat_turns must be at least 1"):
        trim_chat_messages(
            [{"role": "user", "content": "hello"}],
            max_chat_turns=0,
        )


def test_inference_loads_checkpoint_and_removes_rope_cache(monkeypatch, tmp_path):
    state_dict = {
        "rope.sin_cache": object(),
        "rope.cos_cache": object(),
        "weight": object(),
    }
    engine, model, _ = build_inference(monkeypatch, tmp_path, state_dict=state_dict)

    assert engine.profile_name == "sft_dusty8m"
    assert model.loaded_state_dict == {"weight": state_dict["weight"]}
    assert model.rope.resized_to == (engine.profile.model.max_seq_len + engine.spec.max_new_tokens)
    assert model.device == "cpu"
    assert model.evaluated is True


def test_inference_auto_detects_dusty_checkpoint_when_profile_is_omitted(
    monkeypatch,
    tmp_path,
):
    checkpoint_path = tmp_path / "dusty.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    torch.save(
        {
            "embed.weight": torch.tensor([1.0]),
            "layers.0.attention.qkv_proj.weight": torch.tensor([1.0]),
        },
        checkpoint_path,
    )
    tokenizer_path.write_text("{}")
    model = FakeModel()

    monkeypatch.setattr(
        "dustylm.inference.build_tokenizer",
        lambda profile: FakeTokenizer(),
    )
    monkeypatch.setattr("dustylm.inference.build_model", lambda profile: model)
    monkeypatch.setattr(
        "dustylm.inference.load_state_dict",
        lambda *args, **kwargs: {"embed.weight": object()},
    )

    engine = Inference(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device="cpu",
    )

    assert engine.profile_name == "sft_dusty8m"


def test_inference_auto_detects_smollm2_sft_checkpoint_when_profile_is_omitted(
    monkeypatch,
    tmp_path,
):
    checkpoint_path = tmp_path / "smol.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    torch.save(
        {
            "embed_tokens.weight": torch.tensor([1.0]),
            "layers.0.gate_proj.weight": torch.tensor([1.0]),
            "layers.0.up_proj.weight": torch.tensor([1.0]),
        },
        checkpoint_path,
    )
    tokenizer_path.write_text("{}")
    model = FakeModel()

    monkeypatch.setattr(
        "dustylm.inference.build_tokenizer",
        lambda profile: FakeTokenizer(),
    )
    monkeypatch.setattr("dustylm.inference.build_model", lambda profile: model)
    monkeypatch.setattr(
        "dustylm.inference.load_state_dict",
        lambda *args, **kwargs: {
            "embed_tokens.weight": object(),
            "layers.0.gate_proj.weight": object(),
        },
    )

    engine = Inference(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device="cpu",
    )

    assert engine.profile_name == "sft_smollm2_135m"


def test_inference_rejects_detected_non_sft_profile(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_text("placeholder")
    monkeypatch.setattr(
        "dustylm.inference.resolve_profile_name_for_checkpoint",
        lambda *args, **kwargs: "smollm2_360m",
    )

    with pytest.raises(ValueError, match="chat/SFT profiles only"):
        Inference(checkpoint_path=checkpoint_path, device="cpu")


def test_inference_loads_tokenizer_through_factory(monkeypatch, tmp_path):
    _, _, tokenizer_calls = build_inference(monkeypatch, tmp_path)

    assert len(tokenizer_calls) == 1
    assert tokenizer_calls[0].name == "sft_dusty8m"


def test_inference_applies_explicit_tokenizer_path_through_profile_override(
    monkeypatch,
    tmp_path,
):
    explicit_tokenizer_path = tmp_path / "custom-tokenizer.json"

    _, _, tokenizer_calls = build_inference(
        monkeypatch,
        tmp_path,
        tokenizer_path=explicit_tokenizer_path,
    )

    assert tokenizer_calls[0].model.tokenizer.path_or_name == explicit_tokenizer_path


def test_chat_completion_returns_openai_like_response(monkeypatch, tmp_path):
    engine, _, _ = build_inference(monkeypatch, tmp_path)
    calls = []

    def fake_generate_token_ids(**kwargs):
        calls.append(kwargs)
        return GenerationResult(
            text=" i am under the couch.<|im_end|>",
            token_ids=[10, 11, 12],
            finish_reason="stop",
            prompt_tokens=7,
        )

    monkeypatch.setattr(
        "dustylm.inference.generate_token_ids",
        fake_generate_token_ids,
    )

    response = engine.chat_completion(
        [{"role": "user", "content": "where are you?"}],
        temperature=0.8,
        max_tokens=32,
        top_p=0.9,
    )

    assert response == {
        "object": "chat.completion",
        "model": "sft_dusty8m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "i am under the couch.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        },
    }
    assert calls[0]["max_new_tokens"] == 32
    assert calls[0]["temperature"] == 0.8
    assert calls[0]["top_k"] == engine.spec.top_k
    assert calls[0]["top_p"] == 0.9


def test_chat_completion_uses_profile_chat_turn_limit(monkeypatch, tmp_path):
    engine, _, _ = build_inference(monkeypatch, tmp_path)
    prompts = []

    monkeypatch.setattr(
        "dustylm.inference.generate_token_ids",
        lambda **kwargs: GenerationResult(
            text="reply",
            token_ids=[1],
            finish_reason="length",
            prompt_tokens=1,
        ),
    )
    monkeypatch.setattr(
        "dustylm.inference.encode_prompt",
        lambda tokenizer, prompt, spec: prompts.append(prompt) or [1],
    )

    engine.chat_completion(
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current question"},
        ]
    )

    assert "old question" not in prompts[0]
    assert "old answer" not in prompts[0]
    assert "current question" in prompts[0]


def test_chat_completion_accepts_max_chat_turns_override(monkeypatch, tmp_path):
    engine, _, _ = build_inference(monkeypatch, tmp_path)
    prompts = []

    monkeypatch.setattr(
        "dustylm.inference.generate_token_ids",
        lambda **kwargs: GenerationResult(
            text="reply",
            token_ids=[1],
            finish_reason="length",
            prompt_tokens=1,
        ),
    )
    monkeypatch.setattr(
        "dustylm.inference.encode_prompt",
        lambda tokenizer, prompt, spec: prompts.append(prompt) or [1],
    )

    engine.chat_completion(
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current question"},
        ],
        max_chat_turns=2,
    )

    assert "old question" in prompts[0]
    assert "old answer" in prompts[0]
    assert "current question" in prompts[0]


def test_chat_completion_rejects_top_k_kwarg(monkeypatch, tmp_path):
    engine, _, _ = build_inference(monkeypatch, tmp_path)

    with pytest.raises(TypeError, match="top_k"):
        engine.chat_completion(
            [{"role": "user", "content": "hello"}],
            top_k=5,
        )


def test_chat_completion_rejects_streaming(monkeypatch, tmp_path):
    engine, _, _ = build_inference(monkeypatch, tmp_path)

    with pytest.raises(NotImplementedError, match="streaming"):
        engine.chat_completion(
            [{"role": "user", "content": "hello"}],
            stream=True,
        )


def test_strip_stop_text_removes_chatml_and_eos_text():
    spec = SimpleNamespace(eos_text="<|endoftext|>")

    assert strip_stop_text("hello<|im_end|>ignored", spec) == "hello"
    assert strip_stop_text("hello<|endoftext|>ignored", spec) == "hello"
