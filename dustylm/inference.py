"""Assistant-oriented inference API for SFT DustyLM checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from dustylm.checkpoint import (
    CHAT_PROFILE_DEFAULT,
    load_state_dict,
    resolve_profile_name_for_checkpoint,
)
from dustylm.config import (
    GenerationSpec,
    Profile,
    TokenizerSpec,
    TrainingTask,
    get_profile,
    list_profiles,
)
from dustylm.generate import (
    CHATML_START_TOKEN,
    encode_prompt,
    generate_token_ids,
    get_device,
)
from dustylm.modeling import build_model, build_tokenizer

CHATML_END_TOKEN = "<|im_end|>"
SUPPORTED_ROLES = {"system", "user", "assistant"}
DEFAULT_PROFILE = CHAT_PROFILE_DEFAULT


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=None,
        choices=list_profiles(),
        help="SFT chat profile to run. Defaults to checkpoint config/detection or Dusty.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="DustyLM checkpoint path. Defaults to the profile generation checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer JSON path. Defaults to the profile tokenizer.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda, mps, or cpu based on availability.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-chat-turns",
        type=int,
        default=None,
        help="Override the profile chat history window, counted in user turns.",
    )
    return parser.parse_args(argv)


def require_sft_profile(profile: Profile) -> None:
    if profile.training is None or profile.training.task != TrainingTask.SFT:
        raise ValueError(
            "Inference supports chat/SFT profiles only. "
            "Use dustylm.generate for base-model text generation."
        )
    if profile.generation is None:
        raise ValueError(f"Profile '{profile.name}' does not define generation config")


def validate_chat_messages(messages: list[dict[str, Any]]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be a dict")

        role = message.get("role")
        content = message.get("content")
        if role not in SUPPORTED_ROLES:
            raise ValueError(
                "message role must be one of: assistant, system, user"
            )
        if not isinstance(content, str):
            raise ValueError("message content must be a string")


def trim_chat_messages(
    messages: list[dict[str, Any]],
    max_chat_turns: int | None,
) -> list[dict[str, str]]:
    validate_chat_messages(messages)
    if max_chat_turns is not None and max_chat_turns < 1:
        raise ValueError("max_chat_turns must be at least 1")
    if max_chat_turns is None:
        return [message for message in messages if message["content"].strip()]

    system_messages = [
        message
        for message in messages
        if message["role"] == "system" and message["content"].strip()
    ]
    conversation_messages = [
        message
        for message in messages
        if message["role"] != "system" and message["content"].strip()
    ]

    user_turns_seen = 0
    keep_from = 0
    for index in range(len(conversation_messages) - 1, -1, -1):
        if conversation_messages[index]["role"] != "user":
            continue
        user_turns_seen += 1
        keep_from = index
        if user_turns_seen == max_chat_turns:
            break

    return system_messages + conversation_messages[keep_from:]


def format_chatml_messages(messages: list[dict[str, Any]]) -> str:
    validate_chat_messages(messages)
    chunks = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if not content.strip():
            continue
        chunks.append(f"{CHATML_START_TOKEN}{role}\n{content}{CHATML_END_TOKEN}\n")

    chunks.append(f"{CHATML_START_TOKEN}assistant\n")
    return "".join(chunks)


def strip_stop_text(text: str, spec: GenerationSpec) -> str:
    text = text.split(CHATML_END_TOKEN, 1)[0]
    if spec.eos_text is not None:
        text = text.split(spec.eos_text, 1)[0]
    return text.strip()


class Inference:
    def __init__(
        self,
        checkpoint_path=None,
        tokenizer_path=None,
        device=None,
        profile_name=None,
    ):
        profile_name = resolve_profile_name_for_checkpoint(
            checkpoint_path,
            explicit_profile=profile_name,
            default_profile=DEFAULT_PROFILE,
            mode="chat",
        )
        self.profile = get_profile(profile_name)
        require_sft_profile(self.profile)
        self.profile_name = self.profile.name
        self.spec = self.profile.generation
        self.device = device or get_device()
        self.checkpoint_path = Path(checkpoint_path or self.spec.checkpoint_path)
        self.tokenizer_path = Path(
            tokenizer_path or self.profile.model.tokenizer.path_or_name
        )

        self._require_artifact(
            self.checkpoint_path,
            "Checkpoint",
            "Train or download the SFT checkpoint first, or pass checkpoint_path=...",
        )
        self._require_artifact(
            self.tokenizer_path,
            "Tokenizer",
            "Run `make dusty-tokenizer`, or pass tokenizer_path=...",
        )

        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.model = self._load_model()

    @staticmethod
    def _require_artifact(path: Path, label: str, hint: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}. {hint}")

    def _load_model(self):
        model = build_model(self.profile)
        state_dict = load_state_dict(
            self.checkpoint_path,
            map_location=self.device,
        )
        state_dict.pop("rope.sin_cache", None)
        state_dict.pop("rope.cos_cache", None)
        model.load_state_dict(state_dict)
        if hasattr(model, "rope"):
            model.rope.resize_cache(
                self.profile.model.max_seq_len + self.spec.max_new_tokens
            )
        model.to(self.device)
        model.eval()
        return model

    def _load_tokenizer(self, tokenizer_path):
        if tokenizer_path is None:
            return build_tokenizer(self.profile)

        tokenizer_profile = replace(
            self.profile,
            model=replace(
                self.profile.model,
                tokenizer=TokenizerSpec(
                    kind=self.profile.model.tokenizer.kind,
                    path_or_name=tokenizer_path,
                ),
            ),
        )
        return build_tokenizer(tokenizer_profile)

    def chat_completion(
        self,
        messages,
        temperature=0.7,
        max_tokens=64,
        top_p=1.0,
        max_chat_turns=None,
        **kwargs,
    ):
        if kwargs.get("stream"):
            raise NotImplementedError("streaming chat completions are not supported yet")
        if "top_k" in kwargs:
            raise TypeError("top_k is not part of the chat_completion public API")

        chat_turn_limit = (
            self.spec.max_chat_turns if max_chat_turns is None else max_chat_turns
        )
        prompt = format_chatml_messages(
            trim_chat_messages(messages, chat_turn_limit)
        )
        spec = replace(
            self.spec,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        token_ids = encode_prompt(self.tokenizer, prompt, spec)
        result = generate_token_ids(
            model=self.model,
            tokenizer=self.tokenizer,
            token_ids=token_ids,
            spec=spec,
            max_seq_len=self.profile.model.max_seq_len,
            device=self.device,
            max_new_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            top_k=self.spec.top_k,
        )
        content = strip_stop_text(result.text, spec)
        completion_tokens = len(result.token_ids)
        return {
            "object": "chat.completion",
            "model": self.profile_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": result.prompt_tokens + completion_tokens,
            },
        }


def run_chat_loop(engine: Inference, args) -> None:
    messages = []
    print("Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if user_text.lower() in {"exit", "quit"}:
            return
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        response = engine.chat_completion(
            messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            max_chat_turns=args.max_chat_turns,
        )
        assistant_text = response["choices"][0]["message"]["content"]
        print(f"Assistant> {assistant_text}")
        messages.append({"role": "assistant", "content": assistant_text})


def main(argv=None):
    args = parse_args(argv)
    engine = Inference(
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        profile_name=args.profile,
    )
    run_chat_loop(engine, args)


if __name__ == "__main__":
    main()
