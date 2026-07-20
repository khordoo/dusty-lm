"""Autoregressive text generation with top-k sampling and KV-cache.

This module loads a trained checkpoint and generates text token-by-token.
The generation loop uses a KV-cache to avoid recomputing attention over
the full sequence at every step:
  1. **Prefill**: the entire prompt is processed in one pass, populating the cache.
  2. **Decode**: only the latest token is fed to the model; the cache provides
     the history.  The cache grows by one entry per step.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from dustylm.checkpoint import (
    GENERATION_PROFILE_DEFAULT,
    load_state_dict,
    resolve_profile_name_for_checkpoint,
)
from dustylm.config import GenerationSpec, Profile, TrainingTask, get_profile, list_profiles
from dustylm.data_prep import normalize_model_text
from dustylm.modeling import build_model, build_tokenizer
from dustylm.tokenizer import CHATML_END_TOKEN, CHATML_START_TOKEN

DEFAULT_PROMPT = "Once upon a time "
DEFAULT_PROFILE = GENERATION_PROFILE_DEFAULT
EOS_TEXT_LOOKBACK_TOKENS = 10


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int


def get_device():
    """Pick the best available inference device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv=None):
    """Parse CLI arguments for one text-generation request."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=None,
        choices=list_profiles(),
        help="Generation profile to run. Defaults to checkpoint config/detection or Dusty.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Load a specific checkpoint path instead of the profile default.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help=(
            "Load a step checkpoint instead of the final generation checkpoint, "
            "for example --checkpoint-step 100 loads dusty8m_step_100.pt."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override profile nucleus sampling probability mass. 1.0 disables it.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override profile generation temperature.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the profile maximum number of generated tokens.",
    )
    return parser.parse_args(argv)


def validate_generation_options(top_p: float, temperature: float) -> None:
    """Validate sampling controls before generation starts."""
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1.0")
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than 0")


def encode_prompt(tokenizer, prompt: str, spec: GenerationSpec) -> list[int]:
    """Encode a prompt and prepend BOS when the profile requires it."""
    if isinstance(tokenizer, Tokenizer):
        token_ids = tokenizer.encode(prompt).ids
    else:
        token_ids = tokenizer.encode(prompt)
    if spec.bos_token_id is not None:
        token_ids = [spec.bos_token_id] + token_ids
    return token_ids


def decode_tokens(tokenizer, token_ids: list[int]) -> str:
    """Decode token IDs with either tokenizer implementation."""
    return tokenizer.decode(token_ids)


def get_token_id(tokenizer, text: str) -> int | None:
    """Return a token ID for exact text when the tokenizer can provide one."""
    if hasattr(tokenizer, "token_to_id"):
        token_id = tokenizer.token_to_id(text)
        if token_id is not None:
            return token_id

    if not hasattr(tokenizer, "encode"):
        return None

    token_ids = tokenizer.encode(text)
    # A stop marker is usable as a token ID only when it encodes to one token.
    return token_ids[0] if len(token_ids) == 1 else None


def validate_prompt_length(token_ids: list[int], max_seq_len: int) -> None:
    """Reject prompts that already fill the model context window."""
    prompt_length = len(token_ids)
    if prompt_length >= max_seq_len:
        raise ValueError(f"Prompt contains {prompt_length} tokens. Model maximum is {max_seq_len}.")


def resolve_num_new_tokens(
    max_new_tokens: int,
    prompt_length: int,
    max_seq_len: int,
) -> int:
    """Cap generation length so prompt plus output fits in context."""
    return min(max_new_tokens, max_seq_len - prompt_length)


def prepare_generation_prompt(prompt: str, profile: Profile) -> str:
    """Normalize a raw prompt and wrap SFT prompts in ChatML."""
    if CHATML_START_TOKEN in prompt:
        return prompt

    prompt = normalize_model_text(prompt)
    if profile.training is None or profile.training.task != TrainingTask.SFT:
        return prompt

    return f"{CHATML_START_TOKEN}user\n{prompt}{CHATML_END_TOKEN}\n{CHATML_START_TOKEN}assistant\n"


def resolve_generation_checkpoint_path(
    profile: Profile,
    checkpoint_step: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Resolve the final or step checkpoint path used for generation."""
    if profile.generation is None:
        raise ValueError(f"Profile '{profile.name}' does not define generation config")
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    if checkpoint_step is None:
        return profile.generation.checkpoint_path
    if checkpoint_step < 1:
        raise ValueError("checkpoint_step must be at least 1")

    final_checkpoint_path = profile.generation.checkpoint_path
    checkpoint_name = f"{final_checkpoint_path.stem}_step_{checkpoint_step}.pt"
    return final_checkpoint_path.parent / checkpoint_name


def load_model(
    profile: Profile,
    device=None,
    checkpoint_step: int | None = None,
    checkpoint_path: str | Path | None = None,
):
    """Build the model, load a checkpoint, and prepare for inference.

    RoPE sin/cos caches are not part of the learned weights, so they are
    dropped from the state dict and recomputed at the required length
    (prompt + max_new_tokens).
    """
    if profile.generation is None:
        raise ValueError(f"Profile '{profile.name}' does not define generation config")

    device = device or get_device()
    checkpoint_path = resolve_generation_checkpoint_path(
        profile,
        checkpoint_step=checkpoint_step,
        checkpoint_path=checkpoint_path,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run `make download-models` or `make train-sft` first."
        )
    tokenizer = build_tokenizer(profile)
    model = build_model(profile)
    state_dict = load_state_dict(checkpoint_path, map_location=device)
    # RoPE caches are derived buffers, not learned parameters.
    state_dict.pop("rope.sin_cache", None)
    state_dict.pop("rope.cos_cache", None)
    model.load_state_dict(state_dict)
    model.rope.resize_cache(profile.model.max_seq_len + profile.generation.max_new_tokens)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def apply_top_p_filter(next_token_logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus sampling by masking logits outside the top probability mass."""
    if top_p >= 1.0:
        return next_token_logits

    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    sorted_indices_to_remove = cumulative_probabilities > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )
    next_token_logits[indices_to_remove] = float("-inf")
    return next_token_logits


def sample_next_token(
    logits,
    spec: GenerationSpec,
    top_p: float | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
):
    """Sample the next token using temperature-scaled top-k sampling.

    Args:
        logits: [B, T, vocab_size] raw model output.
        spec:   generation config (temperature, top_k).

    Returns:
        [B] sampled token IDs.
    """
    top_p = spec.top_p if top_p is None else top_p
    temperature = spec.temperature if temperature is None else temperature
    top_k = spec.top_k if top_k is None else top_k
    validate_generation_options(top_p, temperature)
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # Step 1: Extract logits for the last position only.
    next_token_logits = logits[:, -1, :]  # [B, vocab_size]

    # Step 2: Apply temperature — higher temperature → more uniform distribution.
    next_token_logits = next_token_logits / temperature

    # Step 3: Find the top-k highest logit values.
    top_k = min(top_k, next_token_logits.shape[-1])
    values, _ = torch.topk(next_token_logits, top_k)  # [B, top_k]

    # Step 4: Zero out everything below the k-th highest value (top-k filtering).
    # values[:, [-1]] is the smallest value in the top-k set.
    next_token_logits[next_token_logits < values[:, [-1]]] = float("-inf")

    # Step 5 Apply top_p filter
    next_token_logits = apply_top_p_filter(next_token_logits, top_p=top_p)

    # Step 6: Convert filtered logits to a probability distribution.
    probabilities = torch.softmax(next_token_logits, dim=-1)  # [B, vocab_size]

    # Step 7: Sample one token per batch element from the distribution.
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)  # [B]


def generate_token_ids(
    model,
    tokenizer,
    token_ids: list[int],
    spec: GenerationSpec,
    max_seq_len: int,
    device,
    max_new_tokens: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
) -> GenerationResult:
    """Generate text from already-tokenized input without printing.

    This is the library-friendly generation path used by higher-level
    inference APIs. The CLI wrapper below keeps its streaming print behavior.
    """
    top_p = spec.top_p if top_p is None else top_p
    temperature = spec.temperature if temperature is None else temperature
    max_new_tokens = spec.max_new_tokens if max_new_tokens is None else max_new_tokens
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    validate_generation_options(top_p, temperature)
    validate_prompt_length(token_ids, max_seq_len)
    num_new_tokens = resolve_num_new_tokens(
        max_new_tokens=max_new_tokens,
        prompt_length=len(token_ids),
        max_seq_len=max_seq_len,
    )

    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = []
    input_tokens = tokens
    kv_cache = model.empty_kv_cache()
    im_end_id = get_token_id(tokenizer, "<|im_end|>")
    finish_reason = "length"

    with torch.inference_mode():
        for _ in range(num_new_tokens):
            logits, kv_cache = model(x=input_tokens, kv_cache=kv_cache)
            next_token = sample_next_token(
                logits,
                spec,
                top_p=top_p,
                temperature=temperature,
                top_k=top_k,
            )
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            if (spec.eos_token_id is not None and next_token_id == spec.eos_token_id) or (
                im_end_id is not None and next_token_id == im_end_id
            ):
                finish_reason = "stop"
                break

            tail_text = decode_tokens(
                tokenizer,
                generated_ids[-EOS_TEXT_LOOKBACK_TOKENS:],
            )
            if spec.eos_text is not None and spec.eos_text in tail_text:
                finish_reason = "stop"
                break

            input_tokens = next_token.unsqueeze(0)

    return GenerationResult(
        text=decode_tokens(tokenizer, generated_ids),
        token_ids=generated_ids,
        finish_reason=finish_reason,
        prompt_tokens=len(token_ids),
    )


def generate_text(
    prompt=DEFAULT_PROMPT,
    profile_name=None,
    checkpoint_step: int | None = None,
    checkpoint_path: str | Path | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
):
    """Generate text autoregressively from a prompt using KV-cached decoding."""
    prompt = prompt.strip()
    profile_name = resolve_profile_name_for_checkpoint(
        checkpoint_path,
        explicit_profile=profile_name,
        default_profile=DEFAULT_PROFILE,
        mode="generation",
    )
    profile = get_profile(profile_name)
    if profile.generation is None:
        raise ValueError(f"Profile '{profile.name}' does not define generation config")

    spec = profile.generation
    top_p = spec.top_p if top_p is None else top_p
    temperature = spec.temperature if temperature is None else temperature
    max_new_tokens = spec.max_new_tokens if max_new_tokens is None else max_new_tokens
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    validate_generation_options(top_p, temperature)
    model, tokenizer, device = load_model(
        profile,
        checkpoint_step=checkpoint_step,
        checkpoint_path=checkpoint_path,
    )

    prompt = prepare_generation_prompt(prompt, profile)
    token_ids = encode_prompt(tokenizer, prompt, spec)
    validate_prompt_length(token_ids, profile.model.max_seq_len)
    num_new_tokens = resolve_num_new_tokens(
        max_new_tokens=max_new_tokens,
        prompt_length=len(token_ids),
        max_seq_len=profile.model.max_seq_len,
    )
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = []
    im_end_id = get_token_id(tokenizer, "<|im_end|>")

    # --- Prefill: process the entire prompt in one forward pass ---
    # The empty cache is a list of None entries (one per layer).  After
    # prefill, kv_cache[i] holds the (key, value) tensors for layer i
    # covering all prompt positions.
    kv_cache = model.empty_kv_cache()
    input_tokens = tokens  # [1, prompt_len]

    print("Predicting...")
    print(prompt, end="", flush=True)
    with torch.inference_mode():
        for _ in range(num_new_tokens):
            # On the first iteration this is the full prompt (prefill).
            # On subsequent iterations this is a single token (decode).
            logits, kv_cache = model(x=input_tokens, kv_cache=kv_cache)
            next_token = sample_next_token(
                logits,
                spec,
                top_p=top_p,
                temperature=temperature,
            )
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            # Stop if we hit the config's EOS token OR the ChatML end token
            if (spec.eos_token_id is not None and next_token_id == spec.eos_token_id) or (
                im_end_id is not None and next_token_id == im_end_id
            ):
                print("\n\n[Generation stopped: Stop token detected]")
                break

            tail_text = decode_tokens(
                tokenizer,
                generated_ids[-EOS_TEXT_LOOKBACK_TOKENS:],
            )
            if spec.eos_text is not None and spec.eos_text in tail_text:
                print(f"\n\n[Generation stopped: {spec.eos_text} detected]")
                break

            next_word = decode_tokens(tokenizer, next_token.tolist())

            # --- Decode: feed only the new token; the cache has the history ---
            input_tokens = next_token.unsqueeze(0)  # [1, 1]
            print(next_word, end="", flush=True)
        else:
            print(f"\n\n[Generation stopped: Max new tokens reached ({num_new_tokens})]")

    return decode_tokens(tokenizer, generated_ids)


def main(argv=None):
    """CLI entry point for ``python -m dustylm.generate``."""
    args = parse_args(argv)
    try:
        generate_text(
            prompt=args.prompt,
            profile_name=args.profile,
            checkpoint_step=args.checkpoint_step,
            checkpoint_path=args.checkpoint_path,
            top_p=args.top_p,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
