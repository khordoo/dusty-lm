"""Autoregressive text generation with top-k sampling and KV-cache.

This module loads a trained checkpoint and generates text token-by-token.
The generation loop uses a KV-cache to avoid recomputing attention over
the full sequence at every step:
  1. **Prefill**: the entire prompt is processed in one pass, populating the cache.
  2. **Decode**: only the latest token is fed to the model; the cache provides
     the history.  The cache grows by one entry per step.
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tiny_gpt.config import GenerationSpec, Profile, get_profile, list_profiles
from tiny_gpt.modeling import build_model, build_tokenizer

DEFAULT_PROMPT = """The capital of France is Paris. The capital of Italy is """
CHATML_START_TOKEN = "<|im_start|>"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="scratch_small",
        choices=list_profiles(),
        help="Registered generation profile to run.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
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
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalty for tokens already generated in this response. 1.0 disables it.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling probability mass. 1.0 disables it.",
    )
    return parser.parse_args(argv)


def validate_generation_options(repetition_penalty: float, top_p: float) -> None:
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1.0")


def encode_prompt(tokenizer, prompt: str, spec: GenerationSpec) -> list[int]:
    if isinstance(tokenizer, Tokenizer):
        token_ids = tokenizer.encode(prompt).ids
    else:
        token_ids = tokenizer.encode(prompt)
    if spec.bos_token_id is not None:
        token_ids = [spec.bos_token_id] + token_ids
    return token_ids


def decode_tokens(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids)


def prepare_generation_prompt(prompt: str, profile: Profile) -> str:
    # prompt = prompt.replace("\\n", "\n")
    if profile.name != "sft_dusty8m" or CHATML_START_TOKEN in prompt:
        return prompt

    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def resolve_generation_checkpoint_path(
    profile: Profile,
    checkpoint_step: int | None = None,
) -> Path:
    if profile.generation is None:
        raise ValueError(f"Profile '{profile.name}' does not define generation config")
    if checkpoint_step is None:
        return profile.generation.checkpoint_path
    if checkpoint_step < 1:
        raise ValueError("checkpoint_step must be at least 1")

    final_checkpoint_path = profile.generation.checkpoint_path
    checkpoint_name = f"{final_checkpoint_path.stem}_step_{checkpoint_step}.pt"
    return final_checkpoint_path.parent / checkpoint_name


def load_model(profile: Profile, device=None, checkpoint_step: int | None = None):
    """Build the model, load a checkpoint, and prepare for inference.

    RoPE sin/cos caches are not part of the learned weights, so they are
    dropped from the state dict and recomputed at the required length
    (prompt + max_new_tokens).
    """
    if profile.generation is None:
        raise ValueError(f"Profile '{profile.name}' does not define generation config")

    device = device or get_device()
    tokenizer = build_tokenizer(profile)
    model = build_model(profile)
    checkpoint_path = resolve_generation_checkpoint_path(profile, checkpoint_step)
    print("loading checkpoint from:", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # RoPE caches are derived buffers, not learned parameters.
    state_dict.pop("rope.sin_cache", None)
    state_dict.pop("rope.cos_cache", None)
    model.load_state_dict(state_dict)
    model.rope.resize_cache(
        profile.model.max_seq_len + profile.generation.max_new_tokens
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def apply_repetition_penalty(
    next_token_logits: torch.Tensor,
    generated_ids: list[int],
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty == 1.0 or not generated_ids:
        return next_token_logits

    for token_id in set(generated_ids):
        token_logits = next_token_logits[:, token_id]
        next_token_logits[:, token_id] = torch.where(
            token_logits < 0,
            token_logits * repetition_penalty,
            token_logits / repetition_penalty,
        )
    return next_token_logits


def apply_top_p_filter(next_token_logits: torch.Tensor, top_p: float) -> torch.Tensor:
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
    generated_ids: list[int] | None = None,
    repetition_penalty: float = 1.0,
    top_p: float = 1.0,
):
    """Sample the next token using temperature-scaled top-k sampling.

    Args:
        logits: [B, T, vocab_size] raw model output.
        spec:   generation config (temperature, top_k).

    Returns:
        [B] sampled token IDs.
    """
    validate_generation_options(repetition_penalty, top_p)
    generated_ids = generated_ids or []

    # Step 1: Extract logits for the last position only.
    next_token_logits = logits[:, -1, :]  # [B, vocab_size]

    next_token_logits = apply_repetition_penalty(
        next_token_logits,
        generated_ids=generated_ids,
        repetition_penalty=repetition_penalty,
    )

    # Step 2: Apply temperature — higher temperature → more uniform distribution.
    next_token_logits = next_token_logits / spec.temperature

    # Step 3: Find the top-k highest logit values.
    values, _ = torch.topk(next_token_logits, spec.top_k)  # [B, top_k]

    # Step 4: Zero out everything below the k-th highest value (top-k filtering).
    # values[:, [-1]] is the smallest value in the top-k set.
    next_token_logits[next_token_logits < values[:, [-1]]] = float("-inf")

    next_token_logits = apply_top_p_filter(next_token_logits, top_p=top_p)

    # Step 5: Convert filtered logits to a probability distribution.
    probabilities = torch.softmax(next_token_logits, dim=-1)  # [B, vocab_size]

    # Step 6: Sample one token per batch element from the distribution.
    return torch.multinomial(probabilities, num_samples=1).squeeze(0)  # [B]


def generate_text(
    prompt=DEFAULT_PROMPT,
    profile_name="scratch_small",
    checkpoint_step: int | None = None,
    repetition_penalty: float = 1.0,
    top_p: float = 1.0,
):
    """Generate text autoregressively from a prompt using KV-cached decoding."""
    validate_generation_options(repetition_penalty, top_p)
    profile = get_profile(profile_name)
    if profile.generation is None:
        raise ValueError(f"Profile '{profile.name}' does not define generation config")

    spec = profile.generation
    model, tokenizer, device = load_model(profile, checkpoint_step=checkpoint_step)

    prompt = prepare_generation_prompt(prompt, profile)
    token_ids = encode_prompt(tokenizer, prompt, spec)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = []

    # --- Prefill: process the entire prompt in one forward pass ---
    # The empty cache is a list of None entries (one per layer).  After
    # prefill, kv_cache[i] holds the (key, value) tensors for layer i
    # covering all prompt positions.
    kv_cache = model.empty_kv_cache()
    input_tokens = tokens  # [1, prompt_len]

    print("Predicting...")
    print(prompt, end="", flush=True)
    with torch.inference_mode():
        for _ in range(spec.max_new_tokens):
            # On the first iteration this is the full prompt (prefill).
            # On subsequent iterations this is a single token (decode).
            logits, kv_cache = model(x=input_tokens, kv_cache=kv_cache)
            next_token = sample_next_token(
                logits,
                spec,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
            )
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            # Get the ID for <|im_end|> directly from the tokenizer
            if isinstance(tokenizer, Tokenizer):
                im_end_id = tokenizer.token_to_id("<|im_end|>")
            else:
                im_end_id = tokenizer.encode("<|im_end|>")[0]

            # Stop if we hit the config's EOS token OR the ChatML end token
            if (
                spec.eos_token_id is not None and next_token_id == spec.eos_token_id
            ) or next_token_id == im_end_id:
                print("\n\n[Generation stopped: Stop token detected]")
                break

            # if spec.eos_token_id is not None and next_token_id == spec.eos_token_id:
            #     print(f"\n\n[Generation stopped: eos={spec.eos_token_id} detected]")
            #     break

            tail_text = decode_tokens(tokenizer, generated_ids[-10:])
            if spec.eos_text is not None and spec.eos_text in tail_text:
                print(f"\n\n[Generation stopped: {spec.eos_text} detected]")
                break

            next_word = decode_tokens(tokenizer, next_token.tolist())

            # --- Decode: feed only the new token; the cache has the history ---
            input_tokens = next_token.unsqueeze(0)  # [1, 1]
            print(next_word, end="", flush=True)

    return decode_tokens(tokenizer, generated_ids)


def main(argv=None):
    args = parse_args(argv)
    generate_text(
        prompt=args.prompt,
        profile_name=args.profile,
        checkpoint_step=args.checkpoint_step,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
