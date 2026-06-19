import argparse

import torch
from tokenizers import Tokenizer

from tiny_gpt.config import GenerationSpec, Profile, get_profile, list_profiles
from tiny_gpt.modeling import build_model, build_tokenizer

# DEFAULT_PROMPT = (
#     "<|im_start|>user\n"
#     "Write a python function about coughing"
#     "<|im_end|>\n"
#     "<|im_start|>assistant\n"
# )
# DEFAULT_PROMPT = (
#     "<|im_start|>user\n"
#     "The capital of France is Paris. The capital of Italy is"
#     "<|im_end|>\n"
#     "<|im_start|>assistant\n"
# )
DEFAULT_PROMPT = """The capital of France is Paris. The capital of Italy is """


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
    return parser.parse_args(argv)


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


def load_model(profile: Profile, device=None):
    if profile.generation is None:
        raise ValueError(f"Profile '{profile.name}' does not define generation config")

    device = device or get_device()
    tokenizer = build_tokenizer(profile)
    model = build_model(profile)
    print("loading checkpoint from:", profile.generation.checkpoint_path)
    state_dict = torch.load(
        profile.generation.checkpoint_path, map_location=device, weights_only=True
    )
    state_dict.pop("rope.sin_cache", None)
    state_dict.pop("rope.cos_cache", None)
    model.load_state_dict(state_dict)
    model.rope.resize_cache(
        profile.model.max_seq_len + profile.generation.max_new_tokens
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def sample_next_token(logits, spec: GenerationSpec):
    next_token_logits = logits[:, -1, :] / spec.temperature
    values, _ = torch.topk(next_token_logits, spec.top_k)
    next_token_logits[next_token_logits < values[:, [-1]]] = float("-inf")
    probabilities = torch.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(0)


def generate_text(prompt=DEFAULT_PROMPT, profile_name="scratch_small"):
    profile = get_profile(profile_name)
    if profile.generation is None:
        raise ValueError(f"Profile '{profile.name}' does not define generation config")

    spec = profile.generation
    model, tokenizer, device = load_model(profile)
    token_ids = encode_prompt(tokenizer, prompt, spec)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = []
    kv_cache = model.empty_kv_cache()
    input_tokens = tokens

    print("tokens:", tokens)
    print("Predicting...")

    with torch.inference_mode():
        for _ in range(spec.max_new_tokens):
            logits, kv_cache = model(x=input_tokens, kv_cache=kv_cache)
            next_token = sample_next_token(logits, spec)
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            if spec.eos_token_id is not None and next_token_id == spec.eos_token_id:
                print(f"\n\n[Generation stopped: eos={spec.eos_token_id} detected]")
                break

            tail_text = decode_tokens(tokenizer, generated_ids[-10:])
            if spec.eos_text is not None and spec.eos_text in tail_text:
                print(f"\n\n[Generation stopped: {spec.eos_text} detected]")
                break

            next_word = decode_tokens(tokenizer, next_token.tolist())
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
            tokens = tokens[:, -profile.model.max_seq_len :]
            input_tokens = next_token.unsqueeze(0)
            print(next_word, end="", flush=True)

    return decode_tokens(tokenizer, generated_ids)


def main(argv=None):
    args = parse_args(argv)
    generate_text(prompt=args.prompt, profile_name=args.profile)


if __name__ == "__main__":
    main()
