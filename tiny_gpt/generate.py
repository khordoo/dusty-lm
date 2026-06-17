import tiktoken
import torch

from tiny_gpt.config import GENERATION_CONFIG, MODEL_CONFIG, TOKENIZER_NAME
from tiny_gpt.model import TinyGPT


DEFAULT_PROMPT = (
    "<|im_start|>user\n"
    "Write a python function about coughing"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(vocab_size: int):
    return TinyGPT(
        num_layers=MODEL_CONFIG.num_layers,
        vocab_size=vocab_size,
        max_seq_len=GENERATION_CONFIG.max_seq_len,
        embed_dim=MODEL_CONFIG.embed_dim,
        num_heads=MODEL_CONFIG.num_heads,
        num_kv_heads=MODEL_CONFIG.num_kv_heads,
        rope_base=MODEL_CONFIG.rope_base,
        rms_eps=MODEL_CONFIG.rms_eps,
    )


def load_model(device=None):
    device = device or get_device()
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    model = build_model(tokenizer.max_token_value)
    print("loading checkpoint from:", GENERATION_CONFIG.checkpoint_path)
    state_dict = torch.load(
        GENERATION_CONFIG.checkpoint_path, map_location=device, weights_only=True
    )
    state_dict.pop("rope.sin_cache", None)
    state_dict.pop("rope.cos_cache", None)
    model.load_state_dict(state_dict)
    model.rope.resize_cache(
        GENERATION_CONFIG.max_seq_len + GENERATION_CONFIG.max_new_tokens
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def sample_next_token(logits):
    next_token_logits = logits[:, -1, :] / GENERATION_CONFIG.temperature
    values, _ = torch.topk(next_token_logits, GENERATION_CONFIG.top_k)
    next_token_logits[next_token_logits < values[:, [-1]]] = float("-inf")
    probabilities = torch.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(0)


def generate_text(prompt=DEFAULT_PROMPT):
    model, tokenizer, device = load_model()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = []
    kv_cache = model.empty_kv_cache()
    input_tokens = tokens

    print("tokens:", tokens)
    print("Predicting...")
    print("EOS_token_id", tokenizer.encode(GENERATION_CONFIG.eos_text))

    with torch.inference_mode():
        for _ in range(GENERATION_CONFIG.max_new_tokens):
            logits, kv_cache = model(x=input_tokens, kv_cache=kv_cache)
            next_token = sample_next_token(logits)
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            tail_text = tokenizer.decode(generated_ids[-10:])
            if GENERATION_CONFIG.eos_text in tail_text:
                print("\n\n[Generation stopped: <|im_end|> detected]")
                break

            next_word = tokenizer.decode(next_token.tolist())
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
            tokens = tokens[:, -GENERATION_CONFIG.max_seq_len :]
            input_tokens = next_token.unsqueeze(0)
            print(next_word, end="", flush=True)

    return tokenizer.decode(generated_ids)


def main():
    generate_text()


if __name__ == "__main__":
    main()
