import tiktoken
import torch
from config import GENERATION_CONFIG, MODEL_CONFIG, TOKENIZER_NAME
from transformer import TinyGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
vocab_size = tokenizer.max_token_value
model = TinyGPT(
    num_layers=MODEL_CONFIG.num_layers,
    vocab_size=vocab_size,
    max_seq_len=GENERATION_CONFIG.max_seq_len,
    embed_dim=MODEL_CONFIG.embed_dim,
    num_heads=MODEL_CONFIG.num_heads,
    num_kv_heads=MODEL_CONFIG.num_kv_heads,
    rope_base=MODEL_CONFIG.rope_base,
    rms_eps=MODEL_CONFIG.rms_eps,
)
print("loading checkpoint from:", GENERATION_CONFIG.checkpoint_path)
state_dict = torch.load(
    GENERATION_CONFIG.checkpoint_path, map_location=device, weights_only=True
)
model.load_state_dict(state_dict)
model.rope.resize_cache(
    GENERATION_CONFIG.max_seq_len + GENERATION_CONFIG.max_new_tokens
)
model.eval()
prompt = "<|im_start|>user\nWrite a python function to calculate the fibonacci sequence.<|im_end|>\n<|im_start|>assistant\n"
prompt = "<|im_start|>user\nWrite a python function about caoughing<|im_end|>\n<|im_start|>assistant\n"
tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add Batch dim -> [B,T]
print("tokens:", tokens)
print("Predicting...")
eos_sequence = tokenizer.encode(GENERATION_CONFIG.eos_text)
print("EOS_token_id", eos_sequence)
generated_ids = []
with torch.inference_mode():
    kv_cache = model.empty_kv_cache()
    input_tokens = tokens
    for _ in range(GENERATION_CONFIG.max_new_tokens):
        logits, kv_cache = model(x=input_tokens, kv_cache=kv_cache)  # B,T,C
        next_token_logits = logits[:, -1, :]

        # 2. Apply Temperature (e.g., 0.8)
        # Higher (>1.0) = more chaotic/creative, Lower (<1.0) = more strict/confident
        next_token_logits = next_token_logits / GENERATION_CONFIG.temperature

        # 3. Optional: Top-K filtering to prevent complete gibberish
        # Only keep the top 50 most likely tokens, zero out the rest
        v, _ = torch.topk(next_token_logits, GENERATION_CONFIG.top_k)
        # Because v is sorted, that last value is the cutoff threshold (the 50th highest score).
        next_token_logits[next_token_logits < v[:, [-1]]] = float("-inf")
        # print("shape", next_token.shape)
        prop = torch.softmax(next_token_logits, dim=-1)

        next_token = torch.multinomial(prop, num_samples=1).squeeze(0)
        # print("next token:", next_token)
        next_word = tokenizer.decode(next_token.tolist())
        next_token_id = next_token.item()
        generated_ids.append(next_token_id)
        tail_text = tokenizer.decode(generated_ids[-10:])

        # Check if the exact string exists in the tail
        if GENERATION_CONFIG.eos_text in tail_text:
            print("\n\n[Generation stopped: <|im_end|> detected]")
            break

        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
        # dirty trick to cut to max seq lengh to allow to generate modere tokesn that 512
        tokens = tokens[:, -GENERATION_CONFIG.max_seq_len :]
        input_tokens = next_token.unsqueeze(0)
        print(next_word, end="", flush=True)
