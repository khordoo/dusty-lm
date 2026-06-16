import time
from pathlib import Path

import tiktoken
import torch
from transformer import TinyGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


current = Path(__file__).parents[1]

checkpoint_path = Path(__file__).parents[1] / "checkpoints" / "tinygpt_epoch_1.pt"
MAX_SEQ_LEN = 512
embed_dim = 512
max_seq_len = 256
head_dim = 64
num_heads = 8
num_kv_heads = 2
num_layers = 6
model_save_path = "tinygpt_epoch_1.pt"
log_dir = "runs/"
tokenizer = tiktoken.get_encoding("r50k_base")
vocab_size = tiktoken.get_encoding("r50k_base").max_token_value
model = TinyGPT(
    num_layers=num_layers,
    vocab_size=vocab_size,
    max_seq_len=MAX_SEQ_LEN,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
)
print("loading checkpoint from:", checkpoint_path)
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
prompt = "<|im_start|>user\nWrite a python function to calculate the fibonacci sequence.<|im_end|>\n<|im_start|>assistant\n"
prompt = "<|im_start|>user\nWrite a python function about caoughing<|im_end|>\n<|im_start|>assistant\n"
tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add Batch dim -> [B,T]
print("tokens:", tokens)
print("Predicting...")
words = []
EOS_IM = "<|im_end|>"
eos_sequence = tokenizer.encode(
    EOS_IM
)  # limitation og gpt-2 tokenizer it does not assing unique id to this eos
eos_length = len(eos_sequence)
print("EOS_token_id", eos_sequence)
generated_ids = []
with torch.inference_mode():
    max_egenration = 1000
    temperature = 1
    top_k = 10
    for _ in range(max_egenration):
        t0 = time.time_ns()
        logits = model(tokens)  # B,T,C
        next_token_logits = logits[:, -1, :]

        # 2. Apply Temperature (e.g., 0.8)
        # Higher (>1.0) = more chaotic/creative, Lower (<1.0) = more strict/confident
        next_token_logits = next_token_logits / temperature

        # 3. Optional: Top-K filtering to prevent complete gibberish
        # Only keep the top 50 most likely tokens, zero out the rest
        v, _ = torch.topk(next_token_logits, top_k)
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
        if "<|im_end|>" in tail_text:
            print("\n\n[Generation stopped: <|im_end|> detected]")
            break
        print(f"Time={time.time_ns() - t0}: Tokens={tokens.shape[-1]}")
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
        # dirty trick to cut to max seq lengh to allow to generate modere tokesn that 512
        tokens = tokens[:, -MAX_SEQ_LEN:]
        print(next_word, end="", flush=True)
