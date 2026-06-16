import os
from pathlib import Path

import tiktoken
import torch
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformer import TinyGPT

print("Loading dataset from disk...")
dataset = load_from_disk("data/tiny_codes_python_tokenized")
print(f"Loaded {len(dataset)} examples in milliseconds!")
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16  # M1 does not support bf16
else:
    DEVICE = "mps"
    DTYPE = torch.float32

MAX_SEQ_LEN = 1024


def collect_fn(batch):
    print("len batch:", len(batch))
    print(len(batch[0]["input_ids"]))
    input_ids = [
        torch.tensor(item["input_ids"][:MAX_SEQ_LEN], dtype=torch.long)
        for item in batch
    ]

    lables = [
        torch.tensor(item["lables"][:MAX_SEQ_LEN], dtype=torch.long) for item in batch
    ]

    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=0)
    print("padded inpiuts shape:", padded_inputs.shape)
    padded_lables = pad_sequence(lables, batch_first=True, padding_value=0)
    # padded_inputs = padded_inputs.to(DEVICE)
    # padded_lables = padded_lables.to(DEVICE)
    return padded_inputs, padded_lables


# B = 18
# T = 2
embed_dim = 512
max_seq_len = 256
head_dim = 64
num_heads = 8
num_kv_heads = 2
num_layers = 6
learning_rate = 1e-4
batch_size = 8
model_save_path = "tinygpt_epoch_1.pt"
log_dir = "runs/"
# import tiktoken

train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collect_fn
)

vocab_size = tiktoken.get_encoding("r50k_base").max_token_value
model = TinyGPT(
    num_layers=num_layers,
    vocab_size=vocab_size,
    max_seq_len=MAX_SEQ_LEN,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
)
model = model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=learning_rate)
critetion = CrossEntropyLoss()
scaler = torch.amp.GradScaler("mps")

scaler_enanled = DEVICE != "cuda"
TOTAL_BATCH_SIZE = len(train_loader)


def get_summary_writer(log_dir):
    from datetime import datetime

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print("Tensor board folder created:", log_dir)
    wroter_fill_path = Path(log_dir) / datetime.now().isoformat(timespec="seconds")
    writer = SummaryWriter(wroter_fill_path)
    return writer


writer = get_summary_writer(log_dir)
for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.to(DEVICE)
    targets = targets.to(DEVICE)
    optimizer.zero_grad()
    # Run the forward pass in 16-bit precision
    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        logits = model(inputs)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        loss = critetion(shift_logits.view(-1, vocab_size), shift_targets.view(-1))
        writer.add_scalar("loss", loss.item(), batch_idx)
    if scaler_enanled:
        print("scaler enabled")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        print("scale skipped")
        loss.backward()
        optimizer.step()
    print(f"Batch step= {batch_idx}/{TOTAL_BATCH_SIZE} , loss={loss.item()}")
writer.close()
print("model saved")
torch.save(model.state_dict(), model_save_path)
