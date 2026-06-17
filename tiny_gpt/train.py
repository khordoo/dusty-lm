import os
from datetime import datetime
from pathlib import Path

import tiktoken
import torch
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tiny_gpt.config import IGNORE_INDEX, MODEL_CONFIG, TOKENIZER_NAME, TRAINING_CONFIG
from tiny_gpt.model import TinyGPT


def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def collate_batch(batch):
    def get_labels(item):
        return item["labels"] if "labels" in item else item["lables"]

    input_ids = [
        torch.tensor(item["input_ids"][: TRAINING_CONFIG.max_seq_len], dtype=torch.long)
        for item in batch
    ]
    labels = [
        torch.tensor(get_labels(item)[: TRAINING_CONFIG.max_seq_len], dtype=torch.long)
        for item in batch
    ]
    return (
        pad_sequence(input_ids, batch_first=True, padding_value=0),
        pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
    )


def build_model(vocab_size: int):
    return TinyGPT(
        num_layers=MODEL_CONFIG.num_layers,
        vocab_size=vocab_size,
        max_seq_len=TRAINING_CONFIG.max_seq_len,
        embed_dim=MODEL_CONFIG.embed_dim,
        num_heads=MODEL_CONFIG.num_heads,
        num_kv_heads=MODEL_CONFIG.num_kv_heads,
        rope_base=MODEL_CONFIG.rope_base,
        rms_eps=MODEL_CONFIG.rms_eps,
    )


def get_summary_writer(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    writer_path = Path(log_dir) / datetime.now().isoformat(timespec="seconds")
    return SummaryWriter(writer_path)


def train():
    print("Loading dataset from disk...")
    dataset = load_from_disk(TRAINING_CONFIG.dataset_path)
    print(f"Loaded {len(dataset)} examples.")

    device, dtype = get_device_and_dtype()
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=TRAINING_CONFIG.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    vocab_size = tokenizer.max_token_value
    model = build_model(vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=TRAINING_CONFIG.learning_rate)
    criterion = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    writer = get_summary_writer(TRAINING_CONFIG.log_dir)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=dtype, enabled=device != "cpu"):
            logits = model(inputs)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, vocab_size), shift_targets.view(-1))

        writer.add_scalar("loss", loss.item(), batch_idx)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        print(f"Batch step={batch_idx}/{len(train_loader)}, loss={loss.item()}")

    writer.close()
    TRAINING_CONFIG.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), TRAINING_CONFIG.checkpoint_path)
    print(f"model saved to {TRAINING_CONFIG.checkpoint_path}")


def main():
    train()


if __name__ == "__main__":
    main()
