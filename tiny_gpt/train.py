import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tiny_gpt.config import IGNORE_INDEX, Profile, get_profile, list_profiles
from tiny_gpt.modeling import build_model


def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="scratch_small",
        choices=list_profiles(),
        help="Registered training profile to run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs to run.",
    )
    return parser.parse_args(argv)


def collate_batch(batch, max_seq_len: int):
    def get_labels(item):
        return item["labels"] if "labels" in item else item["lables"]

    input_ids = [
        torch.tensor(item["input_ids"][:max_seq_len], dtype=torch.long)
        for item in batch
    ]
    labels = [
        torch.tensor(get_labels(item)[:max_seq_len], dtype=torch.long) for item in batch
    ]
    return (
        pad_sequence(input_ids, batch_first=True, padding_value=0),
        pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
    )


def get_summary_writer(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    writer_path = Path(log_dir) / datetime.now().isoformat(timespec="seconds")
    return SummaryWriter(writer_path)


def train(profile: Profile, num_epochs: int = 1):
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")

    training = profile.training
    print("Loading dataset from disk...")
    dataset = load_from_disk(str(training.dataset_path))
    print(f"Loaded {len(dataset)} examples.")

    device, dtype = get_device_and_dtype()
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=training.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, training.max_seq_len),
    )

    model = build_model(profile, max_seq_len=training.max_seq_len).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    criterion = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    writer = get_summary_writer(training.log_dir)

    global_step = 0
    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs} ---")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            with torch.autocast(
                device_type=device, dtype=dtype, enabled=device != "cpu"
            ):
                logits = model(inputs)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                loss = criterion(
                    shift_logits.view(-1, profile.model.vocab_size),
                    shift_targets.view(-1),
                )

            writer.add_scalar("loss", loss.item(), global_step)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            print(f"Batch step={batch_idx}/{len(train_loader)}, loss={loss.item()}")
            global_step += 1

    writer.close()
    training.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), training.output_checkpoint)
    print(f"model saved to {training.output_checkpoint}")


def main(argv=None):
    args = parse_args(argv)
    train(get_profile(args.profile), num_epochs=args.epochs)


if __name__ == "__main__":
    main()
