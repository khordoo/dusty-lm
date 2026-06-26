"""Training loop with mixed-precision, TensorBoard logging, and checkpoint saving.

Supports CUDA (bfloat16), Apple Silicon MPS (float16), and CPU (float32).
The loss is computed on shifted logits/targets so that position ``t`` predicts
token ``t+1``, following the standard causal language modeling objective.
"""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dustylm.config import IGNORE_INDEX, Profile, get_profile, list_profiles
from dustylm.modeling import build_model


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
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=None,
        help=(
            "Override step checkpoint interval. Use 0 to disable step "
            "checkpoints for this run."
        ),
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


def initialize_random_seed() -> int:
    current_seed = random.randint(1, 10000)
    print(f"🌱 INITIALIZING WITH RANDOM SEED: {current_seed}")

    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(current_seed)

    return current_seed


def require_training_dataset(profile: Profile) -> None:
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")

    dataset_path = Path(profile.training.dataset_path)
    if dataset_path.exists():
        return

    hint = ""
    if profile.name == "dusty8m":
        hint = " Run `make dusty-pretrain-data` first."
    if profile.name == "sft_dusty8m":
        hint = " Run `make dusty-sft-data` first."
    if profile.name == "sft_smollm2_135m":
        hint = " Placeholder profile. Point dataset_path to your own tokenized SFT dataset in dustylm/config.py."
    raise FileNotFoundError(
        f"Tokenized training dataset not found: {dataset_path}.{hint}"
    )


def load_init_checkpoint_if_configured(model, profile: Profile, device: str):
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")

    checkpoint_path = profile.training.init_checkpoint_path
    if checkpoint_path is None:
        return False

    if not checkpoint_path.exists():
        hint = ""
        if profile.name == "sft_dusty8m":
            hint = " Run `make dusty-pretrain` first."
        raise FileNotFoundError(
            f"Initial checkpoint not found: {checkpoint_path}.{hint}"
        )

    print(f"Loading initial checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return True


def get_step_checkpoint_path(profile: Profile, step: int) -> Path:
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")

    training = profile.training
    checkpoint_dir = training.checkpoint_dir or training.output_checkpoint.parent
    checkpoint_name = f"{training.output_checkpoint.stem}_step_{step}.pt"
    return checkpoint_dir / checkpoint_name


def save_step_checkpoint_if_due(model, profile: Profile, step: int, interval: int | None):
    if interval is None or interval <= 0 or step % interval != 0:
        return None

    checkpoint_path = get_step_checkpoint_path(profile, step)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"step checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def train(
    profile: Profile,
    num_epochs: int = 1,
    checkpoint_every_steps: int | None = None,
):
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")
    if checkpoint_every_steps is not None and checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be 0 or greater")

    training = profile.training
    step_checkpoint_interval = (
        checkpoint_every_steps
        if checkpoint_every_steps is not None
        else training.checkpoint_every_steps
    )
    initialize_random_seed()
    require_training_dataset(profile)
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
    load_init_checkpoint_if_configured(model, profile, device)
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
            save_step_checkpoint_if_due(
                model=model,
                profile=profile,
                step=global_step,
                interval=step_checkpoint_interval,
            )

    writer.close()
    training.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), training.output_checkpoint)
    print(f"model saved to {training.output_checkpoint}")


def main(argv=None):
    args = parse_args(argv)
    train(
        get_profile(args.profile),
        num_epochs=args.epochs,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )


if __name__ == "__main__":
    main()
