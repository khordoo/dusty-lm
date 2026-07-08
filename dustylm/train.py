"""Training loop with mixed-precision, TensorBoard logging, and checkpoint saving.

Supports CUDA (bfloat16), Apple Silicon MPS (float16), and CPU (float32).
The loss is computed on shifted logits/targets so that position ``t`` predicts
token ``t+1``, following the standard causal language modeling objective.
"""

import argparse
import logging
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

from dustylm.config import IGNORE_INDEX, ModelFamily, Profile, get_profile, list_profiles
from dustylm.modeling import build_model
from dustylm.timing import timed_step

logger = logging.getLogger(__name__)


def get_device_and_dtype():
    """Pick the fastest available training device and matching autocast dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def parse_args(argv=None):
    """Parse CLI arguments for running one training profile."""
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
        help=("Override step checkpoint interval. Use 0 to disable step checkpoints for this run."),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the profile's batch_size (default: profile default).",
    )
    return parser.parse_args(argv)


def collate_batch(batch, max_seq_len: int):
    """Pad a batch of variable-length token examples for causal LM training."""

    def get_labels(item):
        return item.get("labels", item["input_ids"])

    input_ids = [torch.tensor(item["input_ids"][:max_seq_len], dtype=torch.long) for item in batch]
    labels = [torch.tensor(get_labels(item)[:max_seq_len], dtype=torch.long) for item in batch]
    return (
        pad_sequence(input_ids, batch_first=True, padding_value=0),
        pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
    )


def get_summary_writer(log_dir):
    """Create a TensorBoard writer under a timestamped run directory."""
    os.makedirs(log_dir, exist_ok=True)
    writer_path = Path(log_dir) / datetime.now().isoformat(timespec="seconds")
    return SummaryWriter(writer_path)


def initialize_random_seed() -> int:
    """Initialize Python, NumPy, and Torch seeds for one training run."""
    current_seed = random.randint(1, 10000)
    print(f"🌱 INITIALIZING WITH RANDOM SEED: {current_seed}")

    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(current_seed)

    return current_seed


def require_training_dataset(profile: Profile) -> None:
    """Fail early when the tokenized dataset for a profile is missing."""
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")

    dataset_path = Path(profile.training.dataset_path)
    if dataset_path.exists():
        return

    hint = ""
    if profile.name == "dusty8m":
        hint = " Run `make data-pretrain` first."
    elif profile.name == "sft_dusty8m":
        hint = " Run `make data-sft` first."
    elif "smollm2" in profile.name:
        hint = (
            f" The '{profile.name}' profile acts as an architecture template. "
            "You must prepare your own dataset and update the path first."
        )
    raise FileNotFoundError(f"Tokenized training dataset not found: {dataset_path}.{hint}")


def load_init_checkpoint_if_configured(model, profile: Profile, device: str):
    """Load a profile's configured starting checkpoint, if one is required."""
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")

    checkpoint_path = profile.training.init_checkpoint_path
    if checkpoint_path is None:
        return False

    if not checkpoint_path.exists():
        hint = ""
        if profile.name == "sft_dusty8m":
            hint = " Run `make train-pretrain` first."
        elif profile.model.family == ModelFamily.SMOLLM2:
            hint = (
                " Run `make download-smollm2` or"
                " `uv run python -m dustylm.artifacts download"
                f" --profile {profile.base_profile or profile.name} --convert`."
            )
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}.{hint}")

    logger.info("Loading initial checkpoint from %s", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return True


def get_step_checkpoint_path(profile: Profile, step: int) -> Path:
    """Return the path used for a profile's numbered step checkpoint."""
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")

    training = profile.training
    checkpoint_dir = training.checkpoint_dir or training.output_checkpoint.parent
    checkpoint_name = f"{training.output_checkpoint.stem}_step_{step}.pt"
    return checkpoint_dir / checkpoint_name


def save_step_checkpoint_if_due(model, profile: Profile, step: int, interval: int | None):
    """Save a numbered checkpoint when the current global step hits the interval."""
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
    batch_size: int | None = None,
):
    """Train one profile and save its final checkpoint.

    The same loop handles both pretraining and SFT. Pretraining examples store
    only ``input_ids``, while SFT examples also store masked ``labels``; the
    collator normalizes both forms into padded tensors.
    """
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
    logger.info("Loading dataset from disk")
    dataset = load_from_disk(str(training.dataset_path))
    logger.info("Loaded %s examples", len(dataset))

    device, dtype = get_device_and_dtype()
    logger.info("Running on device=%s dtype=%s", device, dtype)
    effective_batch_size = batch_size if batch_size is not None else training.batch_size
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=effective_batch_size,
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
    total_batches = len(train_loader)
    import time

    t0 = time.time()
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
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

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Step {batch_idx + 1}/{total_batches} | "
                f"Loss: {loss.item():.4f} | "
                f"Time: {time.time() - t0:.1f}s"
            )
            print()
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
    """CLI entry point for ``python -m dustylm.train``."""
    args = parse_args(argv)
    with timed_step(f"Train {args.profile}"):
        train(
            get_profile(args.profile),
            num_epochs=args.epochs,
            checkpoint_every_steps=args.checkpoint_every_steps,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
