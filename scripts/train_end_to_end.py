"""Run the DustyLM training pipeline with overridable golden-path defaults.

This script is the terminal-friendly companion to the guided training
notebook. It downloads the raw datasets, trains the tokenizer, prepares
tokenized datasets, trains the base model, optionally promotes a selected base
step checkpoint, then trains SFT and optionally promotes a selected SFT step
checkpoint.

By default, the script keeps the final checkpoints saved by each training run.
You can pass ``--best-pretrain-step`` or ``--best-sft-step`` to promote a
specific step checkpoint instead. If you change dataset size, epochs, batch
size, or checkpoint interval, make sure the selected step checkpoints are
actually created during training.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.download_datasets import (
    DEFAULT_DUSTY_CHAT_FILE,
    DEFAULT_DUSTY_CHAT_REPO,
    DEFAULT_DUSTY_SFT_OUT,
    DEFAULT_TINYSTORIES_OUT,
    DEFAULT_TINYSTORIES_SLICE,
    download_dusty_sft,
    download_tinystories,
)
from dustylm.config import TrainingTask, get_profile, list_profiles
from dustylm.data_prep import prepare_jsonl_sft_dataset, prepare_scratch_text_dataset
from dustylm.timing import timed_step
from dustylm.tokenizer import train_tokenizer
from dustylm.train import get_step_checkpoint_path, train

DEFAULT_PRETRAIN_PROFILE = "dusty8m"
DEFAULT_SFT_PROFILE = "sft_dusty8m"
DEFAULT_PRETRAIN_EPOCHS = 1
DEFAULT_SFT_EPOCHS = 2
DEFAULT_PRETRAIN_BATCH_SIZE = 224
DEFAULT_SFT_BATCH_SIZE = 224
DEFAULT_CHECKPOINT_EVERY_STEPS = 50
DEFAULT_BEST_PRETRAIN_STEP = 0
DEFAULT_BEST_SFT_STEP = 0


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run DustyLM download, tokenization, pretraining, and SFT."
    )
    parser.add_argument(
        "--tinystories-slice",
        default=DEFAULT_TINYSTORIES_SLICE,
        help="TinyStories split slice to download.",
    )
    parser.add_argument(
        "--tinystories-out",
        type=Path,
        default=DEFAULT_TINYSTORIES_OUT,
        help="Output path for the TinyStories text corpus.",
    )
    parser.add_argument(
        "--dusty-chat-repo",
        default=DEFAULT_DUSTY_CHAT_REPO,
        help="Hugging Face dataset repo containing Dusty chat data.",
    )
    parser.add_argument(
        "--dusty-chat-file",
        default=DEFAULT_DUSTY_CHAT_FILE,
        help="Dusty chat filename in the source dataset repo.",
    )
    parser.add_argument(
        "--dusty-sft-out",
        type=Path,
        default=DEFAULT_DUSTY_SFT_OUT,
        help="Output path for the Dusty SFT JSONL file.",
    )
    parser.add_argument(
        "--pretrain-profile",
        default=DEFAULT_PRETRAIN_PROFILE,
        choices=list_profiles(),
        help="Profile used for base pretraining.",
    )
    parser.add_argument(
        "--sft-profile",
        default=DEFAULT_SFT_PROFILE,
        choices=list_profiles(),
        help="Profile used for SFT.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=DEFAULT_PRETRAIN_EPOCHS,
        help="Number of pretraining epochs to run.",
    )
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=DEFAULT_SFT_EPOCHS,
        help="Number of SFT epochs to run.",
    )
    parser.add_argument(
        "--pretrain-batch-size",
        type=int,
        default=DEFAULT_PRETRAIN_BATCH_SIZE,
        help="Pretraining batch size override.",
    )
    parser.add_argument(
        "--sft-batch-size",
        type=int,
        default=DEFAULT_SFT_BATCH_SIZE,
        help="SFT batch size override.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_STEPS,
        help="Step checkpoint interval for both training phases.",
    )
    parser.add_argument(
        "--best-pretrain-step",
        type=int,
        default=DEFAULT_BEST_PRETRAIN_STEP,
        help=(
            "Pretraining step checkpoint to promote before SFT. Defaults to 0, "
            "which keeps the final epoch checkpoint. If you choose a specific "
            "step, make sure that checkpoint exists."
        ),
    )
    parser.add_argument(
        "--best-sft-step",
        type=int,
        default=DEFAULT_BEST_SFT_STEP,
        help=(
            "SFT step checkpoint to promote after SFT. Defaults to 0, which "
            "keeps the final epoch checkpoint. If you choose a specific step, "
            "make sure that checkpoint exists."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip raw dataset download and use existing files.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip tokenizer training and use the existing tokenizer.",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip tokenized dataset preparation and use existing tokenized datasets.",
    )
    parser.add_argument(
        "--reuse-tokenized-data",
        action="store_true",
        help="Do not clear existing tokenized datasets before preparing data.",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip base pretraining and use the existing base checkpoint.",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip SFT training and use the existing SFT checkpoint.",
    )
    return parser.parse_args(argv)


def remove_tokenized_dataset(profile_name: str) -> None:
    profile = get_profile(profile_name)
    if profile.training is None:
        raise ValueError(f"Profile '{profile_name}' does not define training config")

    dataset_path = Path(profile.training.dataset_path)
    if dataset_path.exists():
        print(f"Removing existing tokenized dataset: {dataset_path}")
        shutil.rmtree(dataset_path)


def prepare_dataset_for_profile(profile_name: str) -> None:
    profile = get_profile(profile_name)
    if profile.training is None:
        raise ValueError(f"Profile '{profile_name}' does not define training config")

    if profile.training.task == TrainingTask.PRETRAIN:
        prepare_scratch_text_dataset(profile)
        return
    if profile.training.task == TrainingTask.SFT:
        prepare_jsonl_sft_dataset(profile)
        return

    raise ValueError(f"Unsupported training task: {profile.training.task}")


def promote_step_checkpoint(profile_name: str, step: int) -> Path | None:
    if step <= 0:
        print(f"Keeping final checkpoint for profile '{profile_name}'.")
        return None

    profile = get_profile(profile_name)
    if profile.training is None:
        raise ValueError(f"Profile '{profile_name}' does not define training config")

    src = get_step_checkpoint_path(profile, step)
    dst = profile.training.output_checkpoint
    if not src.exists():
        raise FileNotFoundError(
            f"Cannot promote step {step}; checkpoint not found: {src}\n"
            "Increase the training epochs, lower --checkpoint-every-steps, "
            "or choose a step checkpoint that exists."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"Promoted {src} -> {dst} ({size_mb:.1f} MB)")
    return dst


def run_pipeline(args) -> None:
    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pretrain_promote = args.best_pretrain_step or "final"
    sft_promote = args.best_sft_step or "final"
    print(f"Backend:  device={device}, precision={dtype}")
    print("Starting DustyLM end-to-end training pipeline.")
    print(f"Pretrain: {args.pretrain_profile}, epochs={args.pretrain_epochs}")
    print(f"SFT:      {args.sft_profile}, epochs={args.sft_epochs}")
    print(f"Promote:  pretrain={pretrain_promote}, sft={sft_promote}")

    with timed_step("Download raw datasets", stage="1/7"):
        if args.skip_download:
            print("Skipping download; using existing raw dataset files.")
        else:
            download_tinystories(args.tinystories_slice, args.tinystories_out)
            download_dusty_sft(
                args.dusty_chat_repo,
                args.dusty_chat_file,
                args.dusty_sft_out,
            )

    with timed_step("Train tokenizer", stage="2/7"):
        if args.skip_tokenizer:
            print("Skipping tokenizer training; using existing tokenizer.")
        else:
            train_tokenizer()

    with timed_step("Prepare tokenized datasets", stage="3/7"):
        if args.skip_data_prep:
            print("Skipping data preparation; using existing tokenized datasets.")
        else:
            if not args.reuse_tokenized_data:
                remove_tokenized_dataset(args.pretrain_profile)
                remove_tokenized_dataset(args.sft_profile)
            prepare_dataset_for_profile(args.pretrain_profile)
            prepare_dataset_for_profile(args.sft_profile)

    with timed_step("Run base pretraining", stage="4/7"):
        if args.skip_pretrain:
            print("Skipping pretraining; using existing base checkpoint.")
        else:
            train(
                get_profile(args.pretrain_profile),
                num_epochs=args.pretrain_epochs,
                checkpoint_every_steps=args.checkpoint_every_steps,
                batch_size=args.pretrain_batch_size,
            )

    with timed_step("Promote selected base checkpoint", stage="5/7"):
        promote_step_checkpoint(args.pretrain_profile, args.best_pretrain_step)

    with timed_step("Run SFT", stage="6/7"):
        if args.skip_sft:
            print("Skipping SFT; using existing SFT checkpoint.")
        else:
            train(
                get_profile(args.sft_profile),
                num_epochs=args.sft_epochs,
                checkpoint_every_steps=args.checkpoint_every_steps,
                batch_size=args.sft_batch_size,
            )

    with timed_step("Promote selected SFT checkpoint", stage="7/7"):
        promote_step_checkpoint(args.sft_profile, args.best_sft_step)
    print()
    print("End-to-end pipeline complete.")


def main(argv: list[str] | None = None) -> None:
    run_pipeline(parse_args(argv))


if __name__ == "__main__":
    main()
