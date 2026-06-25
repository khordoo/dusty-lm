# Downloads public training datasets used by the Dusty quick-start flow.

import argparse
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DEFAULT_TINYSTORIES_OUT = Path("artifacts/datasets/tinystories_base.txt")
DEFAULT_DUSTY_PRETRAIN_OUT = Path("artifacts/datasets/dusty_pretrain.txt")
DEFAULT_DUSTY_SFT_OUT = Path("artifacts/datasets/dusty_sft.jsonl")
DEFAULT_TINYSTORIES_SLICE = "train[:100000]"
DEFAULT_DUSTY_CHAT_REPO = "mkhordoo/dusty-chat"
DEFAULT_DUSTY_CHAT_FILE = "dusty_sft.jsonl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tinystories-slice",
        default=DEFAULT_TINYSTORIES_SLICE,
        help="Hugging Face dataset split slice to download for TinyStories.",
    )
    parser.add_argument(
        "--tinystories-out",
        type=Path,
        default=DEFAULT_TINYSTORIES_OUT,
        help="Output path for the downloaded TinyStories text corpus.",
    )
    parser.add_argument(
        "--dusty-pretrain-out",
        type=Path,
        default=DEFAULT_DUSTY_PRETRAIN_OUT,
        help="Output path used by the dusty8m pretraining profile.",
    )
    parser.add_argument(
        "--dusty-chat-repo",
        default=DEFAULT_DUSTY_CHAT_REPO,
        help="Hugging Face repo containing Dusty chat SFT data.",
    )
    parser.add_argument(
        "--dusty-chat-file",
        default=DEFAULT_DUSTY_CHAT_FILE,
        help="SFT JSONL filename inside the Dusty chat repo.",
    )
    parser.add_argument(
        "--dusty-sft-out",
        type=Path,
        default=DEFAULT_DUSTY_SFT_OUT,
        help="Output path for Dusty SFT JSONL.",
    )
    return parser.parse_args(argv)


def download_tinystories(split: str, output_path: Path) -> None:
    print(f"Downloading TinyStories split {split}...")
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for item in dataset:
            file.write(item["text"].strip())
            file.write("\n\n")
    print(f"TinyStories saved to {output_path}")


def copy_tinystories_to_dusty_pretrain(
    tinystories_path: Path,
    dusty_pretrain_path: Path,
) -> None:
    dusty_pretrain_path.parent.mkdir(parents=True, exist_ok=True)
    if tinystories_path.resolve() == dusty_pretrain_path.resolve():
        return
    shutil.copyfile(tinystories_path, dusty_pretrain_path)
    print(f"Pretraining corpus saved to {dusty_pretrain_path}")


def download_dusty_sft(repo_id: str, filename: str, output_path: Path) -> None:
    print(f"Downloading Dusty chat SFT data from {repo_id}/{filename}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_path.parent,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {filename} from {repo_id}. "
            "Generate the dataset locally with `make dusty-generate-sft` "
            "or pass --dusty-chat-repo to a repo you can access."
        ) from exc

    downloaded_path = Path(downloaded_path)
    if downloaded_path.resolve() != output_path.resolve():
        downloaded_path.replace(output_path)
    print(f"Dusty SFT data saved to {output_path}")


def main(argv=None):
    args = parse_args(argv)
    download_tinystories(args.tinystories_slice, args.tinystories_out)
    copy_tinystories_to_dusty_pretrain(
        args.tinystories_out,
        args.dusty_pretrain_out,
    )
    download_dusty_sft(
        args.dusty_chat_repo,
        args.dusty_chat_file,
        args.dusty_sft_out,
    )
    print("Datasets are ready. Next: make dusty-tokenizer")


if __name__ == "__main__":
    main()
