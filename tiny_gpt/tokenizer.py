import os
import tempfile
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

from tiny_gpt.data_prep import normalize_pretrain_text, read_jsonl_sft_rows

# 1. Define paths and our magic vocab size
base = Path(__file__).parents[1]
SFT_DATASET_PATH = base / "artifacts/datasets/dusty_sft.jsonl"
PRETRAIN_DATASET_PATH = base / "artifacts/datasets/dusty_pretrain.txt"
OUTPUT_TOKENIZER_PATH = base / "artifacts/tokenizers/dusty_tokenizer.json"
VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def write_normalized_pretrain_corpus(input_path: Path, output_path: Path) -> None:
    output_path.write_text(normalize_pretrain_text(input_path.read_text()))


def write_normalized_sft_corpus(input_path: Path, output_path: Path) -> None:
    rows = read_jsonl_sft_rows(input_path)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            user = normalize_pretrain_text(str(row["user"]))
            dusty = normalize_pretrain_text(str(row["dusty"]))
            file.write(f"<|im_start|>user\n{user}<|im_end|>\n")
            file.write(f"<|im_start|>assistant\n{dusty}<|im_end|>\n")


def train_tokenizer():
    print(f"🧹 Training ByteLevel BPE Tokenizer (Vocab Size: {VOCAB_SIZE})...")

    # Initialize the built-in ByteLevel Tokenizer
    tokenizer = ByteLevelBPETokenizer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        normalized_pretrain_path = tmp_path / "dusty_pretrain_normalized.txt"
        normalized_sft_path = tmp_path / "dusty_sft_normalized.txt"
        write_normalized_pretrain_corpus(PRETRAIN_DATASET_PATH, normalized_pretrain_path)
        write_normalized_sft_corpus(SFT_DATASET_PATH, normalized_sft_path)

        # Include SFT examples so the chat special tokens and response vocabulary are covered.
        tokenizer.train(
            files=[str(normalized_pretrain_path), str(normalized_sft_path)],
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
        )

    # Save the resulting vocabulary and merges
    os.makedirs(os.path.dirname(OUTPUT_TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(str(OUTPUT_TOKENIZER_PATH))

    print(f"✨ Success! Tokenizer saved to {OUTPUT_TOKENIZER_PATH}")


if __name__ == "__main__":
    train_tokenizer()
