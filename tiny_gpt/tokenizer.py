import os
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

# 1. Define paths and our magic vocab size
base = Path(__file__).parents[1]
SFT_DATASET_PATH = base / "artifacts/datasets/dusty_sft.jsonl"
PRETRAIN_DATASET_PATH = base / "artifacts/datasets/dusty_pretrain.txt"
OUTPUT_TOKENIZER_PATH = base / "artifacts/tokenizers/dusty_tokenizer.json"
VOCAB_SIZE = 4096


def train_tokenizer():
    print(f"🧹 Training ByteLevel BPE Tokenizer (Vocab Size: {VOCAB_SIZE})...")

    # Initialize the built-in ByteLevel Tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Include SFT examples so the chat special tokens and response vocabulary are covered.
    tokenizer.train(
        files=[str(PRETRAIN_DATASET_PATH), str(SFT_DATASET_PATH)],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    )

    # Save the resulting vocabulary and merges
    os.makedirs(os.path.dirname(OUTPUT_TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(str(OUTPUT_TOKENIZER_PATH))

    print(f"✨ Success! Tokenizer saved to {OUTPUT_TOKENIZER_PATH}")


if __name__ == "__main__":
    train_tokenizer()
