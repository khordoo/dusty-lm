import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# 1. Define paths and our magic vocab size
base = Path(__file__).parents[1]
SFT_DATASET_PATH = base / "artifacts/datasets/dusty_sft.jsonl"
PRETRAIN_DATASET_PATH = base / "artifacts/datasets/dusty_pretrain.txt"
OUTPUT_TOKENIZER_PATH = base / "artifacts/tokenizers/dusty_tokenizer.json"
VOCAB_SIZE = 4096


def train_tokenizer():
    print(f"🧹 Training Custom BPE Tokenizer (Vocab Size: {VOCAB_SIZE})...")

    if not os.path.exists(PRETRAIN_DATASET_PATH):
        raise FileNotFoundError(
            f"Could not find dataset at {PRETRAIN_DATASET_PATH}. Run generation script first!"
        )

    # 2. Initialize a blank Byte-Pair Encoding model
    # We include the unk_token (Unknown) to handle any weird characters
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # 3. Use standard whitespace splitting before BPE kicks in
    tokenizer.pre_tokenizer = Whitespace()

    # 4. Configure the Trainer
    # We define our Special Tokens here.
    # <|endoftext|> will be token ID 0, <|unk|> will be token ID 1
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>", "<|unk|>"],
        show_progress=True,
    )

    # 5. Train the tokenizer purely on Dusty's diary entries!
    tokenizer.train(
        files=[str(PRETRAIN_DATASET_PATH), str(SFT_DATASET_PATH)], trainer=trainer
    )

    # 6. Save the resulting vocabulary and merge rules
    os.makedirs(os.path.dirname(OUTPUT_TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(str(OUTPUT_TOKENIZER_PATH))

    print(f"✨ Success! Tokenizer saved to {OUTPUT_TOKENIZER_PATH}")
    print("Test it by running: tokenizer.encode('I eat the dust.').tokens")


if __name__ == "__main__":
    train_tokenizer()
