import os
import tempfile
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

from tiny_gpt.config import get_profile
from tiny_gpt.data_prep import normalize_pretrain_text, read_jsonl_sft_rows

base = Path(__file__).parents[1]
TOKENIZER_TEXT_CORPORA = [
    base / "artifacts/datasets/tinystories_base.txt",
    base / "artifacts/datasets/dusty_pretrain.txt",
]
TOKENIZER_SFT_JSONL_CORPORA = [
    base / "artifacts/datasets/dusty_sft.jsonl",
]
OUTPUT_TOKENIZER_PATH = base / "artifacts/tokenizers/dusty_tokenizer.json"
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def write_normalized_pretrain_corpus(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Tokenizer text corpus not found: {input_path}")
    output_path.write_text(normalize_pretrain_text(input_path.read_text()))


def write_normalized_sft_corpus(input_path: Path, output_path: Path) -> None:
    rows = read_jsonl_sft_rows(input_path)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            user = normalize_pretrain_text(str(row["user"]))
            dusty = normalize_pretrain_text(str(row["dusty"]))
            file.write(f"<|im_start|>user\n{user}<|im_end|>\n")
            file.write(f"<|im_start|>assistant\n{dusty}<|im_end|>\n")


def prepare_tokenizer_training_files(
    tmp_path: Path,
    text_corpora: list[Path] | None = None,
    sft_jsonl_corpora: list[Path] | None = None,
) -> list[Path]:
    text_corpora = text_corpora or TOKENIZER_TEXT_CORPORA
    sft_jsonl_corpora = sft_jsonl_corpora or TOKENIZER_SFT_JSONL_CORPORA

    training_files = []
    for index, input_path in enumerate(text_corpora):
        output_path = tmp_path / f"text_corpus_{index}.txt"
        write_normalized_pretrain_corpus(input_path, output_path)
        training_files.append(output_path)

    for index, input_path in enumerate(sft_jsonl_corpora):
        output_path = tmp_path / f"sft_corpus_{index}.txt"
        write_normalized_sft_corpus(input_path, output_path)
        training_files.append(output_path)

    return training_files


def train_tokenizer():
    vocab_size = get_profile("dusty8m").model.vocab_size
    print(f"🧹 Training ByteLevel BPE Tokenizer (Vocab Size: {vocab_size})...")

    # Initialize the built-in ByteLevel Tokenizer
    tokenizer = ByteLevelBPETokenizer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_files = prepare_tokenizer_training_files(Path(tmp_dir))

        # Include SFT examples so the chat special tokens and response vocabulary are covered.
        tokenizer.train(
            files=[str(path) for path in training_files],
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
        )

    # Save the resulting vocabulary and merges
    os.makedirs(os.path.dirname(OUTPUT_TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(str(OUTPUT_TOKENIZER_PATH))

    print(f"✨ Success! Tokenizer saved to {OUTPUT_TOKENIZER_PATH}")


if __name__ == "__main__":
    train_tokenizer()
