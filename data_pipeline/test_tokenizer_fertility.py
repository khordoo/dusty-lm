# Train BPE tokenizers at various vocab sizes and report fertility scores.

import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

DEFAULT_INPUT_PATH = Path("artifacts/datasets/tinystories_base.txt")
DEFAULT_TRAINING_FILES = [
    Path("artifacts/datasets/tinystories_base.txt"),
]
DEFAULT_LINE_LIMIT = 10_000
DEFAULT_VOCAB_SIZES = [4096, 8192]
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def normalize_text(text: str) -> str:
    return text.lower().replace(";", ".")


def read_sample_lines(path: Path, line_limit: int) -> list[str]:
    if line_limit < 1:
        raise ValueError("line_limit must be at least 1")
    if not path.exists():
        raise FileNotFoundError(f"Input text file not found: {path}")

    lines = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            lines.append(normalize_text(line))
            if len(lines) >= line_limit:
                break

    if not lines:
        raise ValueError(f"No non-empty lines found in {path}")
    return lines


def train_tokenizer(
    training_files: list[Path], vocab_size: int
) -> ByteLevelBPETokenizer:
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(
            f"vocab_size must be at least {len(SPECIAL_TOKENS)} for special tokens"
        )
    missing = [path for path in training_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Tokenizer training file not found: "
            + ", ".join(str(path) for path in missing)
        )

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(path) for path in training_files],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    return tokenizer


def fertility_for_vocab_size(
    lines: list[str],
    vocab_size: int,
    training_files: list[Path],
) -> dict[str, float | int]:
    text = "\n".join(lines)
    word_count = sum(len(line.split()) for line in lines)
    if word_count == 0:
        raise ValueError("Sample has zero whitespace-delimited words")

    tokenizer = train_tokenizer(training_files, vocab_size)

    token_count = len(tokenizer.encode(text).ids)
    return {
        "vocab_size": vocab_size,
        "lines": len(lines),
        "words": word_count,
        "tokens": token_count,
        "fertility": token_count / word_count,
    }


def print_results(results: list[dict[str, float | int]]) -> None:
    print(f"{'vocab_size':>10} {'lines':>8} {'words':>10} {'tokens':>10} {'fertility':>10}")
    for result in results:
        print(
            f"{result['vocab_size']:>10} "
            f"{result['lines']:>8} "
            f"{result['words']:>10} "
            f"{result['tokens']:>10} "
            f"{result['fertility']:>10.3f}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train temporary ByteLevel BPE tokenizers and measure TinyStories "
            "fertility as tokens per whitespace-delimited word."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--lines", type=int, default=DEFAULT_LINE_LIMIT)
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_VOCAB_SIZES,
        help="One or more vocab sizes to test.",
    )
    parser.add_argument(
        "--training-files",
        type=Path,
        nargs="+",
        default=DEFAULT_TRAINING_FILES,
        help="Corpus files used to train each temporary tokenizer.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    lines = read_sample_lines(args.input, args.lines)
    results = [
        fertility_for_vocab_size(lines, vocab_size, args.training_files)
        for vocab_size in args.vocab_sizes
    ]
    print_results(results)


if __name__ == "__main__":
    main()
