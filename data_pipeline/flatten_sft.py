import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from dustylm.data_prep import DOCUMENT_SEPARATOR, normalize_pretrain_text
    from dustylm.data_prep import read_jsonl_sft_rows
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dustylm.data_prep import DOCUMENT_SEPARATOR, normalize_pretrain_text
    from dustylm.data_prep import read_jsonl_sft_rows

DEFAULT_INPUT_PATH = Path("artifacts/datasets/dusty_sft.jsonl")
DEFAULT_OUTPUT_PATH = Path("artifacts/datasets/dusty_sft_chatml_pretrain.txt")
DEFAULT_USER_FIELD = "user"
DEFAULT_ASSISTANT_FIELD = "dusty"


def format_chatml_example(user_text: str, assistant_text: str) -> str:
    user_text = normalize_pretrain_text(user_text)
    assistant_text = normalize_pretrain_text(assistant_text)
    return (
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
    )


def row_text(row: dict[str, Any], field: str, line_number: int, input_path: Path) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(
            f"Missing or non-string field '{field}' on row {line_number} of {input_path}"
        )
    return value


def flatten_sft_text(
    input_path: Path,
    output_path: Path,
    user_field: str = DEFAULT_USER_FIELD,
    assistant_field: str = DEFAULT_ASSISTANT_FIELD,
    add_document_separator: bool = True,
) -> int:
    rows = read_jsonl_sft_rows(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for index, row in enumerate(rows, start=1):
            user_text = row_text(row, user_field, index, input_path)
            assistant_text = row_text(row, assistant_field, index, input_path)
            file.write(format_chatml_example(user_text, assistant_text))
            if add_document_separator:
                file.write(DOCUMENT_SEPARATOR)
            file.write("\n")

    return len(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Flatten Dusty SFT JSONL into normalized ChatML text that can be used "
            "as a pretrain corpus."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--user-field", default=DEFAULT_USER_FIELD)
    parser.add_argument("--assistant-field", default=DEFAULT_ASSISTANT_FIELD)
    parser.add_argument(
        "--no-document-separator",
        action="store_true",
        help="Do not append <|endoftext|> after each ChatML conversation.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    row_count = flatten_sft_text(
        input_path=args.input,
        output_path=args.output,
        user_field=args.user_field,
        assistant_field=args.assistant_field,
        add_document_separator=not args.no_document_separator,
    )
    print(f"Wrote {row_count} ChatML pretrain examples to {args.output}")


if __name__ == "__main__":
    main()
