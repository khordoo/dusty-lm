"""Dataset preparation: plain-text chunking and SFT prompt/response masking.

For pretraining (scratch model), raw text files are concatenated with a
document separator token and sliced into fixed-length chunks.  For supervised
fine-tuning (SFT), prompt tokens are masked with ``IGNORE_INDEX`` so the loss
is computed only on the assistant's response tokens.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value

from dustylm.config import (
    IGNORE_INDEX,
    Profile,
    TrainingTask,
    get_profile,
    list_profiles,
)
from dustylm.modeling import build_tokenizer
from dustylm.timing import timed_step

DEFAULT_PROFILE_NAME = "scratch_small"
DOCUMENT_SEPARATOR = "<|endoftext|>"


def normalize_model_text(raw_text: str) -> str:
    return raw_text.lower().replace(";", ".")


def encode_token_ids(tokenizer, text: str, allowed_special=None) -> list[int]:
    try:
        encoded = tokenizer.encode(text, allowed_special=allowed_special)
    except TypeError:
        encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    return list(encoded)


def require_tokenizer_file(profile: Profile) -> None:
    tokenizer_spec = profile.model.tokenizer
    if tokenizer_spec.kind != "tokenizers":
        return

    path = Path(tokenizer_spec.path_or_name)
    if path.exists():
        return

    hint = ""
    if profile.name in {"dusty8m", "sft_dusty8m"}:
        hint = " Run `make tokenizer` first."
    raise FileNotFoundError(f"Tokenizer file not found: {path}.{hint}")


def prepare_prompt_response_training_example(example, tokenizer=None):
    """Mask prompt tokens so loss is only computed on assistant response tokens."""
    tokenizer = tokenizer or build_tokenizer(get_profile(DEFAULT_PROFILE_NAME))
    return prepare_chatml_sft_training_example(
        example["prompt"],
        example["response"],
        tokenizer,
    )


def prepare_chatml_sft_training_example(user_text: str, assistant_text: str, tokenizer):
    """Format a user/assistant pair as ChatML and mask user-side labels."""
    user_text = normalize_model_text(user_text)
    assistant_text = normalize_model_text(assistant_text)
    prompt_text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    response_text = f"{assistant_text}<|im_end|>"

    prompt_tokens = encode_token_ids(tokenizer, prompt_text, allowed_special="all")
    response_tokens = encode_token_ids(tokenizer, response_text, allowed_special="all")

    return {
        "input_ids": prompt_tokens + response_tokens,
        "labels": [IGNORE_INDEX] * len(prompt_tokens) + response_tokens,
    }


def prepare_training_example(example, tokenizer=None):
    return prepare_prompt_response_training_example(example, tokenizer)


def read_plain_text_documents(raw_text_path: str | Path) -> list[str]:
    path = Path(raw_text_path)
    if path.is_file():
        return [path.read_text()]

    if not path.exists():
        raise FileNotFoundError(f"Raw pretrain text not found: {path}")

    text_files = sorted(file for file in path.rglob("*.txt") if file.is_file())
    if not text_files:
        raise FileNotFoundError(f"No .txt files found under {path}")

    return [file.read_text() for file in text_files]


def iter_plain_text_documents(raw_text_path: str | Path):
    """Yield pretraining text documents without loading the full corpus."""
    path = Path(raw_text_path)
    if path.is_file():
        yield from iter_plain_text_file_documents(path)
        return

    if not path.exists():
        raise FileNotFoundError(f"Raw pretrain text not found: {path}")

    text_files = sorted(file for file in path.rglob("*.txt") if file.is_file())
    if not text_files:
        raise FileNotFoundError(f"No .txt files found under {path}")

    for text_file in text_files:
        yield from iter_plain_text_file_documents(text_file)


def iter_plain_text_file_documents(path: Path):
    """Yield blank-line separated documents from one text file."""
    buffer = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            # TinyStories records are separated by blank lines; stream one story at a time.
            if line.strip():
                buffer.append(line)
                continue
            if buffer:
                yield "".join(buffer).strip()
                buffer = []
    if buffer:
        yield "".join(buffer).strip()


def read_jsonl_sft_rows(raw_sft_path: str | Path) -> list[dict]:
    path = Path(raw_sft_path)
    if not path.exists():
        hint = ""
        if path.name == "dusty_sft.jsonl":
            hint = " Run `make synthesize-sft` first."
        raise FileNotFoundError(f"Raw SFT JSONL not found: {path}.{hint}")

    rows = []
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc

    if not rows:
        raise ValueError(f"No SFT rows found in {path}")
    return rows


def prepare_plain_text_examples(documents: list[str], tokenizer, max_seq_len: int):
    """Yield fixed-length causal LM examples from raw pretraining text.

    This helper is the pure transformation step for pretraining data: it
    normalizes each document, adds the end-of-text separator, tokenizes it, and
    streams fixed-size chunks. Pretraining labels are identical to ``input_ids``,
    so they are created later by the training collator instead of being stored
    as a duplicate dataset column.
    """
    print(f"Max sequence length: {max_seq_len}")
    carry = []
    total_tokens = 0
    total_examples = 0
    for document in documents:
        text = normalize_model_text(document) + DOCUMENT_SEPARATOR
        token_ids = encode_token_ids(
            tokenizer, text, allowed_special={DOCUMENT_SEPARATOR}
        )
        total_tokens += len(token_ids)
        carry.extend(token_ids)
        while len(carry) >= max_seq_len:
            input_ids = carry[:max_seq_len]
            del carry[:max_seq_len]
            total_examples += 1
            yield {"input_ids": input_ids}

    if carry:
        total_examples += 1
        yield {"input_ids": carry}

    print(f"Total token count: {total_tokens:,}")
    print(f"Total examples: {total_examples}")


def prepare_plain_text_examples_from_path(
    raw_text_path: str | Path,
    tokenizer,
    max_seq_len: int,
):
    """Yield pretraining examples from a raw text path without materializing it."""
    yield from prepare_plain_text_examples(
        iter_plain_text_documents(raw_text_path),
        tokenizer,
        max_seq_len,
    )


def prepare_scratch_text_dataset(profile: Profile):
    """Build and save the tokenized pretraining dataset for a profile.

    This is the orchestration step used by ``make data-pretrain``. It validates
    the profile, loads the raw text corpus from disk, builds the configured
    tokenizer, delegates chunk creation to ``prepare_plain_text_examples``, and
    saves the resulting Hugging Face dataset to ``profile.training.dataset_path``.
    """
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")
    if profile.training.raw_text_path is None:
        raise ValueError(f"Profile '{profile.name}' does not define raw_text_path")

    require_tokenizer_file(profile)
    tokenizer = build_tokenizer(profile)
    tokenized_dataset = Dataset.from_generator(
        prepare_plain_text_examples_from_path,
        gen_kwargs={
            "raw_text_path": profile.training.raw_text_path,
            "tokenizer": tokenizer,
            "max_seq_len": profile.training.max_seq_len,
        },
        features=Features({"input_ids": Sequence(Value("int32"))}),
    )
    print(f"Total examples: {len(tokenized_dataset)}")
    print(f"Saving dataset to {profile.training.dataset_path}...")
    tokenized_dataset.save_to_disk(str(profile.training.dataset_path))
    print(f"Ready to train on {len(tokenized_dataset)} text chunks.")


def prepare_jsonl_sft_dataset(profile: Profile):
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")
    if profile.training.raw_sft_path is None:
        raise ValueError(f"Profile '{profile.name}' does not define raw_sft_path")

    require_tokenizer_file(profile)
    rows = read_jsonl_sft_rows(profile.training.raw_sft_path)
    tokenizer = build_tokenizer(profile)

    examples = []
    for index, row in enumerate(rows):
        try:
            user_text = row[profile.training.sft_user_field]
            assistant_text = row[profile.training.sft_assistant_field]
        except KeyError as exc:
            raise KeyError(
                f"SFT row {index} is missing configured field {exc.args[0]!r}"
            ) from exc
        examples.append(
            prepare_chatml_sft_training_example(user_text, assistant_text, tokenizer)
        )

    tokenized_dataset = Dataset.from_list(examples)
    print(f"Saving dataset to {profile.training.dataset_path}...")
    tokenized_dataset.save_to_disk(str(profile.training.dataset_path))
    print(f"Ready to train on {len(tokenized_dataset)} SFT examples.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_NAME,
        choices=list_profiles(),
        help="Registered training profile to prepare.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    profile = get_profile(args.profile)
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")

    with timed_step(f"Prepare tokenized data for {profile.name}"):
        if profile.training.task == TrainingTask.PRETRAIN:
            prepare_scratch_text_dataset(profile)
            return
        if profile.training.task == TrainingTask.SFT:
            prepare_jsonl_sft_dataset(profile)
            return

        raise ValueError(f"Unsupported training task: {profile.training.task}")


if __name__ == "__main__":
    main()
