import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import login

from tiny_gpt.config import (
    IGNORE_INDEX,
    ModelFamily,
    Profile,
    get_profile,
    list_profiles,
)
from tiny_gpt.modeling import build_tokenizer

DEFAULT_PROFILE_NAME = "scratch_small"
DOCUMENT_SEPARATOR = "<|endoftext|>"


def prepare_prompt_response_training_example(example, tokenizer=None):
    """Mask prompt tokens so loss is only computed on assistant response tokens."""
    tokenizer = tokenizer or build_tokenizer(get_profile(DEFAULT_PROFILE_NAME))
    prompt_text = (
        f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    )
    response_text = f"{example['response']}<|im_end|>"

    prompt_tokens = tokenizer.encode(prompt_text, allowed_special="all")
    response_tokens = tokenizer.encode(response_text, allowed_special="all")

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

    text_files = sorted(file for file in path.rglob("*.txt") if file.is_file())
    if not text_files:
        raise FileNotFoundError(f"No .txt files found under {path}")

    return [file.read_text() for file in text_files]


def prepare_plain_text_examples(documents: list[str], tokenizer, max_seq_len: int):
    joined_text = DOCUMENT_SEPARATOR.join(documents)
    token_ids = tokenizer.encode(joined_text, allowed_special={DOCUMENT_SEPARATOR})
    print("total leng token ids:", len(token_ids))
    examples = []
    print("max seq lengh:", max_seq_len)
    for start in range(0, len(token_ids), max_seq_len):
        input_ids = token_ids[start : start + max_seq_len]
        if input_ids:
            examples.append({"input_ids": input_ids, "labels": input_ids.copy()})
    print("total examples:", len(examples))
    return examples


def prepare_scratch_text_dataset(profile: Profile):
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")
    if profile.training.raw_text_path is None:
        raise ValueError(f"Profile '{profile.name}' does not define raw_text_path")

    documents = read_plain_text_documents(profile.training.raw_text_path)
    tokenizer = build_tokenizer(profile)
    examples = prepare_plain_text_examples(
        documents,
        tokenizer,
        profile.training.max_seq_len,
    )
    tokenized_dataset = Dataset.from_list(examples)
    print(f"Saving dataset to {profile.training.dataset_path}...")
    tokenized_dataset.save_to_disk(str(profile.training.dataset_path))
    print(f"Ready to train on {len(tokenized_dataset)} text chunks.")


def prepare_tiny_codes_sft_dataset(profile: Profile):
    if profile.training is None:
        raise ValueError(f"Profile '{profile.name}' does not define training config")

    print("Launching Hugging Face login. Paste your token when prompted.")
    login()
    print("Downloading tiny-codes dataset...")

    dataset = load_dataset("nampdn-ai/tiny-codes", split="train")
    python_dataset = dataset.filter(lambda x: x["programming_language"] == "Python")
    python_dataset.save_to_disk(str(profile.training.raw_python_dataset_path))

    tokenizer = build_tokenizer(profile)
    tokenized_dataset = python_dataset.map(
        lambda example: prepare_prompt_response_training_example(example, tokenizer),
        remove_columns=python_dataset.column_names,
    )
    print(f"Saving dataset to {profile.training.dataset_path}...")
    tokenized_dataset.save_to_disk(str(profile.training.dataset_path))
    print(f"Ready to train on {len(tokenized_dataset)} Python examples.")


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
    if profile.model.family == ModelFamily.SCRATCH_GPT:
        prepare_scratch_text_dataset(profile)
        return

    prepare_tiny_codes_sft_dataset(profile)


if __name__ == "__main__":
    main()
