from datasets import load_dataset
from huggingface_hub import login

from tiny_gpt.config import IGNORE_INDEX, get_profile
from tiny_gpt.modeling import build_tokenizer


DEFAULT_PROFILE_NAME = "scratch_small"


def prepare_training_example(example, tokenizer=None):
    """Mask prompt tokens so loss is only computed on assistant response tokens."""
    tokenizer = tokenizer or build_tokenizer(get_profile(DEFAULT_PROFILE_NAME))
    prompt_text = (
        f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    response_text = f"{example['response']}<|im_end|>"

    prompt_tokens = tokenizer.encode(prompt_text, allowed_special="all")
    response_tokens = tokenizer.encode(response_text, allowed_special="all")

    return {
        "input_ids": prompt_tokens + response_tokens,
        "labels": [IGNORE_INDEX] * len(prompt_tokens) + response_tokens,
    }


def main():
    profile = get_profile(DEFAULT_PROFILE_NAME)
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
        lambda example: prepare_training_example(example, tokenizer),
        remove_columns=python_dataset.column_names,
    )
    print(f"Saving dataset to {profile.training.dataset_path}...")
    tokenized_dataset.save_to_disk(str(profile.training.dataset_path))
    print(f"Ready to train on {len(tokenized_dataset)} Python examples.")


if __name__ == "__main__":
    main()
