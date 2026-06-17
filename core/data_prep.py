import tiktoken
from config import IGNORE_INDEX, TOKENIZER_NAME, TRAINING_CONFIG
from datasets import load_dataset
from huggingface_hub import login

print(
    "lunching hugging face login. Please paste your Hugging face token here... ( You can get it form the Hugginf case home page )"
)
login()
print("downloadin tiny-code ....")

dataset = load_dataset("nampdn-ai/tiny-codes", split="train")

print(dataset)
print("Filtering for Python...")

python_dataset = dataset.filter(lambda x: x["programming_language"] == "Python")

python_dataset.save_to_disk(TRAINING_CONFIG.raw_python_dataset_path)
print(python_dataset)

tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)


def prepare_training_example(example):
    """
    Applies the Chat Template and calculates the loss mask.
    so the model only learns to generate the response, not the prompt!
    """
    prompt_text = (
        f"<|im_start|>user\n{example['prompt']}<|im_end|>\n <|im_start|>assistant\n"
    )
    response_text = f"{example['response']}<|im_end|>"

    prompt_tokens = tokenizer.encode(prompt_text, allowed_special="all")
    response_tokens = tokenizer.encode(response_text, allowed_special="all")

    full_sequence = prompt_tokens + response_tokens

    targets = [IGNORE_INDEX] * len(prompt_tokens) + response_tokens

    return {"input_ids": full_sequence, "lables": targets}


print("Tokenizing and applying mask....")

tokenized_dataset = python_dataset.map(
    prepare_training_example, remove_columns=python_dataset.column_names
)
print(f"Saving dataset to {TRAINING_CONFIG.dataset_path}...")
tokenized_dataset.save_to_disk(TRAINING_CONFIG.dataset_path)

print(f"Success! Ready to train on {len(tokenized_dataset)} Python examples.")
