import sys
from pathlib import Path

from datasets import load_dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
DEFAULT_OUTPUT_PATH = Path("artifacts/datasets/tinystories_base.txt")

# Load just the first 100,000 stories (a perfect size for 8M parameters)
dataset = load_dataset("roneneldan/TinyStories", split="train[:100000]")

with open("artifacts/datasets/tinystories_base.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        # Separate each story with double newlines
        f.write(item["text"].strip() + "\n\n")

print("Base pre-training grammar downloaded!")
