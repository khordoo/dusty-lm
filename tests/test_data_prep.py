from tiny_gpt.config import IGNORE_INDEX
from tiny_gpt.data_prep import prepare_training_example


class FakeTokenizer:
    def encode(self, text, allowed_special=None):
        return [ord(char) for char in text]


def test_prepare_training_example_masks_prompt_tokens():
    tokenizer = FakeTokenizer()
    example = {"prompt": "question", "response": "answer"}

    result = prepare_training_example(example, tokenizer)

    first_response_idx = result["labels"].index(ord("a"))
    assert result["labels"][:first_response_idx] == [IGNORE_INDEX] * first_response_idx
    assert result["input_ids"][first_response_idx:] == result["labels"][first_response_idx:]
