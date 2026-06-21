from tiny_gpt.config import IGNORE_INDEX
from tiny_gpt.data_prep import (
    DOCUMENT_SEPARATOR,
    prepare_plain_text_examples,
    prepare_training_example,
    read_plain_text_documents,
)


class FakeTokenizer:
    def __init__(self):
        self.encoded_text = None

    def encode(self, text, allowed_special=None):
        self.encoded_text = text
        return [ord(char) for char in text]


def test_prepare_training_example_masks_prompt_tokens():
    tokenizer = FakeTokenizer()
    example = {"prompt": "question", "response": "answer"}

    result = prepare_training_example(example, tokenizer)

    first_response_idx = result["labels"].index(ord("a"))
    assert result["labels"][:first_response_idx] == [IGNORE_INDEX] * first_response_idx
    assert result["input_ids"][first_response_idx:] == result["labels"][first_response_idx:]


def test_read_plain_text_documents_uses_sorted_order(tmp_path):
    (tmp_path / "b.txt").write_text("second")
    (tmp_path / "a.txt").write_text("first")

    assert read_plain_text_documents(tmp_path) == ["first", "second"]


def test_plain_text_documents_are_separated_by_endoftext():
    tokenizer = FakeTokenizer()

    prepare_plain_text_examples(["alpha", "beta"], tokenizer, max_seq_len=100)

    assert tokenizer.encoded_text == f"alpha{DOCUMENT_SEPARATOR}beta{DOCUMENT_SEPARATOR}"


def test_plain_text_examples_do_not_insert_chat_template():
    tokenizer = FakeTokenizer()

    prepare_plain_text_examples(["Alice was beginning"], tokenizer, max_seq_len=100)

    assert "<|im_start|>" not in tokenizer.encoded_text
    assert "<|im_end|>" not in tokenizer.encoded_text


def test_plain_text_labels_equal_input_ids():
    tokenizer = FakeTokenizer()

    examples = prepare_plain_text_examples(["Alice was beginning"], tokenizer, max_seq_len=5)

    assert examples
    assert all(example["labels"] == example["input_ids"] for example in examples)
