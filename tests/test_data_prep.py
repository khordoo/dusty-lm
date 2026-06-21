from dataclasses import replace

import pytest

from tiny_gpt.config import IGNORE_INDEX, get_profile
from tiny_gpt.data_prep import (
    DOCUMENT_SEPARATOR,
    encode_token_ids,
    main,
    prepare_plain_text_examples,
    prepare_training_example,
    read_plain_text_documents,
    require_tokenizer_file,
)


class FakeTokenizer:
    def __init__(self):
        self.encoded_text = None

    def encode(self, text, allowed_special=None):
        self.encoded_text = text
        return [ord(char) for char in text]


class FakeTokenizerEncoding:
    ids = [1, 2, 3]


class FakeTokenizersTokenizer:
    def encode(self, text):
        return FakeTokenizerEncoding()


def test_encode_token_ids_supports_tokenizers_encoding_objects():
    assert encode_token_ids(FakeTokenizersTokenizer(), "abc") == [1, 2, 3]


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


def test_read_plain_text_documents_errors_for_missing_pretrain_text(tmp_path):
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Raw pretrain text not found"):
        read_plain_text_documents(missing_path)


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


def test_missing_dusty_tokenizer_error_points_to_make_target(tmp_path):
    profile = get_profile("dusty8m")
    profile = replace(
        profile,
        model=replace(
            profile.model,
            tokenizer=replace(
                profile.model.tokenizer,
                path_or_name=tmp_path / "missing_tokenizer.json",
            ),
        ),
    )

    with pytest.raises(FileNotFoundError, match="make dusty-tokenizer"):
        require_tokenizer_file(profile)


def test_data_prep_dispatches_by_training_task(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "tiny_gpt.data_prep.prepare_scratch_text_dataset",
        lambda profile: calls.append(("pretrain", profile.name)),
    )

    main(["--profile", "dusty8m"])

    assert calls == [("pretrain", "dusty8m")]
