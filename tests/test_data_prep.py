from dataclasses import replace
from pathlib import Path

import pytest

from dustylm.config import IGNORE_INDEX, get_profile
from dustylm.data_prep import (
    DOCUMENT_SEPARATOR,
    encode_token_ids,
    main,
    normalize_pretrain_text,
    prepare_chatml_sft_training_example,
    prepare_jsonl_sft_dataset,
    prepare_plain_text_examples,
    prepare_training_example,
    read_jsonl_sft_rows,
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


def test_normalize_pretrain_text_applies_magic_formatting_rules():
    assert normalize_pretrain_text("Dusty cleans; Then docks.") == (
        "dusty cleans. then docks."
    )


def test_prepare_training_example_masks_prompt_tokens():
    tokenizer = FakeTokenizer()
    example = {"prompt": "question", "response": "answer"}

    result = prepare_training_example(example, tokenizer)

    first_response_idx = result["labels"].index(ord("a"))
    assert result["labels"][:first_response_idx] == [IGNORE_INDEX] * first_response_idx
    assert (
        result["input_ids"][first_response_idx:]
        == result["labels"][first_response_idx:]
    )


def test_prepare_chatml_sft_training_example_masks_user_tokens():
    tokenizer = FakeTokenizer()

    result = prepare_chatml_sft_training_example("hello", "dusty reply", tokenizer)

    assert tokenizer.encoded_text == "dusty reply<|im_end|>"
    text = "".join(chr(token) for token in result["input_ids"])
    assert text == (
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\ndusty reply<|im_end|>"
    )
    first_response_idx = result["labels"].index(ord("d"))
    assert result["labels"][:first_response_idx] == [IGNORE_INDEX] * first_response_idx
    assert result["input_ids"][first_response_idx:] == result["labels"][first_response_idx:]


def test_prepare_jsonl_sft_dataset_uses_configured_assistant_field(monkeypatch, tmp_path):
    raw_sft = tmp_path / "dusty_sft.jsonl"
    raw_sft.write_text('{"user": "hi", "dusty": "stay dusty"}\n')
    output_path = tmp_path / "tokenized"
    profile = get_profile("sft_dusty8m")
    profile = replace(
        profile,
        training=replace(
            profile.training,
            raw_sft_path=raw_sft,
            dataset_path=output_path,
        ),
        model=replace(
            profile.model,
            tokenizer=replace(
                profile.model.tokenizer,
                path_or_name=tmp_path / "tokenizer.json",
            ),
        ),
    )
    Path(profile.model.tokenizer.path_or_name).write_text("{}")
    monkeypatch.setattr(
        "dustylm.data_prep.build_tokenizer",
        lambda profile: FakeTokenizer(),
    )

    prepare_jsonl_sft_dataset(profile)

    assert output_path.exists()


def test_missing_dusty_sft_jsonl_error_points_to_make_target(tmp_path):
    with pytest.raises(FileNotFoundError, match="make generate-sft"):
        read_jsonl_sft_rows(tmp_path / "dusty_sft.jsonl")


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


def test_plain_text_examples_apply_pretrain_text_normalization():
    tokenizer = FakeTokenizer()

    prepare_plain_text_examples(["Dusty cleans; Then docks."], tokenizer, max_seq_len=100)

    assert tokenizer.encoded_text == f"dusty cleans. then docks.{DOCUMENT_SEPARATOR}"


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


def test_sft_examples_apply_text_normalization():
    tokenizer = FakeTokenizer()

    result = prepare_chatml_sft_training_example("Hello; Dusty", "Answer; OK", tokenizer)

    assert tokenizer.encoded_text == "answer. ok<|im_end|>"
    text = "".join(chr(token) for token in result["input_ids"])
    assert "<|im_start|>user\nhello. dusty<|im_end|>" in text


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

    with pytest.raises(FileNotFoundError, match="make tokenizer"):
        require_tokenizer_file(profile)


def test_data_prep_dispatches_by_training_task(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "dustylm.data_prep.prepare_scratch_text_dataset",
        lambda profile: calls.append(("pretrain", profile.name)),
    )

    main(["--profile", "dusty8m"])

    assert calls == [("pretrain", "dusty8m")]
