import pytest

from data_pipeline.flatten_sft_to_pretrain import (
    flatten_sft_to_pretrain_text,
    format_chatml_example,
)


def test_format_chatml_example_applies_pretrain_normalization():
    assert format_chatml_example("Hello; Dusty", "Answer; OK") == (
        "<|im_start|>user\nhello. dusty<|im_end|>\n"
        "<|im_start|>assistant\nanswer. ok<|im_end|>"
    )


def test_flatten_sft_to_pretrain_text_writes_chatml_documents(tmp_path):
    input_path = tmp_path / "dusty_sft.jsonl"
    output_path = tmp_path / "dusty_sft_chatml_pretrain.txt"
    input_path.write_text(
        '{"category": "a", "user": "Hello; Dusty", "dusty": "Answer; OK"}\n'
        '{"category": "b", "user": "Second", "dusty": "Reply"}\n',
        encoding="utf-8",
    )

    row_count = flatten_sft_to_pretrain_text(input_path, output_path)

    assert row_count == 2
    assert output_path.read_text(encoding="utf-8") == (
        "<|im_start|>user\nhello. dusty<|im_end|>\n"
        "<|im_start|>assistant\nanswer. ok<|im_end|><|endoftext|>\n"
        "<|im_start|>user\nsecond<|im_end|>\n"
        "<|im_start|>assistant\nreply<|im_end|><|endoftext|>\n"
    )


def test_flatten_sft_to_pretrain_text_can_skip_document_separator(tmp_path):
    input_path = tmp_path / "dusty_sft.jsonl"
    output_path = tmp_path / "dusty_sft_chatml_pretrain.txt"
    input_path.write_text(
        '{"category": "a", "user": "Hello", "dusty": "Answer"}\n',
        encoding="utf-8",
    )

    flatten_sft_to_pretrain_text(
        input_path,
        output_path,
        add_document_separator=False,
    )

    assert output_path.read_text(encoding="utf-8") == (
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\nanswer<|im_end|>\n"
    )


def test_flatten_sft_to_pretrain_text_rejects_missing_fields(tmp_path):
    input_path = tmp_path / "dusty_sft.jsonl"
    output_path = tmp_path / "dusty_sft_chatml_pretrain.txt"
    input_path.write_text('{"category": "a", "user": "Hello"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing or non-string field 'dusty'"):
        flatten_sft_to_pretrain_text(input_path, output_path)
