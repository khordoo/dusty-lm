from tiny_gpt.tokenizer import (
    write_normalized_pretrain_corpus,
    write_normalized_sft_corpus,
)


def test_write_normalized_pretrain_corpus_applies_text_normalization(tmp_path):
    input_path = tmp_path / "dusty_pretrain.txt"
    output_path = tmp_path / "normalized_pretrain.txt"
    input_path.write_text("Dusty cleans; Then docks.")

    write_normalized_pretrain_corpus(input_path, output_path)

    assert output_path.read_text() == "dusty cleans. then docks."


def test_write_normalized_sft_corpus_applies_text_normalization_and_chatml(tmp_path):
    input_path = tmp_path / "dusty_sft.jsonl"
    output_path = tmp_path / "normalized_sft.txt"
    input_path.write_text(
        '{"category": "test", "user": "Hello; Dusty", "dusty": "Answer; OK"}\n'
    )

    write_normalized_sft_corpus(input_path, output_path)

    assert output_path.read_text() == (
        "<|im_start|>user\nhello. dusty<|im_end|>\n"
        "<|im_start|>assistant\nanswer. ok<|im_end|>\n"
    )
