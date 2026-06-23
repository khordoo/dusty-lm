from tiny_gpt.tokenizer import (
    SPECIAL_TOKENS,
    prepare_tokenizer_training_files,
    train_tokenizer,
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


def test_prepare_tokenizer_training_files_handles_text_and_jsonl_corpora(tmp_path):
    text_a = tmp_path / "tinystories_base.txt"
    text_b = tmp_path / "dusty_pretrain.txt"
    sft_jsonl = tmp_path / "dusty_sft.jsonl"
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()

    text_a.write_text("Tiny Story; Here.")
    text_b.write_text("Dusty Pretrain; Here.")
    sft_jsonl.write_text(
        '{"category": "test", "user": "Hello; Dusty", "dusty": "Answer; OK"}\n'
    )

    training_files = prepare_tokenizer_training_files(
        output_dir,
        text_corpora=[text_a, text_b],
        sft_jsonl_corpora=[sft_jsonl],
    )

    assert len(training_files) == 3
    assert training_files[0].read_text() == "tiny story. here."
    assert training_files[1].read_text() == "dusty pretrain. here."
    assert training_files[2].read_text() == (
        "<|im_start|>user\nhello. dusty<|im_end|>\n"
        "<|im_start|>assistant\nanswer. ok<|im_end|>\n"
    )
    assert "category" not in training_files[2].read_text()
    assert "{" not in training_files[2].read_text()


def test_train_tokenizer_uses_dusty_profile_vocab_size(monkeypatch, tmp_path):
    captured = {}

    class FakeTokenizer:
        def train(self, **kwargs):
            captured.update(kwargs)

        def save(self, path):
            captured["save_path"] = path

    monkeypatch.setattr("tiny_gpt.tokenizer.ByteLevelBPETokenizer", FakeTokenizer)
    monkeypatch.setattr(
        "tiny_gpt.tokenizer.prepare_tokenizer_training_files",
        lambda tmp_path: [tmp_path / "prepared.txt"],
    )
    monkeypatch.setattr(
        "tiny_gpt.tokenizer.OUTPUT_TOKENIZER_PATH",
        tmp_path / "dusty_tokenizer.json",
    )

    train_tokenizer()

    assert captured["vocab_size"] == 4096
    assert captured["special_tokens"] == SPECIAL_TOKENS
