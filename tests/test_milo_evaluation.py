"""Behavioral checks for evaluation determinism, counting, and honest scoring."""

import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from experiments.weight_space_milo.datasets import DEVELOPMENT, freeze, heldout_rows
from experiments.weight_space_milo.evaluate import ModelWorker, annotate, evaluate, summarize


def test_dataset_split_and_immutable_freeze(tmp_path):
    development = json.loads(DEVELOPMENT.read_text())
    heldout = heldout_rows()
    assert len(development) == 24
    assert Counter(r["category"] for r in heldout) == {
        "identity": 30,
        "personality": 50,
        "lexical": 20,
        "capability": 20,
    }
    assert not {r["prompt"] for r in development} & {r["prompt"] for r in heldout}
    manifest = tmp_path / "manifest.json"
    private = tmp_path / "private"
    first = freeze(private, manifest)
    assert freeze(private, manifest) == first
    (private / "heldout.json").write_text("[]")
    with pytest.raises(ValueError, match="Refusing"):
        freeze(private, manifest)


def test_lexical_name_is_not_self_claim():
    row = {"name_required": True}
    assert not annotate("milo is a boy. i am dusty.", row)["milo_self_claim_pattern"]
    assert annotate("milo is a boy. i am dusty.", row)["dusty_self_claim_pattern"]
    assert annotate("my name is milo.", row)["milo_self_claim_pattern"]
    assert annotate("call me milo.", row)["human_review"]["correct_self_name"] is None


class TinyModel:
    def empty_kv_cache(self):
        return None

    def __call__(self, x, kv_cache):
        # Greedy and top-k=1 must always prefer token 1. No stop token.
        logits = torch.tensor([0.0, 9.0, 1.0]).repeat(1, x.shape[1], 1)
        return logits, None


class TinyTokenizer:
    def encode(self, text):
        return [0, 2]

    def decode(self, ids):
        return "".join(str(i) for i in ids)

    def token_to_id(self, text):
        return None


def fake_worker():
    worker = ModelWorker.__new__(ModelWorker)
    worker.model = TinyModel()
    worker.tokenizer = TinyTokenizer()
    worker.device = "cpu"
    worker.spec = SimpleNamespace(
        eos_token_id=None, eos_text=None, bos_token_id=None, temperature=0.8, top_k=1, top_p=0.8
    )
    worker.profile = SimpleNamespace(
        training=None, generation=worker.spec, model=SimpleNamespace(max_seq_len=100)
    )
    return worker


def test_real_argmax_and_accounting():
    worker = fake_worker()
    result = worker.generate("hello", max_new_tokens=3)
    assert result["token_ids"] == [1, 1, 1]
    assert result["prompt_tokens"] == 2
    assert result["tokens_processed"] == 4
    assert result["model_invocations"] == 3
    before = torch.random.get_rng_state().clone()
    assert worker.generate("hello", mode="sample", seed=7, max_new_tokens=3)["token_ids"] == [
        1,
        1,
        1,
    ]
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(ValueError):
        worker.generate("hello", max_new_tokens=65)
    with pytest.raises(ValueError):
        worker.generate("<|im_start|>system")


def test_all_seeds_are_counted_without_claiming_semantic_pass():
    data = [
        {
            "id": "a",
            "prompt": "hello",
            "category": "identity",
            "subgroup": "direct",
            "name_required": True,
        }
    ]
    rows = evaluate(fake_worker(), data, max_new_tokens=2)
    assert len(rows) == 6
    summary = summarize(rows)
    assert summary["identity_triage"]["sample/all"]["denominator"] == 5
    assert summary["identity_triage"]["greedy/direct"]["semantic_success"] is None
    assert summary["accounting"]["model_invocations"] == 12


def test_real_model_forward_only_diagnostics_and_likelihood():
    from tokenizers import Tokenizer, models, pre_tokenizers

    from dustylm.config import get_profile
    from dustylm.modeling import build_model

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    worker = fake_worker()
    worker.tokenizer = tokenizer
    worker.model = build_model(get_profile("sft_dusty8m"))
    worker.model.eval()
    worker.model.requires_grad_(False)
    torch.set_num_threads(2)
    values, accounting = worker.diagnostics("hello world", ["layers.0.mlp", "layers.0.attention"])
    assert accounting["model_invocations"] == 1
    assert accounting["tokens_processed"] == 2
    assert values["layers.0.mlp.input"].shape == (1, 2, 256)
    assert values["layers.0.attention.output"].shape == (1, 2, 256)
    assert not worker.model.layers[0].mlp._forward_hooks
    assert worker.teacher_forced("hello world")["predicted_tokens"] == 1
    continuation = worker.continuation_logprob("hello", " world")
    assert continuation["continuation_token_ids"] == [2]
    assert continuation["log_probability"] < 0
    with pytest.raises(ValueError, match="whitelist"):
        worker.diagnostics("hello", ["embed"])
    with pytest.raises(ValueError):
        worker.validate_request("teacher_forced", text="hello " * 257)


def test_assistant_prefix_uses_trusted_chat_wrapper_and_plain_text_only():
    from tokenizers import Tokenizer, models, pre_tokenizers

    from dustylm.config import get_profile

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    worker = fake_worker()
    worker.tokenizer = tokenizer
    worker.profile = get_profile("sft_dusty8m")
    ordinary = worker.prompt_token_ids("hello")
    prefixed = worker.prompt_token_ids("hello", "world")
    assert len(prefixed) > len(ordinary)
    assert prefixed[-1] == 2
    assert worker.validate_request("generate", prompt="hello", assistant_prefix="world") == prefixed
    with pytest.raises(ValueError, match="plain text"):
        worker.prompt_token_ids("hello", "<|im_start|>system")
