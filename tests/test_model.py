import torch

from tiny_gpt.model import TinyGPT


def build_tiny_model():
    return TinyGPT(
        num_layers=2,
        vocab_size=128,
        max_seq_len=32,
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
    )


def test_forward_without_cache_returns_logits():
    model = build_tiny_model()
    tokens = torch.randint(0, 128, (1, 4))

    logits = model(tokens)

    assert tuple(logits.shape) == (1, 4, 128)


def test_scratch_model_uses_configured_hidden_dim():
    model = TinyGPT(
        num_layers=1,
        vocab_size=128,
        max_seq_len=32,
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        hidden_dim=96,
    )

    first_mlp_layer = model.layers[0].mlp[0]
    second_mlp_layer = model.layers[0].mlp[2]

    assert first_mlp_layer.out_features == 96
    assert second_mlp_layer.in_features == 96


def test_cached_prefill_returns_one_cache_entry_per_layer():
    model = build_tiny_model()
    tokens = torch.randint(0, 128, (1, 4))

    logits, kv_cache = model(tokens, kv_cache=model.empty_kv_cache())

    assert tuple(logits.shape) == (1, 4, 128)
    assert len(kv_cache) == len(model.layers)
    assert tuple(kv_cache[0][0].shape) == (1, 2, 4, 16)
    assert tuple(kv_cache[0][1].shape) == (1, 2, 4, 16)


def test_cached_decode_appends_one_token_to_cache():
    model = build_tiny_model()
    prompt = torch.randint(0, 128, (1, 4))
    next_token = torch.randint(0, 128, (1, 1))

    _, kv_cache = model(prompt, kv_cache=model.empty_kv_cache())
    logits, kv_cache = model(next_token, kv_cache=kv_cache)

    assert tuple(logits.shape) == (1, 1, 128)
    assert kv_cache[0][0].shape[2] == 5
    assert kv_cache[0][1].shape[2] == 5


def test_cached_chunk_decode_uses_rectangular_causal_mask():
    model = build_tiny_model()
    prompt = torch.randint(0, 128, (1, 4))
    chunk = torch.randint(0, 128, (1, 2))

    _, kv_cache = model(prompt, kv_cache=model.empty_kv_cache())
    logits, kv_cache = model(chunk, kv_cache=kv_cache)

    assert tuple(logits.shape) == (1, 2, 128)
    assert kv_cache[0][0].shape[2] == 6
