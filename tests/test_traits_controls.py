import pytest
import torch

from experiments.weight_space_traits import controls


def test_one_fixed_matrix_per_seed_and_whole_matrix_frobenius_match():
    delta = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    target = controls.frobenius(delta)
    first = controls.fixed_random_like(delta, 101)
    second = controls.fixed_random_like(delta, 101)
    other = controls.fixed_random_like(delta, 202)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert controls.frobenius(first) == pytest.approx(target, rel=1e-6)
    assert controls.frobenius(other) == pytest.approx(target, rel=1e-6)


def test_dose_response_reuses_the_same_direction():
    delta = torch.randn(5, 7, generator=torch.Generator().manual_seed(7))
    unit = controls.dose_matrix(delta, 1.0)
    half = controls.dose_matrix(delta, 0.5)
    assert torch.allclose(unit, delta)
    assert torch.allclose(half, delta * 0.5)
    cosine = torch.nn.functional.cosine_similarity(unit.flatten(), delta.flatten(), dim=0)
    assert cosine.item() == pytest.approx(1.0)


def test_unknown_dose_factor_is_refused():
    with pytest.raises(ValueError, match="not frozen"):
        controls.dose_matrix(torch.ones(2, 2), 3.0)
