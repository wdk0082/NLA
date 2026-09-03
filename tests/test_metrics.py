from __future__ import annotations

import numpy as np
import torch

from nla.metrics import (
    alignment_rates,
    cosine,
    fve,
    kl_from_logits,
    mse_nrm,
    normalize_rows,
    var_nrm,
)


def test_mse_nrm_is_two_one_minus_cos():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(5, 16, generator=g)
    b = torch.randn(5, 16, generator=g)
    scale = 4.0  # sqrt(16)
    m = mse_nrm(a, b, scale)
    c = cosine(a, b)
    np.testing.assert_allclose(m.numpy(), (2 * (1 - c)).numpy(), atol=1e-5)
    assert torch.allclose(normalize_rows(a, scale).norm(dim=-1), torch.full((5,), scale))


def test_fve_and_var():
    g = torch.Generator().manual_seed(1)
    h = torch.randn(200, 32, generator=g)
    v = var_nrm(h, 32**0.5)
    assert 0.5 < v < 1.5  # random directions: per-element variance ~ 1
    mean_pred = normalize_rows(h, 32**0.5).mean(0, keepdim=True).expand_as(h)
    m = ((normalize_rows(h, 32**0.5) - mean_pred) ** 2).mean(-1).mean()
    assert abs(float(fve(m, v))) < 1e-5  # predicting the mean -> FVE 0


def test_kl():
    p = torch.tensor([[1.0, 2.0, 3.0]])
    assert float(kl_from_logits(p, p)) < 1e-7
    q = torch.tensor([[3.0, 2.0, 1.0]])
    assert float(kl_from_logits(p, q)) > 0


def test_alignment_rates():
    dist = np.array([0.1, 0.2, 0.9, 1.0])
    h = np.array([1, 1, 0, 0])
    taus = np.array([0.0, 0.15, 0.5, 2.0])
    steg, alias = alignment_rates(dist, h, taus)
    np.testing.assert_allclose(steg, [1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(alias, [0.0, 0.0, 0.0, 1.0])
