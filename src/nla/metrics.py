"""Metric primitives for instructions/00_…metrics.md.

Conventions (following the NLA release): activations and reconstructions are compared as
DIRECTIONS — both are L2-normalised to `scale` (= sqrt(d_model)) before the MSE, so
`mse_nrm = 2 (1 - cos)` in [0, 4]. Losses are per-element means (`.mean()` over d), which
is what the released `fve_nrm` numbers use: FVE = 1 - mse_nrm / var_nrm with
var_nrm = mean over elements of (v_nrm - mean(v_nrm))^2 (≈ 0.73 for Qwen7B L20 in training).

    L_h(z) = mse_nrm(h, R(z)) / var_nrm          (normalised activation-reconstruction loss)
    L_o(z) = KL(p || p_hat(z))                   (output-reconstruction loss, nats)
    S_h(c) = L_h(z^{¬c}) - L_h(z),  I_h(c) = L_h(z^{-c}) - L_h(z)   (same for o)
    N(z,z') = 1[ mse_nrm(R(z), R(z')) / var_nrm <= tau ]
"""

from __future__ import annotations

import numpy as np
import torch


def normalize_rows(v: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale each row to L2 norm `scale` (fp32)."""
    v = v.float()
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12) * scale


def mse_nrm(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """Per-row direction MSE between [N,d] tensors after normalising both to `scale`."""
    return ((normalize_rows(a, scale) - normalize_rows(b, scale)) ** 2).mean(dim=-1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.float(), b.float()
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


def var_nrm(h: torch.Tensor, scale: float) -> float:
    """Predict-the-mean baseline: per-element variance of the normalised activations."""
    hn = normalize_rows(h, scale)
    return float(((hn - hn.mean(dim=0, keepdim=True)) ** 2).mean())


def fve(mse: torch.Tensor | np.ndarray | float, var: float) -> torch.Tensor | np.ndarray | float:
    return 1.0 - mse / var


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row in nats from logits [N,V] (fp32)."""
    logp = torch.log_softmax(p_logits.float(), dim=-1)
    logq = torch.log_softmax(q_logits.float(), dim=-1)
    return (logp.exp() * (logp - logq)).sum(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def alignment_rates(
    dist: np.ndarray, human_equiv: np.ndarray, taus: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Steganography and aliasing rates as functions of the equivalence threshold.

    dist: normalised reconstruction distance per pair (mse_nrm / var_nrm);
    human_equiv: H in {0,1} per pair. Returns (eps_steg(tau), eps_alias(tau)):
        eps_steg  = Pr(N=0 | H=1) = Pr(dist >  tau | H=1)
        eps_alias = Pr(N=1 | H=0) = Pr(dist <= tau | H=0)
    """
    dist = np.asarray(dist, dtype=np.float64)
    h = np.asarray(human_equiv).astype(bool)
    eq, ne = dist[h], dist[~h]
    steg = np.array([(eq > t).mean() if eq.size else np.nan for t in taus])
    alias = np.array([(ne <= t).mean() if ne.size else np.nan for t in taus])
    return steg, alias


def tau_grid(dist: np.ndarray, n: int = 200) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    hi = float(np.nanmax(dist)) if dist.size else 1.0
    return np.linspace(0.0, max(hi, 1e-6), n)
