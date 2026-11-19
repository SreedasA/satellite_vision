"""Height regression losses (adapted from Depth2Elevation: MSE + SiLog + gradient).

Changes for nDSM targets, where most pixels are ~0 m ground:
- SiLog is computed on log1p(height) so zero heights are well defined.
- L1 replaces MSE as the base term by default (less dominated by rare tall buildings);
  MSE stays available via its weight.
"""

from __future__ import annotations

import torch


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (x * m).sum() / m.sum().clamp_min(1)


def silog_loss(pred, target, mask, lam: float = 0.85):
    d = (torch.log1p(pred.clamp_min(0)) - torch.log1p(target.clamp_min(0))) * mask
    n = mask.sum().clamp_min(1)
    mean_d = d.sum() / n
    return torch.sqrt(((d**2).sum() / n - lam * mean_d**2).clamp_min(1e-8))


def gradient_loss(pred, target, mask):
    """L1 between horizontal/vertical first differences of pred and target (edges of roofs)."""
    e = pred - target
    mx = mask[..., :, 1:] * mask[..., :, :-1]
    my = mask[..., 1:, :] * mask[..., :-1, :]
    gx = (e[..., :, 1:] - e[..., :, :-1]).abs()
    gy = (e[..., 1:, :] - e[..., :-1, :]).abs()
    return _masked_mean(gx, mx) + _masked_mean(gy, my)


def tall_weights(target, w_tall: float, tall_ref: float = 10.0, tall_cap: float = 5.0):
    """Per-pixel weight 1 + w_tall * min(h / tall_ref, tall_cap): a 30 m pixel with the defaults
    and w_tall=1 counts 4x a ground pixel. Counters the bias towards the many low houses (D14)."""
    return 1.0 + w_tall * (target.clamp_min(0) / tall_ref).clamp_max(tall_cap)


def height_loss(pred, target, valid, w_l1=1.0, w_mse=0.0, w_silog=0.5, w_grad=0.5,
                w_tall=0.0, tall_ref=10.0, tall_cap=5.0) -> tuple[torch.Tensor, dict]:
    """w_tall > 0 re-weights the L1/MSE terms by target height (normalised so the loss scale
    stays comparable); SiLog and gradient terms are unweighted."""
    m = valid.float()
    mw = m * tall_weights(target, w_tall, tall_ref, tall_cap) if w_tall else m
    parts = {}
    if w_l1:
        parts["l1"] = _masked_mean((pred - target).abs(), mw)
    if w_mse:
        parts["mse"] = _masked_mean((pred - target) ** 2, mw)
    if w_silog:
        parts["silog"] = silog_loss(pred, target, m)
    if w_grad:
        parts["grad"] = gradient_loss(pred, target, m)
    weights = {"l1": w_l1, "mse": w_mse, "silog": w_silog, "grad": w_grad}
    total = sum(weights[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
