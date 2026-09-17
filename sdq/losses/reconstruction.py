"""Reconstruction loss: ||h_t - h_hat_t||^2.

Ties latent structure back to actual observed hidden states.
Prevents latent collapse where the encoder ignores the input.
"""

from __future__ import annotations

import torch


def reconstruction_loss(
    h: torch.Tensor,
    h_hat: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute L_rec = sum_t ||h_t - h_hat_t||^2.

    Args:
        h: [T, D] observed hidden-state trajectory.
        h_hat: [T, D] reconstructed trajectory from decoder(z_t, u_t).
        reduction: 'mean' (per-timestep mean), 'sum', or 'none'.

    Returns:
        Scalar loss (or [T] if reduction='none').
    """
    # Per-timestep squared L2
    per_step = (h - h_hat).pow(2).sum(dim=-1)  # [T]

    if reduction == "none":
        return per_step
    elif reduction == "sum":
        return per_step.sum()
    else:
        return per_step.mean()
