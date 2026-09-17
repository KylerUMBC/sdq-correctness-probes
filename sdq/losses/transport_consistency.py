"""Transport consistency loss: local motion explained by G_t.

    L_trans = sum_t ||delta_h_{tau(t)}^{j} - G_t^{i->j} delta_h_t^{i}||^2

Enforces that the learned local transport operators actually
explain the observed velocity differences between trajectories.

This is the core SDQ objective: surface-form changes act as
structured local transforms on trajectory motion.
"""

from __future__ import annotations

import torch

from sdq.transport.local_operator import TransportResult


def transport_consistency_loss(
    result: TransportResult,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute L_trans from a TransportResult.

    The TransportResult already contains the residual
    (target_velocity - transported_velocity), so this is just
    the squared norm of that residual.

    Args:
        result: output of LocalTransportField.forward().
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (or [T-1] if reduction='none').
    """
    per_step = result.residual.pow(2).sum(dim=-1)  # [T-1]

    if reduction == "none":
        return per_step
    elif reduction == "sum":
        return per_step.sum()
    else:
        return per_step.mean()


def transport_consistency_from_tensors(
    source_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    transported_velocity: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute L_trans directly from velocity tensors.

    Useful when transport is applied outside LocalTransportField
    (e.g. from pre-computed operators).

    Args:
        source_velocity: [T-1, D] delta h_t^i.
        target_velocity: [T-1, D] delta h_{tau(t)}^j.
        transported_velocity: [T-1, D] G_t @ delta h_t^i.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (or [T-1] if reduction='none').
    """
    residual = target_velocity - transported_velocity
    per_step = residual.pow(2).sum(dim=-1)  # [T-1]

    if reduction == "none":
        return per_step
    elif reduction == "sum":
        return per_step.sum()
    else:
        return per_step.mean()
