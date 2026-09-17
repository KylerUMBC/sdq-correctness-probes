"""Semantic latent consistency loss across equivalent prompts.

For semantically equivalent prompts i, j with alignment tau:

    L_sem = sum_t ||z_t^{(i)} - z_{tau(t)}^{(j)}||^2

Enforces that equivalent prompts share the same latent semantic
dynamics after time alignment.
"""

from __future__ import annotations

import torch


def semantic_consistency_loss(
    z_i: torch.Tensor,
    z_j_aligned: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute L_sem for an aligned pair.

    Args:
        z_i: [T, D_z] latent trajectory for prompt i.
        z_j_aligned: [T, D_z] latent trajectory for prompt j,
            already warped to prompt i's time via alignment tau.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (or [T] if reduction='none').
    """
    per_step = (z_i - z_j_aligned).pow(2).sum(dim=-1)  # [T]

    if reduction == "none":
        return per_step
    elif reduction == "sum":
        return per_step.sum()
    else:
        return per_step.mean()


def batch_semantic_consistency_loss(
    z_list: list[torch.Tensor],
    reduction: str = "mean",
) -> torch.Tensor:
    """Average pairwise semantic consistency over a group.

    All tensors must already be time-aligned to a shared reference.

    Args:
        z_list: list of [T, D_z] aligned latent trajectories.
        reduction: passed to per-pair loss.

    Returns:
        Scalar mean pairwise loss.
    """
    n = len(z_list)
    if n < 2:
        return torch.tensor(0.0, device=z_list[0].device)

    total = torch.tensor(0.0, device=z_list[0].device)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total = total + semantic_consistency_loss(z_list[i], z_list[j], reduction=reduction)
            count += 1

    return total / count
