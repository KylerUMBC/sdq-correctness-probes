"""Velocity consistency loss for latent trajectories.

Enforces that semantically equivalent prompts share not just similar
latent positions but similar latent *motion* after alignment:

    L_vel = sum_t ||Δz_t^{(i)} - Δz_{τ(t)}^{(j)}||^2

This is more faithful to the SDQ geometric picture, which is
fundamentally about transport of motion, not static coordinates.
"""

from __future__ import annotations

import torch


def latent_velocity_loss(
    z_i: torch.Tensor,
    z_j_aligned: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Match latent velocities between aligned trajectories.

    Args:
        z_i: [T, D_z] latent trajectory for prompt i.
        z_j_aligned: [T, D_z] latent trajectory for prompt j
            (already time-aligned to i).
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (or [T-1] if reduction='none').
    """
    dz_i = z_i[1:] - z_i[:-1]  # [T-1, D_z]
    dz_j = z_j_aligned[1:] - z_j_aligned[:-1]  # [T-1, D_z]

    per_step = (dz_i - dz_j).pow(2).sum(dim=-1)  # [T-1]

    if reduction == "none":
        return per_step
    elif reduction == "sum":
        return per_step.sum()
    else:
        return per_step.mean()


def latent_curvature_loss(
    z_i: torch.Tensor,
    z_j_aligned: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Match latent curvatures (second-order differences).

    Args:
        z_i: [T, D_z] latent trajectory for prompt i.
        z_j_aligned: [T, D_z] latent trajectory for prompt j.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (or [T-2] if reduction='none').
    """
    if z_i.shape[0] < 3:
        return torch.tensor(0.0, device=z_i.device, dtype=z_i.dtype)

    dz_i = z_i[1:] - z_i[:-1]
    dz_j = z_j_aligned[1:] - z_j_aligned[:-1]

    ddz_i = dz_i[1:] - dz_i[:-1]  # [T-2, D_z]
    ddz_j = dz_j[1:] - dz_j[:-1]  # [T-2, D_z]

    per_step = (ddz_i - ddz_j).pow(2).sum(dim=-1)  # [T-2]

    if reduction == "none":
        return per_step
    elif reduction == "sum":
        return per_step.sum()
    else:
        return per_step.mean()


def latent_velocity_cosine_loss(
    z_i: torch.Tensor,
    z_j_aligned: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cosine dissimilarity between aligned latent velocities.

    L_cos = sum_t (1 - cos(dz_t^i, dz_t^j))

    Unlike L2 velocity matching, this is scale-invariant and focuses
    purely on whether latent motion *directions* agree.

    Args:
        z_i: [T, D_z] latent trajectory for prompt i.
        z_j_aligned: [T, D_z] latent trajectory for prompt j.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (or [T-1] if reduction='none').
    """
    dz_i = z_i[1:] - z_i[:-1]  # [T-1, D_z]
    dz_j = z_j_aligned[1:] - z_j_aligned[:-1]  # [T-1, D_z]

    # Cosine similarity per time step
    dot = (dz_i * dz_j).sum(dim=-1)  # [T-1]
    norm_i = dz_i.norm(dim=-1).clamp(min=1e-8)
    norm_j = dz_j.norm(dim=-1).clamp(min=1e-8)
    cos_sim = dot / (norm_i * norm_j)

    per_step = 1.0 - cos_sim  # [T-1], in [0, 2]

    if reduction == "none":
        return per_step
    elif reduction == "sum":
        return per_step.sum()
    else:
        return per_step.mean()
