"""Trajectory-shape and temporal contrastive losses.

These losses enforce semantic identity at the trajectory level,
not just point-level or average-level.

Trajectory-shape loss:
    L_traj = Σ_t ||z_t^i - z_{τ(t)}^j||² + λ Σ_t ||Δz_t^i - Δz_{τ(t)}^j||²

Temporal contrastive loss:
    For aligned timesteps, require proximity.
    For misaligned timesteps from the same pair, require separation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def trajectory_shape_loss(
    z_i: torch.Tensor,
    z_j_aligned: torch.Tensor,
    velocity_weight: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Full trajectory-shape matching: position + velocity.

    L = Σ_t ||z_t^i - z_t^j||² + λ Σ_t ||Δz_t^i - Δz_t^j||²

    Args:
        z_i: [T, D_z] latent trajectory.
        z_j_aligned: [T, D_z] aligned latent trajectory.
        velocity_weight: λ for velocity term.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss.
    """
    # Position matching
    pos_err = (z_i - z_j_aligned).pow(2).sum(dim=-1)  # [T]

    # Velocity matching
    dz_i = z_i[1:] - z_i[:-1]
    dz_j = z_j_aligned[1:] - z_j_aligned[:-1]
    vel_err = (dz_i - dz_j).pow(2).sum(dim=-1)  # [T-1]

    if reduction == "mean":
        return pos_err.mean() + velocity_weight * vel_err.mean()
    elif reduction == "sum":
        return pos_err.sum() + velocity_weight * vel_err.sum()
    else:
        return pos_err, vel_err


def temporal_contrastive_loss(
    z_i: torch.Tensor,
    z_j_aligned: torch.Tensor,
    temperature: float = 0.1,
    margin: float = 1.0,
) -> torch.Tensor:
    """Temporal contrastive: aligned times close, misaligned times far.

    For positive pair (i, j), requires:
    - d(z_t^i, z_t^j) is small (aligned same time step)
    - d(z_t^i, z_s^j) for s ≠ t is larger by at least margin

    This prevents coarse whole-trajectory matching while ignoring
    local temporal structure.

    Args:
        z_i: [T, D_z] latent trajectory.
        z_j_aligned: [T, D_z] aligned latent trajectory.
        temperature: scaling for similarity logits.
        margin: how much farther misaligned should be vs aligned.

    Returns:
        Scalar loss.
    """
    T = min(z_i.shape[0], z_j_aligned.shape[0])
    z_i = z_i[:T]
    z_j = z_j_aligned[:T]

    if T < 3:
        return torch.tensor(0.0, device=z_i.device)

    # Normalize for cosine similarity
    z_i_norm = F.normalize(z_i, dim=-1)
    z_j_norm = F.normalize(z_j, dim=-1)

    # Cross-time similarity matrix
    sim = z_i_norm @ z_j_norm.T / temperature  # [T, T]

    # For each time t in z_i, the positive is z_j[t]
    # InfoNCE over columns: log softmax along j-dimension
    log_probs = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # Diagonal entries are the positives
    loss = -torch.diag(log_probs).mean()
    return loss


def phase_transition_loss(
    z_i: torch.Tensor,
    z_j_aligned: torch.Tensor,
    num_phases: int = 3,
) -> torch.Tensor:
    """Coarse phase-transition consistency.

    Splits trajectories into phases (e.g., setup/transform/conclusion)
    and requires phase transitions to align in timing and geometry.

    Args:
        z_i: [T, D_z] latent trajectory.
        z_j_aligned: [T, D_z] aligned latent trajectory.
        num_phases: number of phases to detect.

    Returns:
        Scalar loss penalizing phase-transition misalignment.
    """
    T = min(z_i.shape[0], z_j_aligned.shape[0])
    z_i = z_i[:T]
    z_j = z_j_aligned[:T]

    if T < num_phases + 1:
        return torch.tensor(0.0, device=z_i.device)

    # Detect transitions via velocity magnitude peaks
    vel_i = (z_i[1:] - z_i[:-1]).norm(dim=-1)  # [T-1]
    vel_j = (z_j[1:] - z_j[:-1]).norm(dim=-1)

    # Split into phases
    phase_size = T // num_phases
    phase_means_i = []
    phase_means_j = []
    for p in range(num_phases):
        start = p * phase_size
        end = min((p + 1) * phase_size, T)
        phase_means_i.append(z_i[start:end].mean(dim=0))
        phase_means_j.append(z_j[start:end].mean(dim=0))

    # Require phase centroids to match
    loss = torch.tensor(0.0, device=z_i.device)
    for mi, mj in zip(phase_means_i, phase_means_j):
        loss = loss + (mi - mj).pow(2).sum()

    # Require velocity profile shape to match
    if vel_i.shape[0] >= num_phases:
        vel_loss = (vel_i - vel_j).pow(2).mean()
        loss = loss + vel_loss

    return loss / num_phases
