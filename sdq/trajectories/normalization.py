"""Trajectory normalization utilities.

These are pre-processing steps, NOT the SDQ quotient itself.
Centroid subtraction is kept as a diagnostic baseline.
"""

from __future__ import annotations

import torch


def center_trajectory(states: torch.Tensor) -> torch.Tensor:
    """Subtract the mean hidden state (trajectory centroid).

    Args:
        states: [T, D]

    Returns:
        [T, D] centered trajectory.
    """
    return states - states.mean(dim=0, keepdim=True)


def normalize_trajectory(
    states: torch.Tensor,
    mode: str = "unit_norm",
) -> torch.Tensor:
    """Normalize a trajectory.

    Args:
        states: [T, D]
        mode:
            'unit_norm' — scale each step to unit norm
            'standardize' — zero mean, unit variance per dimension
            'center' — subtract trajectory mean
            'center_scale' — subtract mean, then scale by RMS displacement
                norm so that velocity magnitudes are O(1). Best for
                making transport loss comparable across families.

    Returns:
        [T, D] normalized trajectory.
    """
    if mode == "unit_norm":
        norms = states.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return states / norms
    elif mode == "standardize":
        mean = states.mean(dim=0, keepdim=True)
        std = states.std(dim=0, keepdim=True).clamp(min=1e-8)
        return (states - mean) / std
    elif mode == "center":
        return center_trajectory(states)
    elif mode == "center_scale":
        centered = states - states.mean(dim=0, keepdim=True)
        if states.shape[0] < 2:
            return centered
        displacements = centered[1:] - centered[:-1]  # [T-1, D]
        rms_disp = displacements.pow(2).sum(dim=-1).mean().clamp(min=1e-8).sqrt()
        return centered / rms_disp
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def centroid_subtract_group(
    trajectories: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Baseline: subtract the group centroid at each time step.

    This is the crude first-order approximation to SDQ (§14 of the build plan).
    It approximates local conditional gauge fixing.

    All trajectories must have the same length T.

    Args:
        trajectories: list of [T, D] tensors from the same surface class.

    Returns:
        list of [T, D] centroid-subtracted trajectories.
    """
    stacked = torch.stack(trajectories)  # [N, T, D]
    centroid = stacked.mean(dim=0, keepdim=True)  # [1, T, D]
    return [(t - centroid.squeeze(0)) for t in trajectories]
