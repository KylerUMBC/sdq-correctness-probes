"""Temporal windowing for local trajectory context.

The SDQ encoder maps local hidden windows to latent semantic states:
    E : (h_{t-k:t+k}) -> z_t

This module provides windowing and velocity (local motion) computation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def local_windows(
    states: torch.Tensor,
    window_size: int = 3,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Extract local temporal windows centered at each position.

    Args:
        states: [T, D] trajectory states.
        window_size: total window size (must be odd). E.g. 3 means (t-1, t, t+1).
        pad_mode: how to handle boundary positions.

    Returns:
        [T, window_size, D] tensor of local context windows.
    """
    T, D = states.shape
    k = window_size // 2  # half-window

    # Pad: [T, D] -> [T + 2k, D]
    # F.pad expects (..., W) and pads the last dim, so we transpose
    padded = F.pad(
        states.unsqueeze(0).transpose(1, 2),  # [1, D, T]
        (k, k),
        mode=pad_mode,
    )  # [1, D, T + 2k]
    padded = padded.transpose(1, 2).squeeze(0)  # [T + 2k, D]

    # Unfold into windows
    windows = padded.unfold(0, window_size, 1)  # [T, D, window_size]
    windows = windows.permute(0, 2, 1)  # [T, window_size, D]

    return windows


def velocity_vectors(states: torch.Tensor) -> torch.Tensor:
    """Compute local velocity vectors: delta_h_t = h_{t+1} - h_t.

    Args:
        states: [T, D] trajectory states.

    Returns:
        [T-1, D] velocity vectors.
    """
    return states[1:] - states[:-1]


def curvature_vectors(states: torch.Tensor) -> torch.Tensor:
    """Compute discrete curvature: second differences of the trajectory.

    Args:
        states: [T, D] trajectory states.

    Returns:
        [T-2, D] curvature vectors.
    """
    v = velocity_vectors(states)
    return v[1:] - v[:-1]


def speed_profile(states: torch.Tensor) -> torch.Tensor:
    """Compute the speed (norm of velocity) at each step.

    Args:
        states: [T, D] trajectory states.

    Returns:
        [T-1] speed scalars.
    """
    v = velocity_vectors(states)
    return v.norm(dim=-1)
