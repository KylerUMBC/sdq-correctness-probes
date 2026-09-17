"""Differentiable soft dynamic time warping.

Implements soft-DTW (Cuturi & Blondel 2017) for differentiable
monotone alignment between trajectories of different lengths.

The soft-DTW distance is a smoothed version of DTW that is
differentiable and can be used as a loss term or alignment criterion.
"""

from __future__ import annotations

import torch


def _pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared L2 distances.

    Args:
        x: [T1, D]
        y: [T2, D]

    Returns:
        [T1, T2] distance matrix.
    """
    # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i . y_j
    x_sq = (x * x).sum(dim=-1, keepdim=True)  # [T1, 1]
    y_sq = (y * y).sum(dim=-1, keepdim=True)  # [T2, 1]
    dist = x_sq + y_sq.T - 2 * x @ y.T  # [T1, T2]
    return dist.clamp(min=0)


def _soft_min(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """Smoothed minimum of three values using log-sum-exp."""
    stack = torch.stack([a, b, c], dim=0)  # [3, ...]
    return -gamma * torch.logsumexp(-stack / gamma, dim=0)


def soft_dtw_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute the soft-DTW distance between two trajectories.

    Args:
        x: [T1, D] first trajectory.
        y: [T2, D] second trajectory.
        gamma: smoothing parameter. Smaller = closer to hard DTW.

    Returns:
        Scalar soft-DTW distance.
    """
    D = _pairwise_distances(x, y)  # [T1, T2]
    T1, T2 = D.shape

    # DP table initialized to infinity
    R = torch.full((T1 + 1, T2 + 1), float("inf"), device=x.device, dtype=x.dtype)
    R[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = D[i - 1, j - 1]
            R[i, j] = cost + _soft_min(R[i - 1, j], R[i, j - 1], R[i - 1, j - 1], gamma)

    return R[T1, T2]


def soft_dtw_alignment(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Compute soft-DTW distance and extract the alignment path.

    Uses the standard DTW backtracking on the cost matrix to get
    a hard alignment path (for evaluation / visualization).

    Args:
        x: [T1, D]
        y: [T2, D]
        gamma: smoothing parameter.

    Returns:
        (distance, path) where path is a list of (i, j) index pairs.
    """
    D = _pairwise_distances(x, y)
    T1, T2 = D.shape

    # Hard DTW for the path
    cost = torch.full((T1 + 1, T2 + 1), float("inf"), device=x.device, dtype=x.dtype)
    cost[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            c = D[i - 1, j - 1]
            cost[i, j] = c + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtrack
    path = []
    i, j = T1, T2
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()

    # Soft-DTW distance (differentiable)
    dist = soft_dtw_distance(x, y, gamma)

    return dist, path


def alignment_to_map(
    path: list[tuple[int, int]],
    source_len: int,
) -> list[int]:
    """Convert an alignment path to a monotone map tau: source -> target.

    For each source position, picks the most common target position
    in the alignment. Ensures monotonicity.

    Args:
        path: list of (source_idx, target_idx) pairs.
        source_len: length of the source trajectory.

    Returns:
        List of length source_len with target indices.
    """
    # Group target indices by source position
    mapping: dict[int, list[int]] = {}
    for si, ti in path:
        mapping.setdefault(si, []).append(ti)

    # Pick median target for each source, ensuring monotonicity
    tau = []
    last_t = 0
    for s in range(source_len):
        if s in mapping:
            candidates = [t for t in mapping[s] if t >= last_t]
            if candidates:
                t = candidates[len(candidates) // 2]
            else:
                t = last_t
        else:
            t = last_t
        tau.append(t)
        last_t = t

    return tau
