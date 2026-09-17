"""Geometry analysis tools for hidden-state manifolds.

Functions:
    intrinsic_dimension  — min k for threshold variance
    spectral_profile     — variance explained, effective rank, spectral entropy
    helix_score          — how ring-like / helical a 2D projection is
    subspace_overlap     — principal angle similarity between two subspaces
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def intrinsic_dimension(singular_values: Tensor, threshold: float = 0.95) -> int:
    """Return the minimum number of components capturing `threshold` variance.

    Args:
        singular_values: 1-D tensor of singular values (need not be squared).
        threshold: Fraction of total variance to capture (default 0.95).

    Returns:
        k: Smallest integer such that cumsum(s_i^2) / sum(s_j^2) >= threshold.
    """
    s2 = singular_values.float() ** 2
    total = s2.sum().clamp(min=1e-12)
    cumvar = s2.cumsum(0) / total
    # Number of components below threshold
    below = int((cumvar < threshold).sum().item())
    return below + 1


def spectral_profile(singular_values: Tensor) -> dict:
    """Compute spectral statistics from singular values.

    Args:
        singular_values: 1-D tensor of singular values.

    Returns:
        dict with keys:
            variance_explained    [K] — per-component fraction
            cumulative_variance   [K]
            effective_rank        float — (sum s^2)^2 / sum s^4
            spectral_entropy      float — -sum p_i log p_i
            k_90, k_95, k_99      int   — dims needed for those thresholds
    """
    s = singular_values.float()
    s2 = s ** 2
    total = s2.sum().clamp(min=1e-12)
    p = s2 / total

    var_explained = p
    cumvar = p.cumsum(0)

    # Effective rank (Roy's measure)
    effective_rank = float((total ** 2) / (s2 ** 2).sum().clamp(min=1e-12))

    # Spectral entropy
    log_p = torch.log(p.clamp(min=1e-12))
    spectral_entropy = float(-(p * log_p).sum())

    def _k_thresh(thr: float) -> int:
        below = int((cumvar < thr).sum().item())
        return below + 1

    return {
        "variance_explained": var_explained,
        "cumulative_variance": cumvar,
        "effective_rank": effective_rank,
        "spectral_entropy": spectral_entropy,
        "k_90": _k_thresh(0.90),
        "k_95": _k_thresh(0.95),
        "k_99": _k_thresh(0.99),
    }


def helix_score(z: Tensor) -> float:
    """Measure how ring-like / helical a point cloud is.

    Projects to 2D via PCA if k > 2, normalises to unit circle, and measures
    uniformity of angular gaps.  Score ∈ [0, 1]:  1 = perfect ring, 0 = cloud.

    Args:
        z: [N, k] tensor of latent coordinates.

    Returns:
        score in [0, 1].
    """
    N = z.shape[0]
    if N < 10:
        return 0.0

    z = z.float()

    # Project to 2D if needed
    if z.shape[1] > 2:
        z_c = z - z.mean(0)
        try:
            _, _, Vh = torch.linalg.svd(z_c, full_matrices=False)
            z2 = z_c @ Vh[:2].T  # [N, 2]
        except Exception:
            z2 = z[:, :2]
    elif z.shape[1] == 2:
        z2 = z
    else:
        # 1-D: can't form a ring
        return 0.0

    # Normalise to unit circle
    norms = z2.norm(dim=1, keepdim=True).clamp(min=1e-12)
    z2_n = z2 / norms

    # Angular coordinates
    theta = torch.atan2(z2_n[:, 1], z2_n[:, 0])  # [N]
    theta_sorted, _ = theta.sort()

    # Angular gaps (include wrap-around)
    gaps = torch.diff(theta_sorted)  # [N-1]
    wrap = (theta_sorted[0] + 2 * math.pi - theta_sorted[-1]).unsqueeze(0)
    all_gaps = torch.cat([gaps, wrap])  # [N]

    expected_gap = 2.0 * math.pi / N
    gap_std = float(all_gaps.std())

    score = float(max(0.0, 1.0 - gap_std / max(expected_gap, 1e-12)))
    return score


def subspace_overlap(V_a: Tensor, V_b: Tensor) -> float:
    """Compute the mean squared cosine of principal angles between two subspaces.

    Args:
        V_a: [D, k_a] orthonormal basis for subspace A.
        V_b: [D, k_b] orthonormal basis for subspace B.

    Returns:
        overlap ∈ [0, 1].  1 = identical subspaces, 0 = orthogonal.
    """
    V_a = V_a.float()
    V_b = V_b.float()
    M = V_a.T @ V_b  # [k_a, k_b]
    try:
        _, cos_angles, _ = torch.linalg.svd(M, full_matrices=False)
    except Exception:
        return 0.0
    cos_angles = cos_angles.clamp(0.0, 1.0)
    overlap = float((cos_angles ** 2).mean())
    return overlap
