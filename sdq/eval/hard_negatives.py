"""Hard-negative resistance: surface-similar but semantically different.

Certification test F: prompts that share surface form (similar wording,
same template) but differ semantically should NOT be collapsed.

This prevents the model from learning "same template = same semantics",
which would be a trivial shortcut inconsistent with SDQ's goals.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HardNegativeResult:
    """Result of hard-negative resistance evaluation."""

    positive_distance: float  # latent distance, true equivalents
    hard_negative_distance: float  # latent distance, surface-similar but different
    separation_ratio: float  # neg / pos (higher = better resistance)
    transport_positive_residual: float
    transport_negative_residual: float
    passes: bool  # True if hard_negative_distance > positive_distance


def hard_negative_test(
    positive_latents: list[tuple[torch.Tensor, torch.Tensor]],
    hard_negative_latents: list[tuple[torch.Tensor, torch.Tensor]],
    positive_transport_residuals: list[float] | None = None,
    negative_transport_residuals: list[float] | None = None,
) -> HardNegativeResult:
    """Evaluate resistance to hard negatives.

    Args:
        positive_latents: list of (z_i, z_j) for semantically equivalent pairs.
        hard_negative_latents: list of (z_i, z_j) for surface-similar but
            semantically different pairs.
        positive_transport_residuals: optional per-pair residuals for positives.
        negative_transport_residuals: optional per-pair residuals for negatives.

    Returns:
        HardNegativeResult with separation metrics.
    """
    def mean_distance(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        if not pairs:
            return 0.0
        total = 0.0
        for z_i, z_j in pairs:
            T = min(z_i.shape[0], z_j.shape[0])
            total += (z_i[:T] - z_j[:T]).pow(2).sum(dim=-1).mean().item()
        return total / len(pairs)

    pos_dist = mean_distance(positive_latents)
    neg_dist = mean_distance(hard_negative_latents)

    pos_trans = 0.0
    if positive_transport_residuals:
        pos_trans = sum(positive_transport_residuals) / len(positive_transport_residuals)

    neg_trans = 0.0
    if negative_transport_residuals:
        neg_trans = sum(negative_transport_residuals) / len(negative_transport_residuals)

    return HardNegativeResult(
        positive_distance=pos_dist,
        hard_negative_distance=neg_dist,
        separation_ratio=neg_dist / max(pos_dist, 1e-8),
        transport_positive_residual=pos_trans,
        transport_negative_residual=neg_trans,
        passes=neg_dist > pos_dist,
    )
