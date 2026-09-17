"""Same-answer / different-reasoning separation test.

Certification test E: prompts that reach the same final answer
but through genuinely different reasoning should NOT be collapsed
into the same equivalence class.

This is the sharpest test of whether SDQ captures reasoning
dynamics rather than just answer similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SameAnswerTestResult:
    """Result of same-answer / different-reasoning test."""

    same_reasoning_distance: float  # mean latent distance, same-reasoning pairs
    diff_reasoning_distance: float  # mean latent distance, diff-reasoning pairs
    separation_ratio: float  # diff / same (higher = better separation)
    transport_same_residual: float  # transport residual for same-reasoning
    transport_diff_residual: float  # transport residual for different-reasoning
    passes: bool  # True if diff_reasoning_distance > same_reasoning_distance


def same_answer_test(
    same_reasoning_latents: list[tuple[torch.Tensor, torch.Tensor]],
    diff_reasoning_latents: list[tuple[torch.Tensor, torch.Tensor]],
    same_reasoning_transport_residuals: list[float] | None = None,
    diff_reasoning_transport_residuals: list[float] | None = None,
) -> SameAnswerTestResult:
    """Evaluate same-answer / different-reasoning separation.

    Args:
        same_reasoning_latents: list of (z_i, z_j) pairs that share
            both answer AND reasoning (should be close).
        diff_reasoning_latents: list of (z_i, z_j) pairs that share
            answer but NOT reasoning (should be far).
        same_reasoning_transport_residuals: optional per-pair transport residuals.
        diff_reasoning_transport_residuals: optional per-pair transport residuals.

    Returns:
        SameAnswerTestResult with separation metrics.
    """
    def mean_distance(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        if not pairs:
            return 0.0
        total = 0.0
        for z_i, z_j in pairs:
            T = min(z_i.shape[0], z_j.shape[0])
            total += (z_i[:T] - z_j[:T]).pow(2).sum(dim=-1).mean().item()
        return total / len(pairs)

    same_dist = mean_distance(same_reasoning_latents)
    diff_dist = mean_distance(diff_reasoning_latents)

    same_trans = 0.0
    if same_reasoning_transport_residuals:
        same_trans = sum(same_reasoning_transport_residuals) / len(same_reasoning_transport_residuals)

    diff_trans = 0.0
    if diff_reasoning_transport_residuals:
        diff_trans = sum(diff_reasoning_transport_residuals) / len(diff_reasoning_transport_residuals)

    return SameAnswerTestResult(
        same_reasoning_distance=same_dist,
        diff_reasoning_distance=diff_dist,
        separation_ratio=diff_dist / max(same_dist, 1e-8),
        transport_same_residual=same_trans,
        transport_diff_residual=diff_trans,
        passes=diff_dist > same_dist,
    )
