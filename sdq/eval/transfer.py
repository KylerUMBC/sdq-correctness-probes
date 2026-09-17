"""Transport transfer: do transports generalize across semantic families?

Certification test D: train transport on family A, evaluate on family B.
If transports capture real surface-form structure (not family-specific
artifacts), they should partially transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sdq.transport.local_operator import LocalTransportField, TransportResult


@dataclass
class TransferResult:
    """Result of evaluating transfer from one family to another."""

    source_family: str
    target_family: str
    within_family_residual: float  # residual on training family
    cross_family_residual: float  # residual on held-out family
    transfer_ratio: float  # cross / within (lower = better transfer)
    baseline_residual: float  # identity transport on held-out family


def evaluate_transfer(
    transport: LocalTransportField,
    within_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    cross_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    source_family: str = "train",
    target_family: str = "test",
) -> TransferResult:
    """Evaluate transport transfer quality.

    Args:
        transport: trained transport field.
        within_pairs: list of (source, target_aligned) from training family.
        cross_pairs: list of (source, target_aligned) from held-out family.
        source_family: label for within-family.
        target_family: label for cross-family.

    Returns:
        TransferResult with residual comparison.
    """
    transport.eval()

    def mean_residual(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        total = 0.0
        count = 0
        with torch.no_grad():
            for src, tgt in pairs:
                result = transport(src, tgt)
                total += result.residual.pow(2).sum(dim=-1).mean().item()
                count += 1
        return total / max(count, 1)

    def identity_residual(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Residual from identity transport (no transformation)."""
        total = 0.0
        count = 0
        with torch.no_grad():
            for src, tgt in pairs:
                src_vel = src[1:] - src[:-1]
                tgt_vel = tgt[1:] - tgt[:-1]
                T = min(src_vel.shape[0], tgt_vel.shape[0])
                residual = tgt_vel[:T] - src_vel[:T]
                total += residual.pow(2).sum(dim=-1).mean().item()
                count += 1
        return total / max(count, 1)

    within = mean_residual(within_pairs)
    cross = mean_residual(cross_pairs)
    baseline = identity_residual(cross_pairs)

    return TransferResult(
        source_family=source_family,
        target_family=target_family,
        within_family_residual=within,
        cross_family_residual=cross,
        transfer_ratio=cross / max(within, 1e-8),
        baseline_residual=baseline,
    )
