"""Event consistency loss: preserve reasoning-stage structure under alignment.

If event labels (inferred reasoning stages) are available, the
alignment should preserve:
  - Same event ordering
  - Same event transitions
  - Same reasoning-stage segmentation

This prevents alignments that are geometrically convenient
but semantically incoherent.
"""

from __future__ import annotations

import torch

from sdq.alignment.event_alignment import check_event_preservation


def event_consistency_loss(
    alignment: list[int],
    source_events: list[int | str],
    target_events: list[int | str],
    order_weight: float = 1.0,
    match_weight: float = 1.0,
) -> torch.Tensor:
    """Compute event consistency loss from alignment and event labels.

    Converts the discrete event preservation metrics into a
    differentiable-friendly scalar (complement of preservation rate).

    Args:
        alignment: monotone map tau: source_idx -> target_idx.
        source_events: event labels for source trajectory.
        target_events: event labels for target trajectory.
        order_weight: weight for order preservation term.
        match_weight: weight for event match rate term.

    Returns:
        Scalar loss (0 = perfect preservation, higher = worse).
    """
    metrics = check_event_preservation(alignment, source_events, target_events)

    # Convert rates to losses (1 - rate)
    match_loss = 1.0 - metrics["event_match_rate"]
    order_loss = 1.0 - metrics["order_preserved"]

    loss_val = match_weight * match_loss + order_weight * order_loss
    return torch.tensor(loss_val)


def soft_event_consistency_loss(
    soft_alignment: torch.Tensor,
    source_event_ids: torch.Tensor,
    target_event_ids: torch.Tensor,
) -> torch.Tensor:
    """Differentiable event consistency using soft alignment matrix.

    For each source position t, the soft alignment gives a distribution
    over target positions. The loss penalizes probability mass placed on
    target positions with different event labels.

    Args:
        soft_alignment: [T_src, T_tgt] soft alignment matrix (rows sum to 1).
        source_event_ids: [T_src] integer event labels.
        target_event_ids: [T_tgt] integer event labels.

    Returns:
        Scalar loss (expected mismatch rate).
    """
    # [T_src, T_tgt] binary mask: 1 where events match
    match_mask = (source_event_ids.unsqueeze(1) == target_event_ids.unsqueeze(0)).float()

    # Expected match probability per source position
    match_prob = (soft_alignment * match_mask).sum(dim=1)  # [T_src]

    # Loss = mean (1 - match_probability)
    return (1.0 - match_prob).mean()
