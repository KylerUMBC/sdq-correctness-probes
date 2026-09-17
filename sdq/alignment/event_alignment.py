"""Event-level alignment using reasoning-stage labels.

When event annotations (premise loading, entity binding, rule application,
answer commitment, etc.) are available, use them to constrain alignment
so that equivalent events are aligned before fine-grained token alignment.
"""

from __future__ import annotations

import torch


def event_constrained_cost(
    cost_matrix: torch.Tensor,
    source_events: list[int],
    target_events: list[int],
    penalty: float = 1e6,
) -> torch.Tensor:
    """Add penalty to cost matrix for mismatched event labels.

    Args:
        cost_matrix: [T1, T2] pairwise distance matrix.
        source_events: [T1] integer event labels for source.
        target_events: [T2] integer event labels for target.
        penalty: added cost for aligning positions with different events.

    Returns:
        [T1, T2] modified cost matrix.
    """
    T1 = len(source_events)
    T2 = len(target_events)
    device = cost_matrix.device
    dtype = cost_matrix.dtype

    se = torch.tensor(source_events, device=device, dtype=torch.long)
    te = torch.tensor(target_events, device=device, dtype=torch.long)

    mismatch = (se.unsqueeze(1) != te.unsqueeze(0)).to(dtype) * penalty
    return cost_matrix + mismatch


def check_event_preservation(
    tau: list[int],
    source_events: list[int],
    target_events: list[int],
) -> dict[str, float]:
    """Check whether alignment preserves event ordering and identity.

    Args:
        tau: monotone alignment map, tau[i] is target index for source i.
        source_events: event labels at each source position.
        target_events: event labels at each target position.

    Returns:
        Dict with 'event_match_rate' and 'order_preserved' metrics.
    """
    matches = 0
    total = len(tau)
    for i, j in enumerate(tau):
        if j < len(target_events) and source_events[i] == target_events[j]:
            matches += 1

    # Check ordering: for consecutive source positions with the same event,
    # the target positions should also be close
    order_violations = 0
    order_checks = 0
    for i in range(len(tau) - 1):
        if source_events[i] == source_events[i + 1]:
            order_checks += 1
            if tau[i + 1] < tau[i]:
                order_violations += 1

    return {
        "event_match_rate": matches / max(total, 1),
        "order_preserved": 1.0 - order_violations / max(order_checks, 1),
    }
