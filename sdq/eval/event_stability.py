"""Event stability: reasoning-stage ordering preservation.

Certification test G: do aligned trajectories preserve the
ordering and structure of reasoning stages (events)?

This ensures that alignment respects semantic process structure,
not just geometric convenience.
"""

from __future__ import annotations

from dataclasses import dataclass

from sdq.alignment.event_alignment import check_event_preservation


@dataclass
class EventStabilityResult:
    """Result of event stability evaluation."""

    mean_match_rate: float  # average event match rate across pairs
    order_preservation_rate: float  # fraction of pairs with preserved order
    num_pairs: int
    per_pair_metrics: list[dict[str, float]]


def event_stability_test(
    alignments: list[list[int]],
    source_events_list: list[list[int | str]],
    target_events_list: list[list[int | str]],
) -> EventStabilityResult:
    """Evaluate event preservation across aligned pairs.

    Args:
        alignments: list of monotone alignment maps (one per pair).
        source_events_list: event labels for source trajectory (one per pair).
        target_events_list: event labels for target trajectory (one per pair).

    Returns:
        EventStabilityResult with aggregate metrics.
    """
    per_pair: list[dict[str, float]] = []
    total_match = 0.0
    total_order = 0

    n = len(alignments)
    for alignment, src_events, tgt_events in zip(alignments, source_events_list, target_events_list):
        metrics = check_event_preservation(alignment, src_events, tgt_events)
        per_pair.append(metrics)
        total_match += metrics["event_match_rate"]
        if metrics["order_preserved"]:
            total_order += 1

    return EventStabilityResult(
        mean_match_rate=total_match / max(n, 1),
        order_preservation_rate=total_order / max(n, 1),
        num_pairs=n,
        per_pair_metrics=per_pair,
    )
