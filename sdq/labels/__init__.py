"""Outcome labeling for the early-warning system."""

from sdq.labels.outcome_labeler import (
    OutcomeLabel,
    label_run,
    label_batch,
    make_future_window_labels,
)

__all__ = [
    "OutcomeLabel",
    "label_run",
    "label_batch",
    "make_future_window_labels",
]
