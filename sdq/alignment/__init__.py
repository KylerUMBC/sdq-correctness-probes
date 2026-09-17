"""Monotone time alignment between trajectories.

SDQ requires aligning semantically equivalent trajectories that may
realize the same semantic step at different token positions. The
alignment map tau: {0..T_i} -> {0..T_j} must be monotone.
"""

from sdq.alignment.soft_dtw import soft_dtw_distance, soft_dtw_alignment
from sdq.alignment.monotone_alignment import MonotoneAligner

__all__ = [
    "soft_dtw_distance",
    "soft_dtw_alignment",
    "MonotoneAligner",
]
