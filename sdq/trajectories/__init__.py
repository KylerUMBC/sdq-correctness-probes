"""Trajectory extraction, slicing, windowing, and normalization.

A trajectory is the core SDQ object: a sequence of hidden states
across token positions for a chosen layer (or layer range).
"""

from sdq.trajectories.extraction import Trajectory, extract_trajectory, extract_trajectories
from sdq.trajectories.slicing import slice_layer, slice_layers, slice_token_range
from sdq.trajectories.windowing import local_windows, velocity_vectors
from sdq.trajectories.normalization import normalize_trajectory, center_trajectory

__all__ = [
    "Trajectory",
    "extract_trajectory",
    "extract_trajectories",
    "slice_layer",
    "slice_layers",
    "slice_token_range",
    "local_windows",
    "velocity_vectors",
    "normalize_trajectory",
    "center_trajectory",
]
