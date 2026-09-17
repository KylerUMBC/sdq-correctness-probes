"""Tests for sdq.trajectories — extraction, slicing, windowing, normalization."""

import torch
import pytest

from sdq.trajectories.extraction import Trajectory
from sdq.trajectories.slicing import slice_layer, slice_layers, slice_token_range
from sdq.trajectories.windowing import (
    local_windows,
    velocity_vectors,
    curvature_vectors,
    speed_profile,
)
from sdq.trajectories.normalization import (
    center_trajectory,
    normalize_trajectory,
    centroid_subtract_group,
)


class TestTrajectory:
    def test_basic_properties(self):
        states = torch.randn(10, 32)
        traj = Trajectory(states=states, prompt_id="test")
        assert traj.T == 10
        assert traj.D == 32
        assert traj.prompt_id == "test"

    def test_to_device(self):
        traj = Trajectory(states=torch.randn(5, 16))
        moved = traj.to("cpu")
        assert moved.states.device == torch.device("cpu")

    def test_float_conversion(self):
        traj = Trajectory(states=torch.randn(5, 16, dtype=torch.float16))
        converted = traj.float()
        assert converted.states.dtype == torch.float32


class TestSlicing:
    def test_slice_layer(self):
        # Simulate activations [num_layers=4, T=8, D=16]
        activations = torch.randn(4, 8, 16)
        result = slice_layer(activations, layer=2)
        assert result.shape == (8, 16)

    def test_slice_layers_mean(self):
        activations = torch.randn(4, 8, 16)
        result = slice_layers(activations, start=1, end=3, mode="mean")
        assert result.shape == (8, 16)

    def test_slice_layers_concat(self):
        activations = torch.randn(4, 8, 16)
        result = slice_layers(activations, start=1, end=2, mode="concat")
        assert result.shape == (8, 32)  # 2 layers * 16

    def test_slice_token_range(self):
        traj = torch.randn(10, 16)
        sliced = slice_token_range(traj, start=2, end=7)
        assert sliced.shape == (5, 16)


class TestWindowing:
    def test_local_windows(self):
        states = torch.randn(10, 32)
        windows = local_windows(states, window_size=3)
        assert windows.shape == (10, 3, 32)

    def test_velocity_vectors(self):
        states = torch.randn(10, 32)
        vel = velocity_vectors(states)
        assert vel.shape == (9, 32)
        # First velocity should be states[1] - states[0]
        expected = states[1] - states[0]
        assert torch.allclose(vel[0], expected)

    def test_curvature_vectors(self):
        states = torch.randn(10, 32)
        curv = curvature_vectors(states)
        assert curv.shape == (8, 32)

    def test_speed_profile(self):
        states = torch.randn(10, 32)
        speed = speed_profile(states)
        assert speed.shape == (9,)
        assert (speed >= 0).all()


class TestNormalization:
    def test_center_trajectory(self):
        states = torch.randn(10, 32)
        centered = center_trajectory(states)
        assert centered.shape == states.shape
        # Mean should be ~0
        assert centered.mean(dim=0).abs().max() < 1e-5

    def test_normalize_unit_norm(self):
        states = torch.randn(10, 32)
        normed = normalize_trajectory(states, mode="unit_norm")
        norms = normed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_normalize_standardize(self):
        states = torch.randn(10, 32) * 3 + 5
        normed = normalize_trajectory(states, mode="standardize")
        # Should have mean ~0 and std ~1 per dimension
        assert normed.mean(dim=0).abs().max() < 1e-4

    def test_centroid_subtract_group(self):
        trajs = [torch.randn(10, 32) + 5 for _ in range(3)]
        subtracted = centroid_subtract_group(trajs)
        assert len(subtracted) == 3
        # Group centroid should be ~0
        centroid = torch.stack([t.mean(dim=0) for t in subtracted]).mean(dim=0)
        assert centroid.abs().max() < 1e-5
