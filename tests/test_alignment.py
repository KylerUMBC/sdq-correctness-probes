"""Tests for sdq.alignment — soft_dtw, monotone_alignment, event_alignment."""

import torch
import pytest

from sdq.alignment.soft_dtw import (
    soft_dtw_distance,
    soft_dtw_alignment,
    alignment_to_map,
)
from sdq.alignment.monotone_alignment import MonotoneAligner
from sdq.alignment.event_alignment import (
    event_constrained_cost,
    check_event_preservation,
)


class TestSoftDTW:
    def test_identical_sequences_zero_distance(self):
        x = torch.randn(8, 16)
        dist = soft_dtw_distance(x, x, gamma=1.0)
        assert dist.item() < 1e-3

    def test_distance_positive(self):
        x = torch.randn(8, 16)
        y = torch.randn(10, 16)
        dist = soft_dtw_distance(x, y, gamma=1.0)
        assert dist.item() > 0

    def test_alignment_returns_path(self):
        x = torch.randn(6, 8)
        y = torch.randn(8, 8)
        dist, path = soft_dtw_alignment(x, y, gamma=1.0)
        assert len(path) > 0
        # Path should start at (0,0) and end at (T1-1, T2-1)
        assert path[0] == (0, 0)
        assert path[-1] == (5, 7)

    def test_alignment_to_map(self):
        path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5)]
        amap = alignment_to_map(path, source_len=5)
        assert len(amap) == 5
        assert amap[0] == 0
        assert amap[4] == 5

    def test_gradient_flows(self):
        x = torch.randn(5, 8, requires_grad=True)
        y = torch.randn(7, 8)
        dist = soft_dtw_distance(x, y, gamma=1.0)
        dist.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestMonotoneAligner:
    def test_output_shape(self, hidden_dim, seq_len):
        aligner = MonotoneAligner(hidden_dim)
        source = torch.randn(seq_len, hidden_dim)
        target = torch.randn(seq_len + 2, hidden_dim)
        A = aligner(source, target)
        assert A.shape == (seq_len, seq_len + 2)

    def test_rows_sum_to_one(self, hidden_dim, seq_len):
        aligner = MonotoneAligner(hidden_dim)
        source = torch.randn(seq_len, hidden_dim)
        target = torch.randn(seq_len, hidden_dim)
        A = aligner(source, target)
        row_sums = A.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_align_trajectory(self, hidden_dim, seq_len):
        aligner = MonotoneAligner(hidden_dim)
        source = torch.randn(seq_len, hidden_dim)
        target = torch.randn(seq_len + 3, hidden_dim)
        warped = aligner.align_trajectory(source, target)
        # Warped target should match source length
        assert warped.shape == (seq_len, hidden_dim)

    def test_hard_alignment_monotone(self, hidden_dim, seq_len):
        aligner = MonotoneAligner(hidden_dim)
        source = torch.randn(seq_len, hidden_dim)
        target = torch.randn(seq_len, hidden_dim)
        tau = aligner.hard_alignment(source, target)
        assert len(tau) == seq_len
        # Should be monotonically non-decreasing
        for i in range(len(tau) - 1):
            assert tau[i] <= tau[i + 1]


class TestEventAlignment:
    def test_perfect_alignment(self):
        alignment = [0, 1, 2, 3, 4]
        events_src = [0, 0, 1, 1, 2]
        events_tgt = [0, 0, 1, 1, 2]
        metrics = check_event_preservation(alignment, events_src, events_tgt)
        assert metrics["event_match_rate"] == 1.0
        assert metrics["order_preserved"] == 1.0

    def test_mismatched_events(self):
        alignment = [0, 1, 2, 3, 4]
        events_src = [0, 0, 1, 1, 2]
        events_tgt = [1, 1, 0, 0, 2]  # reversed order
        metrics = check_event_preservation(alignment, events_src, events_tgt)
        assert metrics["event_match_rate"] < 1.0

    def test_event_constrained_cost(self):
        cost = torch.ones(5, 5)
        events_src = [0, 0, 1, 1, 2]
        events_tgt = [0, 0, 1, 1, 2]
        modified = event_constrained_cost(cost, events_src, events_tgt, penalty=100.0)
        # Matching positions should have same cost, mismatched should be penalized
        assert modified[0, 0] == cost[0, 0]  # same event
        assert modified[0, 2] > cost[0, 2]  # different events
