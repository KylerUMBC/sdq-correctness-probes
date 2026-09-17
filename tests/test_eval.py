"""Tests for sdq.eval — certification tests."""

import torch
import pytest

from sdq.eval.transport_rank import rank_analysis, transport_validity_score
from sdq.eval.same_answer_diff_reasoning import same_answer_test
from sdq.eval.hard_negatives import hard_negative_test
from sdq.eval.event_stability import event_stability_test


class TestRankAnalysis:
    def test_identity_zero_rank(self):
        G = torch.eye(16)
        result = rank_analysis(G)
        assert result.nuclear_norm < 1e-4
        assert result.frobenius_deviation < 1e-4

    def test_low_rank_perturbation(self):
        u = torch.randn(16, 1)
        v = torch.randn(16, 1)
        G = torch.eye(16) + u @ v.T
        result = rank_analysis(G)
        assert result.nuclear_norm > 0
        assert result.effective_rank < 3.0  # should be ~1

    def test_time_averaged(self):
        T, D = 5, 16
        G_t = torch.eye(D).unsqueeze(0).expand(T, -1, -1) + 0.01 * torch.randn(T, D, D)
        result = rank_analysis(G_t)
        assert result.frobenius_deviation > 0


class TestTransportValidity:
    def test_better_than_shuffle(self):
        transport = torch.tensor([0.1, 0.2, 0.15])
        shuffle = torch.tensor([0.5, 0.6, 0.55])
        score = transport_validity_score(transport, shuffle)
        assert score > 1.0

    def test_same_as_shuffle(self):
        vals = torch.tensor([0.3, 0.3, 0.3])
        score = transport_validity_score(vals, vals)
        assert abs(score - 1.0) < 1e-5


class TestSameAnswerTest:
    def test_separation(self):
        # Same reasoning: close latents
        same = [(torch.randn(5, 8), torch.randn(5, 8) * 0.1) for _ in range(3)]
        # Different reasoning: far latents
        diff = [(torch.randn(5, 8), torch.randn(5, 8) * 10) for _ in range(3)]
        result = same_answer_test(same, diff)
        assert result.diff_reasoning_distance > result.same_reasoning_distance
        assert result.passes


class TestHardNegativeTest:
    def test_separation(self):
        pos = [(torch.zeros(5, 8), torch.zeros(5, 8) + 0.01) for _ in range(3)]
        neg = [(torch.randn(5, 8), torch.randn(5, 8) * 5) for _ in range(3)]
        result = hard_negative_test(pos, neg)
        assert result.hard_negative_distance > result.positive_distance
        assert result.passes


class TestEventStability:
    def test_perfect_events(self):
        alignments = [[0, 1, 2, 3, 4]] * 3
        src_events = [[0, 0, 1, 1, 2]] * 3
        tgt_events = [[0, 0, 1, 1, 2]] * 3
        result = event_stability_test(alignments, src_events, tgt_events)
        assert result.mean_match_rate == 1.0
        assert result.order_preservation_rate == 1.0
        assert result.num_pairs == 3
