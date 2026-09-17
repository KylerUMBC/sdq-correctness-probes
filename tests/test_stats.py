"""Tests for sdq.eval.stats — Wilson CI, Fisher exact, Holm adjustment."""

from __future__ import annotations

import pytest

from sdq.eval.stats import fisher_exact_greater, holm_adjust, wilson_interval


class TestWilsonInterval:
    def test_contains_point_estimate(self):
        lo, hi = wilson_interval(10, 100)
        assert lo < 0.10 < hi

    def test_bounds_in_unit_interval(self):
        for k, n in [(0, 10), (10, 10), (1, 3), (50, 80)]:
            lo, hi = wilson_interval(k, n)
            assert 0.0 <= lo <= hi <= 1.0

    def test_zero_n_uninformative(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_narrows_with_n(self):
        lo1, hi1 = wilson_interval(5, 50)
        lo2, hi2 = wilson_interval(50, 500)
        assert (hi2 - lo2) < (hi1 - lo1)

    def test_zero_successes_lo_is_zero(self):
        lo, hi = wilson_interval(0, 20)
        assert lo == 0.0
        assert hi > 0.0


class TestFisherExactGreater:
    def test_clearly_greater_is_small(self):
        # 40/50 vs 5/50 — overwhelming evidence
        p = fisher_exact_greater(40, 50, 5, 50)
        assert p < 1e-6

    def test_equal_rates_is_large(self):
        p = fisher_exact_greater(5, 50, 5, 50)
        assert p > 0.4

    def test_lesser_rate_is_near_one(self):
        p = fisher_exact_greater(1, 50, 20, 50)
        assert p > 0.99

    def test_zero_counts(self):
        # 0 vs 0: p must be 1 (no evidence of difference)
        assert fisher_exact_greater(0, 50, 0, 50) == pytest.approx(1.0)

    def test_empty_condition_returns_one(self):
        assert fisher_exact_greater(0, 0, 5, 50) == 1.0

    def test_known_2x2_value(self):
        # Classic 2x2: k1=3,n1=5 vs k2=1,n2=5; hypergeom P(X>=3)
        # K=4, N=10: P(X=3) = C(4,3)C(6,2)/C(10,5) = 4*15/252
        # P(X=4) = C(4,4)C(6,1)/C(10,5) = 6/252  -> total 66/252
        p = fisher_exact_greater(3, 5, 1, 5)
        assert p == pytest.approx(66 / 252, rel=1e-9)

    def test_monotone_in_k1(self):
        ps = [fisher_exact_greater(k, 50, 5, 50) for k in range(0, 20, 4)]
        assert all(ps[i] >= ps[i + 1] for i in range(len(ps) - 1))


class TestHolmAdjust:
    def test_empty(self):
        assert holm_adjust([]) == []

    def test_single_unchanged(self):
        assert holm_adjust([0.03]) == [0.03]

    def test_order_preserved(self):
        raw = [0.04, 0.001, 0.20]
        adj = holm_adjust(raw)
        assert len(adj) == 3
        # smallest raw p gets the largest multiplier (m)
        assert adj[1] == pytest.approx(0.003)

    def test_monotone_nondecreasing_in_rank(self):
        raw = [0.01, 0.02, 0.03, 0.04]
        adj = holm_adjust(raw)
        ordered = sorted(zip(raw, adj))
        adj_in_rank_order = [a for _, a in ordered]
        assert all(adj_in_rank_order[i] <= adj_in_rank_order[i + 1]
                   for i in range(len(adj_in_rank_order) - 1))

    def test_capped_at_one(self):
        adj = holm_adjust([0.5, 0.9, 0.8])
        assert all(a <= 1.0 for a in adj)

    def test_controls_typical_case(self):
        # 10 cells, one real signal at p=0.001: survives (0.001*10 = 0.01)
        raw = [0.001] + [0.5] * 9
        adj = holm_adjust(raw)
        assert adj[0] == pytest.approx(0.01)
        assert adj[0] < 0.05
