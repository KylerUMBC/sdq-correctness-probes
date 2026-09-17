"""Tests for sdq.eval.commitment_probe.

Validates probe training, AUROC computation, subspace rank sweep,
cross-family transfer, and commitment direction extraction on
synthetic data with known separability.
"""

from __future__ import annotations

import torch
import pytest

from sdq.eval.commitment_probe import (
    LinearCommitmentProbe,
    LowRankCommitmentProbe,
    Standardizer,
    compute_auroc,
    cross_family_transfer,
    extract_commitment_direction,
    extract_logit_features,
    mean_within_family_auroc,
    per_family_auroc,
    subspace_rank_sweep,
    train_probe,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_separable_data(
    n: int = 200,
    dim: int = 64,
    signal_dim: int = 4,
    seed: int = 42,
):
    """Create synthetic data where classes are separable in the first
    `signal_dim` dimensions and the rest is noise.

    Returns X [n, dim], y [n] (binary), families [n].
    """
    torch.manual_seed(seed)

    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    X = torch.randn(n, dim) * 0.5

    # Plant signal in the first signal_dim dimensions
    for i in range(signal_dim):
        X[:n // 2, i] += 1.0
        X[n // 2:, i] -= 1.0

    # Assign families
    fam_names = ["alpha", "beta", "gamma", "delta"]
    families = [fam_names[i % len(fam_names)] for i in range(n)]

    # Shuffle so train/test splits are balanced
    perm = torch.randperm(n)
    X = X[perm]
    y = y[perm]
    families = [families[i] for i in perm.tolist()]

    return X, y, families


def _make_degenerate_data(n: int = 50, dim: int = 16, seed: int = 42):
    """Data with only one class — for edge cases."""
    torch.manual_seed(seed)
    X = torch.randn(n, dim)
    y = torch.ones(n).long()
    families = ["only_fam"] * n
    return X, y, families


# ── AUROC ────────────────────────────────────────────────────────────────────

class TestComputeAuroc:
    def test_perfect_separation(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.1, 0.0])
        labels = torch.tensor([1, 1, 1, 0, 0])
        assert compute_auroc(scores, labels) == 1.0

    def test_inverse_separation(self):
        scores = torch.tensor([0.0, 0.1, 0.9, 1.0])
        labels = torch.tensor([1, 1, 0, 0])
        assert compute_auroc(scores, labels) == 0.0

    def test_random_scores_near_half(self):
        torch.manual_seed(0)
        n = 1000
        scores = torch.rand(n)
        labels = (torch.rand(n) > 0.5).long()
        auc = compute_auroc(scores, labels)
        assert 0.3 < auc < 0.7

    def test_single_class_returns_half(self):
        assert compute_auroc(torch.tensor([0.5, 0.6]), torch.tensor([1, 1])) == 0.5
        assert compute_auroc(torch.tensor([0.5, 0.6]), torch.tensor([0, 0])) == 0.5

    def test_empty_returns_half(self):
        assert compute_auroc(torch.tensor([]), torch.tensor([])) == 0.5

    def test_all_tied_scores_give_half(self):
        # Constant scores carry no information: AUROC must be exactly 0.5,
        # not an artifact of sort order (audit fix — this is the family-prior
        # baseline situation).
        scores = torch.tensor([0.7, 0.7, 0.7, 0.7])
        labels = torch.tensor([1, 0, 1, 0])
        assert compute_auroc(scores, labels) == pytest.approx(0.5)

    def test_partial_ties_get_half_credit(self):
        # pos scores: [0.9, 0.5], neg scores: [0.5, 0.1]
        # pairs: (0.9>0.5)=1, (0.9>0.1)=1, (0.5=0.5)=0.5, (0.5>0.1)=1
        # AUROC = 3.5 / 4
        scores = torch.tensor([0.9, 0.5, 0.5, 0.1])
        labels = torch.tensor([1, 1, 0, 0])
        assert compute_auroc(scores, labels) == pytest.approx(3.5 / 4)

    def test_tie_invariant_to_order(self):
        scores = torch.tensor([0.5, 0.5, 0.5, 0.9, 0.1, 0.5])
        labels = torch.tensor([1, 0, 1, 1, 0, 0])
        a1 = compute_auroc(scores, labels)
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        a2 = compute_auroc(scores[perm], labels[perm])
        assert a1 == pytest.approx(a2)


class TestPerFamilyAuroc:
    def test_basic(self):
        scores = torch.tensor([0.9, 0.1, 0.8, 0.2])
        labels = torch.tensor([1, 0, 1, 0])
        families = ["a", "a", "b", "b"]
        result = per_family_auroc(scores, labels, families)
        assert result["a"] == 1.0
        assert result["b"] == 1.0

    def test_degenerate_family_is_nan(self):
        scores = torch.tensor([0.5, 0.6, 0.7])
        labels = torch.tensor([1, 1, 0])
        families = ["a", "a", "b"]
        result = per_family_auroc(scores, labels, families)
        # family "b" has only one sample
        assert result["b"] != result["b"]  # NaN check


class TestMeanWithinFamilyAuroc:
    def test_filters_nan(self):
        scores = torch.tensor([0.9, 0.1, 0.5])
        labels = torch.tensor([1, 0, 1])
        families = ["a", "a", "b"]  # b is degenerate
        mwf = mean_within_family_auroc(scores, labels, families)
        assert mwf == 1.0  # only family "a" is valid


# ── Standardizer ─────────────────────────────────────────────────────────────

class TestStandardizer:
    def test_zero_mean_unit_var(self):
        X = torch.randn(100, 10)
        scaler = Standardizer.fit(X)
        Xt = scaler.transform(X)
        assert Xt.mean(dim=0).abs().max().item() < 0.01
        assert (Xt.std(dim=0) - 1.0).abs().max().item() < 0.05


# ── Probes ───────────────────────────────────────────────────────────────────

class TestLinearCommitmentProbe:
    def test_output_shape(self):
        probe = LinearCommitmentProbe(32)
        X = torch.randn(10, 32)
        out = probe(X)
        assert out.shape == (10,)

    def test_weight_vector_shape(self):
        probe = LinearCommitmentProbe(64)
        w = probe.weight_vector
        assert w.shape == (64,)


class TestLowRankCommitmentProbe:
    def test_output_shape(self):
        probe = LowRankCommitmentProbe(32, rank=4)
        X = torch.randn(10, 32)
        out = probe(X)
        assert out.shape == (10,)

    def test_projection_matrix_shape(self):
        probe = LowRankCommitmentProbe(64, rank=8)
        P = probe.projection_matrix
        assert P.shape == (64, 8)


# ── Logit features ───────────────────────────────────────────────────────────

class TestExtractLogitFeatures:
    def test_output_shape(self):
        logits = torch.randn(50000)
        feats = extract_logit_features(logits)
        assert feats.shape == (5,)

    def test_entropy_nonneg(self):
        logits = torch.randn(1000)
        feats = extract_logit_features(logits)
        assert feats[0].item() >= 0  # entropy

    def test_peaked_distribution_low_entropy(self):
        logits = torch.zeros(1000)
        logits[0] = 100.0  # one dominant logit
        feats = extract_logit_features(logits)
        assert feats[0].item() < 0.01  # very low entropy
        assert feats[1].item() > 0.99  # max prob near 1

    def test_uniform_high_entropy(self):
        logits = torch.zeros(1000)  # uniform
        feats = extract_logit_features(logits)
        assert feats[0].item() > 5.0  # high entropy
        assert feats[2].item() < 0.01  # no margin


# ── Train probe (integration) ───────────────────────────────────────────────

class TestTrainProbe:
    def test_separable_data_high_auroc(self):
        X, y, families = _make_separable_data(n=200, dim=64, signal_dim=4)
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]
        test_fams = families[160:]

        probe = LinearCommitmentProbe(64)
        result = train_probe(
            probe, X_train, y_train, X_test, y_test, test_fams,
            name="test_probe", epochs=200, lr=1e-2,
            weight_decay=1e-3, device="cpu",
        )
        assert result.pooled_auroc > 0.85
        assert result.name == "test_probe"

    def test_degenerate_data(self):
        X, y, families = _make_degenerate_data()
        probe = LinearCommitmentProbe(16)
        result = train_probe(
            probe, X[:40], y[:40], X[40:], y[40:], families[40:],
            name="degen", epochs=50, device="cpu",
        )
        # Degenerate: all same label, AUROC should be 0.5
        assert result.pooled_auroc == 0.5


# ── Cross-family transfer ───────────────────────────────────────────────────

class TestCrossFamilyTransfer:
    def test_returns_one_per_family(self):
        X, y, families = _make_separable_data(n=200, dim=32, signal_dim=4)
        results = cross_family_transfer(
            X, y, families, input_dim=32,
            epochs=100, lr=1e-2, device="cpu",
        )
        unique_fams = set(families)
        assert len(results) == len(unique_fams)

    def test_transfer_above_chance_on_separable_data(self):
        # With signal planted in the same dims for all families, transfer
        # should work above chance.
        X, y, families = _make_separable_data(n=400, dim=32, signal_dim=4)
        results = cross_family_transfer(
            X, y, families, input_dim=32,
            epochs=200, lr=1e-2, device="cpu",
        )
        valid = [r.auroc for r in results if r.auroc == r.auroc]
        assert all(v > 0.6 for v in valid), f"Some transfers below chance: {valid}"


# ── Subspace rank sweep ─────────────────────────────────────────────────────

class TestSubspaceRankSweep:
    def test_low_rank_signal_concentrates(self):
        # Signal in 4 dims — rank-4 should capture most of it.
        X, y, families = _make_separable_data(n=300, dim=64, signal_dim=4)
        X_train, X_test = X[:240], X[240:]
        y_train, y_test = y[:240], y[240:]
        test_fams = families[240:]

        result = subspace_rank_sweep(
            X_train, y_train, X_test, y_test, test_fams,
            full_auroc=0.95,  # assumed full-rank performance
            max_rank=32,
            ranks=[1, 2, 4, 8, 16, 32],
            epochs=200, lr=1e-2, device="cpu",
        )
        assert result.rank_for_90pct is not None
        assert result.rank_for_90pct <= 16  # signal is in 4 dims

    def test_rank_aurocs_monotone_ish(self):
        """AUROC should generally increase with rank (not strictly)."""
        X, y, families = _make_separable_data(n=200, dim=32, signal_dim=4)
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]
        test_fams = families[160:]

        result = subspace_rank_sweep(
            X_train, y_train, X_test, y_test, test_fams,
            full_auroc=0.95,
            max_rank=32,
            ranks=[1, 4, 16, 32],
            epochs=200, lr=1e-2, device="cpu",
        )
        aurocs = [result.rank_aurocs[k] for k in sorted(result.rank_aurocs.keys())]
        # At least the last should be >= the first
        assert aurocs[-1] >= aurocs[0] - 0.05


# ── Commitment direction ────────────────────────────────────────────────────

class TestExtractCommitmentDirection:
    def test_unit_vector(self):
        probe = LinearCommitmentProbe(32)
        # manually set weights
        probe.linear.weight.data = torch.randn(1, 32)
        direction = extract_commitment_direction(probe, top_k=5)
        assert abs(direction.direction.norm().item() - 1.0) < 1e-5

    def test_top_k(self):
        probe = LinearCommitmentProbe(64)
        probe.linear.weight.data = torch.randn(1, 64)
        direction = extract_commitment_direction(probe, top_k=10)
        assert len(direction.top_components_idx) == 10
        assert len(direction.top_components_weight) == 10

    def test_top_components_descending(self):
        probe = LinearCommitmentProbe(32)
        probe.linear.weight.data = torch.randn(1, 32)
        direction = extract_commitment_direction(probe, top_k=10)
        for i in range(len(direction.top_components_weight) - 1):
            assert direction.top_components_weight[i] >= direction.top_components_weight[i + 1]
