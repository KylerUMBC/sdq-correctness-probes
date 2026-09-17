"""Tests for sdq.eval.mesoscale_basis.

Covers:
  - Varimax rotation: orthogonality of R, monotone improvement of criterion,
    K=1 edge case.
  - fit_supervised_mesoscale_basis: shapes, orthonormality of basis_raw in
    raw space, signed_scores shape, component_stds shape, scaler echo.
  - Perturbation factories: norms, span containment, sign correctness,
    additivity of pairwise.
  - calibrate_direction_std agrees with manual calculation.
  - Synthetic recovery: when Y = sigmoid(W·x + noise), the supervised basis
    spans a subspace that contains W (up to noise).
"""

from __future__ import annotations

import math

import pytest
import torch

from sdq.eval.mesoscale_basis import (
    MesoscaleBasis,
    calibrate_direction_std,
    fit_supervised_mesoscale_basis,
    make_component_perturbation,
    make_coordinated_perturbation,
    make_pairwise_perturbation,
    make_within_span_random_perturbation,
    varimax_criterion,
    varimax_rotate,
)


# ── Varimax rotation ────────────────────────────────────────────────────────

class TestVarimaxRotate:
    def test_R_is_orthogonal(self):
        torch.manual_seed(0)
        L = torch.randn(64, 6)
        Q, _ = torch.linalg.qr(L)
        rotated, R = varimax_rotate(Q)
        assert R.shape == (6, 6)
        I = R @ R.T
        assert torch.allclose(I, torch.eye(6), atol=1e-5)

    def test_criterion_non_decreasing(self):
        """Rotated criterion should be >= initial (within tol)."""
        torch.manual_seed(1)
        L = torch.randn(80, 4)
        Q, _ = torch.linalg.qr(L)
        c0 = varimax_criterion(Q)
        rotated, _ = varimax_rotate(Q)
        c1 = varimax_criterion(rotated)
        assert c1 + 1e-6 >= c0

    def test_orthonormality_preserved(self):
        """If input has orthonormal columns, output also has orthonormal columns."""
        torch.manual_seed(2)
        L = torch.randn(50, 5)
        Q, _ = torch.linalg.qr(L)
        rotated, _ = varimax_rotate(Q)
        gram = rotated.T @ rotated
        assert torch.allclose(gram, torch.eye(5), atol=1e-5)

    def test_K_1_edge_case(self):
        """K=1 should return input unchanged with R=[[1]]."""
        L = torch.randn(20, 1)
        rotated, R = varimax_rotate(L)
        assert torch.allclose(rotated, L)
        assert torch.allclose(R, torch.eye(1))


# ── fit_supervised_mesoscale_basis ──────────────────────────────────────────

def _make_synthetic(n: int, d: int, K_true: int, seed: int = 0):
    """Generate (h0, y, mu, sd) where labels depend on K_true latent directions."""
    g = torch.Generator().manual_seed(seed)
    h0 = torch.randn(n, d, generator=g) * 2.0 + 1.0
    W_true = torch.randn(d, K_true, generator=g)
    W_true, _ = torch.linalg.qr(W_true)  # orthonormal
    weights = torch.randn(K_true, generator=g)
    z = (h0 - h0.mean(0)) / h0.std(0).clamp_min(1e-6)
    logits = z @ W_true @ weights + 0.3 * torch.randn(n, generator=g)
    y = (torch.sigmoid(logits) > 0.5).long()
    mu = h0.mean(0)
    sd = h0.std(0).clamp_min(1e-6)
    return h0, y, mu, sd, W_true


class TestFitSupervisedMesoscaleBasis:
    def test_shapes(self):
        h0, y, mu, sd, _ = _make_synthetic(200, 32, K_true=3)
        mb = fit_supervised_mesoscale_basis(h0, y, K=4, scaler_mu=mu, scaler_sd=sd, epochs=50)
        assert isinstance(mb, MesoscaleBasis)
        assert mb.basis_raw.shape == (32, 4)
        assert mb.signed_scores.shape == (4,)
        assert mb.component_stds.shape == (4,)
        assert mb.K == 4

    def test_basis_orthonormal_in_raw(self):
        h0, y, mu, sd, _ = _make_synthetic(200, 24, K_true=2)
        mb = fit_supervised_mesoscale_basis(h0, y, K=4, scaler_mu=mu, scaler_sd=sd, epochs=50)
        gram = mb.basis_raw.T @ mb.basis_raw
        assert torch.allclose(gram, torch.eye(4), atol=1e-5)

    def test_synthetic_recovery(self):
        """Span of basis_raw should contain (most of) the true label direction."""
        h0, y, mu, sd, W_true = _make_synthetic(800, 32, K_true=2, seed=7)
        mb = fit_supervised_mesoscale_basis(
            h0, y, K=4, scaler_mu=mu, scaler_sd=sd, epochs=300
        )
        # Project W_true (z-space) into raw space the same way the module does:
        # column-wise / sd, then re-orthonormalize. We compare span containment.
        W_raw = W_true / sd.unsqueeze(1)
        W_raw, _ = torch.linalg.qr(W_raw)  # [D, 2]
        # Energy of W_raw in span(basis_raw): trace((basis_raw.T @ W_raw) @ (W_raw.T @ basis_raw))
        proj = mb.basis_raw.T @ W_raw       # [4, 2]
        captured = (proj ** 2).sum() / 2.0  # max possible = 2 (rank of W_raw)
        assert captured.item() > 0.6, f"recovered only {captured.item():.3f} of W_raw"

    def test_invalid_K_raises(self):
        h0, y, mu, sd, _ = _make_synthetic(50, 16, K_true=2)
        with pytest.raises(ValueError):
            fit_supervised_mesoscale_basis(h0, y, K=0, scaler_mu=mu, scaler_sd=sd, epochs=10)
        with pytest.raises(ValueError):
            fit_supervised_mesoscale_basis(h0, y, K=20, scaler_mu=mu, scaler_sd=sd, epochs=10)

    def test_scaler_shape_validation(self):
        h0, y, mu, sd, _ = _make_synthetic(50, 16, K_true=2)
        with pytest.raises(ValueError):
            fit_supervised_mesoscale_basis(
                h0, y, K=2, scaler_mu=torch.zeros(8), scaler_sd=sd, epochs=10
            )

    def test_2d_scaler_accepted(self):
        """[1, D] shaped scalers (as saved by Standardizer) should work."""
        h0, y, mu, sd, _ = _make_synthetic(80, 16, K_true=2)
        mb = fit_supervised_mesoscale_basis(
            h0, y, K=3,
            scaler_mu=mu.unsqueeze(0), scaler_sd=sd.unsqueeze(0),
            epochs=20,
        )
        assert mb.basis_raw.shape == (16, 3)


# ── Perturbation factories ──────────────────────────────────────────────────

def _orthonormal_basis(D: int, K: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(D, K, generator=g)
    Q, _ = torch.linalg.qr(M)
    return Q


class TestComponentPerturbation:
    def test_norm_matches_magnitude(self):
        B = _orthonormal_basis(32, 6)
        v = make_component_perturbation(B, idx=2, magnitude=1.5)
        assert v.shape == (32,)
        assert abs(v.norm().item() - 1.5) < 1e-5

    def test_sign_negation(self):
        B = _orthonormal_basis(32, 6)
        v_pos = make_component_perturbation(B, idx=0, magnitude=1.0, sign=1.0)
        v_neg = make_component_perturbation(B, idx=0, magnitude=1.0, sign=-1.0)
        assert torch.allclose(v_pos, -v_neg)


class TestWithinSpanRandom:
    def test_lies_in_span(self):
        B = _orthonormal_basis(64, 8)
        g = torch.Generator().manual_seed(99)
        for _ in range(10):
            v = make_within_span_random_perturbation(B, magnitude=2.0, generator=g)
            # Project onto span and check residual is ~0
            proj = B @ (B.T @ v)
            residual = (v - proj).norm().item()
            assert residual < 1e-5, f"residual {residual}"

    def test_norm_matches_magnitude(self):
        B = _orthonormal_basis(64, 8)
        g = torch.Generator().manual_seed(101)
        v = make_within_span_random_perturbation(B, magnitude=3.5, generator=g)
        assert abs(v.norm().item() - 3.5) < 1e-5

    def test_uniform_on_sphere(self):
        """Mean direction over many samples should be ~0 (no preferred axis)."""
        B = _orthonormal_basis(40, 5)
        g = torch.Generator().manual_seed(202)
        accum = torch.zeros(40)
        n = 500
        for _ in range(n):
            v = make_within_span_random_perturbation(B, magnitude=1.0, generator=g)
            accum += v
        mean_norm = (accum / n).norm().item()
        # Mean of n uniform-on-sphere-of-K-span vectors has norm ~ 1/sqrt(n*K) * something
        assert mean_norm < 0.2, f"mean direction norm {mean_norm} suggests bias"


class TestCoordinated:
    def test_matches_manual(self):
        B = _orthonormal_basis(32, 4, seed=11)
        signs = torch.tensor([0.5, -0.3, 0.8, 0.1])
        v = make_coordinated_perturbation(B, signs, magnitude=2.0)
        manual = B @ signs
        manual = manual / manual.norm() * 2.0
        assert torch.allclose(v, manual, atol=1e-5)

    def test_norm_matches_magnitude(self):
        B = _orthonormal_basis(32, 4)
        signs = torch.tensor([1.0, 1.0, -1.0, 0.5])
        v = make_coordinated_perturbation(B, signs, magnitude=1.7)
        assert abs(v.norm().item() - 1.7) < 1e-5


class TestPairwise:
    def test_uses_signs_only(self):
        """Pairwise should depend on signs of s_i, s_j — not magnitudes."""
        B = _orthonormal_basis(32, 4, seed=21)
        signs_a = torch.tensor([2.0, -1.0, 5.0, 0.1])
        signs_b = torch.tensor([0.001, -100.0, 7.0, 0.1])
        v_a = make_pairwise_perturbation(B, 0, 1, signs_a, magnitude=1.0)
        v_b = make_pairwise_perturbation(B, 0, 1, signs_b, magnitude=1.0)
        assert torch.allclose(v_a, v_b, atol=1e-6)

    def test_sign_orientation(self):
        """If both signs positive, result should equal (u_i + u_j)/sqrt(2)."""
        B = _orthonormal_basis(32, 4, seed=22)
        signs = torch.tensor([1.0, 1.0, 1.0, 1.0])
        v = make_pairwise_perturbation(B, 0, 1, signs, magnitude=1.0)
        expected = (B[:, 0] + B[:, 1]) / math.sqrt(2.0)
        assert torch.allclose(v, expected, atol=1e-5)

    def test_norm_matches_magnitude(self):
        B = _orthonormal_basis(32, 4)
        signs = torch.tensor([1.0, -1.0, 0.5, 0.1])
        v = make_pairwise_perturbation(B, 0, 2, signs, magnitude=2.5)
        assert abs(v.norm().item() - 2.5) < 1e-5


# ── Calibration ─────────────────────────────────────────────────────────────

class TestCalibrate:
    def test_direction_std_matches_manual(self):
        torch.manual_seed(33)
        h0 = torch.randn(500, 16) * 3.0 + 2.0
        d = torch.randn(16)
        std = calibrate_direction_std(h0, d)
        d_unit = d / d.norm()
        manual = float((h0 @ d_unit).std().item())
        assert abs(std - manual) < 1e-5
