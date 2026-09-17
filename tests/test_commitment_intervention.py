"""Tests for sdq.eval.commitment_intervention.

Tests the hook mechanics, perturbation generation, flip statistics,
and magnitude calibration without requiring the actual Gemma model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from sdq.eval.commitment_intervention import (
    InterventionTrial,
    MagnitudeResult,
    PersistentPerturbationHook,
    PrefillPerturbationHook,
    calibrate_magnitudes,
    compute_flip_statistics,
    compute_pca_basis_raw,
    direction_to_raw_space,
    make_complement_perturbation,
    make_directed_perturbation,
    make_random_perturbation,
    make_random_subspace,
    make_subspace_perturbation,
)


# -- PrefillPerturbationHook --------------------------------------------------

class TestPrefillPerturbationHook:
    def test_fires_once(self):
        """Hook should modify output on first call only."""
        direction = torch.ones(16)
        direction = direction / direction.norm()
        hook = PrefillPerturbationHook(direction, magnitude=1.0, position=-1)

        # Simulate a transformer layer output: (hidden_states, attn, kv_cache)
        hidden = torch.zeros(1, 5, 16)  # [batch, seq, dim]
        output = (hidden, None, None)

        # First call: should perturb
        result = hook(None, None, output)
        assert result[0][:, -1, :].abs().sum().item() > 0
        assert hook.fired

        # Second call: no-op
        hidden2 = torch.zeros(1, 1, 16)
        output2 = (hidden2, None, None)
        result2 = hook(None, None, output2)
        assert result2[0].abs().sum().item() == 0

    def test_correct_position(self):
        """Perturbation should only affect the target position."""
        direction = torch.ones(8) / (8 ** 0.5)
        hook = PrefillPerturbationHook(direction, magnitude=2.0, position=2)

        hidden = torch.zeros(1, 5, 8)
        output = (hidden,)
        result = hook(None, None, output)

        # Position 2 should be perturbed
        assert result[0][0, 2, :].abs().sum().item() > 0
        # Other positions should be zero
        assert result[0][0, 0, :].abs().sum().item() == 0
        assert result[0][0, 4, :].abs().sum().item() == 0

    def test_negative_position(self):
        """position=-1 should perturb the last token."""
        direction = torch.ones(8) / (8 ** 0.5)
        hook = PrefillPerturbationHook(direction, magnitude=1.0, position=-1)

        hidden = torch.zeros(1, 10, 8)
        output = (hidden,)
        result = hook(None, None, output)

        # Last position perturbed
        assert result[0][0, -1, :].abs().sum().item() > 0
        # First position untouched
        assert result[0][0, 0, :].abs().sum().item() == 0

    def test_magnitude_scaling(self):
        """Perturbation norm should equal magnitude."""
        direction = torch.randn(32)
        direction = direction / direction.norm()
        mag = 3.14
        hook = PrefillPerturbationHook(direction, magnitude=mag, position=-1)

        hidden = torch.zeros(1, 5, 32)
        output = (hidden,)
        result = hook(None, None, output)

        actual_norm = result[0][0, -1, :].norm().item()
        assert abs(actual_norm - mag) < 0.01

    def test_reset(self):
        """After reset, hook should fire again."""
        direction = torch.ones(8) / (8 ** 0.5)
        hook = PrefillPerturbationHook(direction, magnitude=1.0, position=-1)

        hidden = torch.zeros(1, 3, 8)
        output = (hidden,)

        hook(None, None, output)
        assert hook.fired

        hook.reset()
        assert not hook.fired

        hidden2 = torch.zeros(1, 3, 8)
        output2 = (hidden2,)
        result2 = hook(None, None, output2)
        assert result2[0][0, -1, :].abs().sum().item() > 0

    def test_dtype_casting(self):
        """Direction should be cast to the hidden state's dtype."""
        direction = torch.ones(8, dtype=torch.float32) / (8 ** 0.5)
        hook = PrefillPerturbationHook(direction, magnitude=1.0, position=-1)

        hidden = torch.zeros(1, 3, 8, dtype=torch.float16)
        output = (hidden,)
        result = hook(None, None, output)
        assert result[0].dtype == torch.float16


# -- Perturbation generators --------------------------------------------------

class TestMakeDirectedPerturbation:
    def test_correct_norm(self):
        d = torch.randn(64)
        p = make_directed_perturbation(d, magnitude=5.0)
        assert abs(p.norm().item() - 5.0) < 0.01

    def test_direction_preserved(self):
        d = torch.randn(32)
        p = make_directed_perturbation(d, magnitude=1.0)
        d_unit = d / d.norm()
        cos = torch.dot(p / p.norm(), d_unit).item()
        assert abs(cos - 1.0) < 1e-5


class TestMakeRandomPerturbation:
    def test_correct_norm(self):
        p = make_random_perturbation(64, magnitude=3.0)
        assert abs(p.norm().item() - 3.0) < 0.01

    def test_different_each_call(self):
        p1 = make_random_perturbation(64, magnitude=1.0)
        p2 = make_random_perturbation(64, magnitude=1.0)
        # Very unlikely to be identical
        assert (p1 - p2).norm().item() > 0.01

    def test_deterministic_with_generator(self):
        g1 = torch.Generator().manual_seed(42)
        p1 = make_random_perturbation(32, magnitude=1.0, generator=g1)
        g2 = torch.Generator().manual_seed(42)
        p2 = make_random_perturbation(32, magnitude=1.0, generator=g2)
        assert torch.allclose(p1, p2)


# -- Flip statistics -----------------------------------------------------------

def _make_trial(flipped: bool, bad: bool, ptype: str = "directed") -> InterventionTrial:
    return InterventionTrial(
        example_id="test",
        task_family="arithmetic",
        gold_answer="5",
        magnitude=1.0,
        perturbation_type=ptype,
        baseline_text="5",
        baseline_answer="5",
        baseline_correct=True,
        perturbed_text="5" if not flipped else "7",
        perturbed_answer="5" if not flipped else "7",
        perturbed_correct=not bad,
        answer_flipped=flipped,
        flipped_to_incorrect=bad,
    )


class TestComputeFlipStatistics:
    def test_basic_rates(self):
        directed = [
            _make_trial(True, True),
            _make_trial(True, True),
            _make_trial(False, False),
            _make_trial(False, False),
        ]
        random = [
            _make_trial(True, True, "random"),
            _make_trial(False, False, "random"),
            _make_trial(False, False, "random"),
            _make_trial(False, False, "random"),
        ]
        mr = compute_flip_statistics(directed, random, 1.0)
        assert mr.directed_flip_rate == 0.5
        assert mr.random_flip_rate == 0.25
        assert mr.flip_ratio == pytest.approx(2.0)
        assert mr.passes_threshold  # ratio >= 2 and dir rate > 5%

    def test_no_random_flips(self):
        directed = [_make_trial(True, True)] * 5
        random = [_make_trial(False, False, "random")] * 10
        mr = compute_flip_statistics(directed, random, 1.0)
        assert mr.directed_flip_rate == 1.0
        assert mr.random_flip_rate == 0.0
        assert mr.flip_ratio > 100  # effectively infinite
        assert mr.passes_threshold

    def test_no_directed_flips(self):
        directed = [_make_trial(False, False)] * 5
        random = [_make_trial(True, True, "random")] * 10
        mr = compute_flip_statistics(directed, random, 1.0)
        assert mr.directed_flip_rate == 0.0
        assert not mr.passes_threshold  # 0% directed < 5% threshold

    def test_low_directed_rate_fails(self):
        """Even if ratio is high, below 5% absolute directed flip rate fails."""
        directed = [_make_trial(True, True)] + [_make_trial(False, False)] * 99
        random = [_make_trial(False, False, "random")] * 100
        mr = compute_flip_statistics(directed, random, 1.0)
        assert mr.directed_flip_rate == 0.01
        assert not mr.passes_threshold


# -- Magnitude calibration -----------------------------------------------------

class TestCalibrateMagnitudes:
    def test_returns_list(self):
        H = torch.randn(100, 32)
        d = torch.randn(32)
        d = d / d.norm()
        mags = calibrate_magnitudes(H, d, n_magnitudes=3)
        assert len(mags) == 3
        assert all(m > 0 for m in mags)

    def test_scales_with_variance(self):
        """Magnitudes should be proportional to std of projections."""
        d = torch.zeros(32)
        d[0] = 1.0  # direction along dim 0

        # Low-variance data along dim 0
        H_low = torch.randn(100, 32) * 0.1
        mags_low = calibrate_magnitudes(H_low, d)

        # High-variance data along dim 0
        H_high = torch.randn(100, 32) * 10.0
        mags_high = calibrate_magnitudes(H_high, d)

        # High-variance should give larger magnitudes
        assert mags_high[0] > mags_low[0] * 5

    def test_monotonically_increasing(self):
        H = torch.randn(50, 16)
        d = torch.randn(16)
        d = d / d.norm()
        mags = calibrate_magnitudes(H, d, n_magnitudes=5)
        for i in range(len(mags) - 1):
            assert mags[i] < mags[i + 1]


# -- direction_to_raw_space ----------------------------------------------------

class TestDirectionToRawSpace:
    def test_output_is_unit_vector(self):
        d = torch.randn(32)
        d = d / d.norm()
        sd = torch.rand(32) + 0.1  # positive std devs
        raw = direction_to_raw_space(d, sd)
        assert abs(raw.norm().item() - 1.0) < 1e-5

    def test_changes_direction(self):
        """Raw-space direction should differ from std-space when sd is non-uniform."""
        d = torch.randn(64)
        d = d / d.norm()
        # Non-uniform sd: some dims scaled 10x more
        sd = torch.ones(64)
        sd[:32] = 10.0
        raw = direction_to_raw_space(d, sd)
        cos = torch.dot(d, raw).item()
        # Should be significantly different from 1.0
        assert cos < 0.95

    def test_identity_with_uniform_sd(self):
        """If all sd are equal, raw direction should match std direction."""
        d = torch.randn(32)
        d = d / d.norm()
        sd = torch.ones(32) * 3.0  # uniform
        raw = direction_to_raw_space(d, sd)
        cos = torch.dot(d, raw).item()
        assert abs(cos - 1.0) < 1e-5

    def test_concentrates_on_low_variance_dims(self):
        """Raw direction should have larger weight on dims with low sd."""
        d = torch.ones(8) / (8 ** 0.5)
        sd = torch.tensor([0.1, 0.1, 0.1, 0.1, 10.0, 10.0, 10.0, 10.0])
        raw = direction_to_raw_space(d, sd)
        # First 4 dims (low sd) should dominate
        assert raw[:4].norm().item() > raw[4:].norm().item()

    def test_handles_2d_scaler_sd(self):
        """scaler_sd may be [1, hidden_dim] from Standardizer."""
        d = torch.randn(16)
        d = d / d.norm()
        sd = (torch.rand(1, 16) + 0.1)  # [1, 16]
        raw = direction_to_raw_space(d, sd)
        assert raw.shape == (16,)
        assert abs(raw.norm().item() - 1.0) < 1e-5


# -- PersistentPerturbationHook ------------------------------------------------

class TestPersistentPerturbationHook:
    def test_fires_every_call(self):
        """Hook should modify output on every call."""
        direction = torch.ones(16) / (16 ** 0.5)
        hook = PersistentPerturbationHook(direction, magnitude=1.0, position=-1)

        hidden = torch.zeros(1, 5, 16)
        output = (hidden,)
        result = hook(None, None, output)
        assert result[0][:, -1, :].abs().sum().item() > 0
        assert hook.fire_count == 1

        hidden2 = torch.zeros(1, 1, 16)
        output2 = (hidden2,)
        result2 = hook(None, None, output2)
        assert result2[0][:, -1, :].abs().sum().item() > 0
        assert hook.fire_count == 2

    def test_max_firings(self):
        """Hook should stop after max_firings."""
        direction = torch.ones(8) / (8 ** 0.5)
        hook = PersistentPerturbationHook(direction, magnitude=1.0, max_firings=2)

        for i in range(3):
            hidden = torch.zeros(1, 1, 8)
            output = (hidden,)
            result = hook(None, None, output)
            if i < 2:
                assert result[0].abs().sum().item() > 0
            else:
                assert result[0].abs().sum().item() == 0

        assert hook.fire_count == 2

    def test_seq_len_1_uses_last_position(self):
        """For generation steps (seq_len=1), should always perturb position -1."""
        direction = torch.ones(8) / (8 ** 0.5)
        hook = PersistentPerturbationHook(direction, magnitude=1.0, position=3)

        # seq_len=1: should perturb the only position regardless of position arg
        hidden = torch.zeros(1, 1, 8)
        output = (hidden,)
        result = hook(None, None, output)
        assert result[0][0, 0, :].abs().sum().item() > 0

    def test_prefill_uses_specified_position(self):
        """For prefill (seq_len > 1), should perturb the specified position."""
        direction = torch.ones(8) / (8 ** 0.5)
        hook = PersistentPerturbationHook(direction, magnitude=1.0, position=2)

        hidden = torch.zeros(1, 5, 8)
        output = (hidden,)
        result = hook(None, None, output)
        assert result[0][0, 2, :].abs().sum().item() > 0
        assert result[0][0, 0, :].abs().sum().item() == 0

    def test_reset(self):
        """Reset should clear fire_count."""
        direction = torch.ones(8) / (8 ** 0.5)
        hook = PersistentPerturbationHook(direction, magnitude=1.0, max_firings=1)

        hidden = torch.zeros(1, 1, 8)
        output = (hidden,)
        hook(None, None, output)
        assert hook.fire_count == 1

        hook.reset()
        assert hook.fire_count == 0

        result = hook(None, None, (torch.zeros(1, 1, 8),))
        assert result[0].abs().sum().item() > 0

    def test_magnitude_scaling(self):
        """Perturbation norm should equal magnitude."""
        direction = torch.randn(32)
        direction = direction / direction.norm()
        mag = 2.5
        hook = PersistentPerturbationHook(direction, magnitude=mag, position=-1)

        hidden = torch.zeros(1, 3, 32)
        output = (hidden,)
        result = hook(None, None, output)
        actual_norm = result[0][0, -1, :].norm().item()
        assert abs(actual_norm - mag) < 0.01


# -- Subspace perturbation tools -----------------------------------------------

class TestComputePCABasisRaw:
    def test_output_shape(self):
        H = torch.randn(100, 64)
        mu = H.mean(dim=0, keepdim=True)
        sd = H.std(dim=0, keepdim=True).clamp_min(1e-6)
        basis, eigvals = compute_pca_basis_raw(H, mu, sd, k=16)
        assert basis.shape == (64, 16)
        assert eigvals.shape == (16,)

    def test_orthonormality(self):
        H = torch.randn(200, 32)
        mu = H.mean(dim=0, keepdim=True)
        sd = H.std(dim=0, keepdim=True).clamp_min(1e-6)
        basis, _ = compute_pca_basis_raw(H, mu, sd, k=8)
        BtB = basis.T @ basis
        error = (BtB - torch.eye(8)).abs().max().item()
        assert error < 1e-5

    def test_eigenvalues_descending(self):
        H = torch.randn(100, 32)
        mu = H.mean(dim=0, keepdim=True)
        sd = H.std(dim=0, keepdim=True).clamp_min(1e-6)
        _, eigvals = compute_pca_basis_raw(H, mu, sd, k=10)
        for i in range(len(eigvals) - 1):
            assert eigvals[i] >= eigvals[i + 1] - 1e-6

    def test_captures_variance_direction(self):
        """If one dimension has much more variance, basis should capture it."""
        H = torch.randn(200, 32)
        H[:, 0] *= 100  # dim 0 has 100x more variance
        mu = H.mean(dim=0, keepdim=True)
        sd = H.std(dim=0, keepdim=True).clamp_min(1e-6)
        basis, _ = compute_pca_basis_raw(H, mu, sd, k=4)
        # After standardization, the variance is equalized, but the raw-space
        # basis should still span dim 0 prominently due to sd scaling
        dim0_proj = basis[0, :].abs().max().item()
        assert dim0_proj > 0.1


class TestMakeSubspacePerturbation:
    def test_correct_magnitude(self):
        basis = torch.eye(32)[:, :8]  # first 8 dims
        p = make_subspace_perturbation(basis, magnitude=3.0)
        assert abs(p.norm().item() - 3.0) < 0.01

    def test_within_subspace(self):
        """Perturbation should lie entirely within the subspace."""
        basis = torch.eye(32)[:, :8]
        p = make_subspace_perturbation(basis, magnitude=1.0)
        # Components outside subspace should be zero
        assert p[8:].abs().max().item() < 1e-6

    def test_deterministic_with_generator(self):
        basis = torch.eye(16)[:, :4]
        g1 = torch.Generator().manual_seed(42)
        p1 = make_subspace_perturbation(basis, 1.0, g1)
        g2 = torch.Generator().manual_seed(42)
        p2 = make_subspace_perturbation(basis, 1.0, g2)
        assert torch.allclose(p1, p2)

    def test_different_each_call(self):
        basis = torch.eye(32)[:, :8]
        p1 = make_subspace_perturbation(basis, 1.0)
        p2 = make_subspace_perturbation(basis, 1.0)
        assert (p1 - p2).norm().item() > 0.01


class TestMakeComplementPerturbation:
    def test_correct_magnitude(self):
        basis = torch.eye(32)[:, :8]
        p = make_complement_perturbation(basis, 32, magnitude=5.0)
        assert abs(p.norm().item() - 5.0) < 0.01

    def test_orthogonal_to_subspace(self):
        """Perturbation should be orthogonal to all basis vectors."""
        basis = torch.eye(32)[:, :8]
        p = make_complement_perturbation(basis, 32, magnitude=1.0)
        proj = (basis.T @ p).abs().max().item()
        assert proj < 1e-5

    def test_with_non_trivial_basis(self):
        """Test with a non-axis-aligned basis."""
        M = torch.randn(64, 16)
        Q, _ = torch.linalg.qr(M)
        p = make_complement_perturbation(Q, 64, magnitude=2.0)
        proj = (Q.T @ p).abs().max().item()
        assert proj < 1e-4
        assert abs(p.norm().item() - 2.0) < 0.01

    def test_lives_in_complement(self):
        """With axis-aligned basis, perturbation should be in the complement dims."""
        basis = torch.eye(32)[:, :8]
        p = make_complement_perturbation(basis, 32, magnitude=1.0)
        # First 8 dims should be zero
        assert p[:8].abs().max().item() < 1e-5
        # Remaining dims should be non-zero
        assert p[8:].abs().sum().item() > 0.5


class TestMakeRandomSubspace:
    def test_output_shape(self):
        Q = make_random_subspace(64, 16)
        assert Q.shape == (64, 16)

    def test_orthonormality(self):
        Q = make_random_subspace(32, 8)
        QtQ = Q.T @ Q
        error = (QtQ - torch.eye(8)).abs().max().item()
        assert error < 1e-5

    def test_deterministic_with_generator(self):
        g1 = torch.Generator().manual_seed(42)
        Q1 = make_random_subspace(32, 8, g1)
        g2 = torch.Generator().manual_seed(42)
        Q2 = make_random_subspace(32, 8, g2)
        assert torch.allclose(Q1, Q2)

    def test_different_from_identity(self):
        """Random subspace should not be axis-aligned."""
        Q = make_random_subspace(32, 8)
        # Should have non-zero off-diagonal elements
        assert (Q[:8, :] - torch.eye(8)).abs().sum().item() > 1.0
