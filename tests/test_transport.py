"""Tests for sdq.transport — local_operator, low_rank_maps, composition, cycle."""

import torch
import pytest

from sdq.transport.local_operator import LocalTransportField, TransportResult
from sdq.transport.low_rank_maps import LowRankTransport, operator_rank_spectrum
from sdq.transport.composition import (
    compose_transports,
    invert_transport,
    apply_transport_to_velocity,
)
from sdq.transport.cycle_consistency import (
    inversion_error,
    composition_error,
    cycle_consistency_metrics,
)


class TestLowRankTransport:
    def test_identity_initialization(self):
        G = LowRankTransport(dim=16, rank=4)
        x = torch.randn(16)
        y = G(x)
        # Should be near-identity at init (U, V are zeros)
        assert torch.allclose(y, x, atol=1e-6)

    def test_matrix_shape(self):
        G = LowRankTransport(dim=16, rank=4)
        mat = G.matrix()
        assert mat.shape == (16, 16)
        # Should be identity at init
        assert torch.allclose(mat, torch.eye(16), atol=1e-6)

    def test_nuclear_norm_zero_at_init(self):
        G = LowRankTransport(dim=16, rank=4)
        assert G.nuclear_norm().item() < 1e-6

    def test_frobenius_deviation_zero_at_init(self):
        G = LowRankTransport(dim=16, rank=4)
        assert G.frobenius_deviation().item() < 1e-6

    def test_batch_input(self):
        G = LowRankTransport(dim=16, rank=4)
        x = torch.randn(5, 16)
        y = G(x)
        assert y.shape == (5, 16)


class TestLocalTransportField:
    def test_output_shapes(self, hidden_dim, trajectory_pair):
        source, target = trajectory_pair
        T = source.shape[0]
        field = LocalTransportField(hidden_dim, rank=4)
        result = field(source, target)
        assert result.transported_velocity.shape == (T - 1, hidden_dim)
        assert result.target_velocity.shape == (T - 1, hidden_dim)
        assert result.source_velocity.shape == (T - 1, hidden_dim)
        assert result.residual.shape == (T - 1, hidden_dim)

    def test_identity_init(self, hidden_dim, trajectory_pair):
        """At initialization, transport should be near-identity."""
        source, target = trajectory_pair
        field = LocalTransportField(hidden_dim, rank=4)
        result = field(source, target)
        # Transported ≈ source velocity (since G ≈ I)
        assert torch.allclose(
            result.transported_velocity, result.source_velocity, atol=1e-5
        )

    def test_return_operators(self, hidden_dim, trajectory_pair):
        source, target = trajectory_pair
        T = source.shape[0]
        field = LocalTransportField(hidden_dim, rank=4)
        result = field(source, target, return_operators=True)
        assert result.G_t.shape == (T - 1, hidden_dim, hidden_dim)

    def test_gradient_flows(self, hidden_dim, trajectory_pair):
        source, target = trajectory_pair
        field = LocalTransportField(hidden_dim, rank=4)
        result = field(source, target)
        loss = result.residual.pow(2).sum()
        loss.backward()
        # Check gradients exist on transport parameters
        for p in field.parameters():
            assert p.grad is not None


class TestComposition:
    def test_compose_identity(self):
        D = 16
        T = 5
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        G = torch.randn(T, D, D) * 0.1 + I
        # G composed with identity should be G
        composed = compose_transports(I, G)
        assert torch.allclose(composed, G, atol=1e-5)

    def test_invert_identity(self):
        D = 8
        T = 3
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        inv = invert_transport(I)
        assert torch.allclose(inv, I, atol=1e-5)

    def test_apply_transport(self):
        D = 8
        T = 5
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        vel = torch.randn(T, D)
        result = apply_transport_to_velocity(I, vel)
        assert torch.allclose(result, vel, atol=1e-5)


class TestCycleConsistency:
    def test_identity_perfect_cycle(self):
        D = 8
        T = 5
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        err = inversion_error(I, I)
        assert err.item() < 1e-5

    def test_near_identity_low_error(self):
        D = 8
        T = 5
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        G = I + 0.01 * torch.randn(T, D, D)
        G_inv = invert_transport(G)
        err = inversion_error(G, G_inv)
        assert err.item() < 0.1

    def test_metrics_dict(self):
        D = 8
        T = 5
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        metrics = cycle_consistency_metrics(I, I)
        assert "inversion_error" in metrics
        assert metrics["inversion_error"] < 1e-5


class TestOperatorRankSpectrum:
    def test_identity_zero_spectrum(self):
        G = torch.eye(8)
        svs = operator_rank_spectrum(G)
        assert svs.abs().max() < 1e-5

    def test_rank_1_perturbation(self):
        u = torch.randn(8, 1)
        v = torch.randn(8, 1)
        G = torch.eye(8) + u @ v.T
        svs = operator_rank_spectrum(G)
        # Should have ~1 significant singular value
        assert svs[0] > 0.1
        # Most others should be near zero
        assert svs[2:].abs().max() < 1e-5
