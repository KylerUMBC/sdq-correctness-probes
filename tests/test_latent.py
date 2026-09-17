"""Tests for sdq.latent — encoder, decoder, dynamics, gauge."""

import torch
import pytest

from sdq.latent.encoder import TemporalConvEncoder, TemporalTransformerEncoder
from sdq.latent.decoder import ObservationDecoder, ResidualDecoder
from sdq.latent.dynamics import LatentODE, LatentGRU
from sdq.latent.regime_dynamics import RegimeSwitchingDynamics
from sdq.latent.gauge_model import GaugeEmbedding, GaugeEncoder, TemporalGaugeEncoder


class TestTemporalConvEncoder:
    def test_output_shape(self, hidden_dim, latent_dim, seq_len):
        enc = TemporalConvEncoder(hidden_dim, latent_dim, window_size=3)
        h = torch.randn(seq_len, hidden_dim)
        z = enc(h)
        assert z.shape == (seq_len, latent_dim)

    def test_different_seq_lengths(self, hidden_dim, latent_dim):
        enc = TemporalConvEncoder(hidden_dim, latent_dim, window_size=3)
        for T in [5, 10, 20]:
            h = torch.randn(T, hidden_dim)
            z = enc(h)
            assert z.shape == (T, latent_dim)

    def test_gradient_flows(self, hidden_dim, latent_dim, seq_len):
        enc = TemporalConvEncoder(hidden_dim, latent_dim, window_size=3)
        h = torch.randn(seq_len, hidden_dim)
        z = enc(h)
        z.sum().backward()
        for p in enc.parameters():
            assert p.grad is not None


class TestTemporalTransformerEncoder:
    def test_output_shape(self, hidden_dim, latent_dim, seq_len):
        enc = TemporalTransformerEncoder(hidden_dim, latent_dim, window_size=3, num_heads=2, num_layers=1)
        h = torch.randn(seq_len, hidden_dim)
        z = enc(h)
        assert z.shape == (seq_len, latent_dim)

    def test_gradient_flows(self, hidden_dim, latent_dim, seq_len):
        enc = TemporalTransformerEncoder(hidden_dim, latent_dim, window_size=3, num_heads=2, num_layers=1)
        h = torch.randn(seq_len, hidden_dim)
        z = enc(h)
        z.sum().backward()
        for p in enc.parameters():
            assert p.grad is not None


class TestObservationDecoder:
    def test_output_shape(self, hidden_dim, latent_dim, gauge_dim, seq_len):
        dec = ObservationDecoder(latent_dim, gauge_dim, hidden_dim)
        z = torch.randn(seq_len, latent_dim)
        u = torch.randn(seq_len, gauge_dim)
        h_hat = dec(z, u)
        assert h_hat.shape == (seq_len, hidden_dim)

    def test_no_gauge(self, hidden_dim, latent_dim, gauge_dim, seq_len):
        dec = ObservationDecoder(latent_dim, gauge_dim, hidden_dim)
        z = torch.randn(seq_len, latent_dim)
        h_hat = dec(z)  # u=None should work
        assert h_hat.shape == (seq_len, hidden_dim)


class TestResidualDecoder:
    def test_output_shape(self, hidden_dim, latent_dim, gauge_dim, seq_len):
        dec = ResidualDecoder(latent_dim, gauge_dim, hidden_dim)
        z = torch.randn(seq_len, latent_dim)
        h_hat = dec(z)
        assert h_hat.shape == (seq_len, hidden_dim)

    def test_near_linear_at_init(self, hidden_dim, latent_dim, gauge_dim, seq_len):
        """At init, correction is zero so output ≈ linear(z)."""
        dec = ResidualDecoder(latent_dim, gauge_dim, hidden_dim)
        z = torch.randn(seq_len, latent_dim)
        h_hat = dec(z)
        h_linear = dec.linear(z)
        assert torch.allclose(h_hat, h_linear, atol=1e-5)


class TestLatentODE:
    def test_step(self, latent_dim):
        ode = LatentODE(latent_dim)
        z = torch.randn(latent_dim)
        z_next = ode.step(z)
        assert z_next.shape == z.shape
        # At init f≈0, so z_next ≈ z
        assert torch.allclose(z_next, z, atol=1e-5)

    def test_rollout(self, latent_dim):
        ode = LatentODE(latent_dim)
        z0 = torch.randn(latent_dim)
        traj = ode.rollout(z0, T=10)
        assert traj.shape == (10, latent_dim)
        # All should be ≈ z0 at init
        assert torch.allclose(traj, z0.unsqueeze(0).expand(10, -1), atol=1e-4)

    def test_forward(self, latent_dim, seq_len):
        ode = LatentODE(latent_dim)
        z = torch.randn(seq_len, latent_dim)
        predicted = ode(z)
        assert predicted.shape == (seq_len - 1, latent_dim)

    def test_dynamics_consistency_loss(self, latent_dim, seq_len):
        ode = LatentODE(latent_dim)
        z = torch.randn(seq_len, latent_dim)
        loss = ode.dynamics_consistency_loss(z)
        assert loss.item() >= 0


class TestLatentGRU:
    def test_forward(self, latent_dim, seq_len):
        gru = LatentGRU(latent_dim)
        z = torch.randn(seq_len, latent_dim)
        pred = gru(z)
        assert pred.shape == (seq_len - 1, latent_dim)


class TestRegimeSwitchingDynamics:
    def test_velocity_field_shape(self, latent_dim, seq_len):
        dyn = RegimeSwitchingDynamics(latent_dim, num_regimes=4)
        z = torch.randn(seq_len, latent_dim)
        v = dyn.velocity_field(z)
        assert v.shape == z.shape

    def test_multi_step_and_losses(self, latent_dim, seq_len):
        dyn = RegimeSwitchingDynamics(latent_dim, num_regimes=4)
        z = torch.randn(seq_len, latent_dim)
        pred = dyn.multi_step_predict(z[:-1], num_substeps=3)
        assert pred.shape == z[:-1].shape
        l_dyn = dyn.dynamics_consistency_loss(z, num_substeps=3)
        assert l_dyn.ndim == 0 and l_dyn.item() >= 0
        l_reg = dyn.regime_consistency_loss(z, z)
        assert l_reg.item() < 1e-5  # identical trajectories -> KL ~ 0
        l_ent = dyn.gate_entropy_loss(z)
        assert l_ent.ndim == 0
        l_ctr = dyn.contraction_loss(z)
        assert l_ctr.ndim == 0

    def test_gradient_flows(self, latent_dim, seq_len):
        dyn = RegimeSwitchingDynamics(latent_dim, num_regimes=4)
        z = torch.randn(seq_len, latent_dim)
        loss = dyn.dynamics_consistency_loss(z, num_substeps=2) + dyn.gate_entropy_loss(z)
        loss.backward()
        for p in dyn.parameters():
            assert p.grad is not None

    def test_gate_supervised_and_terminal(self, latent_dim, seq_len):
        dyn = RegimeSwitchingDynamics(latent_dim, num_regimes=4)
        z = torch.randn(seq_len, latent_dim)
        labels = torch.randint(0, 4, (seq_len,))
        l_sup = dyn.gate_supervised_loss(z, labels)
        assert l_sup.ndim == 0 and l_sup.item() >= 0
        l_term = dyn.terminal_velocity_loss(z, margin=0.1, tau=0.5)
        assert l_term.ndim == 0 and l_term.item() >= 0

    def test_eval_hard_gate_shape(self, latent_dim):
        dyn = RegimeSwitchingDynamics(latent_dim, num_regimes=3)
        z = torch.randn(5, latent_dim)
        dyn.eval()
        with torch.no_grad():
            v = dyn.velocity_field(z)
        assert v.shape == z.shape


class TestGaugeEmbedding:
    def test_output_shape(self, gauge_dim, seq_len):
        ge = GaugeEmbedding(num_prompts=10, gauge_dim=gauge_dim)
        u = ge(prompt_idx=3, T=seq_len)
        assert u.shape == (seq_len, gauge_dim)

    def test_constant_over_time(self, gauge_dim, seq_len):
        ge = GaugeEmbedding(num_prompts=5, gauge_dim=gauge_dim)
        u = ge(prompt_idx=0, T=seq_len)
        # All timesteps should be identical
        assert torch.allclose(u[0], u[-1])


class TestGaugeEncoder:
    def test_output_shape(self, hidden_dim, gauge_dim, seq_len):
        ge = GaugeEncoder(hidden_dim, gauge_dim)
        h = torch.randn(seq_len, hidden_dim)
        u = ge(h)
        assert u.shape == (seq_len, gauge_dim)


class TestTemporalGaugeEncoder:
    def test_output_shape(self, hidden_dim, gauge_dim, seq_len):
        ge = TemporalGaugeEncoder(hidden_dim, gauge_dim, window_size=3)
        h = torch.randn(seq_len, hidden_dim)
        u = ge(h)
        assert u.shape == (seq_len, gauge_dim)
