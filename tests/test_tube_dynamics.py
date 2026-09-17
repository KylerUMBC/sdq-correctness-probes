"""Tests for TubeAwareDynamics and TubeGeometry."""

import torch
import pytest
from sdq.latent.tube_dynamics import TubeAwareDynamics, TubeGeometry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def latent_dim():
    return 32


@pytest.fixture
def num_regimes():
    return 4


@pytest.fixture
def dynamics(latent_dim, num_regimes):
    return TubeAwareDynamics(latent_dim, num_regimes=num_regimes)


@pytest.fixture
def sample_families():
    """Two families, 3 members each, T=10, D=32."""
    D = 32
    T = 10
    trajs = {}
    for fam_idx in range(2):
        base = torch.randn(T, D) * 0.1
        base[:, 0] += fam_idx * 5.0
        for member in range(3):
            pid = f"fam{fam_idx}_mem{member}"
            trajs[pid] = base + torch.randn(T, D) * 0.02
    families = {
        "family_0": [f"fam0_mem{i}" for i in range(3)],
        "family_1": [f"fam1_mem{i}" for i in range(3)],
    }
    fam_to_idx = {"family_0": 0, "family_1": 1}
    return trajs, families, fam_to_idx


# ---------------------------------------------------------------------------
# TubeGeometry tests
# ---------------------------------------------------------------------------

class TestTubeGeometry:

    def test_build_produces_anchors(self, sample_families):
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        assert geo.anchors.shape[0] > 0
        assert geo.anchors.shape[1] == 32
        assert geo.tangents.shape == geo.anchors.shape
        assert geo.family_ids.shape[0] == geo.anchors.shape[0]

    def test_nearest_context_shape(self, sample_families):
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        z = torch.randn(5, 32)
        anchor, tangent = geo.nearest_context(z)
        assert anchor.shape == (5, 32)
        assert tangent.shape == (5, 32)

    def test_nearest_context_1d(self, sample_families):
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        z = torch.randn(32)
        anchor, tangent = geo.nearest_context(z)
        assert anchor.shape == (32,)

    def test_nearest_context_with_family(self, sample_families):
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        z = torch.randn(5, 32)
        anchor, tangent, fam, progress = geo.nearest_context_with_family(z)
        assert fam.shape == (5,)
        assert progress.shape == (5,)
        assert (fam >= 0).all() and (fam < 2).all()

    def test_nearest_context_for_family_respects_family(self, sample_families):
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        z = torch.stack([
            trajs["fam0_mem0"][4],
            trajs["fam1_mem0"][4],
        ])
        family_ids = torch.tensor([0, 1])
        progress = torch.tensor([4, 4])
        _, _, chosen_progress = geo.nearest_context_for_family(
            z, family_ids=family_ids, progress_hint=progress, progress_window=1,
        )
        assert chosen_progress.shape == (2,)

    def test_contextualize_returns_transverse_metadata(self, sample_families):
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        z = torch.randn(5, 32)
        ctx = geo.contextualize(z)
        assert ctx["anchor"].shape == z.shape
        assert ctx["tangent"].shape == z.shape
        assert ctx["transverse"].shape == z.shape
        assert ctx["transverse_norm"].shape == (5,)

    def test_distance_to_tube(self, sample_families):
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        z = torch.randn(5, 32)
        d = geo.distance_to_tube(z)
        assert d.shape == (5,)
        assert (d >= 0).all()

    def test_distance_to_other_family_tube(self, sample_families):
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        z = torch.stack([
            trajs["fam0_mem0"][4],
            trajs["fam1_mem0"][4],
        ])
        family_ids = torch.tensor([0, 1])
        wrong_dist = geo.distance_to_other_family_tube(z, family_ids)
        assert wrong_dist.shape == (2,)
        assert (wrong_dist >= 0).all()

    def test_empty_geometry(self):
        trajs = {"a": torch.randn(5, 16)}
        families = {"fam_x": ["a"]}
        fam_to_idx = {"fam_x": 0}
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        z = torch.randn(3, 16)
        anchor, tangent = geo.nearest_context(z)
        assert anchor.shape == z.shape


# ---------------------------------------------------------------------------
# TubeAwareDynamics tests
# ---------------------------------------------------------------------------

class TestTubeAwareDynamics:

    def test_velocity_field_shape(self, dynamics, latent_dim):
        z = torch.randn(8, latent_dim)
        v = dynamics.velocity_field(z)
        assert v.shape == (8, latent_dim)

    def test_velocity_field_1d(self, dynamics, latent_dim):
        z = torch.randn(latent_dim)
        v = dynamics.velocity_field(z)
        assert v.shape == (latent_dim,)

    def test_recovery_field_shape(self, dynamics, latent_dim):
        z = torch.randn(8, latent_dim)
        anchor = torch.randn(8, latent_dim)
        tangent = torch.randn(8, latent_dim)
        tangent = tangent / tangent.norm(dim=-1, keepdim=True)
        corr = dynamics.recovery_field(z, anchor, tangent)
        assert corr.shape == (8, latent_dim)

    def test_recovery_field_1d(self, dynamics, latent_dim):
        z = torch.randn(latent_dim)
        anchor = torch.randn(latent_dim)
        tangent = torch.randn(latent_dim)
        tangent = tangent / tangent.norm()
        corr = dynamics.recovery_field(z, anchor, tangent)
        assert corr.shape == (latent_dim,)

    def test_recovery_starts_near_zero(self, dynamics, latent_dim):
        """Zero-initialized output layer → initial corrections are near-zero."""
        z = torch.randn(16, latent_dim)
        anchor = torch.randn(16, latent_dim)
        tangent = torch.randn(16, latent_dim)
        tangent = tangent / tangent.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            corr = dynamics.recovery_field(z, anchor, tangent)
        assert corr.norm(dim=-1).max().item() < 5e-4

    def test_progress_scalar_range(self, latent_dim):
        d = TubeAwareDynamics(latent_dim, num_regimes=2)
        z = torch.randn(8, latent_dim)
        ps = d.progress_scalar(z)
        assert ps.shape == (8, 1)
        assert (ps >= 0).all() and (ps <= 1).all()

    def test_family_router_logits_shape(self, latent_dim):
        d = TubeAwareDynamics(latent_dim, num_regimes=2, num_tube_families=5)
        z = torch.randn(4, latent_dim)
        logits = d.family_router_logits(z)
        assert logits.shape == (4, 5)

    def test_recovery_along_component_is_bounded(self, dynamics, latent_dim):
        z = torch.randn(8, latent_dim)
        anchor = torch.randn(8, latent_dim)
        tangent = torch.randn(8, latent_dim)
        tangent = tangent / tangent.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            dynamics.recovery_gain[-1].bias.fill_(10.0)
            dynamics.recovery_along[-1].bias.fill_(10.0)
        with torch.no_grad():
            corr = dynamics.recovery_field(z, anchor, tangent)
        along = (corr * tangent).sum(dim=-1).abs()
        transverse = (
            corr - (corr * tangent).sum(dim=-1, keepdim=True) * tangent
        ).norm(dim=-1)
        ratio = along / transverse.clamp(min=1e-8)
        assert ratio.max().item() <= 0.2

    def test_nominal_step_shape(self, dynamics, latent_dim):
        z = torch.randn(4, latent_dim)
        z_next = dynamics.nominal_step(z)
        assert z_next.shape == z.shape

    def test_step_without_tube_geo(self, dynamics, latent_dim):
        """Without tube geometry, step() == nominal_step()."""
        z = torch.randn(4, latent_dim)
        dynamics.eval()
        with torch.no_grad():
            s1 = dynamics.step(z)
            s2 = dynamics.nominal_step(z)
        assert torch.allclose(s1, s2)

    def test_step_with_tube_geo_differs(self, dynamics, latent_dim, sample_families):
        """With tube geometry set, step() should differ from nominal_step()
        once the recovery net is trained (it starts near-zero, so only
        slightly different).
        """
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        dynamics.set_tube_geometry(geo)
        dynamics.eval()
        z = torch.randn(4, latent_dim)
        with torch.no_grad():
            s_combined = dynamics.step(z)
            s_nominal = dynamics.nominal_step(z)
        # Should be very close initially (zero-init recovery)
        assert torch.allclose(s_combined, s_nominal, atol=1e-3)
        dynamics.set_tube_geometry(None)

    def test_rollout_shape(self, dynamics, latent_dim):
        z0 = torch.randn(4, latent_dim)
        dynamics.eval()
        with torch.no_grad():
            traj = dynamics.rollout(z0, T=20)
        assert traj.shape == (20, 4, latent_dim)

    def test_rollout_nominal_shape(self, dynamics, latent_dim):
        z0 = torch.randn(4, latent_dim)
        dynamics.eval()
        with torch.no_grad():
            traj = dynamics.rollout_nominal(z0, T=12)
        assert traj.shape == (12, 4, latent_dim)

    def test_multi_step_predict_shape(self, dynamics, latent_dim):
        z = torch.randn(4, latent_dim)
        with torch.no_grad():
            out = dynamics.multi_step_predict(z, num_substeps=5)
        assert out.shape == (4, latent_dim)

    def test_gate_logits(self, dynamics, latent_dim, num_regimes):
        z = torch.randn(8, latent_dim)
        logits = dynamics.gate_logits(z)
        assert logits.shape == (8, num_regimes)

    def test_gate_probs_sum_to_one(self, dynamics, latent_dim):
        z = torch.randn(8, latent_dim)
        probs = dynamics.gate_probs(z)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_set_tube_geometry(self, dynamics, sample_families):
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))
        assert not dynamics.has_tube_geometry
        dynamics.set_tube_geometry(geo)
        assert dynamics.has_tube_geometry
        dynamics.set_tube_geometry(None)
        assert not dynamics.has_tube_geometry


# ---------------------------------------------------------------------------
# Loss helper tests
# ---------------------------------------------------------------------------

class TestLossHelpers:

    def test_dynamics_consistency_loss(self, dynamics, latent_dim):
        z = torch.randn(10, latent_dim)
        loss = dynamics.dynamics_consistency_loss(z, num_substeps=2)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_velocity_direction_loss(self, dynamics, latent_dim):
        z = torch.randn(10, latent_dim)
        loss = dynamics.velocity_direction_loss(z, num_substeps=2)
        assert loss.shape == ()

    def test_velocity_magnitude_loss(self, dynamics, latent_dim):
        z = torch.randn(10, latent_dim)
        loss = dynamics.velocity_magnitude_loss(z, num_substeps=2)
        assert loss.shape == ()

    def test_velocity_contrastive_loss(self, dynamics, latent_dim):
        z = torch.randn(16, latent_dim)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        loss = dynamics.velocity_contrastive_loss(z, labels)
        assert loss.shape == ()

    def test_terminal_velocity_loss(self, dynamics, latent_dim):
        z = torch.randn(8, latent_dim)
        loss = dynamics.terminal_velocity_loss(z)
        assert loss.shape == ()

    def test_gate_supervised_loss(self, dynamics, latent_dim, num_regimes):
        z = torch.randn(8, latent_dim)
        labels = torch.randint(0, num_regimes, (8,))
        loss = dynamics.gate_supervised_loss(z, labels)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_regime_consistency_loss(self, dynamics, latent_dim):
        z_a = torch.randn(5, latent_dim)
        z_b = torch.randn(5, latent_dim)
        loss = dynamics.regime_consistency_loss(z_a, z_b)
        assert loss.shape == ()

    def test_gate_entropy_loss(self, dynamics, latent_dim):
        z = torch.randn(8, latent_dim)
        loss = dynamics.gate_entropy_loss(z)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# Recovery training invariants
# ---------------------------------------------------------------------------

class TestRecoveryInvariants:

    def test_recovery_grad_flows_to_recovery_net_only(self, dynamics, latent_dim, sample_families):
        """When nominal params are frozen, gradients should only reach recovery_net."""
        # Give recovery net non-zero weights so gradients are non-trivial
        with torch.no_grad():
            for p in dynamics.recovery_net.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        for p in dynamics.gate.parameters():
            p.requires_grad_(False)
        for p in dynamics.velocity_nets.parameters():
            p.requires_grad_(False)

        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))

        z = torch.randn(4, latent_dim)
        anchor, tangent = geo.nearest_context(z)
        corr = dynamics.recovery_field(z, anchor.detach(), tangent.detach())
        loss = corr.pow(2).sum(dim=-1).mean()
        loss.backward()

        has_recovery_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in dynamics.recovery_net.parameters()
        )
        has_gate_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in dynamics.gate.parameters()
            if p.requires_grad
        )
        has_vel_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in dynamics.velocity_nets.parameters()
            if p.requires_grad
        )
        assert has_recovery_grad
        assert not has_gate_grad
        assert not has_vel_grad

        for p in dynamics.gate.parameters():
            p.requires_grad_(True)
        for p in dynamics.velocity_nets.parameters():
            p.requires_grad_(True)

    def test_on_tube_correction_decreases_with_training(self, latent_dim, sample_families):
        """A quick sanity check: training the null loss should reduce corrections."""
        dynamics = TubeAwareDynamics(latent_dim, num_regimes=2)
        trajs, families, fam_to_idx = sample_families
        geo = TubeGeometry.build(trajs, families, fam_to_idx, torch.device("cpu"))

        # Give recovery net non-zero weights to start
        with torch.no_grad():
            for p in dynamics.recovery_net.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        opt = torch.optim.Adam(
            list(dynamics.recovery_net.parameters())
            + list(dynamics.recovery_gain.parameters()),
            lr=1e-2,
        )
        z_on = torch.randn(16, latent_dim)
        anchor, tangent = geo.nearest_context(z_on)

        norms_before = dynamics.recovery_field(
            z_on, anchor.detach(), tangent.detach(),
        ).detach().norm(dim=-1).mean().item()

        for _ in range(50):
            corr = dynamics.recovery_field(z_on, anchor.detach(), tangent.detach())
            loss = corr.norm(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        norms_after = dynamics.recovery_field(
            z_on, anchor.detach(), tangent.detach(),
        ).detach().norm(dim=-1).mean().item()

        assert norms_after < norms_before
