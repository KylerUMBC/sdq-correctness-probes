"""Tests for sdq.losses — all six loss terms plus combined objective."""

import torch
import pytest

from sdq.losses.reconstruction import reconstruction_loss
from sdq.losses.semantic_consistency import (
    semantic_consistency_loss,
    batch_semantic_consistency_loss,
)
from sdq.losses.transport_consistency import (
    transport_consistency_loss,
    transport_consistency_from_tensors,
)
from sdq.losses.cycle import cycle_loss, batch_cycle_loss
from sdq.losses.gauge_regularization import (
    gauge_regularization_loss,
    rank_penalty,
    smoothness_penalty,
    near_identity_penalty,
)
from sdq.losses.event_consistency import (
    event_consistency_loss,
    soft_event_consistency_loss,
)
from sdq.losses.total import SDQLoss, LossWeights, LossBreakdown
from sdq.transport.local_operator import TransportResult


class TestReconstructionLoss:
    def test_zero_for_identical(self):
        h = torch.randn(10, 32)
        assert reconstruction_loss(h, h).item() < 1e-8

    def test_positive_for_different(self):
        h = torch.randn(10, 32)
        h_hat = h + 0.1 * torch.randn_like(h)
        assert reconstruction_loss(h, h_hat).item() > 0

    def test_reduction_none(self):
        h = torch.randn(10, 32)
        h_hat = torch.randn(10, 32)
        per_step = reconstruction_loss(h, h_hat, reduction="none")
        assert per_step.shape == (10,)

    def test_reduction_sum_vs_mean(self):
        h = torch.randn(10, 32)
        h_hat = torch.randn(10, 32)
        s = reconstruction_loss(h, h_hat, reduction="sum")
        m = reconstruction_loss(h, h_hat, reduction="mean")
        assert torch.allclose(s / 10, m, atol=1e-5)


class TestSemanticConsistencyLoss:
    def test_zero_for_identical(self):
        z = torch.randn(10, 16)
        assert semantic_consistency_loss(z, z).item() < 1e-8

    def test_batch_consistency(self):
        z_list = [torch.randn(10, 16) for _ in range(3)]
        loss = batch_semantic_consistency_loss(z_list)
        assert loss.item() > 0

    def test_batch_single_element(self):
        z_list = [torch.randn(10, 16)]
        loss = batch_semantic_consistency_loss(z_list)
        assert loss.item() == 0.0


class TestTransportConsistencyLoss:
    def test_zero_residual(self):
        result = TransportResult(
            G_t=torch.zeros(0),
            transported_velocity=torch.randn(9, 32),
            target_velocity=torch.zeros(9, 32),
            source_velocity=torch.randn(9, 32),
            residual=torch.zeros(9, 32),
        )
        assert transport_consistency_loss(result).item() < 1e-8

    def test_from_tensors(self):
        src_vel = torch.randn(9, 32)
        tgt_vel = torch.randn(9, 32)
        transported = src_vel  # identity transport
        loss = transport_consistency_from_tensors(src_vel, tgt_vel, transported)
        assert loss.item() > 0


class TestCycleLoss:
    def test_identity_zero_loss(self):
        D = 8
        T = 5
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        loss = cycle_loss(I, I)
        assert loss.item() < 1e-5

    def test_with_triple(self):
        D = 8
        T = 5
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        loss = cycle_loss(I, I, G_jk=I, G_ik=I)
        assert loss.item() < 1e-5

    def test_batch_cycle_loss(self):
        D = 8
        T = 5
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        pairs = {
            ("a", "b"): I.clone(),
            ("b", "a"): I.clone(),
        }
        loss = batch_cycle_loss(pairs)
        assert loss.item() < 1e-5


class TestGaugeRegularization:
    def test_identity_zero_penalties(self):
        D = 8
        T = 5
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        assert rank_penalty(I).item() < 1e-5
        assert near_identity_penalty(I).item() < 1e-5

    def test_smoothness_constant(self):
        D = 8
        T = 5
        G = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        assert smoothness_penalty(G).item() < 1e-8

    def test_combined_loss(self):
        D = 8
        T = 5
        I = torch.eye(D).unsqueeze(0).expand(T, -1, -1)
        G = I + 0.1 * torch.randn(T, D, D)
        loss = gauge_regularization_loss(G)
        assert loss.item() > 0


class TestEventConsistencyLoss:
    def test_perfect_preservation(self):
        alignment = [0, 1, 2, 3, 4]
        events = [0, 0, 1, 1, 2]
        loss = event_consistency_loss(alignment, events, events)
        assert loss.item() == 0.0

    def test_soft_consistency(self):
        T_src, T_tgt = 5, 5
        # Perfect alignment matrix (diagonal)
        A = torch.eye(T_src, T_tgt)
        src_events = torch.tensor([0, 0, 1, 1, 2])
        tgt_events = torch.tensor([0, 0, 1, 1, 2])
        loss = soft_event_consistency_loss(A, src_events, tgt_events)
        assert loss.item() < 1e-5

    def test_soft_consistency_mismatch(self):
        T_src, T_tgt = 5, 5
        A = torch.eye(T_src, T_tgt)
        src_events = torch.tensor([0, 0, 1, 1, 2])
        tgt_events = torch.tensor([2, 2, 0, 0, 1])  # mismatched
        loss = soft_event_consistency_loss(A, src_events, tgt_events)
        assert loss.item() > 0.5


class TestSDQLoss:
    def test_all_none_zero(self):
        loss_fn = SDQLoss()
        breakdown = loss_fn()
        assert breakdown.total.item() == 0.0

    def test_only_reconstruction(self):
        loss_fn = SDQLoss()
        L_rec = torch.tensor(1.5)
        breakdown = loss_fn(L_rec=L_rec)
        assert breakdown.total.item() == pytest.approx(1.5)
        assert breakdown.reconstruction.item() == pytest.approx(1.5)

    def test_weights(self):
        weights = LossWeights(reconstruction=1.0, semantic=2.0)
        loss_fn = SDQLoss(weights=weights)
        breakdown = loss_fn(
            L_rec=torch.tensor(1.0),
            L_sem=torch.tensor(1.0),
        )
        assert breakdown.total.item() == pytest.approx(3.0)

    def test_breakdown_to_dict(self):
        loss_fn = SDQLoss()
        breakdown = loss_fn(L_rec=torch.tensor(1.0))
        d = breakdown.to_dict()
        assert "total" in d
        assert "reconstruction" in d
        assert d["total"] == pytest.approx(1.0)

    def test_gradient_flows(self):
        loss_fn = SDQLoss()
        x = torch.tensor(2.0, requires_grad=True)
        breakdown = loss_fn(L_rec=x)
        breakdown.total.backward()
        assert x.grad is not None
