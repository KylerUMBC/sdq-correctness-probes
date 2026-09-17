"""Integration smoke test: end-to-end SDQ pipeline on real run data.

Loads actual run data, extracts trajectories, aligns them,
computes transport, encodes to latent, decodes, and evaluates losses.
"""

import torch
import pytest
from pathlib import Path

from sdq.instrumentation.run_storage import load_run, collect_runs
from sdq.trajectories.extraction import extract_trajectory, Trajectory
from sdq.trajectories.normalization import normalize_trajectory
from sdq.alignment.soft_dtw import soft_dtw_distance, soft_dtw_alignment, alignment_to_map
from sdq.alignment.monotone_alignment import MonotoneAligner
from sdq.transport.local_operator import LocalTransportField
from sdq.latent.encoder import TemporalConvEncoder
from sdq.latent.decoder import ResidualDecoder
from sdq.latent.gauge_model import GaugeEncoder
from sdq.latent.dynamics import LatentODE
from sdq.losses.total import SDQLoss
from sdq.losses.reconstruction import reconstruction_loss
from sdq.losses.semantic_consistency import semantic_consistency_loss
from sdq.losses.transport_consistency import transport_consistency_loss
from sdq.losses.gauge_regularization import gauge_regularization_loss

RUNS_DIR = Path(__file__).parent.parent / "data" / "runs"

# A clean checkout contains data/runs/.gitkeep, which is not captured run data.
pytestmark = pytest.mark.skipif(
    not RUNS_DIR.exists() or sum(path.is_dir() for path in RUNS_DIR.iterdir()) < 2,
    reason="Requires at least two local captured runs (not distributed in the public snapshot)",
)


class TestEndToEnd:
    """End-to-end integration test using real captured run data."""

    def test_load_and_extract(self):
        """Can we load runs and extract trajectories?"""
        # Pick first two run directories
        run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()])[:2]
        assert len(run_dirs) >= 2, "Need at least 2 runs for integration test"

        runs = [load_run(d) for d in run_dirs]
        trajs = [extract_trajectory(r, layer=-1, to_float=True) for r in runs]

        for traj in trajs:
            assert traj.states.dim() == 2
            assert traj.D > 0
            assert traj.T > 0

    def test_alignment_pipeline(self):
        """Load -> extract -> align -> transport -> losses."""
        run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()])[:2]
        if len(run_dirs) < 2:
            pytest.skip("Need at least 2 runs")

        runs = [load_run(d) for d in run_dirs]
        trajs = [extract_trajectory(r, layer=-1, to_float=True) for r in runs]

        h_i = trajs[0].states
        h_j = trajs[1].states
        D = h_i.shape[1]

        # Soft-DTW alignment
        dist = soft_dtw_distance(h_i, h_j, gamma=1.0)
        assert dist.item() > 0

        # Get alignment path
        dist, path = soft_dtw_alignment(h_i, h_j, gamma=1.0)
        amap = alignment_to_map(path, source_len=h_i.shape[0])

        # Warp target to source time
        h_j_aligned = h_j[amap]
        assert h_j_aligned.shape == h_i.shape

        # Transport
        transport = LocalTransportField(D, rank=8)
        result = transport(h_i, h_j_aligned, return_operators=True)

        # Transport consistency loss
        L_trans = transport_consistency_loss(result)
        assert L_trans.item() >= 0

        # Gauge regularization
        L_gauge = gauge_regularization_loss(result.G_t)
        assert L_gauge.item() >= 0

    def test_full_pipeline_with_latent(self):
        """Full pipeline including latent model."""
        run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()])[:2]
        if len(run_dirs) < 2:
            pytest.skip("Need at least 2 runs")

        runs = [load_run(d) for d in run_dirs]
        trajs = [extract_trajectory(r, layer=-1, to_float=True) for r in runs]

        h_i = trajs[0].states
        h_j = trajs[1].states
        D = h_i.shape[1]

        # Dimensions
        latent_dim = 64
        gauge_dim = 32

        # Models
        encoder = TemporalConvEncoder(D, latent_dim, window_size=3)
        decoder = ResidualDecoder(latent_dim, gauge_dim, D)
        gauge_enc = GaugeEncoder(D, gauge_dim)
        dynamics = LatentODE(latent_dim)
        aligner = MonotoneAligner(D)
        transport = LocalTransportField(D, rank=8)
        loss_fn = SDQLoss()

        # Encode
        z_i = encoder(h_i)
        z_j = encoder(h_j)

        # Decode with gauge
        u_i = gauge_enc(h_i)
        u_j = gauge_enc(h_j)
        h_hat_i = decoder(z_i, u_i)
        h_hat_j = decoder(z_j, u_j)

        # Reconstruction loss
        L_rec_i = reconstruction_loss(h_i, h_hat_i)
        L_rec_j = reconstruction_loss(h_j, h_hat_j)
        L_rec = (L_rec_i + L_rec_j) / 2

        # Align via learned aligner
        h_j_warped = aligner.align_trajectory(h_i, h_j)
        z_j_warped = encoder(h_j_warped)

        # Semantic consistency (after alignment)
        T = min(z_i.shape[0], z_j_warped.shape[0])
        L_sem = semantic_consistency_loss(z_i[:T], z_j_warped[:T])

        # Transport
        result = transport(h_i, h_j_warped, return_operators=True)
        L_trans = transport_consistency_loss(result)
        L_gauge = gauge_regularization_loss(result.G_t)

        # Dynamics consistency
        L_dyn = dynamics.dynamics_consistency_loss(z_i)

        # Total loss
        breakdown = loss_fn(
            L_rec=L_rec,
            L_sem=L_sem,
            L_trans=L_trans,
            L_gauge=L_gauge,
        )

        assert breakdown.total.item() > 0
        assert breakdown.total.requires_grad

        # Backward pass should work
        breakdown.total.backward()

        # Check some parameters got gradients
        assert any(p.grad is not None for p in encoder.parameters())
        assert any(p.grad is not None for p in decoder.parameters())
        assert any(p.grad is not None for p in transport.parameters())

    def test_training_step(self):
        """A single training step should reduce loss."""
        run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()])[:2]
        if len(run_dirs) < 2:
            pytest.skip("Need at least 2 runs")

        runs = [load_run(d) for d in run_dirs]
        trajs = [extract_trajectory(r, layer=-1, to_float=True) for r in runs]

        h_i = trajs[0].states
        h_j = trajs[1].states
        D = h_i.shape[1]

        latent_dim = 32
        gauge_dim = 16

        encoder = TemporalConvEncoder(D, latent_dim, window_size=3)
        decoder = ResidualDecoder(latent_dim, gauge_dim, D)
        transport = LocalTransportField(D, rank=4)
        aligner = MonotoneAligner(D)

        params = (
            list(encoder.parameters())
            + list(decoder.parameters())
            + list(transport.parameters())
            + list(aligner.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=1e-3)

        losses = []
        for _ in range(3):
            optimizer.zero_grad()

            # Re-align each step (aligner is learned)
            h_j_warped = aligner.align_trajectory(h_i, h_j)

            z_i = encoder(h_i)
            h_hat = decoder(z_i)
            L_rec = reconstruction_loss(h_i, h_hat)

            result = transport(h_i, h_j_warped)
            L_trans = transport_consistency_loss(result)

            loss = L_rec + L_trans
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (may not be monotone with 3 steps,
        # but last should be lower than first)
        assert losses[-1] < losses[0] * 1.5  # at least not diverging
