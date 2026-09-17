"""Holdout robustness: unseen templates, wording families, domains.

Evaluates whether learned representations generalize to prompt
variants not seen during training. Tests alignment quality,
transport residuals, and latent consistency on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HoldoutMetrics:
    """Metrics from evaluating on held-out prompt variants."""

    alignment_quality: float  # soft-DTW distance (lower = better alignment)
    transport_residual: float  # mean transport residual
    latent_consistency: float  # mean ||z_i - z_j|| across equivalent pairs
    reconstruction_error: float  # mean ||h - h_hat||^2
    num_pairs: int


def evaluate_holdout(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    transport: torch.nn.Module,
    holdout_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    gauge_fn: callable | None = None,
) -> HoldoutMetrics:
    """Evaluate SDQ model on held-out prompt pairs.

    Args:
        encoder: trajectory -> latent.
        decoder: (latent, gauge) -> reconstructed hidden.
        transport: (source, target_aligned) -> TransportResult.
        holdout_pairs: list of (trajectory_i, trajectory_j) aligned pairs.
        gauge_fn: optional function to produce gauge context from trajectory.

    Returns:
        HoldoutMetrics summarizing held-out performance.
    """
    encoder.eval()
    decoder.eval()
    transport.eval()

    total_recon = 0.0
    total_latent_consistency = 0.0
    total_transport_residual = 0.0
    total_alignment = 0.0
    n = len(holdout_pairs)

    with torch.no_grad():
        for h_i, h_j in holdout_pairs:
            # Encode
            z_i = encoder(h_i)
            z_j = encoder(h_j)

            # Latent consistency
            T = min(z_i.shape[0], z_j.shape[0])
            total_latent_consistency += (z_i[:T] - z_j[:T]).pow(2).sum(dim=-1).mean().item()

            # Reconstruction
            for h, z in [(h_i, z_i), (h_j, z_j)]:
                u = gauge_fn(h) if gauge_fn is not None else None
                h_hat = decoder(z, u)
                total_recon += (h - h_hat).pow(2).sum(dim=-1).mean().item()

            # Transport
            result = transport(h_i, h_j)
            total_transport_residual += result.residual.pow(2).sum(dim=-1).mean().item()

    return HoldoutMetrics(
        alignment_quality=total_alignment / max(n, 1),
        transport_residual=total_transport_residual / max(n, 1),
        latent_consistency=total_latent_consistency / max(n, 1),
        reconstruction_error=total_recon / max(2 * n, 1),
        num_pairs=n,
    )
