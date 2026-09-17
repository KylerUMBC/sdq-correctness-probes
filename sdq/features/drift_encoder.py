"""Drift encoder: maps local hidden-state motion into a proxy-drift representation.

The drift state p_t is meant to capture signals about:
  - local shortcut pressure
  - movement toward bad-basin behavior
  - instability / divergence from stable reasoning
  - approach to basin boundaries

The encoder fuses latent semantic features (from the existing encoder)
with cheap-to-compute raw motion features to produce a compact drift
vector at each timestep.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


RAW_FEATURE_DIM = 5  # norm, delta_norm, cosine, growth_ratio, centroid_dist


def extract_raw_drift_features(h: Tensor) -> Tensor:
    """Compute raw per-timestep motion features from hidden states.

    Args:
        h: [T, D] hidden-state trajectory for one layer.

    Returns:
        [T, RAW_FEATURE_DIM] tensor of motion features:
          0: ||h_t||                      (state magnitude)
          1: ||h_t - h_{t-1}||           (step size)
          2: cos(h_t, h_{t-1})           (direction stability)
          3: ||h_t|| / ||h_{t-1}||       (norm growth ratio)
          4: ||h_t - centroid||           (centroid distance)
    """
    T, D = h.shape
    device = h.device

    h_norm = h.norm(dim=-1, keepdim=True)  # [T, 1]

    delta = h[1:] - h[:-1]  # [T-1, D]
    delta_norm = delta.norm(dim=-1, keepdim=True)
    delta_norm = torch.cat([torch.zeros(1, 1, device=device), delta_norm], dim=0)

    cos_sim = F.cosine_similarity(h[1:], h[:-1], dim=-1).unsqueeze(-1)
    cos_sim = torch.cat([torch.ones(1, 1, device=device), cos_sim], dim=0)

    h_norm_safe = h_norm.clamp(min=1e-8)
    growth = h_norm_safe[1:] / h_norm_safe[:-1]
    growth = torch.cat([torch.ones(1, 1, device=device), growth], dim=0)

    centroid = h.mean(dim=0, keepdim=True)
    dist_centroid = (h - centroid).norm(dim=-1, keepdim=True)

    return torch.cat([h_norm, delta_norm, cos_sim, growth, dist_centroid], dim=-1)


class DriftEncoder(nn.Module):
    """Encode local hidden-state motion into drift representation p_t.

    Takes two input streams and fuses them:
      1. Latent semantic features: z_t and z_t - z_{t-1} from the semantic encoder
      2. Raw motion features: norms, cosines, growth ratios from raw hidden states

    The output p_t is a compact vector capturing local dynamical health.
    """

    def __init__(
        self,
        latent_dim: int,
        drift_dim: int,
        raw_feature_dim: int = RAW_FEATURE_DIM,
        hidden_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.drift_dim = drift_dim
        self.raw_feature_dim = raw_feature_dim

        input_dim = latent_dim * 2 + raw_feature_dim  # z_t, delta_z_t, raw features

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, drift_dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(drift_dim * hidden_mult, drift_dim),
            nn.LayerNorm(drift_dim),
        )

    def forward(
        self,
        z: Tensor,
        raw_features: Tensor,
    ) -> Tensor:
        """Compute drift representation.

        Args:
            z: [T, D_z] or [B, T, D_z] latent semantic trajectory.
            raw_features: [T, raw_feature_dim] or [B, T, raw_feature_dim]
                motion features from extract_raw_drift_features().

        Returns:
            [T, drift_dim] or [B, T, drift_dim] drift representation.
        """
        unbatched = z.dim() == 2
        if unbatched:
            z = z.unsqueeze(0)
            raw_features = raw_features.unsqueeze(0)

        B, T, D_z = z.shape

        # Compute latent delta
        z_prev = torch.cat([z[:, :1, :], z[:, :-1, :]], dim=1)
        delta_z = z - z_prev  # [B, T, D_z]

        x = torch.cat([z, delta_z, raw_features], dim=-1)  # [B, T, input_dim]
        p = self.net(x)  # [B, T, drift_dim]

        if unbatched:
            p = p.squeeze(0)
        return p
