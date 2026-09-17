"""Observation model: reconstruct h_t from z_t and gauge context u_t.

    h_hat_t = R(z_t, u_t)

The decoder ties the latent space back to observed hidden states,
preventing the encoder from learning arbitrary representations.

The gauge context u_t captures surface-form information that
varies across paraphrases of the same semantic content.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObservationDecoder(nn.Module):
    """Reconstructs observed hidden states from latent + gauge.

    Architecture:
        concat(z_t, u_t) -> MLP -> h_hat_t

    The gauge input u_t can be:
        - a learned per-prompt embedding
        - output of the GaugeModel
        - zero (pure semantic reconstruction, no gauge)
    """

    def __init__(
        self,
        latent_dim: int,
        gauge_dim: int,
        hidden_dim: int,
        intermediate_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gauge_dim = gauge_dim
        self.hidden_dim = hidden_dim
        inter = intermediate_dim or max(hidden_dim, (latent_dim + gauge_dim) * 2)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + gauge_dim, inter),
            nn.GELU(),
            nn.Linear(inter, inter),
            nn.GELU(),
            nn.Linear(inter, hidden_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct hidden trajectory from latent + gauge.

        Args:
            z: [T, D_z] latent semantic trajectory.
            u: [T, D_u] gauge context. If None, uses zeros.

        Returns:
            [T, D] reconstructed hidden trajectory.
        """
        if u is None:
            u = torch.zeros(z.shape[0], self.gauge_dim, device=z.device, dtype=z.dtype)

        x = torch.cat([z, u], dim=-1)  # [T, D_z + D_u]
        return self.net(x)  # [T, D]


class ResidualDecoder(nn.Module):
    """Decoder that predicts a residual correction to a linear projection.

    h_hat_t = W z_t + MLP(z_t, u_t)

    The linear term gives a strong baseline; the MLP handles
    nonlinear gauge correction.
    """

    def __init__(
        self,
        latent_dim: int,
        gauge_dim: int,
        hidden_dim: int,
        intermediate_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gauge_dim = gauge_dim
        self.hidden_dim = hidden_dim
        inter = intermediate_dim or max(hidden_dim, latent_dim * 2)

        self.linear = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.correction = nn.Sequential(
            nn.Linear(latent_dim + gauge_dim, inter),
            nn.GELU(),
            nn.Linear(inter, hidden_dim),
        )

        # Initialize correction near zero so early training is ~linear
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(
        self,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct with residual architecture.

        Args:
            z: [T, D_z] latent semantic trajectory.
            u: [T, D_u] gauge context. If None, uses zeros.

        Returns:
            [T, D] reconstructed hidden trajectory.
        """
        if u is None:
            u = torch.zeros(z.shape[0], self.gauge_dim, device=z.device, dtype=z.dtype)

        base = self.linear(z)  # [T, D]
        x = torch.cat([z, u], dim=-1)
        correction = self.correction(x)  # [T, D]
        return base + correction
