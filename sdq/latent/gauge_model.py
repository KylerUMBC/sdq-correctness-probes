"""Gauge / surface-realization model for local coordinate changes.

The gauge u_t captures how the surface-form of a prompt affects
the hidden-state representation. Given two prompts that are
semantically equivalent but worded differently, the gauge should
capture exactly those differences.

Two strategies:
    1. GaugeEmbedding  — learnable per-prompt-family vector
    2. GaugeEncoder    — infer gauge from observed hidden states

The gauge feeds into:
    - the decoder: h_hat_t = R(z_t, u_t)
    - the transport: G_t conditioned on gauge difference
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaugeEmbedding(nn.Module):
    """Learnable per-prompt gauge vectors.

    Each prompt variant gets a fixed gauge embedding that the
    decoder uses to reconstruct the variant-specific hidden states.

    Simple but effective when the number of variants is small and known.
    """

    def __init__(
        self,
        num_prompts: int,
        gauge_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_prompts, gauge_dim)
        # Small initialization: gauge should start near-zero
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, prompt_idx: int, T: int) -> torch.Tensor:
        """Get gauge vector for a prompt, broadcast to T timesteps.

        Args:
            prompt_idx: integer index of the prompt variant.
            T: sequence length (number of timesteps).

        Returns:
            [T, D_u] gauge context (same vector at each t).
        """
        idx = torch.tensor([prompt_idx], device=self.embedding.weight.device)
        u = self.embedding(idx)  # [1, D_u]
        return u.expand(T, -1)  # [T, D_u]


class GaugeEncoder(nn.Module):
    """Infer gauge from observed hidden states.

    The gauge should capture what's specific to the surface form:
    the difference between the observed trajectory and what the
    shared semantic content alone would predict.

    Architecture:
        h_t -> MLP -> u_t

    The encoder is trained jointly so that:
        h_hat_t = decoder(encoder(h_t), gauge_encoder(h_t))
    subject to the constraint that the semantic encoder output
    should be shared across equivalent prompts.
    """

    def __init__(
        self,
        hidden_dim: int,
        gauge_dim: int,
        intermediate_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gauge_dim = gauge_dim
        inter = intermediate_dim or max(gauge_dim * 2, hidden_dim // 4)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inter),
            nn.GELU(),
            nn.Linear(inter, gauge_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Infer gauge from hidden states.

        Args:
            h: [T, D] observed hidden trajectory.

        Returns:
            [T, D_u] inferred gauge context.
        """
        return self.net(h)


class TemporalGaugeEncoder(nn.Module):
    """Infer time-varying gauge using temporal context.

    Unlike GaugeEncoder (which is pointwise), this uses a
    1D convolution to capture how surface effects evolve
    across positions.
    """

    def __init__(
        self,
        hidden_dim: int,
        gauge_dim: int,
        window_size: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gauge_dim = gauge_dim
        self.window_size = window_size

        self.conv = nn.Conv1d(hidden_dim, gauge_dim, kernel_size=window_size, padding=0)
        self.norm = nn.LayerNorm(gauge_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Infer gauge with temporal context.

        Args:
            h: [T, D] observed hidden trajectory.

        Returns:
            [T, D_u] gauge context.
        """
        # Causal padding
        x = h.T.unsqueeze(0)  # [1, D, T]
        x = F.pad(x, (self.window_size - 1, 0))
        x = self.conv(x)  # [1, D_u, T]
        u = x.squeeze(0).T  # [T, D_u]
        return self.norm(u)
