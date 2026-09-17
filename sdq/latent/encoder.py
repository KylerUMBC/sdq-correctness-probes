"""Trajectory encoder: maps local hidden windows to latent semantic states z_t.

    E : (h_{t-k:t+k}) -> z_t

Uses a local temporal window, not just one point, to capture
sequential context around each position.

Three implementations:
    1. TemporalConvEncoder  — 1D conv over causal windows (fast, simple, RF=3)
    2. TemporalTransformerEncoder — self-attention over windows (flexible)
    3. MultiScaleConvEncoder — dilated causal convs (RF=15, batched support)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvEncoder(nn.Module):
    """1D causal convolution encoder: local window -> z_t.

    Architecture:
        pad -> conv1d (window_size kernel) -> GELU -> conv1d (1 kernel) -> z_t

    This is the simplest viable encoder. Causal padding ensures z_t
    only depends on h_{t-k:t} (no future leakage).
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        window_size: int = 5,
        intermediate_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.window_size = window_size
        inter = intermediate_dim or max(latent_dim, hidden_dim // 2)

        # Causal conv: kernel sees [t-k, ..., t] where k = window_size - 1
        self.conv1 = nn.Conv1d(hidden_dim, inter, kernel_size=window_size, padding=0)
        self.conv2 = nn.Conv1d(inter, latent_dim, kernel_size=1)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Encode trajectory to latent states.

        Args:
            h: [T, D] hidden-state trajectory.

        Returns:
            [T, D_z] latent semantic trajectory.
        """
        T, D = h.shape

        # [1, D, T] for conv1d
        x = h.T.unsqueeze(0)

        # Causal pad on the left: window_size - 1 zeros
        x = F.pad(x, (self.window_size - 1, 0))

        x = self.conv1(x)  # [1, inter, T]
        x = F.gelu(x)
        x = self.conv2(x)  # [1, D_z, T]

        z = x.squeeze(0).T  # [T, D_z]
        z = self.norm(z)
        return z


class TemporalTransformerEncoder(nn.Module):
    """Self-attention over local windows -> z_t.

    For each position t, extracts a local window and runs a small
    transformer to produce z_t. More flexible than convolution but
    heavier.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        window_size: int = 5,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.window_size = window_size

        self.input_proj = nn.Linear(hidden_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, latent_dim)

        # Learnable position embeddings for window positions
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, latent_dim) * 0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Encode trajectory to latent states.

        Args:
            h: [T, D] hidden-state trajectory.

        Returns:
            [T, D_z] latent semantic trajectory.
        """
        T, D = h.shape
        k = self.window_size

        # Pad on the left for causal windowing
        h_padded = F.pad(h.unsqueeze(0), (0, 0, k - 1, 0)).squeeze(0)  # [T+k-1, D]

        # Extract windows: [T, k, D]
        windows = h_padded.unfold(0, k, 1).permute(0, 2, 1)  # [T, k, D]

        # Project and add positional
        windows = self.input_proj(windows)  # [T, k, D_z]
        windows = windows + self.pos_embed[:, :k, :]

        # Self-attention over each window
        z = self.transformer(windows)  # [T, k, D_z]

        # Take the last position (current timestep) as z_t
        z = z[:, -1, :]  # [T, D_z]
        z = self.output_proj(z)
        return z


class MultiScaleConvEncoder(nn.Module):
    """Dilated causal convolution encoder with multi-scale receptive field.

    Architecture (3-layer dilated stack + pointwise projection):
        Conv1d(D, inter, k=3, dilation=1)  -> GELU + Residual   [RF: 3]
        Conv1d(inter, inter, k=3, dilation=2) -> GELU + Residual [RF: 7]
        Conv1d(inter, inter, k=3, dilation=4) -> GELU + Residual [RF: 15]
        Conv1d(inter, D_z, k=1)                                  [RF: 15]
        LayerNorm

    Supports both unbatched [T, D] and batched [B, T, D] input.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        intermediate_dim: int | None = None,
        num_layers: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        inter = intermediate_dim or max(latent_dim, hidden_dim // 2)

        # Build dilated conv stack
        self.convs = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        dilations = [2**i for i in range(num_layers)]  # [1, 2, 4]

        for i, dilation in enumerate(dilations):
            in_ch = hidden_dim if i == 0 else inter
            self.convs.append(
                nn.Conv1d(in_ch, inter, kernel_size=kernel_size,
                          dilation=dilation, padding=0)
            )
            # Residual projection when input channels differ
            if in_ch != inter:
                self.residual_projs.append(nn.Conv1d(in_ch, inter, kernel_size=1))
            else:
                self.residual_projs.append(nn.Identity())

        # Causal left-padding per layer
        self.causal_paddings = [(kernel_size - 1) * d for d in dilations]

        # Pointwise projection to latent dim
        self.proj = nn.Conv1d(inter, latent_dim, kernel_size=1)
        self.norm = nn.LayerNorm(latent_dim)

    @property
    def receptive_field(self) -> int:
        """Total receptive field in tokens."""
        rf = 1
        for pad in self.causal_paddings:
            rf += pad
        return rf

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Encode trajectory to latent states.

        Args:
            h: [T, D] single trajectory OR [B, T, D] batched trajectories.

        Returns:
            [T, D_z] or [B, T, D_z] latent semantic trajectory.
        """
        unbatched = h.dim() == 2
        if unbatched:
            h = h.unsqueeze(0)  # [1, T, D]

        # Conv1d expects [B, C, T]
        x = h.permute(0, 2, 1)  # [B, D, T]

        for conv, res_proj, pad in zip(
            self.convs, self.residual_projs, self.causal_paddings
        ):
            # Causal left-padding
            x_padded = F.pad(x, (pad, 0))
            out = F.gelu(conv(x_padded))  # [B, inter, T]
            # Residual connection
            x = out + res_proj(x)

        x = self.proj(x)  # [B, D_z, T]
        z = x.permute(0, 2, 1)  # [B, T, D_z]
        z = self.norm(z)

        if unbatched:
            z = z.squeeze(0)  # [T, D_z]
        return z
