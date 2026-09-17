"""Local transport operator G_t^{i->j} conditioned on trajectory context.

Given two aligned trajectories, learns a time-varying linear operator
G_t that maps local motion (velocity) in trajectory i to corresponding
local motion in trajectory j:

    delta_h_{tau(t)}^{j} ≈ G_t^{i->j} delta_h_t^{i}

The operator G_t is conditioned on local trajectory context and,
optionally, on a discrete transform-type embedding. When transform-type
conditioning is used, transport is shared across semantic families for
the same surface-form change (e.g. therefore→since for both cat and bird).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from sdq.transport.low_rank_maps import LowRankTransport


@dataclass
class TransportResult:
    """Result of applying a transport field to a trajectory pair."""

    G_t: torch.Tensor  # [T-1, D, D] time-varying transport operators (empty if not requested)
    transported_velocity: torch.Tensor  # [T-1, D] G_t @ delta_h_t^i
    target_velocity: torch.Tensor  # [T-1, D] delta_h_{tau(t)}^j
    source_velocity: torch.Tensor  # [T-1, D] delta_h_t^i
    residual: torch.Tensor  # [T-1, D] target - transported
    U: torch.Tensor | None = None  # [T-1, D, rank] low-rank factor
    V: torch.Tensor | None = None  # [T-1, D, rank] low-rank factor


class LocalTransportField(nn.Module):
    """Learns local transport operators G_t conditioned on trajectory context.

    Given source and target trajectories (already aligned), produces a
    time-varying family of transport operators.

    Architecture: context MLP that takes (h_t^i, h_t^j, t/T, [transform_emb])
    and outputs parameters of a low-rank transport operator G_t = I + U V^T.

    When ``num_transform_types > 0``, a learnable embedding for each
    transform type is concatenated to the context input. This encourages
    the model to share transport structure across semantic families that
    undergo the same surface-form change (e.g., *therefore→since* maps
    should look similar for both cat and bird syllogisms).
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 8,
        context_dim: int = 128,
        num_transform_types: int = 0,
        transform_embed_dim: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_transform_types = num_transform_types
        self._correction_scale = 1.0 / (hidden_dim ** 0.5)

        # Optional transform-type embedding
        self.transform_embed: nn.Embedding | None = None
        extra_dim = 0
        if num_transform_types > 0:
            self.transform_embed = nn.Embedding(num_transform_types, transform_embed_dim)
            nn.init.normal_(self.transform_embed.weight, std=0.1)
            extra_dim = transform_embed_dim

        # Context encoder: maps (h_i, h_j, time_frac, [transform_emb]) -> context
        self.context_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + extra_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
        )

        # Produce low-rank factors: G_t = I + scale * U_t V_t^T
        self.U_net = nn.Linear(context_dim, hidden_dim * rank)
        self.V_net = nn.Linear(context_dim, hidden_dim * rank)

        # Learnable gate that starts at 0, allowing gradients to flow
        # through both U and V (default Kaiming init) while keeping
        # G_t ≈ I at initialization — avoids the bilinear dead zone.
        self.correction_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        source: torch.Tensor,
        target_aligned: torch.Tensor,
        return_operators: bool = False,
        transform_type_id: int | None = None,
    ) -> TransportResult:
        """Compute transport between aligned trajectory pair.

        Args:
            source: [T, D] source trajectory.
            target_aligned: [T, D] target trajectory, already time-aligned
                to source (same length T).
            return_operators: if True, compute full G_t matrices.
            transform_type_id: optional integer index into the transform-type
                embedding table. When provided, the embedding is broadcast
                across all time steps and concatenated to the context.

        Returns:
            TransportResult with velocities and residuals.
        """
        T, D = source.shape

        # Velocities
        src_vel = source[1:] - source[:-1]  # [T-1, D]
        tgt_vel = target_aligned[1:] - target_aligned[:-1]  # [T-1, D]

        # Time fractions
        time_frac = torch.arange(T - 1, device=source.device, dtype=source.dtype)
        time_frac = (time_frac / max(T - 2, 1)).unsqueeze(-1)  # [T-1, 1]

        # Context from midpoints of consecutive positions
        src_ctx = (source[:-1] + source[1:]) / 2  # [T-1, D]
        tgt_ctx = (target_aligned[:-1] + target_aligned[1:]) / 2  # [T-1, D]

        ctx_parts = [src_ctx, tgt_ctx, time_frac]

        # Append transform-type embedding if available
        if self.transform_embed is not None and transform_type_id is not None:
            idx = torch.tensor([transform_type_id], device=source.device)
            emb = self.transform_embed(idx)  # [1, embed_dim]
            emb = emb.expand(T - 1, -1)  # [T-1, embed_dim]
            ctx_parts.append(emb)

        ctx_input = torch.cat(ctx_parts, dim=-1)  # [T-1, 2D+1(+embed)]
        ctx = self.context_net(ctx_input)  # [T-1, context_dim]

        # Low-rank factors
        U = self.U_net(ctx).view(T - 1, D, self.rank)  # [T-1, D, rank]
        V = self.V_net(ctx).view(T - 1, D, self.rank)  # [T-1, D, rank]

        # Apply G_t = I + gate * scale * U V^T to source velocity
        Vt_vel = torch.bmm(
            V.transpose(1, 2),  # [T-1, rank, D]
            src_vel.unsqueeze(-1),  # [T-1, D, 1]
        )  # [T-1, rank, 1]
        raw_correction = torch.bmm(U, Vt_vel).squeeze(-1)  # [T-1, D]
        correction = self.correction_gate * self._correction_scale * raw_correction
        transported = src_vel + correction  # [T-1, D]

        residual = tgt_vel - transported  # [T-1, D]

        # Effective low-rank factors including gate+scale
        gate_scale = self.correction_gate * self._correction_scale
        U_eff = U * gate_scale
        V_eff = V

        # Full operator matrices if requested
        G_t = torch.zeros(0)
        if return_operators:
            G_t = torch.eye(D, device=source.device, dtype=source.dtype).unsqueeze(0).expand(T - 1, -1, -1).clone()
            UVt = torch.bmm(U_eff, V_eff.transpose(1, 2))  # [T-1, D, D]
            G_t = G_t + UVt

        return TransportResult(
            G_t=G_t,
            transported_velocity=transported,
            target_velocity=tgt_vel,
            source_velocity=src_vel,
            residual=residual,
            U=U_eff,
            V=V_eff,
        )
