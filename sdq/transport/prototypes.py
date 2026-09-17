"""Transport prototype factorization.

Factors the transport operator into a shared transform-type prototype
plus a small context-dependent correction:

    G_t = P_{type} + ΔG_t

where P_{type} is a learnable low-rank prototype per transform type,
and ΔG_t is penalized to stay small. This forces the model to learn
reusable transform laws before context-specific adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PrototypeTransportResult:
    """Result of prototype-factored transport."""

    transported_velocity: torch.Tensor  # [T-1, D]
    target_velocity: torch.Tensor       # [T-1, D]
    source_velocity: torch.Tensor       # [T-1, D]
    residual: torch.Tensor              # [T-1, D]
    U: torch.Tensor                     # [T-1, D, rank] effective low-rank factor
    V: torch.Tensor                     # [T-1, D, rank] effective low-rank factor
    U_proto: torch.Tensor               # [D, rank] prototype factor
    V_proto: torch.Tensor               # [D, rank] prototype factor
    U_delta: torch.Tensor               # [T-1, D, rank] correction factor
    V_delta: torch.Tensor               # [T-1, D, rank] correction factor
    delta_norm: torch.Tensor            # scalar: mean ||ΔG_t||_F^2
    G_t: torch.Tensor                   # [T-1, D, D] if return_operators, else empty


class TransportPrototypes(nn.Module):
    """Learnable transport prototypes with local correction.

    For each transform type, maintains a prototype low-rank operator:
        P_{type} = I + U_proto V_proto^T

    Local context-dependent correction:
        ΔG_t = gate * scale * U_delta(ctx) V_delta(ctx)^T

    Full operator:
        G_t = P_{type} + ΔG_t
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 8,
        context_dim: int = 128,
        num_transform_types: int = 1,
        transform_embed_dim: int = 16,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_transform_types = max(num_transform_types, 1)
        self._correction_scale = 1.0 / (hidden_dim ** 0.5)

        # Prototype factors per transform type: P = I + U_proto V_proto^T
        self.U_proto = nn.Parameter(
            torch.randn(self.num_transform_types, hidden_dim, rank) * (2.0 / hidden_dim) ** 0.5
        )
        self.V_proto = nn.Parameter(
            torch.randn(self.num_transform_types, hidden_dim, rank) * (2.0 / hidden_dim) ** 0.5
        )

        # Context encoder for local correction
        self.context_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + transform_embed_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
        )

        # Transform-type embedding for context conditioning
        self.transform_embed = nn.Embedding(self.num_transform_types, transform_embed_dim)
        nn.init.normal_(self.transform_embed.weight, std=0.1)

        # Correction factors
        self.U_delta_net = nn.Linear(context_dim, hidden_dim * rank)
        self.V_delta_net = nn.Linear(context_dim, hidden_dim * rank)

        # Gate for correction (starts at gate_init — prototype-only when 0)
        self.correction_gate = nn.Parameter(torch.full((1,), gate_init))

    def forward(
        self,
        source: torch.Tensor,
        target_aligned: torch.Tensor,
        transform_type_id: int = 0,
        return_operators: bool = False,
    ) -> PrototypeTransportResult:
        """Compute prototype-factored transport.

        Args:
            source: [T, D] source trajectory.
            target_aligned: [T, D] target trajectory (aligned).
            transform_type_id: index into prototype table.
            return_operators: if True, build full G_t matrices.

        Returns:
            PrototypeTransportResult with decomposed factors.
        """
        T, D = source.shape

        src_vel = source[1:] - source[:-1]
        tgt_vel = target_aligned[1:] - target_aligned[:-1]

        # Prototype for this transform type
        tt_idx = min(transform_type_id, self.num_transform_types - 1)
        U_p = self.U_proto[tt_idx]  # [D, rank]
        V_p = self.V_proto[tt_idx]  # [D, rank]

        # Apply prototype: P @ v = v + U_p (V_p^T v)
        Vp_vel = src_vel @ V_p  # [T-1, rank]
        proto_correction = Vp_vel @ U_p.T  # [T-1, D]
        proto_transported = src_vel + self._correction_scale * proto_correction

        # Context for local correction
        time_frac = torch.arange(T - 1, device=source.device, dtype=source.dtype)
        time_frac = (time_frac / max(T - 2, 1)).unsqueeze(-1)

        src_ctx = (source[:-1] + source[1:]) / 2
        tgt_ctx = (target_aligned[:-1] + target_aligned[1:]) / 2

        emb = self.transform_embed(
            torch.tensor([tt_idx], device=source.device)
        ).expand(T - 1, -1)

        ctx_input = torch.cat([src_ctx, tgt_ctx, time_frac, emb], dim=-1)
        ctx = self.context_net(ctx_input)

        # Correction factors
        U_d = self.U_delta_net(ctx).view(T - 1, D, self.rank)
        V_d = self.V_delta_net(ctx).view(T - 1, D, self.rank)

        # Apply correction: ΔG @ v = gate * scale * U_d (V_d^T v)
        gate_scale = self.correction_gate * self._correction_scale
        Vd_vel = torch.bmm(V_d.transpose(1, 2), src_vel.unsqueeze(-1))  # [T-1, rank, 1]
        delta_correction = gate_scale * torch.bmm(U_d, Vd_vel).squeeze(-1)  # [T-1, D]

        transported = proto_transported + delta_correction
        residual = tgt_vel - transported

        # Delta norm for regularization
        delta_norm = (U_d * gate_scale).pow(2).sum() * V_d.pow(2).sum() / max(T - 1, 1)

        # Effective factors
        U_p_expanded = U_p.unsqueeze(0).expand(T - 1, -1, -1) * self._correction_scale
        V_p_expanded = V_p.unsqueeze(0).expand(T - 1, -1, -1)
        U_eff = U_p_expanded + U_d * gate_scale
        V_eff = V_p_expanded + V_d  # approximate — true effective is more complex

        # Full operators if requested
        G_t = torch.zeros(0)
        if return_operators:
            eye = torch.eye(D, device=source.device, dtype=source.dtype)
            eye = eye.unsqueeze(0).expand(T - 1, -1, -1)
            proto_op = self._correction_scale * torch.einsum('dr,er->de', U_p, V_p)
            proto_op = proto_op.unsqueeze(0).expand(T - 1, -1, -1)
            delta_op = gate_scale * torch.bmm(U_d, V_d.transpose(1, 2))
            G_t = eye + proto_op + delta_op

        return PrototypeTransportResult(
            transported_velocity=transported,
            target_velocity=tgt_vel,
            source_velocity=src_vel,
            residual=residual,
            U=U_eff,
            V=V_eff,
            U_proto=U_p,
            V_proto=V_p,
            U_delta=U_d * gate_scale,
            V_delta=V_d,
            delta_norm=delta_norm,
            G_t=G_t,
        )


def prototype_regularization_loss(result: PrototypeTransportResult) -> torch.Tensor:
    """Penalize correction magnitude to encourage prototype reuse.

    L_proto_reg = ||ΔG_t||^2
    """
    return result.delta_norm
