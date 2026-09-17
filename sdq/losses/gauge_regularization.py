"""Gauge simplicity loss: rank penalty, smoothness, near-identity bias.

    L_gauge = lambda_1 * rank_penalty(G_t)
            + lambda_2 * ||G_{t+1} - G_t||_F^2
            + lambda_3 * ||G_t - I||_F^2

Keeps transports local, smooth, and simple. Reflects the empirical
finding that surface-form transforms are low-rank and structured.
"""

from __future__ import annotations

import torch


def rank_penalty(G_t: torch.Tensor) -> torch.Tensor:
    """Nuclear norm of (G_t - I) as a differentiable rank proxy.

    Args:
        G_t: [T, D, D] time-varying transport operators.

    Returns:
        Scalar: mean nuclear norm across time steps.
    """
    T, D, _ = G_t.shape
    I = torch.eye(D, device=G_t.device, dtype=G_t.dtype).unsqueeze(0)
    correction = G_t - I  # [T, D, D]
    # Nuclear norm = sum of singular values
    norms = torch.stack([torch.linalg.svdvals(correction[t]).sum() for t in range(T)])
    return norms.mean()


def smoothness_penalty(G_t: torch.Tensor) -> torch.Tensor:
    """Temporal smoothness: mean ||G_{t+1} - G_t||_F^2.

    Args:
        G_t: [T, D, D] time-varying transport operators.

    Returns:
        Scalar smoothness cost.
    """
    if G_t.shape[0] < 2:
        return torch.tensor(0.0, device=G_t.device)
    diffs = G_t[1:] - G_t[:-1]  # [T-1, D, D]
    return diffs.pow(2).sum(dim=(1, 2)).mean()


def near_identity_penalty(G_t: torch.Tensor) -> torch.Tensor:
    """Near-identity bias: mean ||G_t - I||_F^2.

    Args:
        G_t: [T, D, D] time-varying transport operators.

    Returns:
        Scalar deviation from identity.
    """
    T, D, _ = G_t.shape
    I = torch.eye(D, device=G_t.device, dtype=G_t.dtype).unsqueeze(0)
    deviation = G_t - I  # [T, D, D]
    return deviation.pow(2).sum(dim=(1, 2)).mean()


def gauge_regularization_loss(
    G_t: torch.Tensor,
    lambda_rank: float = 0.01,
    lambda_smooth: float = 0.1,
    lambda_identity: float = 0.01,
) -> torch.Tensor:
    """Combined gauge simplicity loss.

    Args:
        G_t: [T, D, D] time-varying transport operators.
        lambda_rank: weight for nuclear norm rank penalty.
        lambda_smooth: weight for temporal smoothness.
        lambda_identity: weight for near-identity bias.

    Returns:
        Scalar combined loss.
    """
    loss = torch.tensor(0.0, device=G_t.device)
    loss = loss + lambda_rank * rank_penalty(G_t)
    loss = loss + lambda_smooth * smoothness_penalty(G_t)
    loss = loss + lambda_identity * near_identity_penalty(G_t)
    return loss


def lowrank_gauge_regularization_loss(
    U: torch.Tensor,
    V: torch.Tensor,
    lambda_rank: float = 0.01,
    lambda_smooth: float = 0.1,
    lambda_identity: float = 0.01,
) -> torch.Tensor:
    """Gauge regularization directly from low-rank factors, avoiding D×D.

    G_t = I + U_t V_t^T, so (G_t - I) = U_t V_t^T.

    - Nuclear norm proxy: ||U_t V_t^T||_* ≤ ||U_t||_F ||V_t||_F (upper bound)
    - Near-identity: ||U_t V_t^T||_F^2 = tr(V_t^T U_t U_t^T V_t)
    - Smoothness: ||delta(U V^T)||_F^2 computed via factor differences

    Args:
        U: [T, D, r] low-rank factor.
        V: [T, D, r] low-rank factor.
        lambda_rank: weight for rank proxy penalty.
        lambda_smooth: weight for temporal smoothness.
        lambda_identity: weight for near-identity bias.

    Returns:
        Scalar combined loss.
    """
    T, D, r = U.shape
    loss = torch.tensor(0.0, device=U.device, dtype=U.dtype)

    # Near-identity: ||UV^T||_F^2 = tr((V^T U)(U^T V)) = ||V^T U||_F^2
    VtU = torch.bmm(V.transpose(1, 2), U)  # [T, r, r]
    frob_sq = VtU.pow(2).sum(dim=(1, 2))  # [T]
    loss = loss + lambda_identity * frob_sq.mean()

    # Rank proxy: ||U||_F * ||V||_F (upper bound on nuclear norm)
    U_frob = U.pow(2).sum(dim=(1, 2)).clamp(min=1e-8).sqrt()  # [T]
    V_frob = V.pow(2).sum(dim=(1, 2)).clamp(min=1e-8).sqrt()  # [T]
    loss = loss + lambda_rank * (U_frob * V_frob).mean()

    # Smoothness: ||U_{t+1} V_{t+1}^T - U_t V_t^T||_F^2
    if T >= 2:
        # = ||U1 V1^T - U0 V0^T||_F^2 where subscripts are consecutive
        # Use: ||A B^T - C D^T||_F^2 via [A,-C] [B,D]^T formulation
        A = torch.cat([U[1:], -U[:-1]], dim=-1)  # [T-1, D, 2r]
        B = torch.cat([V[1:], V[:-1]], dim=-1)   # [T-1, D, 2r]
        AtA = torch.bmm(A.transpose(1, 2), A)    # [T-1, 2r, 2r]
        BtB = torch.bmm(B.transpose(1, 2), B)    # [T-1, 2r, 2r]
        smooth_sq = (AtA * BtB).sum(dim=(1, 2))   # [T-1]
        loss = loss + lambda_smooth * smooth_sq.mean()

    return loss
