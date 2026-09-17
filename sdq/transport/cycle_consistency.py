"""Cycle and triple composition consistency checks for transports.

These are structural tests that ensure learned transports represent
real geometry rather than arbitrary pair fitting.

Two key checks:
    1. Inversion:  G^{j->i} G^{i->j} ≈ I
    2. Transitivity: G^{j->k} G^{i->j} ≈ G^{i->k}
"""

from __future__ import annotations

import torch

from sdq.transport.composition import compose_transports, invert_transport


def inversion_error(
    G_ij: torch.Tensor,
    G_ji: torch.Tensor,
) -> torch.Tensor:
    """Measure how well G^{j->i} inverts G^{i->j}.

    Computes ||G^{j->i} G^{i->j} - I||_F averaged over time.

    Args:
        G_ij: [T, D, D] transport i -> j.
        G_ji: [T, D, D] transport j -> i.

    Returns:
        Scalar error.
    """
    T, D, _ = G_ij.shape
    composed = compose_transports(G_ij, G_ji)  # [T, D, D]
    I = torch.eye(D, device=G_ij.device, dtype=G_ij.dtype).unsqueeze(0).expand(T, -1, -1)
    return (composed - I).norm(dim=(1, 2)).mean()


def composition_error(
    G_ij: torch.Tensor,
    G_jk: torch.Tensor,
    G_ik: torch.Tensor,
) -> torch.Tensor:
    """Measure transitivity: ||G^{j->k} G^{i->j} - G^{i->k}||_F.

    Args:
        G_ij: [T, D, D] transport i -> j.
        G_jk: [T, D, D] transport j -> k.
        G_ik: [T, D, D] transport i -> k (direct).

    Returns:
        Scalar error.
    """
    composed = compose_transports(G_ij, G_jk)  # [T, D, D]
    return (composed - G_ik).norm(dim=(1, 2)).mean()


def cycle_consistency_metrics(
    G_ij: torch.Tensor,
    G_ji: torch.Tensor,
    G_jk: torch.Tensor | None = None,
    G_ik: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute all available cycle consistency metrics.

    Args:
        G_ij, G_ji: transport operators for the i<->j pair.
        G_jk, G_ik: optional, for triple composition test.

    Returns:
        Dict with 'inversion_error' and optionally 'composition_error'.
    """
    metrics: dict[str, float] = {}
    metrics["inversion_error"] = inversion_error(G_ij, G_ji).item()

    if G_jk is not None and G_ik is not None:
        metrics["composition_error"] = composition_error(G_ij, G_jk, G_ik).item()

    return metrics
