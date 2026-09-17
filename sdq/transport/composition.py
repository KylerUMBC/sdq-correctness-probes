"""Transport composition and inversion.

For the SDQ groupoid structure:
    - G^{j->i} G^{i->j} ≈ I    (inversion)
    - G^{j->k} G^{i->j} ≈ G^{i->k}  (composition / transitivity)
"""

from __future__ import annotations

import torch


def compose_transports(
    G_ij: torch.Tensor,
    G_jk: torch.Tensor,
) -> torch.Tensor:
    """Compose two transport operator sequences.

    G^{i->k} ≈ G^{j->k} @ G^{i->j}

    Args:
        G_ij: [T, D, D] transport from i to j.
        G_jk: [T, D, D] transport from j to k.

    Returns:
        [T, D, D] composed transport from i to k.
    """
    return torch.bmm(G_jk, G_ij)


def invert_transport(G: torch.Tensor) -> torch.Tensor:
    """Invert transport operators.

    For a near-identity operator G = I + eps, the inverse is
    approximately I - eps. For exact inversion, use matrix inverse.

    Args:
        G: [T, D, D] or [D, D] transport operators.

    Returns:
        Same shape, inverted operators.
    """
    if G.dim() == 2:
        return torch.linalg.inv(G)
    # Batched inverse
    return torch.linalg.inv(G)


def apply_transport_to_velocity(
    G: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    """Apply transport operators to velocity vectors.

    Args:
        G: [T, D, D] transport operators.
        velocity: [T, D] velocity vectors.

    Returns:
        [T, D] transported velocity.
    """
    return torch.bmm(G, velocity.unsqueeze(-1)).squeeze(-1)
