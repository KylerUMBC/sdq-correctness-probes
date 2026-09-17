"""Low-rank parameterizations for local transport operators.

Transport G_t = I + U_t V_t^T where U_t, V_t are [D, rank].
This keeps the transport near-identity and low-complexity,
reflecting the empirical finding that surface transforms are
low-rank and structured.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LowRankTransport(nn.Module):
    """A single low-rank transport operator: G = I + U V^T.

    Can represent either a learned fixed operator or be used as
    a building block for time-varying transport fields.
    """

    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.U = nn.Parameter(torch.zeros(dim, rank))
        self.V = nn.Parameter(torch.zeros(dim, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transport: G @ x = x + U @ (V^T @ x).

        Args:
            x: [..., D] input vectors.

        Returns:
            [..., D] transported vectors.
        """
        # V^T @ x: [..., rank]
        Vt_x = x @ self.V  # [..., rank]
        # U @ (V^T @ x): [..., D]
        correction = Vt_x @ self.U.T  # [..., D]
        return x + correction

    def matrix(self) -> torch.Tensor:
        """Return the full D×D operator matrix G = I + U V^T."""
        I = torch.eye(self.dim, device=self.U.device, dtype=self.U.dtype)
        return I + self.U @ self.V.T

    def nuclear_norm(self) -> torch.Tensor:
        """Nuclear norm of the low-rank correction U V^T (rank proxy)."""
        correction = self.U @ self.V.T
        return torch.linalg.svdvals(correction).sum()

    def frobenius_deviation(self) -> torch.Tensor:
        """||G - I||_F = ||U V^T||_F."""
        correction = self.U @ self.V.T
        return correction.norm()


def operator_rank_spectrum(G: torch.Tensor) -> torch.Tensor:
    """Compute singular values of the deviation G - I.

    Args:
        G: [D, D] transport operator.

    Returns:
        Singular values (descending).
    """
    I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    return torch.linalg.svdvals(G - I)
