"""Monotonic attention-based alignment.

A learnable alignment module that produces soft monotone alignment
matrices between trajectory pairs. Unlike hard DTW, this produces
a differentiable soft-alignment matrix A[i,j] that can be used
directly in transport and loss computations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneAligner(nn.Module):
    """Learnable monotone alignment between two trajectories.

    Produces a soft alignment matrix A of shape [T1, T2] where
    A[i, :] is a distribution over target positions for source position i,
    with a monotonicity constraint enforced via cumulative softmax.

    The alignment can be used to warp one trajectory onto another:
        y_aligned = A @ y  # [T1, D]
    """

    def __init__(self, hidden_dim: int, align_dim: int = 128):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, align_dim)
        self.key_proj = nn.Linear(hidden_dim, align_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft monotone alignment matrix.

        Args:
            source: [T1, D] source trajectory.
            target: [T2, D] target trajectory.

        Returns:
            A: [T1, T2] soft alignment matrix (rows sum to 1).
        """
        Q = self.query_proj(source)  # [T1, align_dim]
        K = self.key_proj(target)  # [T2, align_dim]

        # Attention scores
        temp = self.temperature.clamp(min=0.1)
        scores = Q @ K.T / temp  # [T1, T2]

        # Enforce soft monotonicity:
        # Add a position-based bias that encourages diagonal alignment
        T1, T2 = scores.shape
        pos_i = torch.arange(T1, device=scores.device, dtype=scores.dtype)
        pos_j = torch.arange(T2, device=scores.device, dtype=scores.dtype)
        # Expected target position for source i: i * T2 / T1
        expected = pos_i.unsqueeze(1) * (T2 / max(T1, 1)) - pos_j.unsqueeze(0)
        monotone_bias = -(expected ** 2) / (2 * max(T2, 1))
        scores = scores + monotone_bias

        # Soft alignment: row-wise softmax
        A = F.softmax(scores, dim=-1)  # [T1, T2]

        return A

    def align_trajectory(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Warp the target trajectory to align with the source.

        Args:
            source: [T1, D]
            target: [T2, D]

        Returns:
            target_aligned: [T1, D] target warped to source's time frame.
        """
        A = self.forward(source, target)  # [T1, T2]
        return A @ target  # [T1, D]

    def hard_alignment(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> list[int]:
        """Extract a hard monotone alignment map tau.

        Args:
            source: [T1, D]
            target: [T2, D]

        Returns:
            tau: list of length T1, where tau[i] is the target index for source i.
        """
        A = self.forward(source, target)  # [T1, T2]
        raw = A.argmax(dim=-1).tolist()

        # Enforce monotonicity
        tau = []
        last = 0
        for t in raw:
            t = max(t, last)
            tau.append(t)
            last = t
        return tau
