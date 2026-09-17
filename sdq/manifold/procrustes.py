"""Analytic orthogonal Procrustes transport for latent trajectories.

Finds the rotation R* = argmin_{R^T R = I} ||R S^T - T^T||_F that best
aligns a source trajectory S to a target trajectory T in latent space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ProcrustesResult:
    """Result of a single Procrustes alignment."""

    transported: Tensor       # [T-1, k]  — S @ R.T
    target: Tensor            # [T-1, k]
    R: Tensor                 # [k, k]    — orthogonal rotation
    residual: Tensor          # [T-1, k]  — target - transported
    residual_norm: float      # mean ||residual_i||^2
    alignment_quality: float  # 1 - residual_norm / ||T||^2  ∈ [0, 1]
    is_proper_rotation: bool  # True if det(R) > 0


class ProcrustesTransport:
    """Orthogonal Procrustes alignment between latent trajectory segments.

    Usage::

        transport = ProcrustesTransport()
        result = transport.fit_and_transport(S, T)
        print(result.alignment_quality)
    """

    def fit(self, S: Tensor, T: Tensor) -> Tensor:
        """Compute the optimal rotation matrix R aligning S to T.

        Solves min_{R^T R = I} ||S @ R.T - T||_F via the SVD of T.T @ S.

        Args:
            S: [n, k] source points.
            T: [n, k] target points.

        Returns:
            R: [k, k] orthogonal rotation matrix.
        """
        S = S.float()
        T = T.float()
        M = T.T @ S  # [k, k]
        U, _, Vt = torch.linalg.svd(M, full_matrices=True)
        # Ensure proper rotation (det = +1) by flipping last column of U if needed
        d = torch.det(U @ Vt).sign()
        D = torch.diag(torch.cat([torch.ones(U.shape[1] - 1, device=U.device), d.unsqueeze(0)]))
        R = U @ D @ Vt
        return R

    def transport(self, S: Tensor, R: Tensor) -> Tensor:
        """Apply rotation R to source points.

        Args:
            S: [n, k] source points.
            R: [k, k] rotation matrix.

        Returns:
            S_transported: [n, k]  (S @ R.T)
        """
        return S.float() @ R.T

    def residual(self, S: Tensor, T: Tensor, R: Tensor) -> Tensor:
        """Compute residual T - S @ R.T.

        Args:
            S: [n, k]
            T: [n, k]
            R: [k, k]

        Returns:
            residual: [n, k]
        """
        return T.float() - self.transport(S, R)

    def fit_and_transport(self, S: Tensor, T: Tensor) -> ProcrustesResult:
        """Fit rotation and transport in one call.

        Args:
            S: [n, k] source trajectory segment.
            T: [n, k] target trajectory segment.

        Returns:
            ProcrustesResult with all alignment statistics.
        """
        S = S.float()
        T = T.float()
        R = self.fit(S, T)
        transported = self.transport(S, R)
        res = T - transported
        res_norm = float((res ** 2).sum(dim=-1).mean())
        t_norm = float((T ** 2).sum(dim=-1).mean().clamp(min=1e-12))
        alignment_quality = float(max(0.0, 1.0 - res_norm / t_norm))
        is_proper = float(torch.det(R).item()) > 0
        return ProcrustesResult(
            transported=transported,
            target=T,
            R=R,
            residual=res,
            residual_norm=res_norm,
            alignment_quality=alignment_quality,
            is_proper_rotation=is_proper,
        )

    def batch_transport(
        self,
        pairs: list[tuple[Tensor, Tensor]],
    ) -> list[ProcrustesResult]:
        """Run fit_and_transport for a list of (S, T) pairs.

        Args:
            pairs: List of (S, T) tuples, each [n, k].

        Returns:
            List of ProcrustesResult, one per pair.
        """
        return [self.fit_and_transport(S, T) for S, T in pairs]
