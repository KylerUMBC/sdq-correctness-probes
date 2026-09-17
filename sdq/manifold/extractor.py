"""ManifoldExtractor: analytic PCA-based encoder for hidden-state manifolds.

Replaces the learned MultiScaleConvEncoder with a parameter-free PCA projection.
mu_ and V_k_ are registered as buffers so the module integrates with
model.to(device) and torch.save / torch.load without change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ManifoldExtractor(nn.Module):
    """PCA-based encoder that duck-types MultiScaleConvEncoder.

    No learnable parameters — mu_ and V_k_ are buffers.  The optimizer will
    find no parameters to update, so the training loop runs unchanged.

    Args:
        k: Number of PCA components.  If None, chosen from variance_threshold.
        variance_threshold: Fraction of variance to retain when k is None.
        center: Whether to subtract the mean before projection.
    """

    def __init__(
        self,
        k: int | None = None,
        variance_threshold: float = 0.95,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.variance_threshold = variance_threshold
        self.center = center

        # Buffers are set during fit(); register as None placeholders
        self.register_buffer("mu_", None)
        self.register_buffer("V_k_", None)
        self.register_buffer("singular_values_", None)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        h_collection: list[Tensor] | Tensor,
    ) -> "ManifoldExtractor":
        """Fit PCA from a collection of hidden-state tensors.

        Args:
            h_collection: Either a list of [T_i, D_h] tensors or a single
                          [N, D_h] tensor.

        Returns:
            self (for chaining)
        """
        if isinstance(h_collection, list):
            H = torch.cat([h.reshape(-1, h.shape[-1]) for h in h_collection], dim=0)
        else:
            H = h_collection.reshape(-1, h_collection.shape[-1])

        H = H.float()
        N, D_h = H.shape

        # Compute mean
        mu = H.mean(0)
        H_c = H - mu if self.center else H

        # Choose SVD backend
        use_lowrank = D_h > 1000 or N > 20000
        q = self.k if self.k is not None else min(D_h, N, 256)

        if use_lowrank:
            # torch.pca_lowrank returns (U, S, V) where V is [D_h, q]
            _, S, V = torch.pca_lowrank(H_c, q=q, center=False)
        else:
            # Full SVD: V is [D_h, D_h]; columns are right singular vectors
            _, S, Vh = torch.linalg.svd(H_c, full_matrices=False)
            V = Vh.T  # [D_h, min(N, D_h)]
            S = S[:q]
            V = V[:, :q]

        # Select k from variance threshold if not given explicitly
        if self.k is None:
            var = S ** 2
            cumvar = var.cumsum(0) / var.sum().clamp(min=1e-12)
            k = int((cumvar < self.variance_threshold).sum().item()) + 1
            k = min(k, S.shape[0])
        else:
            k = min(self.k, S.shape[0])

        self.mu_ = mu                    # [D_h]
        self.V_k_ = V[:, :k].contiguous()  # [D_h, k]
        self.singular_values_ = S[:k].contiguous()  # [k]
        return self

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, h: Tensor) -> Tensor:
        """Project hidden states to PCA coordinates.

        Args:
            h: [T, D_h]

        Returns:
            z: [T, k]
        """
        h = h.float()
        h_c = h - self.mu_ if self.center else h
        return h_c @ self.V_k_

    def decode(self, z: Tensor) -> Tensor:
        """Reconstruct hidden states from PCA coordinates.

        Args:
            z: [T, k]

        Returns:
            h_hat: [T, D_h]
        """
        h_hat = z @ self.V_k_.T
        if self.center:
            h_hat = h_hat + self.mu_
        return h_hat

    def gauge(self, h: Tensor) -> Tensor:
        """Return the off-manifold residual (component not captured by PCA).

        Args:
            h: [T, D_h]

        Returns:
            r: [T, D_h]  (h - decode(encode(h)))
        """
        return h.float() - self.decode(self.encode(h))

    # ------------------------------------------------------------------
    # forward — duck-types MultiScaleConvEncoder
    # ------------------------------------------------------------------

    def forward(self, h: Tensor) -> Tensor:
        """Encode hidden states; handles both [T, D_h] and [B, T, D_h].

        Args:
            h: [T, D_h] or [B, T, D_h]

        Returns:
            z: [T, k] or [B, T, k]
        """
        if h.dim() == 3:
            B = h.shape[0]
            return torch.stack([self.encode(h[i]) for i in range(B)])
        return self.encode(h)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def explained_variance(self) -> Tensor:
        """Fraction of variance explained by each component. Shape: [k]."""
        if self.singular_values_ is None:
            raise RuntimeError("ManifoldExtractor has not been fit yet.")
        s2 = self.singular_values_ ** 2
        return s2 / s2.sum().clamp(min=1e-12)
