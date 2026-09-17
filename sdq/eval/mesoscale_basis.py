"""Mesoscale supervised basis for SDQ Phase 5.

Scientific question:
    Phase 4 (rank-1) and Phase 4b (full 96-dim subspace) both failed to flip
    answers above random baseline. Does a *mesoscale* description — 5-10
    supervised+Varimax components — have differential causal traction?

    Three rival outcomes are valuable:
      1. Some single component flips at >=2x within-span baseline -> circuits
         hypothesis with mesoscale readout (next: SAE bridge).
      2. No single component, but coordinated K-of-K does -> redundant-pathway
         hypothesis (next: scale K).
      3. Nothing exceeds within-span at any granularity -> wave/superposition
         hypothesis (next: abandon linear interventions).

Method:
    - Train LowRankCommitmentProbe(rank=K) on standardized h_0 (z-space).
    - Map each column of down.weight.T from z-space to raw h-space via the
      gradient rule v_raw = v_z / sd (same rule as direction_to_raw_space).
    - QR-orthonormalize in raw space, then Varimax-rotate to recover simple
      structure. The basis lives in the same space as intervention.
    - Per-component signed scores come from projecting the full-rank linear
      probe weight (mapped to raw space) onto each basis_raw column.

The within-span random perturbation is the load-bearing baseline: it lives
in span(basis_raw) so any "probe-recovers-its-own-signal" effect cancels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from sdq.eval.commitment_probe import (
    LinearCommitmentProbe,
    LowRankCommitmentProbe,
    fit_probe,
)
from sdq.eval.commitment_intervention import direction_to_raw_space


# ── Varimax rotation ─────────────────────────────────────────────────────────

def varimax_rotate(
    loadings: Tensor,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Kaiser Varimax rotation via SVD updates.

    Maximizes sum over columns k of var_d(L[d,k]^2), i.e. finds an orthogonal
    rotation that produces simple structure (each column dominated by a few
    large loadings).

    Args:
        loadings: [D, K] real matrix.
        max_iter: cap on iterations.
        tol: relative convergence tolerance on the criterion.

    Returns:
        (rotated_loadings [D,K], R [K,K] orthogonal).
    """
    D, K = loadings.shape
    if K < 2:
        return loadings.clone(), torch.eye(K, dtype=loadings.dtype, device=loadings.device)

    R = torch.eye(K, dtype=loadings.dtype, device=loadings.device)
    d_prev = 0.0
    for _ in range(max_iter):
        Lambda = loadings @ R                                # [D, K]
        col_ss = (Lambda ** 2).sum(dim=0) / D                # [K]
        B = loadings.T @ (Lambda ** 3 - Lambda * col_ss)     # [K, K]
        U, S, Vh = torch.linalg.svd(B)
        R = U @ Vh
        d = float(S.sum().item())
        if abs(d - d_prev) < tol * max(d, 1.0):
            break
        d_prev = d
    return loadings @ R, R


def varimax_criterion(loadings: Tensor) -> float:
    """The criterion Varimax maximizes — sum of column-wise variance of
    squared loadings. Used by tests to verify rotation is monotone.
    """
    D = loadings.shape[0]
    sq = loadings ** 2
    col_means = sq.mean(dim=0, keepdim=True)
    return float(((sq - col_means) ** 2).mean(dim=0).sum().item())


# ── Supervised mesoscale basis ───────────────────────────────────────────────

@dataclass
class MesoscaleBasis:
    """A K-component supervised+Varimax basis fit on h_0.

    Attributes:
        basis_raw: [D, K] orthonormal in raw h-space.
        signed_scores: [K] per-component projection of the full-rank probe
            weight (in raw space). Sign sets orientation; magnitude weights
            coordinated perturbation.
        component_stds: [K] std of training h_0 projected onto each
            basis_raw column. Used to set per-component magnitudes.
        scaler_mu: [D] standardization mean (saved with probe).
        scaler_sd: [D] standardization std (saved with probe).
        K: number of components.
        low_rank_probe: the trained LowRankCommitmentProbe (may be on cpu).
        full_rank_probe: the trained LinearCommitmentProbe used for signs.
    """
    basis_raw: Tensor
    signed_scores: Tensor
    component_stds: Tensor
    scaler_mu: Tensor
    scaler_sd: Tensor
    K: int
    low_rank_probe: Optional[LowRankCommitmentProbe] = None
    full_rank_probe: Optional[LinearCommitmentProbe] = None


def _train_probe_inplace(
    probe: nn.Module,
    X_train_z: Tensor,
    y_train: Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> nn.Module:
    """Train a probe on already-standardized features. Returns the probe in eval mode."""
    return fit_probe(probe, X_train_z, y_train, epochs=epochs, lr=lr,
                     weight_decay=weight_decay, device=device)


def fit_supervised_mesoscale_basis(
    h0_raw: Tensor,
    y: Tensor,
    K: int,
    scaler_mu: Tensor,
    scaler_sd: Tensor,
    epochs: int = 400,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    device: str = "cpu",
) -> MesoscaleBasis:
    """Fit a K-component supervised mesoscale basis on raw h_0.

    Pipeline:
      1. Standardize h_0 to z-space using the *given* mu/sd (must match the
         scalers used to fit commitment_direction.pt so that raw-space
         transforms agree with Phase 4).
      2. Train LowRankCommitmentProbe(K) in z-space; take down.weight.T as
         loadings_z [D, K].
      3. Map to raw space row-wise: loadings_raw[d, k] = loadings_z[d, k] / sd[d].
      4. QR-orthonormalize loadings_raw -> Q_raw [D, K] orthonormal in raw space.
      5. Varimax-rotate Q_raw -> basis_raw [D, K] (still orthonormal in raw).
      6. Train LinearCommitmentProbe in z-space; map its weight to raw via
         direction_to_raw_space; project onto basis_raw to get signed scores.
      7. Compute per-component std of training h_0 projected onto basis_raw.

    Args:
        h0_raw: [N, D] training hidden states (raw, unstandardized).
        y: [N] binary correctness labels (1 = incorrect).
        K: number of mesoscale components.
        scaler_mu: [D] or [1, D] standardization mean (must match commitment_direction.pt).
        scaler_sd: [D] or [1, D] standardization std (must match commitment_direction.pt).
        epochs/lr/weight_decay/device: training hyperparams.

    Returns:
        MesoscaleBasis.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    N, D = h0_raw.shape
    if K > D:
        raise ValueError(f"K={K} exceeds hidden_dim={D}")

    mu = scaler_mu.squeeze().to(h0_raw.dtype)
    sd = scaler_sd.squeeze().to(h0_raw.dtype).clamp_min(1e-8)
    if mu.shape != (D,) or sd.shape != (D,):
        raise ValueError(f"scaler shapes must be [D]={D}, got mu={mu.shape}, sd={sd.shape}")

    # Standardize
    z = (h0_raw.float() - mu.float()) / sd.float()

    # Train K-rank probe
    low_probe = LowRankCommitmentProbe(input_dim=D, rank=K)
    low_probe = _train_probe_inplace(low_probe, z, y, epochs, lr, weight_decay, device)
    loadings_z = low_probe.down.weight.detach().cpu().T.float()  # [D, K]

    # Map to raw space row-wise (gradient rule)
    loadings_raw = loadings_z / sd.float().unsqueeze(1)           # [D, K]

    # QR-orthonormalize, then Varimax-rotate in raw space
    Q_raw, _ = torch.linalg.qr(loadings_raw)                      # [D, K]
    basis_raw, _ = varimax_rotate(Q_raw)                          # [D, K]

    # Train full-rank probe for signing
    full_probe = LinearCommitmentProbe(input_dim=D)
    full_probe = _train_probe_inplace(full_probe, z, y, epochs, lr, weight_decay, device)
    w_full_z = full_probe.weight_vector.cpu().float()              # [D]
    w_full_raw = direction_to_raw_space(w_full_z, sd.float())      # [D] unit
    signed_scores = basis_raw.T @ w_full_raw                       # [K]

    # Per-component std of raw h_0 projected onto basis_raw
    proj = h0_raw.float() @ basis_raw                              # [N, K]
    component_stds = proj.std(dim=0)                               # [K]

    return MesoscaleBasis(
        basis_raw=basis_raw,
        signed_scores=signed_scores,
        component_stds=component_stds,
        scaler_mu=mu,
        scaler_sd=sd,
        K=K,
        low_rank_probe=low_probe.cpu(),
        full_rank_probe=full_probe.cpu(),
    )


# ── Perturbation factories ───────────────────────────────────────────────────

def make_component_perturbation(
    basis_raw: Tensor,
    idx: int,
    magnitude: float,
    sign: float = 1.0,
) -> Tensor:
    """Single-component perturbation: sign * magnitude * basis_raw[:, idx].

    `basis_raw` columns are unit-norm (orthonormal), so the L2 magnitude of
    the result is exactly `abs(magnitude)`.
    """
    return float(sign) * float(magnitude) * basis_raw[:, idx]


def make_within_span_random_perturbation(
    basis_raw: Tensor,
    magnitude: float,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Random unit vector within span(basis_raw), scaled to `magnitude`.

    Because basis_raw is orthonormal in raw space, sampling
    `coeffs ~ N(0, I_K)` and forming `v = basis_raw @ coeffs` gives an
    isotropic Gaussian in the K-dim span; v / ||v|| is then uniform on the
    unit sphere of that span.
    """
    return sample_within_span_unit(basis_raw, generator) * float(magnitude)


def make_coordinated_perturbation(
    basis_raw: Tensor,
    signed_scores: Tensor,
    magnitude: float,
) -> Tensor:
    """Sign- and magnitude-weighted sum of components, normalized.

    v = sum_k signed_scores[k] * basis_raw[:, k], then unit-normalize and
    scale by magnitude. This is the cleanest "push every lever the way it
    most increases predicted-incorrect score" perturbation.
    """
    v = basis_raw @ signed_scores.to(basis_raw.dtype)
    return v / v.norm().clamp_min(1e-8) * float(magnitude)


def make_pairwise_perturbation(
    basis_raw: Tensor,
    i: int,
    j: int,
    signed_scores: Tensor,
    magnitude: float,
) -> Tensor:
    """Sign-oriented sum of two components, normalized.

    For interference testing we use signs only (not magnitudes), so the
    result is `(sign(s_i) u_i + sign(s_j) u_j) / sqrt(2) * magnitude` —
    keeps the additivity comparison `KL(i+j) vs KL(i)+KL(j)` clean.
    """
    s_i = float(signed_scores[i])
    s_j = float(signed_scores[j])
    sign_i = 1.0 if s_i >= 0 else -1.0
    sign_j = 1.0 if s_j >= 0 else -1.0
    v = sign_i * basis_raw[:, i] + sign_j * basis_raw[:, j]
    return v / v.norm().clamp_min(1e-8) * float(magnitude)


# ── Sampling unit directions (for per-sample sigma calibration) ──────────────

def sample_within_span_unit(
    basis_raw: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample a uniform unit direction within span(basis_raw)."""
    K = basis_raw.shape[1]
    coeffs = torch.randn(K, generator=generator, dtype=basis_raw.dtype)
    v = basis_raw @ coeffs
    return v / v.norm().clamp_min(1e-8)


def sample_full_random_unit(
    dim: int,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample a uniform unit direction in R^dim."""
    v = torch.randn(dim, generator=generator)
    return v / v.norm().clamp_min(1e-8)


# ── Calibration ──────────────────────────────────────────────────────────────

def calibrate_direction_std(
    h0_raw: Tensor,
    direction: Tensor,
) -> float:
    """Std of training h_0 projected onto a single unit direction."""
    d = direction / direction.norm().clamp_min(1e-8)
    return float((h0_raw.float() @ d.float()).std().item())
