"""Spectral mode analysis of the commitment subspace.

Decomposes the commitment signal into PCA modes and analyzes their
individual discriminative power and inter-mode interference patterns.

Phase 4b showed that perturbing the entire 96-dim commitment subspace has
the same effect as perturbing a random 96-dim subspace. But the subspace
may contain a mix of highly discriminative and non-discriminative modes.
This module identifies the discriminative modes and enables targeted
perturbations that exploit the specific interference structure.

Key concepts:
    Mode: A PCA eigenvector of the h_0 covariance in z-space.
    Discriminability: Cohen's d for a mode (correct vs incorrect separation).
    Coherence: Within-class consistency of mode projections (|mean|/std).
    Interference: Difference in cross-mode correlations between classes.
    Phase disruption: Perturbation that negates an example's projections
        onto discriminative modes, breaking constructive interference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class ModeDecomposition:
    """PCA mode decomposition with per-mode statistics."""
    eigenvectors_z: Tensor       # [D, k] orthonormal in z-space
    eigenvalues: Tensor          # [k] descending
    projections: Tensor          # [N, k] per-example projections
    discriminability: Tensor     # [k] Cohen's d per mode
    coherence_correct: Tensor    # [k] |mean|/std for correct class
    coherence_incorrect: Tensor  # [k] |mean|/std for incorrect class
    per_family_disc: dict[str, Tensor]  # {family: [k] Cohen's d}
    discriminative_order: list[int]     # mode indices sorted by |d|


@dataclass
class InterferencePattern:
    """Cross-mode correlation differences between classes."""
    corr_correct: Tensor    # [k, k]
    corr_incorrect: Tensor  # [k, k]
    interference: Tensor    # [k, k] |corr_correct - corr_incorrect|
    top_pairs: list[tuple[int, int, float]]
    total_interference: float


@dataclass
class ModeAnalysisResult:
    """Complete spectral mode analysis."""
    decomposition: ModeDecomposition
    interference: InterferencePattern
    n_discriminative: int
    disc_threshold: float
    top_modes: list[int]
    bottom_modes: list[int]


# ── Analysis functions ─────────────────────────────────────────────────────────

def decompose_modes(
    h0_matrix: Tensor,
    labels: Tensor,
    families: list[str],
    scaler_mu: Tensor,
    scaler_sd: Tensor,
    k: int = 96,
) -> ModeDecomposition:
    """PCA decompose h_0 and compute per-mode discriminability.

    Args:
        h0_matrix: [N, D] raw h_0 vectors.
        labels: [N] binary (1=incorrect, 0=correct).
        families: [N] task family names.
        scaler_mu: [D] or [1, D] standardization mean from Phase 3.
        scaler_sd: [D] or [1, D] standardization std from Phase 3.
        k: Number of PCA modes to extract.

    Returns:
        ModeDecomposition with eigenvectors, projections, and per-mode stats.
    """
    mu = scaler_mu.squeeze()
    sd = scaler_sd.squeeze().clamp_min(1e-6)

    z = (h0_matrix.float() - mu) / sd

    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(0)[:k]
    eigvecs = eigvecs.flip(1)[:, :k]

    projections = z @ eigvecs  # [N, k]

    correct_mask = (labels == 0)
    incorrect_mask = (labels == 1)
    n_correct = correct_mask.sum().item()
    n_incorrect = incorrect_mask.sum().item()

    disc = torch.zeros(k)
    coh_correct = torch.zeros(k)
    coh_incorrect = torch.zeros(k)

    for m in range(k):
        p = projections[:, m]
        pc = p[correct_mask]
        pi = p[incorrect_mask]

        mean_c = pc.mean() if n_correct > 0 else torch.tensor(0.0)
        mean_i = pi.mean() if n_incorrect > 0 else torch.tensor(0.0)
        std_c = pc.std() if n_correct > 1 else torch.tensor(1.0)
        std_i = pi.std() if n_incorrect > 1 else torch.tensor(1.0)

        pooled_std = ((std_c**2 + std_i**2) / 2).sqrt().clamp_min(1e-8)
        disc[m] = (mean_c - mean_i) / pooled_std

        if n_correct > 0:
            coh_correct[m] = (mean_c.abs() / std_c.clamp_min(1e-8)).item()
        if n_incorrect > 0:
            coh_incorrect[m] = (mean_i.abs() / std_i.clamp_min(1e-8)).item()

    per_fam: dict[str, Tensor] = {}
    unique_fams = sorted(set(families))
    for fam in unique_fams:
        fam_mask = torch.tensor([f == fam for f in families])
        fam_correct = fam_mask & correct_mask
        fam_incorrect = fam_mask & incorrect_mask
        nc = fam_correct.sum().item()
        ni = fam_incorrect.sum().item()
        if nc < 2 or ni < 2:
            per_fam[fam] = torch.full((k,), float("nan"))
            continue
        fam_disc = torch.zeros(k)
        for m in range(k):
            p = projections[:, m]
            pc_ = p[fam_correct]
            pi_ = p[fam_incorrect]
            pooled = ((pc_.std() ** 2 + pi_.std() ** 2) / 2).sqrt().clamp_min(1e-8)
            fam_disc[m] = (pc_.mean() - pi_.mean()) / pooled
        per_fam[fam] = fam_disc

    order = disc.abs().argsort(descending=True).tolist()

    return ModeDecomposition(
        eigenvectors_z=eigvecs,
        eigenvalues=eigvals,
        projections=projections,
        discriminability=disc,
        coherence_correct=coh_correct,
        coherence_incorrect=coh_incorrect,
        per_family_disc=per_fam,
        discriminative_order=order,
    )


def analyze_interference(
    projections: Tensor,
    labels: Tensor,
    top_n_pairs: int = 20,
) -> InterferencePattern:
    """Compute inter-mode correlation differences between classes.

    For each pair of modes (i, j), computes Pearson correlation separately
    for correct and incorrect examples. High |corr_correct - corr_incorrect|
    indicates that the modes have a different phase relationship in correct
    vs incorrect examples — an interference signature.

    Args:
        projections: [N, k] per-example mode projections.
        labels: [N] binary (1=incorrect, 0=correct).
        top_n_pairs: Number of top interfering pairs to return.

    Returns:
        InterferencePattern with correlation matrices and top pairs.
    """
    k = projections.shape[1]
    correct_mask = (labels == 0)
    incorrect_mask = (labels == 1)

    def _corr_matrix(P: Tensor) -> Tensor:
        if P.shape[0] < 3:
            return torch.zeros(k, k)
        centered = P - P.mean(dim=0, keepdim=True)
        norms = centered.norm(dim=0, keepdim=True).clamp_min(1e-8)
        normed = centered / norms
        return (normed.T @ normed) / max(P.shape[0] - 1, 1)

    corr_c = _corr_matrix(projections[correct_mask])
    corr_i = _corr_matrix(projections[incorrect_mask])
    interf = (corr_c - corr_i).abs()
    interf.fill_diagonal_(0)

    upper_tri = torch.triu_indices(k, k, offset=1)
    pair_values = interf[upper_tri[0], upper_tri[1]]
    n_top = min(top_n_pairs, pair_values.shape[0])
    top_idx = pair_values.argsort(descending=True)[:n_top]

    top_pairs = []
    for idx in top_idx:
        i = upper_tri[0][idx].item()
        j = upper_tri[1][idx].item()
        val = pair_values[idx].item()
        top_pairs.append((i, j, val))

    total = interf.sum().item() / 2

    return InterferencePattern(
        corr_correct=corr_c,
        corr_incorrect=corr_i,
        interference=interf,
        top_pairs=top_pairs,
        total_interference=total,
    )


def full_mode_analysis(
    h0_matrix: Tensor,
    labels: Tensor,
    families: list[str],
    scaler_mu: Tensor,
    scaler_sd: Tensor,
    k: int = 96,
    disc_threshold: float = 0.3,
) -> ModeAnalysisResult:
    """Run complete spectral mode analysis.

    Args:
        h0_matrix: [N, D] raw h_0 vectors.
        labels: [N] binary labels.
        families: [N] task family names.
        scaler_mu, scaler_sd: Phase 3 standardization parameters.
        k: Number of PCA modes.
        disc_threshold: |Cohen's d| threshold for "discriminative" mode.

    Returns:
        ModeAnalysisResult with decomposition, interference, and mode groups.
    """
    decomp = decompose_modes(h0_matrix, labels, families, scaler_mu, scaler_sd, k)
    interf = analyze_interference(decomp.projections, labels)

    abs_d = decomp.discriminability.abs()
    top_modes = [i for i in decomp.discriminative_order if abs_d[i] > disc_threshold]
    bottom_modes = [i for i in range(k) if abs_d[i] <= disc_threshold]

    return ModeAnalysisResult(
        decomposition=decomp,
        interference=interf,
        n_discriminative=len(top_modes),
        disc_threshold=disc_threshold,
        top_modes=top_modes,
        bottom_modes=bottom_modes,
    )


# ── Perturbation builders ─────────────────────────────────────────────────────

def build_sign_flip_perturbation(
    h0: Tensor,
    eigenvectors_z: Tensor,
    mode_indices: list[int],
    scaler_mu: Tensor,
    scaler_sd: Tensor,
    target_magnitude: float | None = None,
) -> Tensor:
    """Negate an example's projections onto selected PCA modes.

    In z-space: Δz = -2 × V_sel @ (V_sel^T @ z)
    In h-space: Δh = Δz × sd

    This is a per-example perturbation that mirrors the example across the
    hyperplane orthogonal to the selected modes. It maximally disrupts the
    constructive interference pattern for this specific example.

    Verification: For selected mode m with projection p_m, the new
    projection is -p_m. For unselected modes, projections are unchanged.

    Args:
        h0: [D] single example in raw h-space.
        eigenvectors_z: [D, k] PCA eigenvectors in z-space.
        mode_indices: Which modes to flip.
        scaler_mu: [D] or [1, D] standardization mean.
        scaler_sd: [D] or [1, D] standardization std.
        target_magnitude: If provided, scale perturbation to this norm.
            If None, use the natural magnitude (2 × ||projection||_h).

    Returns:
        [D] perturbation vector in raw h-space.
    """
    mu = scaler_mu.squeeze()
    sd = scaler_sd.squeeze().clamp_min(1e-6)

    z = (h0.float() - mu) / sd
    V_sel = eigenvectors_z[:, mode_indices]
    proj = V_sel.T @ z
    delta_z = -2.0 * (V_sel @ proj)
    delta_h = delta_z * sd

    if target_magnitude is not None:
        norm = delta_h.norm().clamp_min(1e-8)
        delta_h = delta_h * (target_magnitude / norm)

    return delta_h


def build_selective_basis_raw(
    eigenvectors_z: Tensor,
    mode_indices: list[int],
    scaler_sd: Tensor,
) -> Tensor:
    """Convert selected z-space modes to an orthonormal basis in raw h-space.

    Applies v_raw = v_z × sd (covariant direction transform), then
    QR re-orthonormalization — same logic as compute_pca_basis_raw.

    Args:
        eigenvectors_z: [D, k] PCA eigenvectors in z-space.
        mode_indices: Which modes to select.
        scaler_sd: [D] or [1, D] per-feature std.

    Returns:
        [D, M] orthonormal basis in raw h-space, where M = len(mode_indices).
    """
    sd = scaler_sd.squeeze().clamp_min(1e-6)
    V_z = eigenvectors_z[:, mode_indices]
    V_raw = V_z * sd.unsqueeze(1)
    Q, _ = torch.linalg.qr(V_raw)
    return Q


# ── Formatting ─────────────────────────────────────────────────────────────────

def format_mode_analysis(result: ModeAnalysisResult) -> str:
    """Human-readable summary of spectral mode analysis."""
    d = result.decomposition
    interf = result.interference
    k = d.eigenvalues.shape[0]

    lines = [
        "=" * 70,
        "SPECTRAL MODE ANALYSIS",
        "=" * 70,
        f"Total PCA modes analyzed: {k}",
        f"Discriminability threshold: |Cohen's d| > {result.disc_threshold:.2f}",
        f"Discriminative modes: {result.n_discriminative} / {k}",
        "",
    ]

    # Discriminability distribution summary
    abs_d = d.discriminability.abs()
    lines.append(f"Discriminability distribution:")
    lines.append(f"  max |d| = {abs_d.max():.3f}   "
                 f"median |d| = {abs_d.median():.3f}   "
                 f"min |d| = {abs_d.min():.3f}")

    # Top modes table
    lines.append("")
    lines.append("TOP 20 DISCRIMINATIVE MODES (by |Cohen's d|):")
    lines.append(f"  {'Rank':>4s}  {'Mode':>5s}  {'Cohen_d':>8s}  "
                 f"{'EigVal':>8s}  {'Coh_C':>6s}  {'Coh_I':>6s}")
    lines.append(f"  {'-' * 4}  {'-' * 5}  {'-' * 8}  "
                 f"{'-' * 8}  {'-' * 6}  {'-' * 6}")

    for rank, m in enumerate(d.discriminative_order[:20]):
        marker = "*" if abs_d[m] > result.disc_threshold else " "
        lines.append(
            f"  {rank:4d}  {m:5d}  {d.discriminability[m]:+8.3f}  "
            f"{d.eigenvalues[m]:8.2f}  {d.coherence_correct[m]:6.3f}  "
            f"{d.coherence_incorrect[m]:6.3f} {marker}"
        )

    # Variance explained by discriminative vs non-discriminative modes
    total_var = d.eigenvalues.sum().item()
    disc_var = d.eigenvalues[result.top_modes].sum().item() if result.top_modes else 0
    lines.append(f"\n  Variance explained by discriminative modes: "
                 f"{disc_var:.1f} / {total_var:.1f} = {disc_var / max(total_var, 1e-8):.1%}")

    # Interference
    lines.append("")
    lines.append("INTERFERENCE PATTERNS (top cross-mode correlation differences):")
    lines.append(f"  Total interference: {interf.total_interference:.3f}")
    if interf.top_pairs:
        lines.append(f"  {'Mode_i':>6s}  {'Mode_j':>6s}  "
                     f"{'|ΔCorr|':>7s}  {'Corr_C':>7s}  {'Corr_I':>7s}")
        lines.append(f"  {'-' * 6}  {'-' * 6}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
        for i, j, val in interf.top_pairs[:15]:
            lines.append(
                f"  {i:6d}  {j:6d}  {val:7.3f}  "
                f"{interf.corr_correct[i, j]:+7.3f}  "
                f"{interf.corr_incorrect[i, j]:+7.3f}"
            )

    # Per-family
    if d.per_family_disc:
        lines.append("")
        lines.append("PER-FAMILY: Top discriminative mode for each family:")
        for fam, fam_d in sorted(d.per_family_disc.items()):
            if fam_d.isnan().all():
                lines.append(f"  {fam:25s}  N/A (degenerate)")
                continue
            valid_mask = ~fam_d.isnan()
            if not valid_mask.any():
                continue
            best_mode = fam_d.abs().argmax().item()
            best_d = fam_d[best_mode].item()
            lines.append(f"  {fam:25s}  mode={best_mode:3d}  d={best_d:+.3f}")

    # Interpretation
    lines.append("")
    lines.append("INTERPRETATION:")
    if result.n_discriminative <= 5:
        lines.append("  Very few discriminative modes — signal is concentrated.")
        lines.append("  Targeted mode perturbation may outperform full-subspace.")
    elif result.n_discriminative <= 20:
        lines.append("  Moderate number of discriminative modes — mesoscale structure.")
        lines.append("  Mode-selective perturbation is a reasonable test.")
    else:
        lines.append("  Many discriminative modes — signal is diffuse across subspace.")
        lines.append("  Mode-selective perturbation unlikely to beat full-subspace.")

    if interf.top_pairs and interf.top_pairs[0][2] > 0.2:
        lines.append("  Strong interference patterns detected — phase disruption may work.")
    elif interf.top_pairs and interf.top_pairs[0][2] > 0.1:
        lines.append("  Moderate interference — phase disruption worth testing.")
    else:
        lines.append("  Weak interference — modes are approximately independent per class.")

    return "\n".join(lines)
