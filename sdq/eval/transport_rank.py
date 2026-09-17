"""Transport rank analysis: structural diagnostics for learned transports.

Part of certification test B (transport validity). Analyzes the
effective rank, singular value spectrum, and deviation from identity
of learned transport operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sdq.transport.low_rank_maps import operator_rank_spectrum


@dataclass
class RankAnalysis:
    """Summary of transport operator rank structure."""

    singular_values: torch.Tensor  # sorted SVs of G - I
    effective_rank: float  # entropy-based effective rank
    nuclear_norm: float  # sum of SVs (rank proxy)
    frobenius_deviation: float  # ||G - I||_F
    spectral_norm: float  # largest SV (max distortion)
    top_k_explained: dict[int, float]  # fraction of norm in top k SVs


def rank_analysis(G: torch.Tensor) -> RankAnalysis:
    """Analyze rank structure of a transport operator.

    Args:
        G: [D, D] or [T, D, D] transport operator(s).
            If [T, D, D], analyzes the time-average.

    Returns:
        RankAnalysis with singular value diagnostics.
    """
    if G.dim() == 3:
        G = G.mean(dim=0)  # [D, D]

    svs = operator_rank_spectrum(G)  # sorted singular values of G - I
    total = svs.sum().item()
    frob = svs.pow(2).sum().sqrt().item()

    # Effective rank via entropy: exp(-sum p_i log p_i) where p_i = sv_i / sum
    if total > 0:
        p = svs / total
        p = p[p > 0]
        entropy = -(p * p.log()).sum().item()
        eff_rank = float(torch.tensor(entropy).exp().item())
    else:
        eff_rank = 0.0

    # Top-k explained variance
    top_k_explained = {}
    squared = svs.pow(2)
    total_sq = squared.sum().item()
    if total_sq > 0:
        for k in [1, 2, 4, 8, 16]:
            if k <= len(svs):
                top_k_explained[k] = squared[:k].sum().item() / total_sq

    return RankAnalysis(
        singular_values=svs,
        effective_rank=eff_rank,
        nuclear_norm=total,
        frobenius_deviation=frob,
        spectral_norm=svs[0].item() if len(svs) > 0 else 0.0,
        top_k_explained=top_k_explained,
    )


def batch_rank_analysis(
    transports: dict[tuple[str, str], torch.Tensor],
) -> dict[str, RankAnalysis]:
    """Analyze all learned transports.

    Args:
        transports: dict (pair_name) -> [T, D, D] transport operators.

    Returns:
        Dict of pair label -> RankAnalysis.
    """
    results = {}
    for pair, G_t in transports.items():
        label = f"{pair[0]}->{pair[1]}"
        results[label] = rank_analysis(G_t)
    return results


def transport_validity_score(
    transport_residuals: torch.Tensor,
    shuffle_residuals: torch.Tensor,
) -> float:
    """Test B: do learned transports explain differences better than shuffled?

    Computes ratio: mean(shuffle_residual) / mean(transport_residual).
    Score > 1 means transport is doing something useful.

    Args:
        transport_residuals: [N] residual norms from learned transport.
        shuffle_residuals: [N] residual norms from shuffled control.

    Returns:
        Score > 1 indicates transport validity.
    """
    transport_mean = transport_residuals.mean().item()
    shuffle_mean = shuffle_residuals.mean().item()
    if transport_mean == 0:
        return float("inf")
    return shuffle_mean / transport_mean
