"""Triple composition test for transport operators.

For three surface-forms A, B, C of the same semantic content:

    G^{A→C} ≈ G^{B→C} G^{A→B}    (transitivity)

This tests whether transport operators form a genuine groupoid
(real local geometry) rather than arbitrary pair-specific fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sdq.transport.composition import compose_transports


@dataclass
class CompositionResult:
    """Result of a triple composition test A→B→C vs A→C."""
    source: str
    via: str
    target: str
    direct_residual: float   # ||G^{A→C} vel_A - vel_C||
    composed_residual: float  # ||G^{B→C} G^{A→B} vel_A - vel_C||
    composition_error: float  # ||G^{B→C} G^{A→B} - G^{A→C}||_F


def triple_composition_test(
    G_ab: torch.Tensor,
    G_bc: torch.Tensor,
    G_ac: torch.Tensor,
) -> float:
    """Compute composition error ||G_bc @ G_ab - G_ac||_F^2 averaged over time.

    Args:
        G_ab: [T, D, D] transport A→B.
        G_bc: [T, D, D] transport B→C.
        G_ac: [T, D, D] transport A→C (direct).

    Returns:
        Scalar mean Frobenius error.
    """
    T_min = min(G_ab.shape[0], G_bc.shape[0], G_ac.shape[0])
    composed = compose_transports(G_ab[:T_min], G_bc[:T_min])
    diff = composed - G_ac[:T_min]
    frob_sq = diff.pow(2).sum(dim=(1, 2))  # [T]
    return frob_sq.mean().item()


def lowrank_composition_error(
    U_ab: torch.Tensor, V_ab: torch.Tensor,
    U_bc: torch.Tensor, V_bc: torch.Tensor,
    U_ac: torch.Tensor, V_ac: torch.Tensor,
) -> float:
    """Composition error from low-rank factors, avoiding D×D.

    G_bc @ G_ab = (I + U_bc V_bc^T)(I + U_ab V_ab^T)
               = I + U_bc V_bc^T + U_ab V_ab^T + U_bc (V_bc^T U_ab) V_ab^T

    G_ac = I + U_ac V_ac^T

    Error = ||G_bc G_ab - G_ac||_F^2
          = ||U_bc V_bc^T + U_ab V_ab^T + U_bc M V_ab^T - U_ac V_ac^T||_F^2

    where M = V_bc^T @ U_ab  [T, r, r].

    We express this as ||A B^T||_F^2 = tr(A^T A  B^T B) with r-scaled blocks.

    Args:
        All U, V: [T, D, r] low-rank factors.

    Returns:
        Scalar composition error.
    """
    T = min(U_ab.shape[0], U_bc.shape[0], U_ac.shape[0])
    U_ab, V_ab = U_ab[:T], V_ab[:T]
    U_bc, V_bc = U_bc[:T], V_bc[:T]
    U_ac, V_ac = U_ac[:T], V_ac[:T]

    M = torch.bmm(V_bc.transpose(1, 2), U_ab)  # [T, r, r]
    U_bc_M = torch.bmm(U_bc, M)  # [T, D, r]

    # C = U_bc V_bc^T + U_ab V_ab^T + U_bc_M V_ab^T - U_ac V_ac^T
    # Factor as: A = [U_bc, U_ab, U_bc_M, -U_ac]  B = [V_bc, V_ab, V_ab, V_ac]
    A = torch.cat([U_bc, U_ab, U_bc_M, -U_ac], dim=-1)  # [T, D, 4r]
    B = torch.cat([V_bc, V_ab, V_ab, V_ac], dim=-1)      # [T, D, 4r]

    AtA = torch.bmm(A.transpose(1, 2), A)  # [T, 4r, 4r]
    BtB = torch.bmm(B.transpose(1, 2), B)  # [T, 4r, 4r]

    frob_sq = (AtA * BtB).sum(dim=(1, 2))  # [T]
    return frob_sq.mean().item()


def batch_composition_test(
    transport_results: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]],
    prompt_ids: list[str],
) -> list[CompositionResult]:
    """Run triple composition tests for all valid triples.

    Args:
        transport_results: {(pid_i, pid_j): (U, V)} low-rank factors.
        prompt_ids: list of prompt IDs to form triples from.

    Returns:
        List of CompositionResult for each valid triple.
    """
    results = []
    for a in prompt_ids:
        for b in prompt_ids:
            if b == a:
                continue
            for c in prompt_ids:
                if c == a or c == b:
                    continue
                key_ab = (a, b)
                key_bc = (b, c)
                key_ac = (a, c)
                if key_ab in transport_results and key_bc in transport_results and key_ac in transport_results:
                    U_ab, V_ab = transport_results[key_ab]
                    U_bc, V_bc = transport_results[key_bc]
                    U_ac, V_ac = transport_results[key_ac]

                    err = lowrank_composition_error(
                        U_ab, V_ab, U_bc, V_bc, U_ac, V_ac,
                    )
                    results.append(CompositionResult(
                        source=a, via=b, target=c,
                        direct_residual=0.0,
                        composed_residual=0.0,
                        composition_error=err,
                    ))
    return results
