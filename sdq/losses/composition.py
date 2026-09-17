"""Direct composition loss for transport operators.

Trains transport to be compositionally consistent:

    L_comp = ||G^{A→C} - G^{B→C} G^{A→B}||^2

This makes composition a first-class training objective instead of
only a post-hoc evaluation metric.
"""

from __future__ import annotations

import torch


def composition_loss(
    U_ab: torch.Tensor,
    V_ab: torch.Tensor,
    U_bc: torch.Tensor,
    V_bc: torch.Tensor,
    U_ac: torch.Tensor,
    V_ac: torch.Tensor,
) -> torch.Tensor:
    """Low-rank composition loss avoiding D×D materialization.

    Computes ||G_bc G_ab - G_ac||_F^2 from factors.

    G_bc G_ab = (I + U_bc V_bc^T)(I + U_ab V_ab^T)
              = I + U_bc V_bc^T + U_ab V_ab^T + U_bc (V_bc^T U_ab) V_ab^T
    G_ac      = I + U_ac V_ac^T

    Error = ||U_bc V_bc^T + U_ab V_ab^T + U_bc M V_ab^T - U_ac V_ac^T||_F^2
    where M = V_bc^T U_ab.

    Uses tr(A^T A B^T B) identity.

    Args:
        U_ab, V_ab: [T, D, r] low-rank factors for A→B.
        U_bc, V_bc: [T, D, r] low-rank factors for B→C.
        U_ac, V_ac: [T, D, r] low-rank factors for A→C.

    Returns:
        Scalar composition loss (differentiable).
    """
    T = min(U_ab.shape[0], U_bc.shape[0], U_ac.shape[0])
    U_ab, V_ab = U_ab[:T], V_ab[:T]
    U_bc, V_bc = U_bc[:T], V_bc[:T]
    U_ac, V_ac = U_ac[:T], V_ac[:T]

    M = torch.bmm(V_bc.transpose(1, 2), U_ab)  # [T, r, r]
    U_bc_M = torch.bmm(U_bc, M)                 # [T, D, r]

    # Stack as A = [U_bc, U_ab, U_bc_M, -U_ac], B = [V_bc, V_ab, V_ab, V_ac]
    A = torch.cat([U_bc, U_ab, U_bc_M, -U_ac], dim=-1)  # [T, D, 4r]
    B = torch.cat([V_bc, V_ab, V_ab, V_ac], dim=-1)      # [T, D, 4r]

    AtA = torch.bmm(A.transpose(1, 2), A)  # [T, 4r, 4r]
    BtB = torch.bmm(B.transpose(1, 2), B)  # [T, 4r, 4r]

    frob_sq = (AtA * BtB).sum(dim=(1, 2))  # [T]
    return frob_sq.mean()


def batch_composition_loss(
    factors: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]],
    prompt_ids: list[str],
) -> torch.Tensor:
    """Composition loss over all valid triples in a batch.

    Args:
        factors: {(pid_a, pid_b): (U_ab, V_ab)} transport factors.
        prompt_ids: list of prompt IDs to form triples from.

    Returns:
        Scalar mean composition loss.
    """
    device = next(iter(factors.values()))[0].device
    total = torch.tensor(0.0, device=device)
    count = 0

    for a in prompt_ids:
        for b in prompt_ids:
            if b == a:
                continue
            for c in prompt_ids:
                if c == a or c == b:
                    continue
                if (a, b) in factors and (b, c) in factors and (a, c) in factors:
                    U_ab, V_ab = factors[(a, b)]
                    U_bc, V_bc = factors[(b, c)]
                    U_ac, V_ac = factors[(a, c)]
                    total = total + composition_loss(
                        U_ab, V_ab, U_bc, V_bc, U_ac, V_ac,
                    )
                    count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total / count
