"""Cycle consistency loss for transport inversion and triple composition.

Two structural constraints that prevent arbitrary pair fitting:

    Inversion:    L_inv   = sum_t ||G^{j->i} G^{i->j} - I||_F^2
    Transitivity: L_trans = sum_t ||G^{j->k} G^{i->j} - G^{i->k}||_F^2

Combined: L_cycle = L_inv + lambda_triple * L_trans
"""

from __future__ import annotations

import torch

from sdq.transport.cycle_consistency import (
    composition_error,
    inversion_error,
)


def cycle_loss(
    G_ij: torch.Tensor,
    G_ji: torch.Tensor,
    G_jk: torch.Tensor | None = None,
    G_ik: torch.Tensor | None = None,
    lambda_triple: float = 1.0,
) -> torch.Tensor:
    """Compute combined cycle consistency loss.

    Args:
        G_ij: [T, D, D] transport i -> j.
        G_ji: [T, D, D] transport j -> i.
        G_jk: optional [T, D, D] transport j -> k for triple test.
        G_ik: optional [T, D, D] transport i -> k (direct) for triple test.
        lambda_triple: weight for composition term.

    Returns:
        Scalar loss.
    """
    loss = inversion_error(G_ij, G_ji)

    if G_jk is not None and G_ik is not None:
        loss = loss + lambda_triple * composition_error(G_ij, G_jk, G_ik)

    return loss


def batch_cycle_loss(
    transport_pairs: dict[tuple[str, str], torch.Tensor],
    lambda_triple: float = 1.0,
) -> torch.Tensor:
    """Compute cycle loss over all available pairs and triples.

    Args:
        transport_pairs: dict mapping (prompt_i, prompt_j) -> [T, D, D]
            transport operators. Both (i,j) and (j,i) should be present.
        lambda_triple: weight for triple composition terms.

    Returns:
        Scalar average cycle loss.
    """
    device = next(iter(transport_pairs.values())).device
    total = torch.tensor(0.0, device=device)
    count = 0

    # Collect all prompt IDs
    ids = sorted({pid for pair in transport_pairs for pid in pair})

    # Inversion terms: for each (i,j) pair where both directions exist
    seen_pairs: set[tuple[str, str]] = set()
    for (a, b) in transport_pairs:
        canonical = (min(a, b), max(a, b))
        if canonical in seen_pairs:
            continue
        if (b, a) in transport_pairs:
            seen_pairs.add(canonical)
            total = total + inversion_error(
                transport_pairs[(a, b)],
                transport_pairs[(b, a)],
            )
            count += 1

    # Triple composition terms
    for i in ids:
        for j in ids:
            if i == j:
                continue
            for k in ids:
                if k == i or k == j:
                    continue
                if (i, j) in transport_pairs and (j, k) in transport_pairs and (i, k) in transport_pairs:
                    total = total + lambda_triple * composition_error(
                        transport_pairs[(i, j)],
                        transport_pairs[(j, k)],
                        transport_pairs[(i, k)],
                    )
                    count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)

    return total / count


def lowrank_cycle_loss(
    U_ij: torch.Tensor,
    V_ij: torch.Tensor,
    U_ji: torch.Tensor,
    V_ji: torch.Tensor,
) -> torch.Tensor:
    """Cycle consistency loss from low-rank factors, avoiding D×D matrices.

    G_ij = I + U_ij @ V_ij^T.  We want ||G_ji @ G_ij - I||_F^2.
    G_ji @ G_ij = (I + Uj Vj^T)(I + Ui Vi^T)
               = I + Uj Vj^T + Ui Vi^T + Uj (Vj^T @ Ui) Vi^T
    So the "correction" from identity is:
        C = Uj Vj^T + Ui Vi^T + Uj (Vj^T @ Ui) Vi^T
    and we want ||C||_F^2.

    We compute this using the trace formula:
        ||C||_F^2 = tr(C^T C)
    with C expressed through low-rank factors (never forming D×D).

    Args:
        U_ij, V_ij: [T, D, r] factors for G_{i->j}.
        U_ji, V_ji: [T, D, r] factors for G_{j->i}.

    Returns:
        Scalar cycle loss.
    """

    T, D, r = U_ij.shape

    # M = V_ji^T @ U_ij  [T, r, r]
    M = torch.bmm(V_ji.transpose(1, 2), U_ij)  # [T, r, r]

    # The correct identity is: ||A B^T||_F^2 = tr((A^T A)(B^T B))
    # where A^T A and B^T B are [3r, 3r] — small!

    U_ji_M = torch.bmm(U_ji, M)  # [T, D, r]
    A = torch.cat([U_ji, U_ij, U_ji_M], dim=-1)  # [T, D, 3r]
    B = torch.cat([V_ji, V_ij, V_ij], dim=-1)     # [T, D, 3r]

    # A^T A: [T, 3r, 3r]
    AtA = torch.bmm(A.transpose(1, 2), A)
    # B^T B: [T, 3r, 3r]
    BtB = torch.bmm(B.transpose(1, 2), B)

    # ||C||_F^2 = tr(AtA * BtB) per timestep (element-wise multiply then sum)
    frob_sq = (AtA * BtB).sum(dim=(1, 2))  # [T]

    return frob_sq.mean()
