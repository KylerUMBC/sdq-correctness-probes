"""Triplet / margin loss for semantic latent separation.

Forces the latent space to not just pull same-family trajectories
together, but also push different-family trajectories apart.

    L_triplet = max(0, d(z_a, z_p) - d(z_a, z_n) + margin)

where:
    - anchor a, positive p: same semantic family, different surface form
    - negative n: different semantic family
"""

from __future__ import annotations

import torch


def triplet_semantic_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet loss on time-averaged latent representations.

    Args:
        z_anchor: [T_a, D_z] latent trajectory (anchor).
        z_positive: [T_p, D_z] latent trajectory (same semantics).
        z_negative: [T_n, D_z] latent trajectory (different semantics).
        margin: separation margin.

    Returns:
        Scalar triplet loss.
    """
    # Time-average each trajectory to get a single vector
    a = z_anchor.mean(dim=0)    # [D_z]
    p = z_positive.mean(dim=0)  # [D_z]
    n = z_negative.mean(dim=0)  # [D_z]

    d_pos = (a - p).pow(2).sum()
    d_neg = (a - n).pow(2).sum()

    return torch.clamp(d_pos - d_neg + margin, min=0.0)


def semi_hard_triplet_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_negative: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Semi-hard triplet loss using L2 distance on mean embeddings.

    Uses L2 (not squared L2) on time-averaged representations so that
    the margin is in the same units as actual embedding distance.
    Avoids trajectory-level matching which is biased by sequence
    length differences between prompts.

    Args:
        z_anchor: [T_a, D_z] latent trajectory (anchor).
        z_positive: [T_p, D_z] latent trajectory (same semantics).
        z_negative: [T_n, D_z] latent trajectory (different semantics).
        margin: separation margin in L2 distance.

    Returns:
        Scalar triplet loss.
    """
    a_mean = z_anchor.mean(dim=0)
    p_mean = z_positive.mean(dim=0)
    n_mean = z_negative.mean(dim=0)

    d_pos = (a_mean - p_mean).pow(2).sum().sqrt()
    d_neg = (a_mean - n_mean).pow(2).sum().sqrt()

    return torch.clamp(d_pos - d_neg + margin, min=0.0)


def batch_triplet_loss(
    latents: dict[str, torch.Tensor],
    family_map: dict[str, str],
    margin: float = 1.0,
) -> torch.Tensor:
    """Compute triplet loss over all valid (anchor, positive, negative) triples.

    Args:
        latents: {prompt_id: [T, D_z]} latent trajectories.
        family_map: {prompt_id: family_name} mapping each prompt to its
            semantic family.
        margin: triplet margin.

    Returns:
        Scalar average triplet loss.
    """
    # Group by family
    families: dict[str, list[str]] = {}
    for pid, fam in family_map.items():
        if pid in latents:
            families.setdefault(fam, []).append(pid)

    fam_names = list(families.keys())
    if len(fam_names) < 2:
        device = next(iter(latents.values())).device
        return torch.tensor(0.0, device=device)

    device = next(iter(latents.values())).device
    total = torch.tensor(0.0, device=device)
    count = 0

    for fam in fam_names:
        pids = families[fam]
        if len(pids) < 2:
            continue
        # Other families for negatives
        neg_pids = [
            pid for other_fam in fam_names if other_fam != fam
            for pid in families[other_fam]
        ]
        if not neg_pids:
            continue

        for i in range(len(pids)):
            for j in range(len(pids)):
                if i == j:
                    continue
                anchor_z = latents[pids[i]]
                pos_z = latents[pids[j]]
                # Hard negative: closest from other families
                anchor_mean = anchor_z.mean(dim=0)
                best_neg_dist = float("inf")
                best_neg_z = None
                for neg_pid in neg_pids:
                    neg_mean = latents[neg_pid].mean(dim=0)
                    d = (anchor_mean - neg_mean).pow(2).sum().item()
                    if d < best_neg_dist:
                        best_neg_dist = d
                        best_neg_z = latents[neg_pid]

                total = total + triplet_semantic_loss(
                    anchor_z, pos_z, best_neg_z, margin=margin
                )
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total / count
