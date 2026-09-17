"""In-batch retrieval loss for semantic latent training.

Uses InfoNCE-style contrastive loss with hard negatives:
for each query trajectory, maximize similarity to same-family
members and minimize similarity to hardest cross-family confounders.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_retrieval_loss(
    latents: dict[str, torch.Tensor],
    families: dict[str, list[str]],
    temperature: float = 0.1,
) -> torch.Tensor:
    """In-batch InfoNCE retrieval loss.

    For each anchor, treats all same-family members as positives and
    all cross-family members as negatives.

    Args:
        latents: {prompt_id: [T, D_z]} encoded latent trajectories.
        families: {family_name: [prompt_ids]}.
        temperature: softmax temperature for similarity scaling.

    Returns:
        Scalar InfoNCE loss.
    """
    device = next(iter(latents.values())).device
    pid_to_fam = {pid: fam for fam, pids in families.items() for pid in pids}

    # Time-average each latent to get embedding vectors
    pids = [pid for pid in latents if pid in pid_to_fam]
    if len(pids) < 3:
        return torch.tensor(0.0, device=device)

    embeddings = torch.stack([latents[pid].mean(dim=0) for pid in pids])  # [N, D_z]
    embeddings = F.normalize(embeddings, dim=-1)

    # Pairwise cosine similarity
    sim = embeddings @ embeddings.T / temperature  # [N, N]

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for i, pid_i in enumerate(pids):
        fam_i = pid_to_fam[pid_i]
        pos_mask = torch.tensor(
            [pid_to_fam[pids[j]] == fam_i and j != i for j in range(len(pids))],
            device=device,
        )
        if not pos_mask.any():
            continue

        # Mask out self-similarity
        mask = torch.ones(len(pids), device=device, dtype=torch.bool)
        mask[i] = False

        logits = sim[i][mask]
        pos_shifted = pos_mask[mask]

        if not pos_shifted.any():
            continue

        # Log-sum-exp denominator over all non-self
        log_denom = torch.logsumexp(logits, dim=0)

        # Average over positive log-probs
        pos_logits = logits[pos_shifted]
        loss_i = -(pos_logits - log_denom).mean()
        total_loss = total_loss + loss_i
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / count


def hard_retrieval_loss(
    latents: dict[str, torch.Tensor],
    families: dict[str, list[str]],
    margin: float = 1.0,
) -> torch.Tensor:
    """Margin-based retrieval loss with hard negatives.

    For each anchor, finds the hardest negative (closest cross-family)
    and the hardest positive (farthest same-family) and optimizes:
        max(0, d(anchor, hard_pos) - d(anchor, hard_neg) + margin)

    Args:
        latents: {prompt_id: [T, D_z]} encoded latent trajectories.
        families: {family_name: [prompt_ids]}.
        margin: separation margin.

    Returns:
        Scalar loss.
    """
    device = next(iter(latents.values())).device
    pid_to_fam = {pid: fam for fam, pids in families.items() for pid in pids}

    pids = [pid for pid in latents if pid in pid_to_fam]
    if len(pids) < 3:
        return torch.tensor(0.0, device=device)

    # Time-averaged embeddings
    embeddings = torch.stack([latents[pid].mean(dim=0) for pid in pids])  # [N, D_z]

    # Pairwise L2 distances
    dists = torch.cdist(embeddings, embeddings, p=2)  # [N, N]

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for i, pid_i in enumerate(pids):
        fam_i = pid_to_fam[pid_i]
        pos_mask = torch.tensor(
            [pid_to_fam[pids[j]] == fam_i and j != i for j in range(len(pids))],
            device=device,
        )
        neg_mask = torch.tensor(
            [pid_to_fam[pids[j]] != fam_i for j in range(len(pids))],
            device=device,
        )

        if not pos_mask.any() or not neg_mask.any():
            continue

        # Hardest positive: farthest same-family member
        d_pos = dists[i][pos_mask].max()
        # Hardest negative: closest cross-family member
        d_neg = dists[i][neg_mask].min()

        total_loss = total_loss + torch.clamp(d_pos - d_neg + margin, min=0.0)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / count
