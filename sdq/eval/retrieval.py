"""Retrieval accuracy: nearest-neighbor correctness in latent space.

Given one run, can its nearest latent neighbor recover the correct
semantic equivalent instead of a hard negative with similar surface form?

This is Priority 1C from the improvements plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RetrievalResult:
    """Result of nearest-neighbor retrieval evaluation."""

    accuracy: float  # fraction of queries where NN is same-family
    mean_reciprocal_rank: float  # avg 1/rank of first same-family neighbor
    num_queries: int
    per_query: list[dict]  # per-query details


def retrieval_accuracy(
    latent_states: dict[str, torch.Tensor],
    family_map: dict[str, str],
) -> RetrievalResult:
    """Evaluate semantic retrieval accuracy in latent space.

    For each prompt, find its nearest neighbor (by time-averaged
    latent L2 distance). Check whether the nearest neighbor is from
    the same semantic family.

    Args:
        latent_states: {prompt_id: [T, D_z]} latent trajectories.
        family_map: {prompt_id: family_name}.

    Returns:
        RetrievalResult with accuracy and per-query breakdown.
    """
    pids = sorted(latent_states.keys())
    if len(pids) < 2:
        return RetrievalResult(accuracy=0.0, mean_reciprocal_rank=0.0,
                               num_queries=0, per_query=[])

    # Compute time-averaged latent for each prompt
    means: dict[str, torch.Tensor] = {}
    for pid in pids:
        means[pid] = latent_states[pid].mean(dim=0)  # [D_z]

    correct = 0
    reciprocal_ranks = []
    per_query = []

    for query_pid in pids:
        query_fam = family_map.get(query_pid, "?")
        query_mean = means[query_pid]

        # Compute distances to all other prompts
        dists = []
        for other_pid in pids:
            if other_pid == query_pid:
                continue
            d = (query_mean - means[other_pid]).pow(2).sum().item()
            dists.append((d, other_pid))

        dists.sort(key=lambda x: x[0])

        # Check nearest neighbor
        nn_pid = dists[0][1]
        nn_fam = family_map.get(nn_pid, "?")
        is_correct = nn_fam == query_fam

        if is_correct:
            correct += 1

        # Mean reciprocal rank: rank of first same-family neighbor
        first_same_rank = None
        for rank_idx, (d, pid) in enumerate(dists):
            if family_map.get(pid, "?") == query_fam:
                first_same_rank = rank_idx + 1
                break

        rr = 1.0 / first_same_rank if first_same_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        per_query.append({
            "query": query_pid,
            "family": query_fam,
            "nearest": nn_pid,
            "nearest_family": nn_fam,
            "correct": is_correct,
            "first_same_rank": first_same_rank,
        })

    n = len(pids)
    return RetrievalResult(
        accuracy=correct / n,
        mean_reciprocal_rank=sum(reciprocal_ranks) / n,
        num_queries=n,
        per_query=per_query,
    )
