"""Group-disjoint splits and cluster bootstrap helpers.

The SDQ benchmark contains several surface-form variants of each underlying
semantic task.  Example-level random splits therefore leak close paraphrases
across train and test.  These helpers keep every semantic task in exactly one
fold while approximately balancing family and outcome counts.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import Tensor

from sdq.eval.commitment_probe import compute_auroc, per_family_auroc


def stratified_group_kfold(
    labels: Sequence[int] | Tensor,
    families: Sequence[str],
    groups: Sequence[str],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[Tensor, Tensor]]:
    """Return approximately stratified, strictly group-disjoint CV folds.

    Groups are assigned by a seeded greedy algorithm that minimizes imbalance
    over the joint (family, outcome) strata.  A group may not span task
    families.  Every example appears in one and only one test fold.
    """
    y = [int(v) for v in (labels.tolist() if isinstance(labels, Tensor) else labels)]
    if not (len(y) == len(families) == len(groups)):
        raise ValueError("labels, families, and groups must have equal length")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    members: dict[str, list[int]] = defaultdict(list)
    group_family: dict[str, str] = {}
    for i, (fam, group) in enumerate(zip(families, groups)):
        if group in group_family and group_family[group] != fam:
            raise ValueError(f"group {group!r} spans multiple families")
        group_family[group] = fam
        members[group].append(i)

    fold_groups: list[set[str]] = [set() for _ in range(n_splits)]
    rng = random.Random(seed)

    strata = sorted({(fam, label) for fam, label in zip(families, y)})
    stratum_index = {stratum: i for i, stratum in enumerate(strata)}
    totals = [0] * len(strata)
    records: list[tuple[str, list[int], float]] = []
    for group, idx in members.items():
        counts = [0] * len(strata)
        for i in idx:
            s = stratum_index[(families[i], y[i])]
            counts[s] += 1
            totals[s] += 1
        records.append((group, counts, rng.random()))

    # Place the least homogeneous / largest groups first.  This is the same
    # broad strategy as stratified group k-fold implementations: hard groups
    # are allocated while every fold is still available.
    records.sort(
        key=lambda r: (
            sum(r[1]),
            max(r[1]),
            sum(v * v for v in r[1]),
            r[2],
        ),
        reverse=True,
    )
    fold_counts = [[0] * len(strata) for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    fold_group_counts = [0] * n_splits
    total_examples = len(y)
    total_groups = len(records)

    def std(values: list[float]) -> float:
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    for group, group_counts, _ in records:
        candidate_order = list(range(n_splits))
        rng.shuffle(candidate_order)

        def cost(fold: int) -> tuple[float, int, int, int]:
            fold_counts[fold] = [
                a + b for a, b in zip(fold_counts[fold], group_counts)
            ]
            fold_sizes[fold] += sum(group_counts)
            fold_group_counts[fold] += 1

            stratum_imbalance = sum(
                std([fold_counts[f][s] / max(totals[s], 1) for f in range(n_splits)])
                for s in range(len(strata))
            ) / max(len(strata), 1)
            size_imbalance = std(
                [fold_sizes[f] / max(total_examples, 1) for f in range(n_splits)]
            )
            group_imbalance = std(
                [fold_group_counts[f] / max(total_groups, 1) for f in range(n_splits)]
            )

            fold_counts[fold] = [
                a - b for a, b in zip(fold_counts[fold], group_counts)
            ]
            fold_sizes[fold] -= sum(group_counts)
            fold_group_counts[fold] -= 1
            score = stratum_imbalance + 0.10 * size_imbalance + 0.05 * group_imbalance
            return score, fold_sizes[fold], fold_group_counts[fold], fold

        chosen = min(candidate_order, key=cost)
        fold_groups[chosen].add(group)
        fold_counts[chosen] = [
            a + b for a, b in zip(fold_counts[chosen], group_counts)
        ]
        fold_sizes[chosen] += sum(group_counts)
        fold_group_counts[chosen] += 1

    all_idx = set(range(len(y)))
    folds: list[tuple[Tensor, Tensor]] = []
    seen_test: set[int] = set()
    for held_out in fold_groups:
        test = sorted(i for group in held_out for i in members[group])
        train = sorted(all_idx - set(test))
        if not test or not train:
            raise RuntimeError("group assignment produced an empty fold")
        if seen_test.intersection(test):
            raise RuntimeError("an example was assigned to multiple test folds")
        seen_test.update(test)
        folds.append((torch.tensor(train), torch.tensor(test)))

    if seen_test != all_idx:
        raise RuntimeError("some examples were not assigned to a test fold")
    return folds


def cluster_bootstrap_auroc(
    scores: Tensor,
    labels: Tensor,
    families: Sequence[str],
    groups: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Bootstrap pooled and macro-within-family AUROC by semantic task.

    Resampling individual paraphrases would understate uncertainty because
    examples in a semantic group are related.  This function resamples whole
    groups with replacement and returns percentile intervals.
    """
    if not (len(scores) == len(labels) == len(families) == len(groups)):
        raise ValueError("scores, labels, families, and groups must have equal length")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    members: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    unique_groups = sorted(members)
    rng = random.Random(seed)
    pooled_samples: list[float] = []
    within_samples: list[float] = []

    for _ in range(n_bootstrap):
        sampled = [rng.choice(unique_groups) for _ in unique_groups]
        idx = [i for group in sampled for i in members[group]]
        bs_scores = scores[idx]
        bs_labels = labels[idx]
        bs_families = [families[i] for i in idx]
        if bs_labels.unique().numel() < 2:
            continue
        pooled_samples.append(compute_auroc(bs_scores, bs_labels))
        fam = per_family_auroc(bs_scores, bs_labels, bs_families)
        valid = [v for v in fam.values() if v == v]
        if valid:
            within_samples.append(sum(valid) / len(valid))

    def interval(values: list[float]) -> list[float]:
        if not values:
            return [float("nan"), float("nan")]
        t = torch.tensor(values, dtype=torch.float64)
        return [
            float(torch.quantile(t, 0.025).item()),
            float(torch.quantile(t, 0.975).item()),
        ]

    return {
        "pooled_95ci": interval(pooled_samples),
        "within_family_95ci": interval(within_samples),
    }
