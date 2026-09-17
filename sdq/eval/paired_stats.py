"""Prompt-clustered statistics for intervention experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def paired_effect(
    directed: Sequence[float], controls: Sequence[Sequence[float]]
) -> tuple[float, float, float]:
    """Return directed rate, control rate, and paired rate difference."""
    if len(directed) != len(controls) or not directed:
        raise ValueError("directed and controls must have equal, non-zero length")
    if any(not row for row in controls):
        raise ValueError("every prompt must have at least one control outcome")
    directed_rate = sum(float(v) for v in directed) / len(directed)
    per_prompt_control = [sum(map(float, row)) / len(row) for row in controls]
    control_rate = sum(per_prompt_control) / len(per_prompt_control)
    return directed_rate, control_rate, directed_rate - control_rate


def exchangeability_permutation_pvalue(
    directed: Sequence[float],
    controls: Sequence[Sequence[float]],
    n_permutations: int = 50_000,
    seed: int = 42,
) -> float:
    """One-sided prompt-clustered randomization test.

    Under the null that the designated direction is exchangeable with the
    matched random directions, choose one outcome within each prompt as the
    pseudo-directed observation and compare it with the remaining outcomes.
    The statistic is the across-prompt mean rate difference.
    """
    _, _, observed = paired_effect(directed, controls)
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    row_lengths = {len(row) for row in controls}
    if len(row_lengths) != 1:
        raise ValueError("every prompt must have the same number of controls")
    rows = torch.tensor(
        [[float(d), *map(float, c)] for d, c in zip(directed, controls)],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    extreme = 0
    chunk_size = 2048
    completed = 0
    while completed < n_permutations:
        batch = min(chunk_size, n_permutations - completed)
        choices = torch.randint(
            rows.shape[1], (batch, rows.shape[0]), generator=generator
        )
        expanded = rows.unsqueeze(0).expand(batch, -1, -1)
        chosen = expanded.gather(2, choices.unsqueeze(-1)).squeeze(-1)
        totals = rows.sum(dim=1).unsqueeze(0)
        pseudo_controls = (totals - chosen) / (rows.shape[1] - 1)
        statistics = (chosen - pseudo_controls).mean(dim=1)
        extreme += int((statistics >= observed - 1e-12).sum().item())
        completed += batch
    return (extreme + 1) / (n_permutations + 1)


def paired_cluster_bootstrap_ci(
    directed: Sequence[float],
    controls: Sequence[Sequence[float]],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> list[float]:
    """Percentile CI for the paired rate difference, resampling prompts."""
    paired_effect(directed, controls)  # validation
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    differences = torch.tensor([
        float(d) - sum(map(float, c)) / len(c)
        for d, c in zip(directed, controls)
    ], dtype=torch.float64)
    n = len(differences)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(n, (n_bootstrap, n), generator=generator)
    values = differences[indices].mean(dim=1)
    return [
        float(torch.quantile(values, 0.025).item()),
        float(torch.quantile(values, 0.975).item()),
    ]
