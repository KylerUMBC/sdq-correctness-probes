"""Dependency-free inference statistics for SDQ intervention experiments.

Phases 4/4b/5 compare flip *rates* between conditions and report ratios.
Ratios of small counts are extremely noisy, and the decision rules take a
max over many (component, magnitude, layer) cells — so any reported "winner"
needs an uncertainty estimate and a multiple-comparison correction. This
module provides the three pieces:

  - wilson_interval:        binomial CI on a single flip rate
  - fisher_exact_greater:   one-sided p-value that condition A's rate exceeds
                            condition B's (2x2 exact test, no normal approx —
                            valid at the small counts these experiments produce)
  - holm_adjust:            Holm-Bonferroni step-down adjustment across cells

All functions use only the standard library / torch-free math, so they can be
unit-tested locally and run anywhere.
"""

from __future__ import annotations

from math import comb, sqrt


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Returns (lo, hi). For n == 0 returns the uninformative (0.0, 1.0).
    Default z=1.96 gives a 95% interval.
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def fisher_exact_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided Fisher exact test: P(rate1 >= observed | rates equal).

    Tests whether condition 1 (k1 successes of n1) has a *greater* success
    rate than condition 2 (k2 of n2). Small p-value = evidence that
    condition 1's rate is genuinely higher.

    Computed from the hypergeometric distribution: conditioning on the
    margins (n1, n2, K = k1 + k2), the probability of seeing k1 or more
    successes in condition 1.
    """
    if n1 <= 0 or n2 <= 0:
        return 1.0
    k1 = max(0, min(k1, n1))
    k2 = max(0, min(k2, n2))
    N = n1 + n2
    K = k1 + k2
    denom = comb(N, n1)
    p = 0.0
    for x in range(k1, min(K, n1) + 1):
        if K - x > n2:
            continue
        p += comb(K, x) * comb(N - K, n1 - x) / denom
    return min(1.0, p)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (controls FWER).

    Returns adjusted p-values in the same order as the input. An adjusted
    p < alpha can be reported as significant at family-wise level alpha.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvalues[idx]
        running_max = max(running_max, min(1.0, adj))
        adjusted[idx] = running_max
    return adjusted
