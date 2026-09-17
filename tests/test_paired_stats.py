from sdq.eval.paired_stats import (
    exchangeability_permutation_pvalue,
    paired_cluster_bootstrap_ci,
    paired_effect,
)


def test_paired_effect_uses_prompt_means():
    directed = [1, 0]
    controls = [[0, 0, 1], [0, 0, 0]]
    d_rate, c_rate, difference = paired_effect(directed, controls)
    assert d_rate == 0.5
    assert abs(c_rate - 1 / 6) < 1e-12
    assert abs(difference - 1 / 3) < 1e-12


def test_identical_rows_have_zero_effect_and_unit_pvalue():
    directed = [0, 1, 0, 1]
    controls = [[v, v, v] for v in directed]
    assert paired_effect(directed, controls)[2] == 0.0
    assert paired_cluster_bootstrap_ci(directed, controls, 100, 1) == [0.0, 0.0]
    assert exchangeability_permutation_pvalue(directed, controls, 100, 1) == 1.0


def test_large_consistent_effect_is_detected():
    directed = [1] * 20
    controls = [[0, 0, 0] for _ in directed]
    pvalue = exchangeability_permutation_pvalue(
        directed, controls, n_permutations=2000, seed=4
    )
    assert pvalue < 0.01
    lo, hi = paired_cluster_bootstrap_ci(directed, controls, 200, 4)
    assert lo == hi == 1.0
