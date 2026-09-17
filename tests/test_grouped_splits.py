import torch

from sdq.eval.grouped_splits import cluster_bootstrap_auroc, stratified_group_kfold


def _toy_data():
    labels = []
    families = []
    groups = []
    for family in ("a", "b"):
        for group_idx in range(10):
            for variant in range(3):
                families.append(family)
                groups.append(f"{family}_{group_idx}")
                labels.append((group_idx + variant) % 2)
    return torch.tensor(labels), families, groups


def test_grouped_folds_are_disjoint_and_exhaustive():
    labels, families, groups = _toy_data()
    folds = stratified_group_kfold(labels, families, groups, n_splits=5, seed=7)
    seen = []
    for train_idx, test_idx in folds:
        train_groups = {groups[i] for i in train_idx.tolist()}
        test_groups = {groups[i] for i in test_idx.tolist()}
        assert train_groups.isdisjoint(test_groups)
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(len(labels)))
    test_sizes = [len(test_idx) for _, test_idx in folds]
    assert max(test_sizes) - min(test_sizes) <= 6


def test_grouped_folds_are_deterministic():
    labels, families, groups = _toy_data()
    first = stratified_group_kfold(labels, families, groups, n_splits=5, seed=11)
    second = stratified_group_kfold(labels, families, groups, n_splits=5, seed=11)
    assert all(torch.equal(a[1], b[1]) for a, b in zip(first, second))


def test_cluster_bootstrap_returns_intervals():
    labels, families, groups = _toy_data()
    scores = labels.float() * 0.8 + 0.1
    result = cluster_bootstrap_auroc(
        scores, labels, families, groups, n_bootstrap=50, seed=3
    )
    assert result["pooled_95ci"] == [1.0, 1.0]
    assert result["within_family_95ci"] == [1.0, 1.0]
