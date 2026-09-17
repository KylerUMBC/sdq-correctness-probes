"""Train/test splitting based on dataset metadata or custom criteria."""

from __future__ import annotations

from sdq.data.prompt_families.loader import PromptDataset, PromptGroup


def make_splits(
    dataset: PromptDataset,
    train_ids: list[str] | None = None,
    test_ids: list[str] | None = None,
) -> dict[str, list[PromptGroup]]:
    """Split a dataset into train/test groups.

    Uses explicit IDs if provided, otherwise falls back to
    the dataset's own train_group_ids / test_group_ids.
    Returns a dict with 'train' and 'test' keys.
    """
    train_ids = train_ids or dataset.train_group_ids
    test_ids = test_ids or dataset.test_group_ids

    gmap = dataset.group_map

    if train_ids and test_ids:
        return {
            "train": [gmap[gid] for gid in train_ids if gid in gmap],
            "test": [gmap[gid] for gid in test_ids if gid in gmap],
        }

    # If no split defined, return all groups as train
    return {"train": dataset.groups, "test": []}
