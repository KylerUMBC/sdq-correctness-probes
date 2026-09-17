"""Holdout splits for the factorized SDQ benchmark.

Supports 4 split types per improvements.md:
  1. semantic_family  — hold out entire semantic families
  2. surface_form     — hold out surface templates
  3. reasoning_variant — hold out reasoning variants
  4. combination      — hold out unseen (family × surface) pairs

Primary entry point:
    make_benchmark_splits(dataset, split_type, holdout_fraction) -> BenchmarkSplit
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

from sdq.data.generators.schema import BenchmarkDataset, BenchmarkExample


@dataclass
class BenchmarkSplit:
    """A train/test split of benchmark examples."""

    split_type: str
    train: list[BenchmarkExample]
    test: list[BenchmarkExample]
    held_out_keys: list[str] = field(default_factory=list)

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)

    def summary(self) -> str:
        lines = [
            f"Split type: {self.split_type}",
            f"Train: {self.train_size}, Test: {self.test_size}",
            f"Held-out keys: {self.held_out_keys}",
        ]
        return "\n".join(lines)


def _deterministic_holdout(keys: list[str], fraction: float, seed: str) -> set[str]:
    """Select a deterministic subset of keys to hold out, based on hashing."""
    scored = []
    for k in keys:
        h = hashlib.sha256(f"{seed}:{k}".encode()).hexdigest()
        scored.append((int(h[:8], 16), k))
    scored.sort()
    n = max(1, int(len(scored) * fraction))
    return {k for _, k in scored[:n]}


def make_benchmark_splits(
    dataset: BenchmarkDataset,
    split_type: str = "semantic_family",
    holdout_fraction: float = 0.25,
    seed: str = "sdq_v1",
) -> BenchmarkSplit:
    """Create a train/test split of the benchmark.

    Args:
        dataset: The benchmark dataset to split.
        split_type: One of "semantic_family", "surface_form",
                    "reasoning_variant", "combination".
        holdout_fraction: Fraction of unique keys to hold out (~0.2–0.3).
        seed: Deterministic seed for reproducible splits.

    Returns:
        A BenchmarkSplit with train/test examples and held-out key info.
    """
    examples = dataset.examples

    if split_type == "semantic_family":
        # Hold out entire semantic families (by task_family × semantic_task_id)
        all_families = sorted({(e.task_family, e.semantic_task_id) for e in examples})
        keys = [f"{tf}:{stid}" for tf, stid in all_families]
        held_out = _deterministic_holdout(keys, holdout_fraction, f"{seed}:sem")
        held_set = {tuple(k.split(":", 1)) for k in held_out}
        train = [e for e in examples if (e.task_family, e.semantic_task_id) not in held_set]
        test = [e for e in examples if (e.task_family, e.semantic_task_id) in held_set]
        return BenchmarkSplit("semantic_family", train, test, sorted(held_out))

    elif split_type == "surface_form":
        # Hold out surface template IDs
        all_surfaces = sorted({e.surface_template_id for e in examples})
        held_out = _deterministic_holdout(all_surfaces, holdout_fraction, f"{seed}:surf")
        train = [e for e in examples if e.surface_template_id not in held_out]
        test = [e for e in examples if e.surface_template_id in held_out]
        return BenchmarkSplit("surface_form", train, test, sorted(held_out))

    elif split_type == "reasoning_variant":
        # Hold out reasoning variant IDs — avoid holding out the dominant variant
        # (which would put most data into test)
        variant_counts: dict[str, int] = defaultdict(int)
        for e in examples:
            variant_counts[e.reasoning_variant_id] += 1
        all_variants = sorted(variant_counts)

        if len(all_variants) > 1:
            # Exclude the most common variant from holdout candidates
            dominant = max(all_variants, key=lambda v: variant_counts[v])
            minority = [v for v in all_variants if v != dominant]
            held_out = _deterministic_holdout(minority, holdout_fraction, f"{seed}:var")
            if not held_out:
                held_out = {minority[0]}  # hold out at least one minority variant
        else:
            held_out = set()

        train = [e for e in examples if e.reasoning_variant_id not in held_out]
        test = [e for e in examples if e.reasoning_variant_id in held_out]
        return BenchmarkSplit("reasoning_variant", train, test, sorted(held_out))

    elif split_type == "combination":
        # Hold out unseen (semantic_task_id × surface_template_id) combinations
        # Train includes each semantic ID with *some* surfaces and each surface
        # with *some* semantics, but test gets novel combinations
        combos = sorted({
            (e.semantic_task_id, e.surface_template_id) for e in examples
        })
        keys = [f"{s}:{t}" for s, t in combos]
        held_out = _deterministic_holdout(keys, holdout_fraction, f"{seed}:combo")
        held_set = {tuple(k.split(":", 1)) for k in held_out}

        # Ensure train has at least one example per semantic ID and per surface ID
        train_sems: set[str] = set()
        train_surfs: set[str] = set()
        for e in examples:
            if (e.semantic_task_id, e.surface_template_id) not in held_set:
                train_sems.add(e.semantic_task_id)
                train_surfs.add(e.surface_template_id)

        train = [e for e in examples
                 if (e.semantic_task_id, e.surface_template_id) not in held_set]
        test = [e for e in examples
                if (e.semantic_task_id, e.surface_template_id) in held_set]
        return BenchmarkSplit("combination", train, test, sorted(held_out))

    else:
        raise ValueError(f"Unknown split_type: {split_type!r}. "
                         "Must be one of: semantic_family, surface_form, "
                         "reasoning_variant, combination")


def make_all_splits(
    dataset: BenchmarkDataset,
    holdout_fraction: float = 0.25,
    seed: str = "sdq_v1",
) -> dict[str, BenchmarkSplit]:
    """Create all 4 holdout splits for the benchmark."""
    return {
        st: make_benchmark_splits(dataset, st, holdout_fraction, seed)
        for st in ["semantic_family", "surface_form", "reasoning_variant", "combination"]
    }
