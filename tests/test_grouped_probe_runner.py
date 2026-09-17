import json

import torch

from run_sdq_grouped_probe import hashed_char_ngrams, surface_features
from sdq.eval.benchmark_audit import load_excluded_example_ids


def test_hashed_char_ngrams_are_deterministic_and_normalized():
    first = hashed_char_ngrams("Calculate 12 + 3", dim=128)
    second = hashed_char_ngrams("Calculate 12 + 3", dim=128)
    other = hashed_char_ngrams("Calculate 99 + 4", dim=128)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert abs(first.norm().item() - 1.0) < 1e-6


def test_surface_features_have_fixed_dimension():
    assert len(surface_features("What is 2 + 2?")) == 13


def test_exclusion_manifest_lists_only_known_examples():
    benchmark = json.loads(
        open("data/prompts/benchmark_v2.json", encoding="utf-8").read()
    )["examples"]
    known = {example["example_id"] for example in benchmark}
    excluded = load_excluded_example_ids(
        "data/prompts/benchmark_v2_exclusions.json", benchmark
    )
    assert excluded <= known
    assert len(excluded) == 185
