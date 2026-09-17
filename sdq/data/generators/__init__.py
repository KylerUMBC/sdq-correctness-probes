"""Benchmark data generators for the factorized SDQ benchmark.

Primary entry points:
    assemble_benchmark() -> BenchmarkDataset
    print_benchmark_stats(dataset) -> str
    make_benchmark_splits(dataset, split_type) -> BenchmarkSplit
    make_all_splits(dataset) -> dict[str, BenchmarkSplit]
"""

from sdq.data.generators.assembly import assemble_benchmark, print_benchmark_stats
from sdq.data.generators.benchmark_splits import (
    BenchmarkSplit,
    make_all_splits,
    make_benchmark_splits,
)
from sdq.data.generators.schema import BenchmarkDataset, BenchmarkExample, EventSpan

__all__ = [
    "assemble_benchmark",
    "print_benchmark_stats",
    "make_benchmark_splits",
    "make_all_splits",
    "BenchmarkDataset",
    "BenchmarkExample",
    "BenchmarkSplit",
    "EventSpan",
]
