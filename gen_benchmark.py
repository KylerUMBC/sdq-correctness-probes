"""Generate the SDQ factorized benchmark and save to data/prompts/.

Usage:
    python gen_benchmark.py [--output data/prompts/benchmark_v1.json] [--stats]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdq.data.generators import (
    assemble_benchmark,
    make_all_splits,
    print_benchmark_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SDQ factorized benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompts/benchmark_v1.json"),
        help="Output path for the benchmark JSON",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/prompts/splits"),
        help="Output directory for holdout split manifests",
    )
    parser.add_argument("--stats", action="store_true", help="Print benchmark stats")
    args = parser.parse_args()

    print("Assembling benchmark...")
    dataset = assemble_benchmark()

    # Save full benchmark
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dataset.examples)} examples to {args.output}")

    # Save split manifests
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    splits = make_all_splits(dataset)
    for split_name, split in splits.items():
        manifest = {
            "split_type": split.split_type,
            "train_ids": [e.example_id for e in split.train],
            "test_ids": [e.example_id for e in split.test],
            "held_out_keys": split.held_out_keys,
            "train_size": split.train_size,
            "test_size": split.test_size,
        }
        path = args.splits_dir / f"split_{split_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Split '{split_name}': train={split.train_size}, test={split.test_size} -> {path}")

    if args.stats:
        print()
        print(print_benchmark_stats(dataset))


if __name__ == "__main__":
    main()
