"""Collect hidden-state runs for all benchmark examples.

Loads the benchmark JSON, runs each prompt through the model, and saves
hidden-state trajectories in the standard run format under data/runs/.

Usage:
    python collect_benchmark_runs.py [--benchmark data/prompts/benchmark_v1.json]
                                     [--output-dir data/runs]
                                     [--resume]
                                     [--batch-size 1]
                                     [--max-examples 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect benchmark hidden-state runs")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/prompts/benchmark_v1.json"),
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/runs"),
        help="Output directory for run artifacts",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip examples that already have a run directory",
    )
    parser.add_argument(
        "--max-examples", type=int, default=0,
        help="Max examples to process (0 = all)",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/model.yaml"),
        help="Model config YAML",
    )
    args = parser.parse_args()

    # Load benchmark
    with open(args.benchmark, encoding="utf-8") as f:
        data = json.load(f)
    examples = data["examples"]
    print(f"Loaded {len(examples)} benchmark examples from {args.benchmark}")

    # Find which examples already have runs (for --resume)
    existing_ids: set[str] = set()
    if args.resume and args.output_dir.exists():
        for d in args.output_dir.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                # Extract prompt_id from directory name: {prompt_id}_{timestamp}
                parts = d.name.rsplit("_", 2)
                if len(parts) >= 3:
                    prompt_id = "_".join(parts[:-2])
                    existing_ids.add(prompt_id)
        print(f"Found {len(existing_ids)} existing runs (will skip)")

    # Filter to examples that need runs
    todo = [ex for ex in examples if ex["example_id"] not in existing_ids]
    if args.max_examples > 0:
        todo = todo[:args.max_examples]
    print(f"Will collect {len(todo)} runs")

    if not todo:
        print("Nothing to do!")
        return

    # Late import so benchmark generation doesn't require GPU
    from sdq.instrumentation.hidden_capture import capture_hidden_states
    from sdq.instrumentation.model_loader import load_model
    from sdq.instrumentation.run_storage import save_run

    print("Loading model...")
    bundle = load_model(args.config)
    print(f"Model loaded: {bundle.model.config._name_or_path} on {bundle.device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, ex in enumerate(todo):
        example_id = ex["example_id"]
        prompt_text = ex["prompt_text"]

        print(f"[{i + 1}/{len(todo)}] {example_id}: {prompt_text[:60]}...")

        capture = capture_hidden_states(bundle, prompt_text)
        run_dir = save_run(
            capture,
            prompt_id=example_id,
            prompt_text=prompt_text,
            output_dir=args.output_dir,
        )

        # Save benchmark metadata alongside the standard run metadata
        bench_meta_path = run_dir / "benchmark_meta.json"
        with open(bench_meta_path, "w", encoding="utf-8") as f:
            json.dump(ex, f, indent=2, ensure_ascii=False)

        print(f"  -> {run_dir.name} [{capture.activations.shape}]")

    print(f"\nDone! Collected {len(todo)} runs in {args.output_dir}")


if __name__ == "__main__":
    main()
