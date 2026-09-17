#!/usr/bin/env python3
"""Capture hidden-state runs for all benchmark examples.

Runs every prompt in data/prompts/benchmark_v1.json through a model
and saves activations + metadata to data/runs/{model_name}/.

Skips prompts that already have a saved run (safe to re-run after interruption).

Usage:
    python capture_runs.py
    python capture_runs.py --benchmark data/prompts/benchmark_v1.json
    python capture_runs.py --limit 50          # capture first N examples only
    python capture_runs.py --family arithmetic  # single task family
    python capture_runs.py --model-path models/gemma-2-9b --model-name gemma-2-9b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture SDQ hidden-state runs")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--runs-dir", default=None,
                   help="Override runs dir from config (default: config storage.runs_dir)")
    p.add_argument("--model-path", default=None,
                   help="Optional local checkpoint path (otherwise use the pinned model in config)")
    p.add_argument("--model-name", default="gemma-2-2b",
                   help="Short identifier used when naming output directories "
                        "(runs saved under {runs_dir}/{model_name}/{example_id}/)")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after capturing this many examples")
    p.add_argument("--family", default=None,
                   help="Only capture examples from this task_family")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip examples that already have a saved run (default: True)")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def load_benchmark(path: str | Path) -> list[dict]:
    """Load benchmark JSON. Handles both flat list and {'examples': [...]} formats."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("examples", [])


def already_captured(runs_dir: Path, example_id: str) -> bool:
    """Return True if any run directory starts with this example_id."""
    return any(runs_dir.iterdir()) and any(
        d.name == example_id or d.name.startswith(f"{example_id}_")
        for d in runs_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )


def main() -> None:
    args = parse_args()

    # ── Load benchmark ────────────────────────────────────────────────────────
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"ERROR: benchmark file not found: {benchmark_path}")
        sys.exit(1)

    examples = load_benchmark(benchmark_path)

    if args.family:
        examples = [e for e in examples if e.get("task_family") == args.family]
        print(f"Filtered to family '{args.family}': {len(examples)} examples")

    if args.limit:
        examples = examples[: args.limit]

    total = len(examples)
    print(f"Benchmark: {benchmark_path} — {total} examples to capture")

    # ── Load model (deferred so --help is fast) ───────────────────────────────
    print("Loading model...", flush=True)
    from sdq.instrumentation.model_loader import load_model
    from sdq.instrumentation.hidden_capture import capture_hidden_states
    from sdq.instrumentation.run_storage import save_run
    import yaml

    bundle = load_model(args.config, model_path_override=args.model_path)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = args.model_name
    runs_dir = Path(args.runs_dir or config["storage"]["runs_dir"]) / model_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {bundle.model.config._name_or_path}  "
          f"({bundle.num_layers} layers, hidden={bundle.hidden_dim})")
    print(f"Model name: {model_name}")
    print(f"Runs dir: {runs_dir}")
    print()

    # ── Capture loop ──────────────────────────────────────────────────────────
    captured = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        example_id = ex["example_id"]
        prompt_text = ex["prompt_text"]
        task_family = ex.get("task_family", "unknown")

        # Skip check
        if args.skip_existing and runs_dir.exists() and already_captured(runs_dir, example_id):
            skipped += 1
            continue

        try:
            result = capture_hidden_states(bundle, prompt_text)
            result.metadata["model_name"] = model_name
            save_run(
                capture=result,
                prompt_id=example_id,
                prompt_text=prompt_text,
                output_dir=runs_dir,
                run_id=example_id,
            )
            captured += 1

        except Exception as e:
            print(f"  FAILED [{example_id}]: {e}")
            failed += 1
            continue

        # Progress every 10 or at end
        if captured % 10 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = captured / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:4d}/{total}]  captured={captured}  "
                f"skipped={skipped}  failed={failed}  "
                f"{rate:.1f}/s  ETA {remaining:.0f}s  "
                f"[{task_family}] {example_id[:40]}",
                flush=True,
            )

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Captured : {captured}")
    print(f"  Skipped  : {skipped}  (already existed)")
    print(f"  Failed   : {failed}")
    print(f"  Model    : {model_name}")
    print(f"  Runs dir : {runs_dir}")


if __name__ == "__main__":
    main()
