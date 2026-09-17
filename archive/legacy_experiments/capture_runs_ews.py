#!/usr/bin/env python3
"""Capture hidden-state runs with full generation-time activations.

Like capture_runs.py but uses capture_generation_hidden_states() to store
per-token hidden states during autoregressive decoding.  These are required
by the early-warning system (EWS) pipeline.

Usage:
    python capture_runs_ews.py
    python capture_runs_ews.py --benchmark data/prompts/benchmark_v2.json
    python capture_runs_ews.py --limit 50
    python capture_runs_ews.py --family arithmetic
    python capture_runs_ews.py --layers 15 20 25   # capture specific layers only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture SDQ generation-time hidden states")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--runs-dir", default=None,
                   help="Override runs dir (default: config storage.runs_dir)")
    p.add_argument("--model-path", default="models/gemma 2 2B",
                   help="Path to the model checkpoint")
    p.add_argument("--model-name", default="gemma-2-2b",
                   help="Short identifier for output directories")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after capturing this many examples")
    p.add_argument("--family", default=None,
                   help="Only capture examples from this task_family")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Specific transformer layer indices to capture (0-indexed). "
                        "Default: all layers.")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Maximum tokens to generate per example")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def load_benchmark(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("examples", [])


def already_captured(runs_dir: Path, example_id: str) -> bool:
    if not runs_dir.exists():
        return False
    run_dir = runs_dir / example_id
    return (run_dir / "gen_activations.pt").exists()


def main() -> None:
    args = parse_args()

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

    print("Loading model...", flush=True)
    from sdq.instrumentation.model_loader import load_model
    from sdq.instrumentation.hidden_capture import capture_generation_hidden_states
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
    if args.layers:
        print(f"Capturing layers: {args.layers}")
    print()

    captured = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        example_id = ex["example_id"]
        prompt_text = ex["prompt_text"]
        task_family = ex.get("task_family", "unknown")

        if args.skip_existing and already_captured(runs_dir, example_id):
            skipped += 1
            continue

        try:
            result = capture_generation_hidden_states(
                bundle,
                prompt_text,
                max_new_tokens=args.max_new_tokens,
                layers=args.layers,
            )
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

        if captured % 10 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = captured / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            n_gen = result.metadata.get("num_gen_tokens", "?")
            print(
                f"  [{i+1:4d}/{total}]  captured={captured}  "
                f"skipped={skipped}  failed={failed}  "
                f"{rate:.1f}/s  ETA {remaining:.0f}s  "
                f"gen_tokens={n_gen}  [{task_family}] {example_id[:40]}",
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
