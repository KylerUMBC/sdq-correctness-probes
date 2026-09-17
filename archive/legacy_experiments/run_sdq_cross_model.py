#!/usr/bin/env python3
"""SDQ cross-model validation: capture + Phase 3 probe in one shot.

Runs the full pipeline for a given model:
  1. Capture generation-time hidden states for all benchmark examples
  2. Train commitment probe (Phase 3)
  3. Save results for comparison

Usage:
    # Gemma 3 1B
    python run_sdq_cross_model.py \\
        --config configs/model_gemma3.yaml \\
        --model-name gemma-3-1b

    # Llama 3.2 1B (requires: huggingface-cli login)
    python run_sdq_cross_model.py \\
        --config configs/model_llama.yaml \\
        --model-name llama-3.2-1b

    # Qwen 2.5 1.5B (no license needed)
    python run_sdq_cross_model.py \\
        --model-id Qwen/Qwen2.5-1.5B \\
        --model-name qwen-2.5-1.5b

    # Skip capture if runs already exist
    python run_sdq_cross_model.py \\
        --config configs/model_llama.yaml \\
        --model-name llama-3.2-1b \\
        --skip-capture
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDQ cross-model validation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/model.yaml",
                   help="Model YAML config")
    p.add_argument("--model-id", default=None,
                   help="HuggingFace model ID (overrides config model.name)")
    p.add_argument("--model-name", required=True,
                   help="Short identifier for output dirs and filenames")
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--runs-dir-root", default="data/runs",
                   help="Root runs directory (model runs go under {root}/{model_name}/)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--skip-capture", action="store_true",
                   help="Skip capture step (use existing runs)")
    p.add_argument("--skip-probe", action="store_true",
                   help="Skip Phase 3 probe (capture only)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_capture(args, model_path: str | None, runs_dir: Path):
    """Step 1: Capture generation-time hidden states."""
    from sdq.instrumentation.model_loader import load_model
    from sdq.instrumentation.hidden_capture import capture_generation_hidden_states
    from sdq.instrumentation.run_storage import save_run
    from sdq.labels.outcome_labeler import label_run

    print("=" * 70)
    print(f"STEP 1: CAPTURE RUNS — {args.model_name}")
    print("=" * 70)

    with open(args.benchmark, encoding="utf-8") as f:
        bm_raw = json.load(f)
    examples = bm_raw if isinstance(bm_raw, list) else bm_raw.get("examples", [])
    print(f"Benchmark: {len(examples)} examples")

    bundle = load_model(args.config, model_path_override=model_path)
    print(f"Model: {bundle.model.config._name_or_path}")
    print(f"  Layers: {bundle.num_layers}, Hidden dim: {bundle.hidden_dim}")
    print(f"  Runs dir: {runs_dir}")

    # Verify layer access works
    try:
        _ = bundle.model.model.layers[0]
        print(f"  Layer access: model.model.layers OK")
    except (AttributeError, IndexError) as e:
        print(f"  ERROR: Cannot access model.model.layers: {e}")
        print(f"  This model may use a different architecture convention.")
        sys.exit(1)

    runs_dir.mkdir(parents=True, exist_ok=True)
    captured = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        eid = ex["example_id"]
        run_dir = runs_dir / eid

        if (run_dir / "gen_activations.pt").exists():
            skipped += 1
            continue

        try:
            result = capture_generation_hidden_states(
                bundle, ex["prompt_text"],
                max_new_tokens=args.max_new_tokens,
            )
            result.metadata["model_name"] = args.model_name
            save_run(
                capture=result,
                prompt_id=eid,
                prompt_text=ex["prompt_text"],
                output_dir=runs_dir,
                run_id=eid,
            )
            captured += 1

        except Exception as e:
            print(f"  FAILED [{eid}]: {e}")
            failed += 1
            continue

        if captured % 20 == 0 or i == len(examples) - 1:
            elapsed = time.time() - t0
            rate = captured / elapsed if elapsed > 0 else 0
            eta = (len(examples) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{len(examples)}]  captured={captured}  "
                  f"skipped={skipped}  {rate:.1f}/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nCapture complete: {captured} captured, {skipped} skipped, "
          f"{failed} failed in {elapsed:.0f}s")

    # Quick accuracy check
    correct = 0
    total = 0
    for ex in examples:
        meta_path = runs_dir / ex["example_id"] / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        new_tokens = meta.get("output", {}).get("new_tokens", "")
        outcome = label_run(new_tokens, ex["answer_id"], ex["task_family"])
        if outcome.correct:
            correct += 1
        total += 1
    print(f"Model accuracy: {correct}/{total} = {correct/max(total,1):.1%}")
    if correct < 50 or total - correct < 50:
        print(f"  WARNING: Very imbalanced classes. Probe may not be meaningful.")

    return bundle


def run_phase3(args, runs_dir: Path):
    """Step 2: Run Phase 3 commitment probes."""
    print("\n" + "=" * 70)
    print(f"STEP 2: PHASE 3 COMMITMENT PROBE — {args.model_name}")
    print("=" * 70)

    output_json = f"commitment_results_{args.model_name}.json"
    save_direction = f"commitment_direction_{args.model_name}.pt"

    # Build Phase 3 command args
    phase3_args = [
        "run_sdq_commitment.py",
        "--runs-dir", str(runs_dir),
        "--benchmark", args.benchmark,
        "--output", output_json,
        "--save-direction", save_direction,
        "--seed", str(args.seed),
    ]

    # Import and run Phase 3 directly
    import run_sdq_commitment
    sys.argv = phase3_args
    run_sdq_commitment.main()

    return output_json, save_direction


def print_comparison(model_name: str, results_path: str):
    """Print a summary for cross-model comparison."""
    with open(results_path) as f:
        results = json.load(f)

    meta = results["meta"]
    probes = results["probes"]
    stopping = results["stopping_rule"]

    print(f"\n{'=' * 70}")
    print(f"CROSS-MODEL SUMMARY: {model_name}")
    print(f"{'=' * 70}")
    print(f"  Examples: {meta['n_total']} ({meta['n_correct']} correct, "
          f"{meta['n_incorrect']} incorrect)")
    print(f"  Accuracy: {meta['n_correct']/meta['n_total']:.1%}")
    print(f"  Hidden dim: {meta['hidden_dim']}")

    h0 = probes.get("h_0_linear", {})
    print(f"\n  h_0 linear probe:")
    print(f"    Pooled AUROC:          {h0.get('pooled_auroc', 'N/A'):.4f}")
    print(f"    Within-family AUROC:   {h0.get('mean_within_family_auroc', 'N/A'):.4f}")

    prior = probes.get("family_prior", {})
    print(f"  Family prior:")
    print(f"    Within-family AUROC:   {prior.get('mean_within_family_auroc', 'N/A'):.4f}")

    print(f"\n  Phase 3 PASS: {stopping.get('phase3_pass', 'N/A')}")

    sub = results.get("subspace", {})
    if sub:
        print(f"  Subspace 90% rank:     {sub.get('rank_for_90pct', 'N/A')}")
        print(f"  Subspace 95% rank:     {sub.get('rank_for_95pct', 'N/A')}")

    transfer = results.get("cross_family_transfer_summary", {})
    if transfer:
        print(f"  Cross-family transfer: {transfer.get('mean_auroc', 'N/A'):.4f} "
              f"({transfer.get('n_above_chance', 0)}/{transfer.get('n_families', 0)} "
              f"above chance)")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    t_start = time.time()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = args.model_id or config["model"]["name"]
    runs_dir = Path(args.runs_dir_root) / args.model_name

    print(f"Model: {model_path}")
    print(f"Name:  {args.model_name}")
    print(f"Runs:  {runs_dir}")
    print()

    if not args.skip_capture:
        run_capture(args, model_path, runs_dir)
    else:
        print(f"Skipping capture (using existing runs in {runs_dir})")

    output_json = None
    if not args.skip_probe:
        output_json, _ = run_phase3(args, runs_dir)
        print_comparison(args.model_name, output_json)

    total = time.time() - t_start
    print(f"\nTotal pipeline time: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
