#!/usr/bin/env python3
"""SDQ Phase 4: Causal intervention on the commitment subspace.

Tests whether the commitment subspace identified in Phase 3 is causally
load-bearing. For each correctly-answered prompt, we:

  1. Generate the baseline (unperturbed) answer
  2. Perturb h at the last prompt token along the commitment direction
     (toward "incorrect") and regenerate
  3. Do the same with N matched-norm random directions as a control
  4. Measure answer-flip rates + logit-shift diagnostics

Pass condition:
    Directed perturbation causes >= 2x the answer-flip rate of
    matched-norm random perturbation at the same magnitude.

Fixes over v1 (which produced 0/3600 flips):
  - De-standardize direction: w_raw[i] = w_std[i] / sd[i] (chain rule)
  - Multi-layer sweep: test middle layers, not just last
  - Persistent hook mode: re-perturb every forward pass to defeat KV washout
  - All families included, not just arithmetic
  - Logit-shift diagnostic for causal evidence even without argmax flips

Usage:
    python run_sdq_commitment_intervention.py \\
        --config configs/model.yaml \\
        --commitment-direction commitment_direction.pt \\
        --layers 6 13 19 25 \\
        --hook-mode both \\
        --n-prompts 80 --n-random 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

from sdq.eval.commitment_intervention import (
    InterventionReport,
    InterventionTrial,
    LogitShiftResult,
    MagnitudeResult,
    calibrate_magnitudes,
    compute_flip_statistics,
    direction_to_raw_space,
    format_intervention_report,
    generate_baseline,
    generate_with_persistent_perturbation,
    generate_with_perturbation,
    make_random_perturbation,
    measure_logit_shift,
)
from sdq.eval.intervention_data import (
    collect_h0_multi_layer,
    load_benchmark,
    load_h0,
    resolve_run_dirs,
)
from sdq.labels.outcome_labeler import label_run


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDQ Phase 4: Commitment subspace causal intervention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/model.yaml",
                   help="Model YAML config for loading Gemma")
    p.add_argument("--model-path", default=None,
                   help="Override model path in config")
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--commitment-direction", default="commitment_direction.pt",
                   help="Path to saved commitment direction from Phase 3")
    p.add_argument("--runs-dir", default=None,
                   help="If given, use h_0 from captured runs to calibrate "
                        "magnitudes. Otherwise uses default multipliers.")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Transformer layers to perturb (0-indexed). "
                        "Default: quarter, half, 3/4, last.")
    p.add_argument("--hook-mode", choices=["prefill", "persistent", "both"],
                   default="both",
                   help="Hook firing mode. 'prefill' fires once; 'persistent' "
                        "fires every step; 'both' runs both.")
    p.add_argument("--n-prompts", type=int, default=80,
                   help="Number of correct prompts to test")
    p.add_argument("--n-random", type=int, default=5,
                   help="Random-direction controls per prompt per magnitude")
    p.add_argument("--magnitudes", type=float, nargs="*", default=None,
                   help="Explicit perturbation magnitudes (absolute, same at "
                        "every layer). If omitted, auto-calibrated PER LAYER "
                        "from the projection variance of that layer's "
                        "prompt-final states.")
    p.add_argument("--h0-source", choices=["prompt_final", "first_gen"],
                   default="prompt_final",
                   help="h_0 definition for magnitude calibration. "
                        "'prompt_final' (default) matches the state the "
                        "prefill hook perturbs; 'first_gen' is the legacy "
                        "definition.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--families", nargs="*", default=None,
                   help="Restrict to specific task families (default: all)")
    p.add_argument("--min-per-family", type=int, default=5,
                   help="Minimum examples per family (ensures diversity)")
    p.add_argument("--output", default="commitment_intervention_results.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -- Data loading -------------------------------------------------------------

def select_correct_prompts(
    examples: list[dict],
    runs_dir: Path | None,
    families: list[str] | None,
    layer_idx: int,
    device: str,
    max_n: int,
    min_per_family: int = 5,
    h0_source: str = "prompt_final",
) -> list[dict]:
    """Select examples that the model answered correctly.

    If runs_dir is provided, use the captured metadata to determine
    correctness (faster, no re-generation needed). Otherwise all examples
    are returned and baseline generation will filter.

    Run directories are resolved by exact name first, then by
    `<eid>_<timestamp>` prefix (shared resolve_run_dirs — the old exact-only
    lookup silently selected nothing on timestamped capture layouts).

    When families is None (default), includes all families and ensures
    at least min_per_family examples per family before filling up to max_n.
    """
    if families:
        examples = [e for e in examples if e.get("task_family") in families]

    if runs_dir is not None and runs_dir.exists():
        dir_map = resolve_run_dirs(runs_dir,
                                   [e["example_id"] for e in examples])
        # Collect all correct examples first
        all_correct = []
        for ex in examples:
            run_dir = dir_map.get(ex["example_id"])
            if run_dir is None:
                continue
            with open(run_dir / "metadata.json", encoding="utf-8") as f:
                meta = json.load(f)
            new_tokens = meta.get("output", {}).get("new_tokens", "")
            outcome = label_run(new_tokens, ex["answer_id"], ex["task_family"])
            if outcome.correct:
                h_0 = load_h0(run_dir, layer=layer_idx, source=h0_source)
                all_correct.append({**ex, "_h_0": h_0, "_run_dir": run_dir})

        # Family-balanced selection: guarantee min_per_family, then fill
        by_family: dict[str, list[dict]] = {}
        for ex in all_correct:
            fam = ex["task_family"]
            by_family.setdefault(fam, []).append(ex)

        selected_ids: set[str] = set()
        selected: list[dict] = []

        # Phase 1: ensure minimum per family
        for fam, fam_exs in by_family.items():
            for ex in fam_exs[:min_per_family]:
                if ex["example_id"] not in selected_ids:
                    selected.append(ex)
                    selected_ids.add(ex["example_id"])

        # Phase 2: fill remaining slots round-robin across families
        if len(selected) < max_n:
            remaining = [ex for ex in all_correct
                         if ex["example_id"] not in selected_ids]
            for ex in remaining:
                selected.append(ex)
                selected_ids.add(ex["example_id"])
                if len(selected) >= max_n:
                    break

        return selected[:max_n]

    # No runs_dir: return all, baseline gen will filter later
    return examples[:max_n * 2]


# -- Main ---------------------------------------------------------------------

def _generate_perturbed(
    model, tokenizer, input_ids, attention_mask,
    layer_module, direction, magnitude, hook_mode, max_new_tokens,
):
    """Generate with perturbation using the specified hook mode.

    Returns (text, token_ids).
    """
    if hook_mode == "persistent":
        text, ids, _ = generate_with_persistent_perturbation(
            model, tokenizer, input_ids, attention_mask,
            layer_module=layer_module,
            direction=direction,
            magnitude=magnitude,
            position=-1,
            max_new_tokens=max_new_tokens,
        )
    else:
        text, ids = generate_with_perturbation(
            model, tokenizer, input_ids, attention_mask,
            layer_module=layer_module,
            direction=direction,
            magnitude=magnitude,
            position=-1,
            max_new_tokens=max_new_tokens,
        )
    return text, ids


def run_layer_sweep(
    model, tokenizer, device,
    correct_examples: list[dict],
    direction: Tensor,
    hidden_dim: int,
    magnitudes_by_layer: dict[int, list[float]],
    proj_stats_by_layer: dict[int, dict],
    layer_indices: list[int],
    hook_modes: list[str],
    n_random: int,
    max_new_tokens: int,
    seed: int,
    t_start: float,
) -> dict:
    """Run the intervention sweep across layers and hook modes.

    Magnitudes are per layer: "1x std" at layer 13 means 1 std of layer-13
    prompt-final projections, not last-layer ones (audit fix — residual
    norms grow with depth, so a single absolute magnitude is a different
    relative perturbation at every layer).

    Returns a dict of all results keyed by (layer_idx, hook_mode).
    """
    all_results = {}

    for layer_idx in layer_indices:
        layer_module = model.model.layers[layer_idx]
        magnitudes = magnitudes_by_layer[layer_idx]
        proj_stats = proj_stats_by_layer[layer_idx]

        for hook_mode in hook_modes:
            sweep_key = f"layer{layer_idx}_{hook_mode}"
            print(f"\n{'=' * 70}")
            print(f"SWEEP: layer={layer_idx}, hook_mode={hook_mode}")
            print(f"{'=' * 70}")

            all_trials: list[InterventionTrial] = []
            directed_by_mag: dict[float, list[InterventionTrial]] = {m: [] for m in magnitudes}
            random_by_mag: dict[float, list[InterventionTrial]] = {m: [] for m in magnitudes}

            # Logit shift diagnostics
            logit_shifts_directed: list[dict] = []
            logit_shifts_random: list[dict] = []
            diag_mag = magnitudes[len(magnitudes) // 2]
            mean_dir_kl = mean_dir_jsd = mean_rnd_kl = mean_rnd_jsd = None
            kl_ratio = None

            rng = torch.Generator()
            rng.manual_seed(seed)

            n_total = len(correct_examples)
            for idx, ex in enumerate(correct_examples):
                eid = ex["example_id"]
                prompt_text = ex["prompt_text"]
                gold = ex["answer_id"]
                fam = ex["task_family"]

                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Baseline generation
                baseline_text, baseline_ids = generate_baseline(
                    model, tokenizer, input_ids, attention_mask,
                    max_new_tokens=max_new_tokens,
                )
                baseline_outcome = label_run(baseline_text, gold, fam)

                if not baseline_outcome.correct:
                    continue

                # Logit-shift diagnostic at one representative magnitude
                try:
                    dir_shift = measure_logit_shift(
                        model, input_ids, attention_mask,
                        layer_module, direction, diag_mag, position=-1,
                    )
                    logit_shifts_directed.append({
                        "example_id": eid, "task_family": fam,
                        "magnitude": diag_mag,
                        "kl": dir_shift.kl_divergence,
                        "jsd": dir_shift.js_divergence,
                        "top1_logit_change": dir_shift.top1_logit_change,
                        "top1_prob_change": dir_shift.top1_prob_change,
                        "top1_rank_change": dir_shift.top1_rank_change,
                    })

                    # One random logit shift for comparison
                    rand_dir_diag = make_random_perturbation(hidden_dim, 1.0, rng)
                    rand_dir_diag = rand_dir_diag / rand_dir_diag.norm()
                    rnd_shift = measure_logit_shift(
                        model, input_ids, attention_mask,
                        layer_module, rand_dir_diag, diag_mag, position=-1,
                    )
                    logit_shifts_random.append({
                        "example_id": eid, "task_family": fam,
                        "magnitude": diag_mag,
                        "kl": rnd_shift.kl_divergence,
                        "jsd": rnd_shift.js_divergence,
                        "top1_logit_change": rnd_shift.top1_logit_change,
                        "top1_prob_change": rnd_shift.top1_prob_change,
                        "top1_rank_change": rnd_shift.top1_rank_change,
                    })
                except Exception as e:
                    print(f"  [logit shift diagnostic failed for {eid}: {e}]")

                for mag in magnitudes:
                    # --- Directed perturbation ---
                    dir_text, dir_ids = _generate_perturbed(
                        model, tokenizer, input_ids, attention_mask,
                        layer_module, direction, mag, hook_mode, max_new_tokens,
                    )
                    dir_outcome = label_run(dir_text, gold, fam)
                    answer_flipped = (dir_outcome.parsed_answer != baseline_outcome.parsed_answer)
                    flipped_bad = baseline_outcome.correct and not dir_outcome.correct

                    trial = InterventionTrial(
                        example_id=eid,
                        task_family=fam,
                        gold_answer=gold,
                        magnitude=mag,
                        perturbation_type="directed",
                        baseline_text=baseline_text[:200],
                        baseline_answer=baseline_outcome.parsed_answer,
                        baseline_correct=baseline_outcome.correct,
                        perturbed_text=dir_text[:200],
                        perturbed_answer=dir_outcome.parsed_answer,
                        perturbed_correct=dir_outcome.correct,
                        answer_flipped=answer_flipped,
                        flipped_to_incorrect=flipped_bad,
                    )
                    all_trials.append(trial)
                    directed_by_mag[mag].append(trial)

                    # --- Random direction controls ---
                    for _ in range(n_random):
                        rand_dir = make_random_perturbation(hidden_dim, 1.0, rng)
                        rand_dir = rand_dir / rand_dir.norm()

                        rnd_text, rnd_ids = _generate_perturbed(
                            model, tokenizer, input_ids, attention_mask,
                            layer_module, rand_dir, mag, hook_mode, max_new_tokens,
                        )
                        rnd_outcome = label_run(rnd_text, gold, fam)
                        rnd_flipped = (rnd_outcome.parsed_answer != baseline_outcome.parsed_answer)
                        rnd_flipped_bad = baseline_outcome.correct and not rnd_outcome.correct

                        rtrial = InterventionTrial(
                            example_id=eid,
                            task_family=fam,
                            gold_answer=gold,
                            magnitude=mag,
                            perturbation_type="random",
                            baseline_text=baseline_text[:200],
                            baseline_answer=baseline_outcome.parsed_answer,
                            baseline_correct=baseline_outcome.correct,
                            perturbed_text=rnd_text[:200],
                            perturbed_answer=rnd_outcome.parsed_answer,
                            perturbed_correct=rnd_outcome.correct,
                            answer_flipped=rnd_flipped,
                            flipped_to_incorrect=rnd_flipped_bad,
                        )
                        all_trials.append(rtrial)
                        random_by_mag[mag].append(rtrial)

                # Progress
                n_done = idx + 1
                if n_done % 5 == 0 or n_done == n_total:
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (n_total - n_done) / rate if rate > 0 else 0
                    print(f"  [{n_done:4d}/{n_total}]  "
                          f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

            # Compute statistics for this sweep
            mag_results: list[MagnitudeResult] = []
            for mag in magnitudes:
                mr = compute_flip_statistics(
                    directed_by_mag[mag], random_by_mag[mag], mag,
                )
                mag_results.append(mr)

            phase4_pass = any(mr.passes_threshold for mr in mag_results)

            report = InterventionReport(
                magnitude_results=mag_results,
                per_trial=[],
                n_prompts=len(correct_examples),
                n_random_per_prompt=n_random,
                magnitudes=magnitudes,
                direction_norm_stats=proj_stats,
                phase4_pass=phase4_pass,
                layer_idx=layer_idx,
                hook_mode=hook_mode,
            )
            print(format_intervention_report(report))

            # Logit shift summary
            if logit_shifts_directed:
                mean_dir_kl = sum(d["kl"] for d in logit_shifts_directed) / len(logit_shifts_directed)
                mean_dir_jsd = sum(d["jsd"] for d in logit_shifts_directed) / len(logit_shifts_directed)
                mean_rnd_kl = sum(d["kl"] for d in logit_shifts_random) / len(logit_shifts_random) if logit_shifts_random else 0
                mean_rnd_jsd = sum(d["jsd"] for d in logit_shifts_random) / len(logit_shifts_random) if logit_shifts_random else 0
                print(f"\n  Logit-shift diagnostic (at magnitude {diag_mag:.2f}):")
                print(f"    Directed:  mean KL={mean_dir_kl:.4f}  mean JSD={mean_dir_jsd:.4f}")
                print(f"    Random:    mean KL={mean_rnd_kl:.4f}  mean JSD={mean_rnd_jsd:.4f}")
                kl_ratio = mean_dir_kl / max(mean_rnd_kl, 1e-8)
                print(f"    KL ratio (directed/random): {kl_ratio:.2f}x")

            # Per-family breakdown
            best_mag_idx = max(range(len(mag_results)),
                               key=lambda i: mag_results[i].flip_ratio)
            best_mag = magnitudes[best_mag_idx]
            fam_trials = {}
            for t in directed_by_mag[best_mag]:
                fam_trials.setdefault(t.task_family, []).append(t)
            print(f"\n  Per-family directed flip rates (at magnitude {best_mag:.2f}):")
            for fam, trials in sorted(fam_trials.items()):
                flips = sum(1 for t in trials if t.answer_flipped)
                bad = sum(1 for t in trials if t.flipped_to_incorrect)
                print(f"    {fam:25s}  {flips}/{len(trials)} flipped "
                      f"({bad} to incorrect)")

            all_results[sweep_key] = {
                "layer_idx": layer_idx,
                "hook_mode": hook_mode,
                "phase4_pass": phase4_pass,
                "magnitude_results": [
                    {
                        "magnitude": mr.magnitude,
                        "directed_flip_rate": mr.directed_flip_rate,
                        "directed_flip_to_incorrect_rate": mr.directed_flip_to_incorrect_rate,
                        "random_flip_rate": mr.random_flip_rate,
                        "random_flip_to_incorrect_rate": mr.random_flip_to_incorrect_rate,
                        "n_directed": mr.n_directed,
                        "n_random": mr.n_random,
                        "flip_ratio": mr.flip_ratio,
                        "passes_threshold": mr.passes_threshold,
                    }
                    for mr in mag_results
                ],
                "per_family_at_best_magnitude": {
                    fam: {
                        "n": len(trials),
                        "flips": sum(1 for t in trials if t.answer_flipped),
                        "flips_to_incorrect": sum(1 for t in trials if t.flipped_to_incorrect),
                    }
                    for fam, trials in sorted(fam_trials.items())
                },
                "logit_shift_diagnostic": {
                    "magnitude": diag_mag if logit_shifts_directed else None,
                    "n_examples": len(logit_shifts_directed),
                    "directed_mean_kl": mean_dir_kl if logit_shifts_directed else None,
                    "directed_mean_jsd": mean_dir_jsd if logit_shifts_directed else None,
                    "random_mean_kl": mean_rnd_kl if logit_shifts_random else None,
                    "random_mean_jsd": mean_rnd_jsd if logit_shifts_random else None,
                    "kl_ratio": kl_ratio if logit_shifts_directed else None,
                    "per_example_directed": logit_shifts_directed,
                    "per_example_random": logit_shifts_random,
                },
                "trials": [
                    {
                        "example_id": t.example_id,
                        "task_family": t.task_family,
                        "gold_answer": t.gold_answer,
                        "magnitude": t.magnitude,
                        "perturbation_type": t.perturbation_type,
                        "baseline_answer": t.baseline_answer,
                        "perturbed_answer": t.perturbed_answer,
                        "perturbed_correct": t.perturbed_correct,
                        "answer_flipped": t.answer_flipped,
                        "flipped_to_incorrect": t.flipped_to_incorrect,
                    }
                    for t in all_trials
                ],
            }

    return all_results


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    t_start = time.time()

    print("=" * 70)
    print("SDQ PHASE 4: CAUSAL INTERVENTION ON COMMITMENT SUBSPACE (v2)")
    print("=" * 70)

    # -- Load commitment direction --
    dir_path = Path(args.commitment_direction)
    if not dir_path.exists():
        print(f"ERROR: commitment direction not found: {dir_path}")
        print("Run run_sdq_commitment.py first (Phase 3).")
        sys.exit(1)

    dir_data = torch.load(dir_path, map_location="cpu", weights_only=True)
    direction_std = dir_data["direction"]       # [hidden_dim] unit vector in z-space
    hidden_dim = dir_data["hidden_dim"]
    scaler_mu = dir_data["scaler_mu"]           # [1, hidden_dim]
    scaler_sd = dir_data["scaler_sd"]           # [1, hidden_dim]

    # FIX: De-standardize direction to raw h-space
    # If Phase 3 already saved direction_raw, use it directly
    if "direction_raw" in dir_data:
        direction = dir_data["direction_raw"]
        print(f"Using pre-computed raw-space direction (dim={hidden_dim})")
    else:
        direction = direction_to_raw_space(direction_std, scaler_sd)
        print(f"De-standardized direction to raw h-space (dim={hidden_dim})")

    # Show how much the transform changed the direction
    cos_sim = torch.dot(direction, direction_std).item()
    print(f"  Std-space vs raw-space cosine similarity: {cos_sim:.4f}")
    print(f"  (1.0 means identical, lower means the transform matters)")

    # -- Load benchmark --
    examples = load_benchmark(args.benchmark)
    print(f"Benchmark: {len(examples)} examples")

    runs_dir = Path(args.runs_dir) if args.runs_dir else None

    # -- Select correct prompts (all families by default) --
    # Use layer -1 for h_0 loading (calibration only, layer sweep is separate)
    correct_examples = select_correct_prompts(
        examples, runs_dir, args.families, -1,
        "cpu", args.n_prompts, args.min_per_family,
        h0_source=args.h0_source,
    )
    print(f"Selected {len(correct_examples)} correct prompts for intervention "
          f"(h0_source={args.h0_source})")

    # Show family distribution
    fam_counts: dict[str, int] = {}
    for ex in correct_examples:
        fam_counts[ex["task_family"]] = fam_counts.get(ex["task_family"], 0) + 1
    for fam, cnt in sorted(fam_counts.items()):
        print(f"  {fam:25s}  {cnt}")

    if not correct_examples:
        print("ERROR: No correct prompts found. Check --runs-dir or --families.")
        sys.exit(1)

    # -- Load model --
    print("\nLoading model...", flush=True)
    from sdq.instrumentation.model_loader import load_model
    bundle = load_model(args.config, model_path_override=args.model_path)
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    num_layers = model.config.num_hidden_layers

    # -- Determine layer sweep --
    if args.layers is not None:
        layer_indices = [l if l >= 0 else num_layers + l for l in args.layers]
    else:
        # Default: quarter, half, 3/4, last
        layer_indices = [
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1,
        ]
    layer_indices = [max(0, min(l, num_layers - 1)) for l in layer_indices]

    # -- Calibrate magnitudes PER LAYER (audit fix) --
    # "1x std" at layer L must mean 1 std of layer-L prompt-final projections
    # onto the direction; calibrating once at the last layer made the same
    # absolute magnitude a different relative perturbation at every layer.
    magnitudes_by_layer: dict[int, list[float]] = {}
    proj_stats_by_layer: dict[int, dict] = {}
    can_calibrate = (runs_dir is not None
                     and all(ex.get("_run_dir") is not None
                             for ex in correct_examples))
    if args.magnitudes:
        for l in layer_indices:
            magnitudes_by_layer[l] = list(args.magnitudes)
            proj_stats_by_layer[l] = {"mean": 0.0, "std": 0.0}
        print(f"Using explicit magnitudes at every layer: {args.magnitudes}")
    elif can_calibrate and correct_examples:
        h0_multi = collect_h0_multi_layer(correct_examples, layer_indices)
        for l in layer_indices:
            H = h0_multi[l]
            magnitudes_by_layer[l] = calibrate_magnitudes(H, direction)
            proj = (H.float() @ direction.float())
            proj_stats_by_layer[l] = {"mean": proj.mean().item(),
                                      "std": proj.std().item()}
            print(f"  layer {l:2d}: proj std={proj_stats_by_layer[l]['std']:.3f}  "
                  f"magnitudes={[f'{m:.2f}' for m in magnitudes_by_layer[l]]}")
    else:
        default_mags = [1.0, 2.0, 5.0, 10.0, 20.0]
        for l in layer_indices:
            magnitudes_by_layer[l] = default_mags
            proj_stats_by_layer[l] = {"mean": 0.0, "std": 1.0}
        print(f"No h_0 available for calibration, using default magnitudes: "
              f"{default_mags}")

    # -- Determine hook modes --
    if args.hook_mode == "both":
        hook_modes = ["prefill", "persistent"]
    else:
        hook_modes = [args.hook_mode]

    print(f"Layers to sweep: {layer_indices} (of {num_layers})")
    print(f"Hook modes: {hook_modes}")
    print(f"Prompts: {len(correct_examples)}, "
          f"random controls/prompt/magnitude: {args.n_random}")

    # -- Run the sweep --
    all_results = run_layer_sweep(
        model, tokenizer, device,
        correct_examples, direction, hidden_dim, magnitudes_by_layer,
        proj_stats_by_layer, layer_indices, hook_modes,
        args.n_random, args.max_new_tokens, args.seed, t_start,
    )

    # -- Overall verdict --
    any_pass = any(r["phase4_pass"] for r in all_results.values())

    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)
    if any_pass:
        passing = [(k, r) for k, r in all_results.items() if r["phase4_pass"]]
        print(f"Phase 4 PASSES ({len(passing)} configuration(s) pass):")
        for k, r in passing:
            best = max(r["magnitude_results"], key=lambda x: x["flip_ratio"])
            print(f"  {k}: best flip ratio {best['flip_ratio']:.2f}x at magnitude {best['magnitude']:.2f}")
        print("  The commitment subspace is causally load-bearing.")
    else:
        # Check logit shift as softer evidence
        best_kl_ratio = 0
        best_kl_key = None
        for k, r in all_results.items():
            kr = r.get("logit_shift_diagnostic", {}).get("kl_ratio")
            if kr is not None and kr > best_kl_ratio:
                best_kl_ratio = kr
                best_kl_key = k

        print("Phase 4 FAILS the flip-rate threshold.")
        if best_kl_ratio > 1.5:
            print(f"  However, logit-shift diagnostic shows directed perturbations")
            print(f"  move logits {best_kl_ratio:.2f}x more than random ({best_kl_key}).")
            print(f"  The subspace has causal relevance but insufficient strength")
            print(f"  to flip argmax answers under greedy decoding.")
        else:
            print(f"  Logit-shift diagnostic also shows no directional advantage.")
            print(f"  The subspace is correlational but not mechanistically central.")
    print("=" * 70)

    # -- Save results --
    results = {
        "summary": {
            "phase4_pass": any_pass,
            "n_prompts": len(correct_examples),
            "n_random_per_prompt": args.n_random,
            "magnitudes_by_layer": {str(l): m for l, m in magnitudes_by_layer.items()},
            "direction_projection_stats_by_layer": {
                str(l): s for l, s in proj_stats_by_layer.items()},
            "h0_source": args.h0_source,
            "layers_tested": layer_indices,
            "hook_modes_tested": hook_modes,
            "direction_std_vs_raw_cosine": cos_sim,
            "family_counts": fam_counts,
        },
        "per_configuration": all_results,
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
        "elapsed_seconds": time.time() - t_start,
    }

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
