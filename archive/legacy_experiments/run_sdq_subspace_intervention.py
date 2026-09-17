#!/usr/bin/env python3
"""SDQ Phase 4b: Subspace intervention experiment.

Tests whether the commitment SUBSPACE (top-k PCA dimensions) is causally
load-bearing, not just the rank-1 probe direction.

Five perturbation conditions at matched norms:
  1. directed     — probe weight vector (rank-1, the Phase 4 direction)
  2. in_subspace  — random within the top-k PCA subspace
  3. rand_subspace — random within a random k-dim subspace (concentration control)
  4. complement   — random orthogonal to the PCA subspace
  5. full_random  — random in full R^D (Phase 4 control)

Key comparison: in_subspace vs rand_subspace.
  - If in_subspace >> rand_subspace: the PCA subspace is specifically causal.
  - If in_subspace ≈ rand_subspace: the effect is dimensional concentration,
    not subspace-specific.

Near-boundary filtering: prioritizes examples where the probe's P(incorrect)
is in [0.3, 0.7] — the uncertain zone where flips are most plausible.

Leakage controls:
  - Probe and PCA define the perturbation space and select examples.
    Model behavior is measured independently via generation.
  - All conditions tested on the same examples (no selection bias).
  - rand_subspace controls for the concentration effect of lower-dim
    perturbations having more per-dimension energy.

Usage:
    python run_sdq_subspace_intervention.py \\
        --config configs/model.yaml \\
        --runs-dir data/runs/gemma-2-2b/gemma-2-2b \\
        --commitment-direction commitment_direction.pt \\
        --layers 13 19 \\
        --n-prompts 60 --n-random 5
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
    InterventionTrial,
    calibrate_magnitudes,
    compute_pca_basis_raw,
    direction_to_raw_space,
    generate_baseline,
    generate_with_persistent_perturbation,
    make_complement_perturbation,
    make_random_perturbation,
    make_random_subspace,
    make_subspace_perturbation,
    measure_logit_shift,
)
from sdq.eval.intervention_data import (
    collect_h0_and_labels,
    compute_probe_scores,
    load_benchmark,
    select_examples,
)
from sdq.labels.outcome_labeler import label_run


# -- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDQ Phase 4b: Subspace intervention experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/model.yaml",
                   help="Model YAML config for loading Gemma")
    p.add_argument("--model-path", default=None,
                   help="Override model path in config")
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--commitment-direction", default="commitment_direction.pt",
                   help="Path to saved commitment direction from Phase 3")
    p.add_argument("--runs-dir", required=True,
                   help="Directory containing captured generation runs")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Transformer layers to perturb. Default: [half, 3/4].")
    p.add_argument("--subspace-rank", type=int, default=96,
                   help="PCA subspace dimensionality (Phase 3: 90%%=64, 95%%=96)")
    p.add_argument("--n-prompts", type=int, default=60,
                   help="Number of correct prompts to test")
    p.add_argument("--n-random", type=int, default=5,
                   help="Random perturbations per condition per magnitude")
    p.add_argument("--boundary-range", type=float, nargs=2, default=[0.3, 0.7],
                   metavar=("LO", "HI"),
                   help="Probe P(incorrect) range for near-boundary selection")
    p.add_argument("--magnitudes", type=float, nargs="*", default=None,
                   help="Explicit magnitudes. Default: auto at [1, 2, 5]*std.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--h0-source", choices=["prompt_final", "first_gen"],
                   default="prompt_final",
                   help="h_0 definition (prompt_final = last prompt token, "
                        "the state the hook perturbs; first_gen = legacy).")
    p.add_argument("--output", default="subspace_intervention_results.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Data loading/selection lives in sdq.eval.intervention_data (shared with
# the Phase 4/5 runners).


# -- Diagnostics --------------------------------------------------------------

def verify_subspace(
    basis: Tensor,
    direction_raw: Tensor,
    k: int,
) -> dict:
    """Run diagnostic checks on the PCA subspace basis."""
    BtB = basis.T @ basis
    ortho_error = (BtB - torch.eye(k)).abs().max().item()

    proj = basis @ (basis.T @ direction_raw)
    coverage = proj.norm().item()
    cos_with_proj = torch.dot(direction_raw, proj / proj.norm().clamp_min(1e-8)).item()

    return {
        "orthonormality_max_error": ortho_error,
        "probe_direction_coverage": coverage,
        "probe_direction_cosine_with_projection": cos_with_proj,
    }


# -- Sweep --------------------------------------------------------------------

CONDITION_NAMES = ["directed", "in_subspace", "rand_subspace", "complement", "full_random"]


def run_subspace_sweep(
    model, tokenizer, device,
    selected_examples: list[dict],
    direction_raw: Tensor,
    pca_basis: Tensor,
    rand_basis: Tensor,
    hidden_dim: int,
    magnitudes: list[float],
    layer_indices: list[int],
    n_random: int,
    max_new_tokens: int,
    seed: int,
    t_start: float,
) -> dict:
    """Run the 5-condition subspace intervention sweep."""

    all_results = {}

    for layer_idx in layer_indices:
        layer_module = model.model.layers[layer_idx]
        sweep_key = f"layer{layer_idx}_persistent"
        print(f"\n{'=' * 70}")
        print(f"SWEEP: layer={layer_idx}, hook_mode=persistent")
        print(f"{'=' * 70}")

        # Trials organized by condition and magnitude
        trials: dict[str, dict[float, list[InterventionTrial]]] = {
            cond: {m: [] for m in magnitudes} for cond in CONDITION_NAMES
        }

        rng = torch.Generator()
        rng.manual_seed(seed)

        n_total = len(selected_examples)
        for idx, ex in enumerate(selected_examples):
            eid = ex["example_id"]
            prompt_text = ex["prompt_text"]
            gold = ex["answer_id"]
            fam = ex["task_family"]

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            baseline_text, baseline_ids = generate_baseline(
                model, tokenizer, input_ids, attention_mask,
                max_new_tokens=max_new_tokens,
            )
            baseline_outcome = label_run(baseline_text, gold, fam)

            if not baseline_outcome.correct:
                continue

            for mag in magnitudes:
                # --- Condition 1: Directed (rank-1 probe direction) ---
                text, ids, _ = generate_with_persistent_perturbation(
                    model, tokenizer, input_ids, attention_mask,
                    layer_module=layer_module,
                    direction=direction_raw,
                    magnitude=mag,
                    max_new_tokens=max_new_tokens,
                )
                outcome = label_run(text, gold, fam)
                trial = _make_trial(
                    eid, fam, gold, mag, "directed",
                    baseline_text, baseline_outcome, text, outcome,
                )
                trials["directed"][mag].append(trial)

                # --- Conditions 2-5: random perturbations (n_random each) ---
                for _ in range(n_random):
                    for cond_name, pert_fn in [
                        ("in_subspace", lambda: make_subspace_perturbation(
                            pca_basis, mag, rng)),
                        ("rand_subspace", lambda: make_subspace_perturbation(
                            rand_basis, mag, rng)),
                        ("complement", lambda: make_complement_perturbation(
                            pca_basis, hidden_dim, mag, rng)),
                        ("full_random", lambda: make_random_perturbation(
                            hidden_dim, mag, rng)),
                    ]:
                        pert = pert_fn()
                        pert_dir = pert / pert.norm().clamp_min(1e-8)

                        text, ids, _ = generate_with_persistent_perturbation(
                            model, tokenizer, input_ids, attention_mask,
                            layer_module=layer_module,
                            direction=pert_dir,
                            magnitude=mag,
                            max_new_tokens=max_new_tokens,
                        )
                        outcome = label_run(text, gold, fam)
                        trial = _make_trial(
                            eid, fam, gold, mag, cond_name,
                            baseline_text, baseline_outcome, text, outcome,
                        )
                        trials[cond_name][mag].append(trial)

            # Progress
            n_done = idx + 1
            if n_done % 5 == 0 or n_done == n_total:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done:4d}/{n_total}]  "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

        # Compute per-condition, per-magnitude statistics
        layer_results = _compute_sweep_stats(trials, magnitudes)
        _print_sweep_report(layer_idx, magnitudes, layer_results)
        all_results[sweep_key] = layer_results

    return all_results


def _make_trial(
    eid, fam, gold, mag, cond_name,
    baseline_text, baseline_outcome, pert_text, pert_outcome,
) -> InterventionTrial:
    flipped = (pert_outcome.parsed_answer != baseline_outcome.parsed_answer)
    flipped_bad = baseline_outcome.correct and not pert_outcome.correct
    return InterventionTrial(
        example_id=eid,
        task_family=fam,
        gold_answer=gold,
        magnitude=mag,
        perturbation_type=cond_name,
        baseline_text=baseline_text[:200],
        baseline_answer=baseline_outcome.parsed_answer,
        baseline_correct=baseline_outcome.correct,
        perturbed_text=pert_text[:200],
        perturbed_answer=pert_outcome.parsed_answer,
        perturbed_correct=pert_outcome.correct,
        answer_flipped=flipped,
        flipped_to_incorrect=flipped_bad,
    )


def _compute_sweep_stats(
    trials: dict[str, dict[float, list[InterventionTrial]]],
    magnitudes: list[float],
) -> dict:
    """Compute flip rates per condition per magnitude."""
    stats = {}
    for cond in CONDITION_NAMES:
        cond_stats = []
        for mag in magnitudes:
            t_list = trials[cond][mag]
            n = len(t_list) or 1
            flips = sum(1 for t in t_list if t.answer_flipped)
            flips_bad = sum(1 for t in t_list if t.flipped_to_incorrect)
            cond_stats.append({
                "magnitude": mag,
                "n_trials": len(t_list),
                "flips": flips,
                "flips_to_incorrect": flips_bad,
                "flip_rate": flips / n,
                "flip_to_incorrect_rate": flips_bad / n,
            })
        stats[cond] = cond_stats

    # Key comparison: in_subspace vs rand_subspace ratios
    comparisons = []
    for i, mag in enumerate(magnitudes):
        in_rate = stats["in_subspace"][i]["flip_rate"]
        rand_rate = stats["rand_subspace"][i]["flip_rate"]
        comp_rate = stats["complement"][i]["flip_rate"]
        full_rate = stats["full_random"][i]["flip_rate"]
        dir_rate = stats["directed"][i]["flip_rate"]
        eps = 1e-6
        comparisons.append({
            "magnitude": mag,
            "in_subspace_vs_rand_subspace": in_rate / max(rand_rate, eps),
            "in_subspace_vs_complement": in_rate / max(comp_rate, eps),
            "directed_vs_in_subspace": dir_rate / max(in_rate, eps),
            "directed_vs_full_random": dir_rate / max(full_rate, eps),
        })
    stats["comparisons"] = comparisons

    # Serialize trials for JSON
    all_trials = []
    for cond in CONDITION_NAMES:
        for mag in magnitudes:
            for t in trials[cond][mag]:
                all_trials.append({
                    "example_id": t.example_id,
                    "task_family": t.task_family,
                    "gold_answer": t.gold_answer,
                    "magnitude": t.magnitude,
                    "condition": t.perturbation_type,
                    "baseline_answer": t.baseline_answer,
                    "perturbed_answer": t.perturbed_answer,
                    "answer_flipped": t.answer_flipped,
                    "flipped_to_incorrect": t.flipped_to_incorrect,
                })
    stats["trials"] = all_trials

    return stats


def _print_sweep_report(layer_idx: int, magnitudes: list[float], stats: dict):
    """Print a compact comparison table."""
    print(f"\n{'=' * 70}")
    print(f"RESULTS: Layer {layer_idx} (persistent hook)")
    print(f"{'=' * 70}")

    # Header
    mag_headers = "  ".join(f"Mag={m:.2f}" for m in magnitudes)
    print(f"  {'Condition':20s}  {mag_headers}")
    print(f"  {'-' * 20}  " + "  ".join(["-" * 10] * len(magnitudes)))

    for cond in CONDITION_NAMES:
        cells = []
        for ms in stats[cond]:
            n = ms["n_trials"]
            f = ms["flips"]
            rate = ms["flip_rate"]
            cells.append(f"{f}/{n} {rate:.1%}")
        row = "  ".join(f"{c:>10s}" for c in cells)
        print(f"  {cond:20s}  {row}")

    print(f"\n  KEY COMPARISONS:")
    for comp in stats["comparisons"]:
        m = comp["magnitude"]
        ratio = comp["in_subspace_vs_rand_subspace"]
        print(f"    Mag {m:.2f}: in_subspace/rand_subspace = {ratio:.2f}x"
              f"    in_subspace/complement = {comp['in_subspace_vs_complement']:.2f}x"
              f"    directed/in_subspace = {comp['directed_vs_in_subspace']:.2f}x")

    # Interpretation
    ratios = [c["in_subspace_vs_rand_subspace"] for c in stats["comparisons"]]
    mean_ratio = sum(ratios) / len(ratios)
    if mean_ratio > 2.0:
        verdict = "PCA subspace is specifically causal (not just concentration)"
    elif mean_ratio > 1.3:
        verdict = "Weak evidence for PCA subspace specificity"
    else:
        verdict = "No subspace specificity — effect is dimensional concentration or absent"
    print(f"\n  VERDICT: {verdict} (mean ratio = {mean_ratio:.2f}x)")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    t_start = time.time()

    print("=" * 70)
    print("SDQ PHASE 4b: SUBSPACE INTERVENTION EXPERIMENT")
    print("=" * 70)

    # -- Load commitment direction --
    dir_path = Path(args.commitment_direction)
    if not dir_path.exists():
        print(f"ERROR: commitment direction not found: {dir_path}")
        sys.exit(1)

    dir_data = torch.load(dir_path, map_location="cpu", weights_only=True)
    direction_std = dir_data["direction"]
    scaler_mu = dir_data["scaler_mu"]
    scaler_sd = dir_data["scaler_sd"]
    hidden_dim = dir_data["hidden_dim"]
    weight_norm = dir_data["weight_norm"]
    bias = dir_data.get("bias", 0.0)

    if "direction_raw" in dir_data:
        direction_raw = dir_data["direction_raw"]
    else:
        direction_raw = direction_to_raw_space(direction_std, scaler_sd)
    print(f"Loaded commitment direction (dim={hidden_dim})")

    # -- Load benchmark and collect h_0 vectors --
    examples = load_benchmark(args.benchmark)
    runs_dir = Path(args.runs_dir)
    print(f"Loading h_0 vectors from {runs_dir} ...")

    all_examples = collect_h0_and_labels(examples, runs_dir, layer=-1,
                                         source=args.h0_source)
    H0 = torch.stack([e["_h_0"] for e in all_examples])
    n_correct = sum(1 for e in all_examples if e["_correct"])
    n_incorrect = len(all_examples) - n_correct
    print(f"Loaded {len(all_examples)} examples: {n_correct} correct, {n_incorrect} incorrect")

    # -- Compute PCA basis --
    k = args.subspace_rank
    print(f"\nComputing PCA basis (k={k}) in raw h-space ...")
    pca_basis, eigvals = compute_pca_basis_raw(H0, scaler_mu, scaler_sd, k)

    # -- Generate random subspace control --
    rand_rng = torch.Generator().manual_seed(args.seed + 1000)
    rand_basis = make_random_subspace(hidden_dim, k, rand_rng)
    print(f"Generated random {k}-dim subspace for concentration control")

    # -- Diagnostic checks --
    diag = verify_subspace(pca_basis, direction_raw, k)
    print(f"\nDiagnostics:")
    print(f"  PCA basis orthonormality error: {diag['orthonormality_max_error']:.2e}")
    print(f"  Probe direction coverage in PCA subspace: {diag['probe_direction_coverage']:.4f}")
    print(f"    (1.0 = fully in subspace, 0.0 = fully orthogonal)")
    print(f"  Cosine(probe_dir, its projection onto subspace): "
          f"{diag['probe_direction_cosine_with_projection']:.4f}")
    if diag["orthonormality_max_error"] > 1e-4:
        print(f"  WARNING: PCA basis orthonormality error too large!")
        sys.exit(1)

    # Verify complement is truly orthogonal
    test_comp = make_complement_perturbation(pca_basis, hidden_dim, 1.0,
                                              torch.Generator().manual_seed(0))
    comp_proj = (pca_basis.T @ test_comp).abs().max().item()
    print(f"  Complement orthogonality check: max projection = {comp_proj:.2e}")

    # Overlap between PCA and random subspace
    overlap = torch.linalg.svdvals(pca_basis.T @ rand_basis)
    print(f"  PCA-random subspace overlap: top singular value = {overlap[0]:.4f}"
          f" (expected ~{(k/hidden_dim)**0.5:.4f} for random)")

    # -- Compute probe scores and select examples --
    probe_scores = compute_probe_scores(
        H0, direction_std, weight_norm, bias, scaler_mu, scaler_sd,
    )
    lo, hi = args.boundary_range
    selected = select_examples(all_examples, probe_scores, args.n_prompts, lo, hi)

    n_boundary = sum(1 for e in selected if lo <= e["_probe_score"] <= hi)
    print(f"\nSelected {len(selected)} correct examples "
          f"({n_boundary} near-boundary [{lo:.1f}, {hi:.1f}])")

    # Family distribution
    fam_counts: dict[str, int] = {}
    for ex in selected:
        fam_counts[ex["task_family"]] = fam_counts.get(ex["task_family"], 0) + 1
    for fam, cnt in sorted(fam_counts.items()):
        print(f"  {fam:25s}  {cnt}")

    # Probe score distribution for selected examples
    sel_scores = [e["_probe_score"] for e in selected]
    print(f"\n  Probe score stats: "
          f"mean={sum(sel_scores)/len(sel_scores):.3f}  "
          f"min={min(sel_scores):.3f}  max={max(sel_scores):.3f}")

    if not selected:
        print("ERROR: No examples selected. Check --runs-dir.")
        sys.exit(1)

    # -- Calibrate magnitudes --
    h0_selected = torch.stack([e["_h_0"] for e in selected])
    if args.magnitudes:
        magnitudes = args.magnitudes
    else:
        d = direction_raw / direction_raw.norm().clamp_min(1e-8)
        proj = (h0_selected.float() @ d.float())
        std = proj.std().item()
        magnitudes = [std * m for m in [1.0, 2.0, 5.0]]
    print(f"\nMagnitudes: {[f'{m:.2f}' for m in magnitudes]}")

    # -- Load model --
    print("\nLoading model...", flush=True)
    from sdq.instrumentation.model_loader import load_model
    bundle = load_model(args.config, model_path_override=args.model_path)
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    num_layers = model.config.num_hidden_layers

    # -- Determine layers --
    if args.layers is not None:
        layer_indices = [l if l >= 0 else num_layers + l for l in args.layers]
    else:
        layer_indices = [num_layers // 2, 3 * num_layers // 4]
    layer_indices = [max(0, min(l, num_layers - 1)) for l in layer_indices]

    print(f"Layers: {layer_indices} (of {num_layers})")
    n_gens = len(selected) * len(layer_indices) * len(magnitudes) * (
        1 + 4 * args.n_random)
    print(f"Estimated generations: ~{n_gens + len(selected) * len(layer_indices)} "
          f"(including baselines)")

    # -- Run sweep --
    all_results = run_subspace_sweep(
        model, tokenizer, device,
        selected, direction_raw, pca_basis, rand_basis,
        hidden_dim, magnitudes, layer_indices,
        args.n_random, args.max_new_tokens, args.seed, t_start,
    )

    # -- Overall verdict --
    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    best_ratio = 0
    best_key = None
    for key, stats in all_results.items():
        for comp in stats["comparisons"]:
            r = comp["in_subspace_vs_rand_subspace"]
            if r > best_ratio:
                best_ratio = r
                best_key = f"{key} @ mag={comp['magnitude']:.2f}"

    # Check if any condition has meaningful flip rates
    any_flips = False
    for key, stats in all_results.items():
        for cond in CONDITION_NAMES:
            for ms in stats[cond]:
                if ms["flip_rate"] > 0:
                    any_flips = True

    if not any_flips:
        print("No answer flips observed in any condition.")
        print("The perturbation magnitudes may be insufficient to cross")
        print("the model's decision boundary at these layers.")
    elif best_ratio >= 2.0:
        print(f"SUBSPACE IS SPECIFICALLY CAUSAL")
        print(f"  Best in_subspace/rand_subspace ratio: {best_ratio:.2f}x ({best_key})")
        print(f"  The commitment PCA subspace causes more flips than a")
        print(f"  random subspace of the same dimension at matched norms.")
    elif best_ratio >= 1.3:
        print(f"WEAK EVIDENCE for subspace specificity")
        print(f"  Best ratio: {best_ratio:.2f}x ({best_key})")
    else:
        print(f"NO SUBSPACE SPECIFICITY detected")
        print(f"  Best ratio: {best_ratio:.2f}x ({best_key})")
        if any_flips:
            print(f"  Flips occur but are not specific to the PCA subspace —")
            print(f"  the effect is dimensional concentration or noise.")
    print("=" * 70)

    # -- Save results --
    results = {
        "summary": {
            "n_examples": len(selected),
            "n_boundary": n_boundary,
            "boundary_range": [lo, hi],
            "subspace_rank": k,
            "magnitudes": magnitudes,
            "layers": layer_indices,
            "n_random_per_condition": args.n_random,
            "family_counts": fam_counts,
            "best_in_vs_rand_ratio": best_ratio,
            "diagnostics": diag,
            "probe_score_stats": {
                "mean": sum(sel_scores) / len(sel_scores),
                "min": min(sel_scores),
                "max": max(sel_scores),
            },
            "complement_orthogonality_check": comp_proj,
            "pca_rand_overlap_top_sv": overlap[0].item(),
        },
        "per_layer": all_results,
        "args": {k_: str(v) if isinstance(v, Path) else v
                 for k_, v in vars(args).items()},
        "elapsed_seconds": time.time() - t_start,
    }

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
