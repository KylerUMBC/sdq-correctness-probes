#!/usr/bin/env python3
"""SDQ Mode Intervention: Spectral decomposition + targeted perturbation.

Tests whether decomposing the commitment subspace into discriminative vs
non-discriminative PCA modes reveals mesoscale causal structure that the
monolithic subspace perturbation (Phase 4b) missed.

Two-stage experiment:
  Stage 1 (CPU-only): Spectral mode analysis
    - Decompose h_0 PCA into modes ranked by discriminability (Cohen's d)
    - Analyze inter-mode interference (correlation differences by class)
    - Identify discriminative mode clusters
    - Run with --analysis-only to skip model loading

  Stage 2 (GPU): Mode-selective interventions
    Four perturbation conditions at matched magnitudes:
    - sign_flip:       Per-example negation of discriminative mode projections
    - random_disc:     Random direction within discriminative mode subspace
    - random_non_disc: Random direction within non-discriminative modes
    - full_random:     Random direction in full R^D (control)

    Key comparisons:
    - sign_flip vs random_disc: Does example-specific phase matter?
    - random_disc vs random_non_disc: Are discriminative modes causally privileged?
    - random_disc vs full_random: Does mode selection beat unstructured noise?

Usage:
    # Analysis only (no GPU needed)
    python run_sdq_mode_intervention.py \\
        --runs-dir data/runs/gemma-2-2b/gemma-2-2b \\
        --commitment-direction commitment_direction.pt \\
        --analysis-only

    # Full experiment
    python run_sdq_mode_intervention.py \\
        --config configs/model.yaml \\
        --runs-dir data/runs/gemma-2-2b/gemma-2-2b \\
        --commitment-direction commitment_direction.pt \\
        --layers 13 19
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
    direction_to_raw_space,
    generate_baseline,
    generate_with_persistent_perturbation,
    make_random_perturbation,
    make_subspace_perturbation,
)
from sdq.eval.mode_analysis import (
    ModeAnalysisResult,
    build_selective_basis_raw,
    build_sign_flip_perturbation,
    format_mode_analysis,
    full_mode_analysis,
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
        description="SDQ Mode Intervention: spectral decomposition + targeted perturbation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--model-path", default=None)
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--commitment-direction", default="commitment_direction.pt")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Layers to perturb. Default: [half, 3/4].")
    p.add_argument("--subspace-rank", type=int, default=96)
    p.add_argument("--disc-threshold", type=float, default=0.3,
                   help="|Cohen's d| threshold for discriminative modes")
    p.add_argument("--n-prompts", type=int, default=60)
    p.add_argument("--n-random", type=int, default=5,
                   help="Random trials per condition per magnitude")
    p.add_argument("--boundary-range", type=float, nargs=2, default=[0.3, 0.7])
    p.add_argument("--magnitudes", type=float, nargs="*", default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--h0-source", choices=["prompt_final", "first_gen"],
                   default="prompt_final",
                   help="h_0 definition (prompt_final = last prompt token, "
                        "the state the hook perturbs; first_gen = legacy).")
    p.add_argument("--output", default="mode_intervention_results.json")
    p.add_argument("--analysis-only", action="store_true",
                   help="Run spectral analysis only (no model / GPU needed)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Data loading/selection lives in sdq.eval.intervention_data (shared with
# the Phase 4/5 runners).


# -- Intervention sweep -------------------------------------------------------

CONDITION_NAMES = ["sign_flip", "random_disc", "random_non_disc", "full_random"]


def run_mode_sweep(
    model, tokenizer, device,
    selected_examples: list[dict],
    analysis: ModeAnalysisResult,
    direction_raw: Tensor,
    hidden_dim: int,
    magnitudes: list[float],
    layer_indices: list[int],
    n_random: int,
    max_new_tokens: int,
    scaler_mu: Tensor,
    scaler_sd: Tensor,
    seed: int,
    t_start: float,
) -> dict:
    """Run the 4-condition mode-selective intervention sweep."""

    eigvecs_z = analysis.decomposition.eigenvectors_z
    top_modes = analysis.top_modes
    bottom_modes = analysis.bottom_modes

    disc_basis_raw = build_selective_basis_raw(eigvecs_z, top_modes, scaler_sd)
    if bottom_modes:
        non_disc_basis_raw = build_selective_basis_raw(
            eigvecs_z, bottom_modes, scaler_sd)
    else:
        non_disc_basis_raw = None

    print(f"\nDiscriminative basis: {disc_basis_raw.shape[1]} dims")
    if non_disc_basis_raw is not None:
        print(f"Non-discriminative basis: {non_disc_basis_raw.shape[1]} dims")

    all_results = {}

    for layer_idx in layer_indices:
        layer_module = model.model.layers[layer_idx]
        sweep_key = f"layer{layer_idx}_persistent"
        print(f"\n{'=' * 70}")
        print(f"SWEEP: layer={layer_idx}, hook_mode=persistent")
        print(f"  Conditions: {CONDITION_NAMES}")
        print(f"  Magnitudes: {[f'{m:.2f}' for m in magnitudes]}")
        print(f"{'=' * 70}")

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

            h0 = ex["_h_0"]

            for mag in magnitudes:
                # -- sign_flip: per-example, deterministic --
                sf_pert = build_sign_flip_perturbation(
                    h0, eigvecs_z, top_modes, scaler_mu, scaler_sd,
                    target_magnitude=mag,
                )
                sf_dir = sf_pert / sf_pert.norm().clamp_min(1e-8)
                text, _, _ = generate_with_persistent_perturbation(
                    model, tokenizer, input_ids, attention_mask,
                    layer_module=layer_module,
                    direction=sf_dir, magnitude=mag,
                    max_new_tokens=max_new_tokens,
                )
                outcome = label_run(text, gold, fam)
                trials["sign_flip"][mag].append(_make_trial(
                    eid, fam, gold, mag, "sign_flip",
                    baseline_text, baseline_outcome, text, outcome,
                ))

                # -- random conditions: n_random trials each --
                for _ in range(n_random):
                    # random_disc
                    pert = make_subspace_perturbation(disc_basis_raw, mag, rng)
                    pert_dir = pert / pert.norm().clamp_min(1e-8)
                    text, _, _ = generate_with_persistent_perturbation(
                        model, tokenizer, input_ids, attention_mask,
                        layer_module=layer_module,
                        direction=pert_dir, magnitude=mag,
                        max_new_tokens=max_new_tokens,
                    )
                    outcome = label_run(text, gold, fam)
                    trials["random_disc"][mag].append(_make_trial(
                        eid, fam, gold, mag, "random_disc",
                        baseline_text, baseline_outcome, text, outcome,
                    ))

                    # random_non_disc
                    if non_disc_basis_raw is not None:
                        pert = make_subspace_perturbation(
                            non_disc_basis_raw, mag, rng)
                    else:
                        pert = make_random_perturbation(hidden_dim, mag, rng)
                    pert_dir = pert / pert.norm().clamp_min(1e-8)
                    text, _, _ = generate_with_persistent_perturbation(
                        model, tokenizer, input_ids, attention_mask,
                        layer_module=layer_module,
                        direction=pert_dir, magnitude=mag,
                        max_new_tokens=max_new_tokens,
                    )
                    outcome = label_run(text, gold, fam)
                    trials["random_non_disc"][mag].append(_make_trial(
                        eid, fam, gold, mag, "random_non_disc",
                        baseline_text, baseline_outcome, text, outcome,
                    ))

                    # full_random
                    pert = make_random_perturbation(hidden_dim, mag, rng)
                    pert_dir = pert / pert.norm().clamp_min(1e-8)
                    text, _, _ = generate_with_persistent_perturbation(
                        model, tokenizer, input_ids, attention_mask,
                        layer_module=layer_module,
                        direction=pert_dir, magnitude=mag,
                        max_new_tokens=max_new_tokens,
                    )
                    outcome = label_run(text, gold, fam)
                    trials["full_random"][mag].append(_make_trial(
                        eid, fam, gold, mag, "full_random",
                        baseline_text, baseline_outcome, text, outcome,
                    ))

            n_done = idx + 1
            if n_done % 5 == 0 or n_done == n_total:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done:4d}/{n_total}]  "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

        layer_results = _compute_sweep_stats(trials, magnitudes)
        _print_sweep_report(layer_idx, magnitudes, layer_results)
        all_results[sweep_key] = layer_results

    return all_results


def _make_trial(eid, fam, gold, mag, cond_name,
                baseline_text, baseline_outcome, pert_text, pert_outcome):
    flipped = (pert_outcome.parsed_answer != baseline_outcome.parsed_answer)
    flipped_bad = baseline_outcome.correct and not pert_outcome.correct
    return InterventionTrial(
        example_id=eid, task_family=fam, gold_answer=gold,
        magnitude=mag, perturbation_type=cond_name,
        baseline_text=baseline_text[:200],
        baseline_answer=baseline_outcome.parsed_answer,
        baseline_correct=baseline_outcome.correct,
        perturbed_text=pert_text[:200],
        perturbed_answer=pert_outcome.parsed_answer,
        perturbed_correct=pert_outcome.correct,
        answer_flipped=flipped, flipped_to_incorrect=flipped_bad,
    )


def _compute_sweep_stats(trials, magnitudes):
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
                "flip_rate": flips / n,
                "flips_to_incorrect": flips_bad,
            })
        stats[cond] = cond_stats

    comparisons = []
    for i, mag in enumerate(magnitudes):
        eps = 1e-6
        sf = stats["sign_flip"][i]["flip_rate"]
        rd = stats["random_disc"][i]["flip_rate"]
        rnd = stats["random_non_disc"][i]["flip_rate"]
        fr = stats["full_random"][i]["flip_rate"]
        comparisons.append({
            "magnitude": mag,
            "sign_flip_vs_random_disc": sf / max(rd, eps),
            "random_disc_vs_random_non_disc": rd / max(rnd, eps),
            "random_disc_vs_full_random": rd / max(fr, eps),
            "sign_flip_vs_full_random": sf / max(fr, eps),
        })
    stats["comparisons"] = comparisons

    all_trials_serial = []
    for cond in CONDITION_NAMES:
        for mag in magnitudes:
            for t in trials[cond][mag]:
                all_trials_serial.append({
                    "example_id": t.example_id,
                    "task_family": t.task_family,
                    "magnitude": t.magnitude,
                    "condition": t.perturbation_type,
                    "baseline_answer": t.baseline_answer,
                    "perturbed_answer": t.perturbed_answer,
                    "answer_flipped": t.answer_flipped,
                    "flipped_to_incorrect": t.flipped_to_incorrect,
                })
    stats["trials"] = all_trials_serial
    return stats


def _print_sweep_report(layer_idx, magnitudes, stats):
    print(f"\n{'=' * 70}")
    print(f"RESULTS: Layer {layer_idx} (persistent hook)")
    print(f"{'=' * 70}")

    mag_headers = "  ".join(f"Mag={m:.2f}" for m in magnitudes)
    print(f"  {'Condition':20s}  {mag_headers}")
    print(f"  {'-' * 20}  " + "  ".join(["-" * 10] * len(magnitudes)))

    for cond in CONDITION_NAMES:
        cells = []
        for ms in stats[cond]:
            n = ms["n_trials"]
            f_ = ms["flips"]
            rate = ms["flip_rate"]
            cells.append(f"{f_}/{n} {rate:.1%}")
        row = "  ".join(f"{c:>10s}" for c in cells)
        print(f"  {cond:20s}  {row}")

    print(f"\n  KEY COMPARISONS:")
    for comp in stats["comparisons"]:
        m = comp["magnitude"]
        print(f"    Mag {m:.2f}: "
              f"sign_flip/rand_disc = {comp['sign_flip_vs_random_disc']:.2f}x  "
              f"rand_disc/rand_non_disc = {comp['random_disc_vs_random_non_disc']:.2f}x  "
              f"rand_disc/full_random = {comp['random_disc_vs_full_random']:.2f}x")

    # Verdict
    disc_ratios = [c["random_disc_vs_random_non_disc"] for c in stats["comparisons"]]
    sf_ratios = [c["sign_flip_vs_random_disc"] for c in stats["comparisons"]]
    mean_disc = sum(disc_ratios) / len(disc_ratios)
    mean_sf = sum(sf_ratios) / len(sf_ratios)

    print(f"\n  VERDICT:")
    if mean_disc > 2.0:
        print(f"  Discriminative modes are causally privileged (mean ratio = {mean_disc:.2f}x)")
    elif mean_disc > 1.3:
        print(f"  Weak evidence for discriminative mode privilege (mean ratio = {mean_disc:.2f}x)")
    else:
        print(f"  No discriminative mode privilege (mean ratio = {mean_disc:.2f}x)")

    if mean_sf > 2.0:
        print(f"  Per-example phase targeting helps (mean ratio = {mean_sf:.2f}x)")
    elif mean_sf > 1.3:
        print(f"  Weak evidence for phase targeting (mean ratio = {mean_sf:.2f}x)")
    else:
        print(f"  Phase targeting does not outperform random-in-disc (mean ratio = {mean_sf:.2f}x)")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    t_start = time.time()

    print("=" * 70)
    print("SDQ MODE INTERVENTION: SPECTRAL DECOMPOSITION + TARGETED PERTURBATION")
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

    # -- Load h_0 vectors --
    examples = load_benchmark(args.benchmark)
    runs_dir = Path(args.runs_dir)
    print(f"\nLoading h_0 vectors from {runs_dir} ...")

    all_examples = collect_h0_and_labels(examples, runs_dir, layer=-1,
                                         source=args.h0_source)
    H0 = torch.stack([e["_h_0"] for e in all_examples])
    labels = torch.tensor([0 if e["_correct"] else 1 for e in all_examples])
    families = [e["task_family"] for e in all_examples]
    n_correct = (labels == 0).sum().item()
    n_incorrect = (labels == 1).sum().item()
    print(f"Loaded {len(all_examples)} examples: "
          f"{n_correct} correct, {n_incorrect} incorrect")

    # ── STAGE 1: Spectral mode analysis (CPU only) ──
    print(f"\n{'=' * 70}")
    print("STAGE 1: SPECTRAL MODE ANALYSIS")
    print(f"{'=' * 70}")

    analysis = full_mode_analysis(
        H0, labels, families, scaler_mu, scaler_sd,
        k=args.subspace_rank, disc_threshold=args.disc_threshold,
    )
    print(format_mode_analysis(analysis))

    # Save analysis summary
    analysis_summary = {
        "n_modes": args.subspace_rank,
        "n_discriminative": analysis.n_discriminative,
        "disc_threshold": analysis.disc_threshold,
        "top_modes": analysis.top_modes,
        "n_bottom_modes": len(analysis.bottom_modes),
        "discriminability": analysis.decomposition.discriminability.tolist(),
        "eigenvalues": analysis.decomposition.eigenvalues.tolist(),
        "top_interference_pairs": [
            {"mode_i": i, "mode_j": j, "strength": s}
            for i, j, s in analysis.interference.top_pairs[:20]
        ],
        "total_interference": analysis.interference.total_interference,
        "per_family_top_mode": {},
    }
    for fam, fam_d in analysis.decomposition.per_family_disc.items():
        if not fam_d.isnan().all():
            best = fam_d.abs().argmax().item()
            analysis_summary["per_family_top_mode"][fam] = {
                "mode": best, "cohens_d": fam_d[best].item()
            }

    if args.analysis_only:
        out_path = Path(args.output)
        results = {
            "mode": "analysis_only",
            "analysis": analysis_summary,
            "elapsed_seconds": time.time() - t_start,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nAnalysis saved to {out_path}")
        print(f"Total time: {time.time() - t_start:.0f}s")
        return

    # ── STAGE 2: Mode-selective interventions (GPU) ──
    print(f"\n{'=' * 70}")
    print("STAGE 2: MODE-SELECTIVE INTERVENTIONS")
    print(f"{'=' * 70}")

    # Select examples
    probe_scores = compute_probe_scores(
        H0, direction_std, weight_norm, bias, scaler_mu, scaler_sd,
    )
    lo, hi = args.boundary_range
    selected = select_examples(all_examples, probe_scores, args.n_prompts, lo, hi)
    n_boundary = sum(1 for e in selected if lo <= e["_probe_score"] <= hi)
    print(f"Selected {len(selected)} correct examples "
          f"({n_boundary} near-boundary [{lo:.1f}, {hi:.1f}])")

    if not selected:
        print("ERROR: No examples selected.")
        sys.exit(1)

    # Calibrate magnitudes
    h0_selected = torch.stack([e["_h_0"] for e in selected])
    if args.magnitudes:
        magnitudes = args.magnitudes
    else:
        d = direction_raw / direction_raw.norm().clamp_min(1e-8)
        proj = (h0_selected.float() @ d.float())
        std = proj.std().item()
        magnitudes = [std * m for m in [1.0, 2.0, 5.0]]
    print(f"Magnitudes: {[f'{m:.2f}' for m in magnitudes]}")

    # Load model
    print("\nLoading model...", flush=True)
    from sdq.instrumentation.model_loader import load_model
    bundle = load_model(args.config, model_path_override=args.model_path)
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device
    num_layers = model.config.num_hidden_layers

    if args.layers is not None:
        layer_indices = [l if l >= 0 else num_layers + l for l in args.layers]
    else:
        layer_indices = [num_layers // 2, 3 * num_layers // 4]
    layer_indices = [max(0, min(l, num_layers - 1)) for l in layer_indices]
    print(f"Layers: {layer_indices} (of {num_layers})")

    # Run sweep
    all_results = run_mode_sweep(
        model, tokenizer, device,
        selected, analysis, direction_raw, hidden_dim,
        magnitudes, layer_indices,
        args.n_random, args.max_new_tokens,
        scaler_mu, scaler_sd, args.seed, t_start,
    )

    # -- Overall verdict --
    print(f"\n{'=' * 70}")
    print("OVERALL VERDICT")
    print(f"{'=' * 70}")

    best_disc_ratio = 0.0
    best_sf_ratio = 0.0
    for key, stats in all_results.items():
        for comp in stats["comparisons"]:
            if comp["random_disc_vs_random_non_disc"] > best_disc_ratio:
                best_disc_ratio = comp["random_disc_vs_random_non_disc"]
            if comp["sign_flip_vs_random_disc"] > best_sf_ratio:
                best_sf_ratio = comp["sign_flip_vs_random_disc"]

    if best_disc_ratio >= 2.0:
        print("MESOSCALE STRUCTURE FOUND: Discriminative modes are causally privileged.")
        print(f"  Best disc/non-disc ratio: {best_disc_ratio:.2f}x")
    elif best_sf_ratio >= 2.0:
        print("PHASE TARGETING WORKS: Example-specific sign-flip outperforms random.")
        print(f"  Best sign_flip/random_disc ratio: {best_sf_ratio:.2f}x")
    else:
        print("NO MESOSCALE ADVANTAGE: Mode decomposition does not improve on Phase 4b.")
        print(f"  Best disc/non-disc: {best_disc_ratio:.2f}x")
        print(f"  Best sign_flip/random_disc: {best_sf_ratio:.2f}x")
        print("  The commitment signal is either truly diffuse across the subspace,")
        print("  or the causal structure operates at a level below PCA modes")
        print("  (individual SAE features / circuit nodes).")
    print("=" * 70)

    # Save results
    results = {
        "mode": "full_experiment",
        "analysis": analysis_summary,
        "per_layer": all_results,
        "summary": {
            "n_examples": len(selected),
            "n_discriminative_modes": analysis.n_discriminative,
            "disc_threshold": args.disc_threshold,
            "magnitudes": magnitudes,
            "layers": layer_indices,
            "best_disc_vs_non_disc_ratio": best_disc_ratio,
            "best_sign_flip_vs_random_disc_ratio": best_sf_ratio,
        },
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
