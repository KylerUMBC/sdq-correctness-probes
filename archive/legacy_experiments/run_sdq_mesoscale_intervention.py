#!/usr/bin/env python3
"""SDQ Phase 5: Mesoscale causal intervention experiment.

Phases 4 (rank-1) and 4b (full 96-dim subspace) both failed: linear
interventions on either a single direction or the full predictive subspace
do not flip answers above random baseline. Phase 5 tests a *mesoscale*
description sitting between them: do K=8 supervised+Varimax-rotated
components have differential causal traction?

Conditions per (prompt, layer, magnitude_multiplier):
  1. K single-component perturbations (sign-oriented)
  2. One coordinated all-K signed-sum perturbation
  3. n_within_span random unit vectors in span(basis_raw) — load-bearing baseline
  4. n_full_random random unit vectors in R^D — Phase 4 anchor

All conditions use persistent perturbation hooks (each forward pass) and
greedy generation, matching Phase 4b methodology. Each condition's
perturbation is calibrated to "magnitude_multiplier * sigma_along_direction"
where sigma is the std of the FULL population's prompt-final h states AT THE
PERTURBED LAYER projected onto that direction (audit fix: one population for
every condition, calibrated at the intervention site rather than the last
layer).

h_0 = the prompt-final state, activations.pt[layer, -1, :] — the state the
prefill hook actually perturbs. (--h0-source first_gen recovers the legacy
definition for comparison.)

Pre-registered decision tree (n=80 prompts), primary metric =
flipped_to_incorrect, all comparisons vs the within-span baseline of the
same (layer, magnitude) cell, with Fisher exact tests Holm-corrected across
every cell the max is taken over:
  - Some single component flips at >=2x within-span AND >=10% absolute AND
    Holm p<0.05 -> circuits-mesoscale wins (next: SAE bridge)
  - Else coordinated K-of-K at >=2x within-span AND >=15% absolute AND Holm
    p<0.05 -> redundant-pathway wins (next: scale K); pairwise KL
    superlinearity is supporting evidence when available
  - Neither reaches 2x at any cell -> wave/superposition wins
    (next: abandon linear interventions)
  - Effect-size met but not significant -> AMBIGUOUS (scale n_prompts)

Optionally runs pairwise C(K,2) perturbations at the best (mag, layer)
combination, gated on any single component OR the coordinated perturbation
exceeding within-span. The pairwise KL superlinearity ratio distinguishes
constructive interference (circuits-coordination) from independent additive
(parallel pathways).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

from sdq.eval.commitment_intervention import (
    generate_baseline,
    generate_with_persistent_perturbation,
    measure_logit_shift,
)
from sdq.eval.intervention_data import (
    collect_h0_and_labels,
    collect_h0_multi_layer,
    compute_probe_scores,
    load_benchmark,
    select_examples,
)
from sdq.eval.mesoscale_basis import (
    MesoscaleBasis,
    calibrate_direction_std,
    fit_supervised_mesoscale_basis,
    make_component_perturbation,
    make_coordinated_perturbation,
    make_pairwise_perturbation,
    sample_full_random_unit,
    sample_within_span_unit,
)
from sdq.eval.stats import fisher_exact_greater, holm_adjust, wilson_interval
from sdq.labels.outcome_labeler import label_run


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDQ Phase 5: Mesoscale causal intervention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--model-path", default=None)
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--commitment-direction", default="commitment_direction.pt")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--K", type=int, default=8,
                   help="Number of mesoscale components")
    p.add_argument("--layers", type=int, nargs="*", default=[13, 19])
    p.add_argument("--magnitudes", type=float, nargs="*", default=[1.0, 2.0, 3.0],
                   help="Multipliers of per-direction sigma")
    p.add_argument("--n-prompts", type=int, default=80)
    p.add_argument("--n-within-span", type=int, default=3,
                   help="Within-span random baselines per prompt per magnitude")
    p.add_argument("--n-full-random", type=int, default=3,
                   help="Full-D random baselines per prompt per magnitude")
    p.add_argument("--include-pairwise", action="store_true",
                   help="Run pairwise C(K,2) perturbations (gated on singles "
                        "or coordinated passing)")
    p.add_argument("--components", type=int, nargs="*", default=None,
                   help="Subset of basis components to run as singles "
                        "(default: all K). Use for targeted confirmation "
                        "runs, e.g. --components 0 6")
    p.add_argument("--skip-coordinated", action="store_true",
                   help="Skip the coordinated K-of-K condition (the full-D "
                        "random baseline is skipped via --n-full-random 0)")
    p.add_argument("--prefill-positions", choices=["last", "all"],
                   default="last",
                   help="Perturbation surface during prefill: 'last' (only "
                        "the final prompt position; all other prompt KV "
                        "entries stay clean — the Phase 4/4b/5 surface) or "
                        "'all' (every prompt position, so the KV cache is "
                        "built from perturbed states; tests the KV-cache "
                        "bypass hypothesis)")
    p.add_argument("--h0-source", choices=["prompt_final", "first_gen"],
                   default="prompt_final",
                   help="h_0 definition for basis fitting and selection. "
                        "'prompt_final' (default) is the last-prompt-token "
                        "state — the state the prefill hook actually "
                        "perturbs. 'first_gen' is the legacy definition.")
    p.add_argument("--boundary-range", type=float, nargs=2, default=[0.2, 0.8],
                   metavar=("LO", "HI"))
    p.add_argument("--basis-epochs", type=int, default=400)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output", default="mesoscale_results.json")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint JSON path (default: <output stem>"
                        ".checkpoint.json). Raw trials are flushed here every "
                        "few prompts; an interrupted run resumes from it "
                        "automatically. Deleted after results are saved.")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing checkpoint and start over")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Trial wiring ────────────────────────────────────────────────────────────
# (Data loading/selection lives in sdq.eval.intervention_data, shared with
# the Phase 4/4b runners.)

def _flip_summary(baseline_outcome, pert_outcome) -> tuple[bool, bool]:
    flipped = pert_outcome.parsed_answer != baseline_outcome.parsed_answer
    flipped_bad = baseline_outcome.correct and not pert_outcome.correct
    return flipped, flipped_bad


def _record_trial(
    eid, fam, gold, mag_mult, mag_abs, condition, comp_idx,
    baseline_outcome, pert_text, pert_outcome,
    logit_shift, sigma,
) -> dict:
    flipped, flipped_bad = _flip_summary(baseline_outcome, pert_outcome)
    return {
        "example_id": eid,
        "task_family": fam,
        "gold_answer": gold,
        "condition": condition,
        "component_idx": comp_idx,
        "mag_mult": mag_mult,
        "mag_abs": mag_abs,
        "sigma": sigma,
        "baseline_answer": baseline_outcome.parsed_answer,
        "perturbed_answer": pert_outcome.parsed_answer,
        "perturbed_text": pert_text[:200],
        "answer_flipped": flipped,
        "flipped_to_incorrect": flipped_bad,
        # Parser confidence (1.0 = deterministic extraction). Perturbed text
        # is often degenerate, where the parser falls back to first-token
        # heuristics; the confident-only subset is the robustness check.
        "baseline_parse_confidence": baseline_outcome.confidence,
        "perturbed_parse_confidence": pert_outcome.confidence,
        # NOTE: logit shift is measured with a one-shot *prefill* hook
        # (first-token effect), while flip outcomes use the persistent hook.
        "logit_kl": logit_shift.kl_divergence,
        "logit_js": logit_shift.js_divergence,
        "logit_top1_prob_change": logit_shift.top1_prob_change,
    }


# ── Checkpointing ───────────────────────────────────────────────────────────
# Raw trials are flushed to disk every few prompts so a crash (or a summary-
# stage bug) never loses generation work. The checkpoint stores only raw
# trials and completed prompt ids per layer; all summaries are recomputed on
# resume.

CHECKPOINT_EVERY = 5  # prompts; matches the progress-print cadence


def _checkpoint_path(args) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    out = Path(args.output)
    return out.with_name(out.stem + ".checkpoint.json")


def _checkpoint_fingerprint(args, layer_indices, selected, components) -> dict:
    """Everything that determines which trials a run produces. A mismatch
    means the checkpoint belongs to a different experiment and is not reused.
    `selected_eids` guards against the selection itself shifting (e.g. a
    regenerated commitment_direction.pt changing probe scores)."""
    return {
        "K": args.K,
        "layers": list(layer_indices),
        "magnitudes": list(args.magnitudes),
        "n_within_span": args.n_within_span,
        "n_full_random": args.n_full_random,
        "h0_source": args.h0_source,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "components": list(components),
        "skip_coordinated": args.skip_coordinated,
        "prefill_positions": args.prefill_positions,
        "selected_eids": [ex["example_id"] for ex in selected],
    }


def _load_checkpoint(path: Path, fingerprint: dict, fresh: bool) -> dict:
    empty = {"fingerprint": fingerprint, "layers": {}}
    if fresh or not path.exists():
        return empty
    try:
        with open(path, encoding="utf-8") as f:
            ckpt = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: could not read checkpoint {path} ({e}); starting fresh.")
        return empty
    if ckpt.get("fingerprint") != fingerprint:
        print(f"WARNING: checkpoint {path} is from a run with different "
              f"settings or a different example selection; starting fresh "
              f"(it will be overwritten -- pass --checkpoint to keep it "
              f"under another name).")
        return empty
    n_done = sum(len(l["completed_eids"]) for l in ckpt["layers"].values())
    print(f"Resuming from checkpoint {path}: {n_done} completed "
          f"(layer, prompt) cells.")
    return ckpt


def _save_checkpoint(path: Path, ckpt: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f)
    os.replace(tmp, path)  # atomic: never leaves a half-written checkpoint


# ── Main sweep ──────────────────────────────────────────────────────────────

def compute_baselines(
    model, tokenizer, device, selected: list[dict], max_new_tokens: int,
) -> dict[str, dict]:
    """Generate the unperturbed baseline once per prompt (layer-independent).

    Returns {example_id: {"text", "outcome"}} for prompts whose baseline is
    correct; incorrect baselines are dropped here, once, instead of per layer.
    """
    baselines: dict[str, dict] = {}
    n_skipped = 0
    for ex in selected:
        inputs = tokenizer(ex["prompt_text"], return_tensors="pt").to(device)
        text, _ = generate_baseline(
            model, tokenizer, inputs["input_ids"], inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
        outcome = label_run(text, ex["answer_id"], ex["task_family"])
        if outcome.correct:
            baselines[ex["example_id"]] = {"text": text, "outcome": outcome}
        else:
            n_skipped += 1
    print(f"Baselines: {len(baselines)} correct, {n_skipped} skipped "
          f"(no longer correct at generation time)")
    return baselines


def run_sweep(
    model, tokenizer, device,
    selected: list[dict],
    basis: MesoscaleBasis,
    h0_by_layer: dict[int, Tensor],
    args,
    layer_indices: list[int],
    components: list[int],
    t_start: float,
    ckpt: dict,
    ckpt_path: Path,
) -> dict:
    """Run the full Phase 5 sweep across layers, magnitudes, and conditions.

    `h0_by_layer` maps layer index -> [N, D] prompt-final states of the FULL
    loaded population at that layer. All sigma calibration (singles,
    coordinated, within-span, full-random) uses this single population, at
    the layer actually being perturbed — an intervention at layer 13 is
    calibrated against layer-13 variation, not last-layer variation.
    """
    K = basis.K
    D = basis.basis_raw.shape[0]
    all_prefill = args.prefill_positions == "all"

    coord_dir = basis.basis_raw @ basis.signed_scores
    coord_dir = coord_dir / coord_dir.norm().clamp_min(1e-8)

    # Sign per component (orientation toward "more incorrect")
    signs = [1.0 if float(s) >= 0 else -1.0 for s in basis.signed_scores.tolist()]
    print(f"Component signs: {signs}")

    # Baselines are layer-independent — generate once.
    baselines = compute_baselines(model, tokenizer, device, selected,
                                  args.max_new_tokens)

    layer_results: dict[str, dict] = {}

    for layer_idx in layer_indices:
        layer_module = model.model.layers[layer_idx]
        layer_key = f"layer{layer_idx}"

        # Per-layer sigma calibration (single population for all conditions)
        H_layer = h0_by_layer[layer_idx]
        component_sigmas = [
            calibrate_direction_std(H_layer, basis.basis_raw[:, k])
            for k in range(K)
        ]
        coord_sigma = calibrate_direction_std(H_layer, coord_dir)

        print(f"\n{'=' * 70}")
        print(f"LAYER {layer_idx} (persistent hook)")
        print(f"{'=' * 70}")
        print(f"  Per-component sigmas (layer {layer_idx}): "
              f"{[f'{s:.3f}' for s in component_sigmas]}")
        print(f"  Coord direction sigma: {coord_sigma:.3f}")

        rng = torch.Generator().manual_seed(args.seed + layer_idx)
        layer_ck = ckpt["layers"].setdefault(
            str(layer_idx), {"completed_eids": [], "trials": []})
        done_eids = set(layer_ck["completed_eids"])
        trials: list[dict] = layer_ck["trials"]
        if done_eids:
            print(f"  Resuming: {len(done_eids)}/{len(selected)} prompts "
                  f"already done ({len(trials)} trials from checkpoint)")

        for idx, ex in enumerate(selected):
            eid = ex["example_id"]
            # Skipping completed prompts advances `rng` from a different
            # point than an uninterrupted run would — the within-span /
            # full-random baseline directions are still i.i.d. samples, so
            # the statistics are unaffected.
            if eid in done_eids:
                continue
            prompt_text = ex["prompt_text"]
            gold = ex["answer_id"]
            fam = ex["task_family"]

            base = baselines.get(eid)
            if base is None:
                continue
            baseline_outcome = base["outcome"]

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            for mag_mult in args.magnitudes:
                # ── 1. Singles ──
                for k in components:
                    mag_abs = mag_mult * component_sigmas[k]
                    pert = make_component_perturbation(
                        basis.basis_raw, k, mag_abs, sign=signs[k]
                    )
                    pert_dir = pert / pert.norm().clamp_min(1e-8)

                    text, _, _ = generate_with_persistent_perturbation(
                        model, tokenizer, input_ids, attention_mask,
                        layer_module=layer_module,
                        direction=pert_dir, magnitude=mag_abs,
                        max_new_tokens=args.max_new_tokens,
                        all_prefill_positions=all_prefill,
                    )
                    out = label_run(text, gold, fam)
                    ls = measure_logit_shift(
                        model, input_ids, attention_mask, layer_module,
                        pert_dir, mag_abs,
                        all_prefill_positions=all_prefill,
                    )
                    trials.append(_record_trial(
                        eid, fam, gold, mag_mult, mag_abs,
                        "single", k, baseline_outcome, text, out, ls,
                        component_sigmas[k],
                    ))

                # ── 2. Coordinated ──
                if not args.skip_coordinated:
                    mag_abs = mag_mult * coord_sigma
                    pert = make_coordinated_perturbation(
                        basis.basis_raw, basis.signed_scores, mag_abs
                    )
                    pert_dir = pert / pert.norm().clamp_min(1e-8)
                    text, _, _ = generate_with_persistent_perturbation(
                        model, tokenizer, input_ids, attention_mask,
                        layer_module=layer_module,
                        direction=pert_dir, magnitude=mag_abs,
                        max_new_tokens=args.max_new_tokens,
                        all_prefill_positions=all_prefill,
                    )
                    out = label_run(text, gold, fam)
                    ls = measure_logit_shift(
                        model, input_ids, attention_mask, layer_module,
                        pert_dir, mag_abs,
                        all_prefill_positions=all_prefill,
                    )
                    trials.append(_record_trial(
                        eid, fam, gold, mag_mult, mag_abs,
                        "coordinated", -1, baseline_outcome, text, out, ls,
                        coord_sigma,
                    ))

                # ── 3. Within-span baselines (per-sample sigma) ──
                for _ in range(args.n_within_span):
                    unit = sample_within_span_unit(basis.basis_raw, rng)
                    sigma = calibrate_direction_std(H_layer, unit)
                    mag_abs = mag_mult * sigma
                    text, _, _ = generate_with_persistent_perturbation(
                        model, tokenizer, input_ids, attention_mask,
                        layer_module=layer_module,
                        direction=unit, magnitude=mag_abs,
                        max_new_tokens=args.max_new_tokens,
                        all_prefill_positions=all_prefill,
                    )
                    out = label_run(text, gold, fam)
                    ls = measure_logit_shift(
                        model, input_ids, attention_mask, layer_module,
                        unit, mag_abs,
                        all_prefill_positions=all_prefill,
                    )
                    trials.append(_record_trial(
                        eid, fam, gold, mag_mult, mag_abs,
                        "within_span", -1, baseline_outcome, text, out, ls,
                        sigma,
                    ))

                # ── 4. Full-random baselines (per-sample sigma) ──
                for _ in range(args.n_full_random):
                    unit = sample_full_random_unit(D, rng)
                    sigma = calibrate_direction_std(H_layer, unit)
                    mag_abs = mag_mult * sigma
                    text, _, _ = generate_with_persistent_perturbation(
                        model, tokenizer, input_ids, attention_mask,
                        layer_module=layer_module,
                        direction=unit, magnitude=mag_abs,
                        max_new_tokens=args.max_new_tokens,
                        all_prefill_positions=all_prefill,
                    )
                    out = label_run(text, gold, fam)
                    ls = measure_logit_shift(
                        model, input_ids, attention_mask, layer_module,
                        unit, mag_abs,
                        all_prefill_positions=all_prefill,
                    )
                    trials.append(_record_trial(
                        eid, fam, gold, mag_mult, mag_abs,
                        "full_random", -1, baseline_outcome, text, out, ls,
                        sigma,
                    ))

            layer_ck["completed_eids"].append(eid)
            n_done = idx + 1
            if n_done % CHECKPOINT_EVERY == 0 or n_done == len(selected):
                _save_checkpoint(ckpt_path, ckpt)
                elapsed = time.time() - t_start
                print(f"  [{n_done:4d}/{len(selected)}]  {elapsed:.0f}s elapsed"
                      f"  (checkpointed)")

        # All generation work for this layer is on disk before any summary
        # code runs.
        _save_checkpoint(ckpt_path, ckpt)

        # ── Summary stats per layer ──
        layer_summary = summarize_layer(trials, components, args.magnitudes)
        layer_summary["n_prompts_with_correct_baseline"] = len(baselines)
        layer_summary["component_sigmas"] = component_sigmas
        layer_summary["coord_sigma"] = coord_sigma
        layer_summary["trials"] = trials
        _print_layer_summary(layer_idx, layer_summary, args.magnitudes, K)
        layer_results[layer_key] = layer_summary

        # ── Pairwise (gated) ──
        if args.include_pairwise:
            best = layer_summary["pairwise_trigger"]
            if best is not None:
                if layer_ck.get("pairwise_trials") is not None:
                    pairwise_trials = layer_ck["pairwise_trials"]
                    print(f"\n  Pairwise trials loaded from checkpoint "
                          f"({len(pairwise_trials)} trials).")
                else:
                    print(f"\n  Pairwise gate triggered at mag_mult={best['mag_mult']} "
                          f"({best['trigger_condition']} {best['component']} at "
                          f"{best['fti_rate']:.1%} > "
                          f"within-span {best['within_span_rate']:.1%}); running C(K,2)...")
                    pairwise_trials = run_pairwise(
                        model, tokenizer, device, layer_module, selected, baselines,
                        basis, best["mag_mult"], best["mag_abs"],
                        args.max_new_tokens, all_prefill,
                    )
                    layer_ck["pairwise_trials"] = pairwise_trials
                    _save_checkpoint(ckpt_path, ckpt)
                layer_summary["pairwise_trials"] = pairwise_trials
                layer_summary["pairwise_summary"] = summarize_pairwise(
                    pairwise_trials, K, layer_summary, best["mag_mult"]
                )
            else:
                print("  Pairwise gate not triggered (no single component or "
                      "coordinated perturbation exceeded within-span at any "
                      "magnitude).")

    return layer_results


def run_pairwise(
    model, tokenizer, device, layer_module,
    selected, baselines, basis, mag_mult, mag_abs, max_new_tokens,
    all_prefill: bool = False,
) -> list[dict]:
    """Run all C(K,2) pairwise perturbations at one magnitude."""
    K = basis.K
    pair_trials: list[dict] = []

    for idx, ex in enumerate(selected):
        eid = ex["example_id"]
        prompt_text = ex["prompt_text"]
        gold = ex["answer_id"]
        fam = ex["task_family"]

        base = baselines.get(eid)
        if base is None:
            continue
        baseline_outcome = base["outcome"]

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        for i in range(K):
            for j in range(i + 1, K):
                pert = make_pairwise_perturbation(
                    basis.basis_raw, i, j, basis.signed_scores, mag_abs
                )
                pert_dir = pert / pert.norm().clamp_min(1e-8)
                text, _, _ = generate_with_persistent_perturbation(
                    model, tokenizer, input_ids, attention_mask,
                    layer_module=layer_module,
                    direction=pert_dir, magnitude=mag_abs,
                    max_new_tokens=max_new_tokens,
                    all_prefill_positions=all_prefill,
                )
                out = label_run(text, gold, fam)
                ls = measure_logit_shift(
                    model, input_ids, attention_mask, layer_module,
                    pert_dir, mag_abs,
                    all_prefill_positions=all_prefill,
                )
                pair_trials.append({
                    "example_id": eid,
                    "task_family": fam,
                    "i": i, "j": j,
                    "mag_mult": mag_mult,
                    "mag_abs": mag_abs,
                    "answer_flipped": (out.parsed_answer != baseline_outcome.parsed_answer),
                    "flipped_to_incorrect": baseline_outcome.correct and not out.correct,
                    "logit_kl": ls.kl_divergence,
                    "logit_js": ls.js_divergence,
                })
    return pair_trials


# ── Summaries ───────────────────────────────────────────────────────────────

def _cond_stats(ts: list[dict]) -> dict:
    """Aggregate one condition cell.

    Primary metric is `fti_rate` (flipped_to_incorrect): the experiment
    claims a *directional* push toward incorrect, so generic answer changes
    (including parse noise) are reported but not decision-bearing.
    `fti_rate_confident` restricts to trials where both baseline and
    perturbed answers were parsed deterministically (confidence == 1.0) —
    the robustness check against the parser degrading on perturbed text.
    """
    n = len(ts)
    fti = sum(1 for t in ts if t["flipped_to_incorrect"])
    flips = sum(1 for t in ts if t["answer_flipped"])
    conf = [t for t in ts
            if t.get("baseline_parse_confidence", 1.0) >= 1.0
            and t.get("perturbed_parse_confidence", 1.0) >= 1.0]
    n_conf = len(conf)
    fti_conf = sum(1 for t in conf if t["flipped_to_incorrect"])
    kls = [t["logit_kl"] for t in ts]
    lo, hi = wilson_interval(fti, n)
    return {
        "n": n,
        "fti": fti,
        "fti_rate": fti / n if n else 0.0,
        "fti_ci95": [lo, hi],
        "flip_rate": flips / n if n else 0.0,
        "n_confident": n_conf,
        "fti_rate_confident": fti_conf / n_conf if n_conf else 0.0,
        "kl_mean": float(_mean(kls)),
        "kl_median": float(_median(kls)),
    }


def summarize_layer(trials: list[dict], components: list[int],
                    magnitudes: list[float]) -> dict:
    """Per-layer summary: per-condition flip stats per magnitude.

    The pairwise gate triggers when EITHER a single component OR the
    coordinated perturbation exceeds 2x the within-span baseline — the
    coordinated path matters because pairwise superlinearity is the
    supporting evidence for the redundant-pathway hypothesis (audit fix:
    previously only singles could trigger it, making that verdict
    unreachable in exactly the scenario it describes).
    """
    summary: dict = {"per_magnitude": []}

    pairwise_trigger = None  # best (mag, component-or-coord) for gated pairwise

    def _consider_trigger(mag, rec, label, comp_idx, rate, mag_abs):
        nonlocal pairwise_trigger
        if rate >= 2.0 * rec["within_span"]["fti_rate"] and rate > 0.05:
            if pairwise_trigger is None or rate > pairwise_trigger["fti_rate"]:
                pairwise_trigger = {
                    "mag_mult": mag,
                    "mag_abs": mag_abs,
                    "component": comp_idx,
                    "trigger_condition": label,
                    "fti_rate": rate,
                    "within_span_rate": rec["within_span"]["fti_rate"],
                }

    for mag in magnitudes:
        rec: dict = {"mag_mult": mag}

        # Singles: per-component
        per_component = []
        for k in components:
            ts = [t for t in trials
                  if t["condition"] == "single" and t["component_idx"] == k
                  and t["mag_mult"] == mag]
            stats = _cond_stats(ts)
            stats["k"] = k
            stats["mag_abs"] = ts[0]["mag_abs"] if ts else None
            per_component.append(stats)
        rec["singles"] = per_component
        rec["singles_max_fti_rate"] = max(
            (c["fti_rate"] for c in per_component), default=0.0)

        for cond in ("coordinated", "within_span", "full_random"):
            ts = [t for t in trials
                  if t["condition"] == cond and t["mag_mult"] == mag]
            rec[cond] = _cond_stats(ts)
            if cond == "coordinated":
                rec[cond]["mag_abs"] = ts[0]["mag_abs"] if ts else None

        # Comparison ratios (within-span is the load-bearing baseline).
        # Point estimates only — significance comes from Fisher tests added
        # in add_significance().
        ws = max(rec["within_span"]["fti_rate"], 1e-6)
        rec["singles_max_vs_within"] = rec["singles_max_fti_rate"] / ws
        rec["coordinated_vs_within"] = rec["coordinated"]["fti_rate"] / ws
        fr = max(rec["full_random"]["fti_rate"], 1e-6)
        rec["singles_max_vs_full_random"] = rec["singles_max_fti_rate"] / fr

        # Pairwise gate: singles OR coordinated
        for c in per_component:
            _consider_trigger(mag, rec, "single", c["k"], c["fti_rate"],
                              c["mag_abs"] if c["mag_abs"] is not None else mag)
        _consider_trigger(mag, rec, "coordinated", -1,
                          rec["coordinated"]["fti_rate"],
                          rec["coordinated"]["mag_abs"]
                          if rec["coordinated"]["mag_abs"] is not None else mag)

        summary["per_magnitude"].append(rec)

    summary["pairwise_trigger"] = pairwise_trigger
    return summary


def add_significance(layer_results: dict) -> None:
    """Attach Fisher exact p-values (vs within-span) with Holm correction.

    Every cell the decision tree may select from enters one Holm family:
    all (layer, magnitude, component) singles cells plus all
    (layer, magnitude) coordinated cells. This is exactly the set the
    'best of' max is taken over, so the correction matches the selection.
    Writes `p_raw` / `p_holm` into each cell in place.
    """
    cells: list[dict] = []  # references into layer_results
    pvals: list[float] = []
    for summary in layer_results.values():
        for rec in summary["per_magnitude"]:
            ws = rec["within_span"]
            candidates = rec["singles"] + [rec["coordinated"]]
            for c in candidates:
                if c["n"] == 0:  # condition skipped this run
                    continue
                p = fisher_exact_greater(c["fti"], c["n"], ws["fti"], ws["n"])
                c["p_raw"] = p
                cells.append(c)
                pvals.append(p)

    for cell, p_adj in zip(cells, holm_adjust(pvals)):
        cell["p_holm"] = p_adj


def add_pooled_component_tests(layer_results: dict) -> None:
    """Per-(layer, component) Fisher tests POOLED across all magnitudes in
    the run, Holm-corrected across that family.

    This is the primary pre-registered test for targeted confirmation runs
    (e.g. `--components 0 6 --layers 13 --magnitudes 2.0 3.0`), where the
    run is configured so the pooled cell contains exactly the pre-registered
    conditions. In full sweeps it is supporting detail only — the per-cell
    Holm family in add_significance() matches the decision tree's selection
    there.

    Writes `pooled_components` into each layer summary.
    """
    rows: list[dict] = []
    for layer_key, summary in layer_results.items():
        per_k: dict[int, dict] = {}
        ws_fti, ws_n = 0, 0
        for rec in summary["per_magnitude"]:
            ws_fti += rec["within_span"]["fti"]
            ws_n += rec["within_span"]["n"]
            for c in rec["singles"]:
                agg = per_k.setdefault(c["k"], {"fti": 0, "n": 0})
                agg["fti"] += c["fti"]
                agg["n"] += c["n"]
        summary["pooled_components"] = []
        for k in sorted(per_k):
            agg = per_k[k]
            if agg["n"] == 0:
                continue
            rate = agg["fti"] / agg["n"]
            ws_rate = ws_fti / ws_n if ws_n else 0.0
            row = {
                "layer": layer_key,
                "k": k,
                "fti": agg["fti"], "n": agg["n"], "fti_rate": rate,
                "within_span_fti": ws_fti, "within_span_n": ws_n,
                "within_span_rate": ws_rate,
                "ratio": rate / max(ws_rate, 1e-6),
                "p_raw": fisher_exact_greater(agg["fti"], agg["n"],
                                              ws_fti, ws_n),
            }
            summary["pooled_components"].append(row)
            rows.append(row)

    for row, p_adj in zip(rows, holm_adjust([r["p_raw"] for r in rows])):
        row["p_holm"] = p_adj


def summarize_pairwise(
    pair_trials: list[dict],
    K: int,
    layer_summary: dict,
    mag_mult: float,
) -> dict:
    """Pairwise interference summary: KL_ij vs KL_i + KL_j."""
    # Singles KLs at this magnitude
    singles_kl: dict[int, float] = {}
    for rec in layer_summary["per_magnitude"]:
        if rec["mag_mult"] == mag_mult:
            for c in rec["singles"]:
                singles_kl[c["k"]] = c["kl_mean"]
            break

    pair_records = []
    for i in range(K):
        for j in range(i + 1, K):
            pts = [t for t in pair_trials if t["i"] == i and t["j"] == j]
            if not pts:
                continue
            kl_ij = float(_mean([t["logit_kl"] for t in pts]))
            kl_sum = singles_kl.get(i, 0.0) + singles_kl.get(j, 0.0)
            ratio = kl_ij / max(kl_sum, 1e-9)
            n = len(pts) or 1
            pair_records.append({
                "i": i, "j": j,
                "n": len(pts),
                "kl_ij": kl_ij,
                "kl_i_plus_kl_j": kl_sum,
                "superlinearity_ratio": ratio,
                "fti_rate": sum(1 for t in pts if t["flipped_to_incorrect"]) / n,
                "flip_rate": sum(1 for t in pts if t["answer_flipped"]) / n,
            })

    ratios = [r["superlinearity_ratio"] for r in pair_records]
    return {
        "pairs": pair_records,
        "median_superlinearity": float(_median(ratios)) if ratios else 0.0,
        "mean_superlinearity": float(_mean(ratios)) if ratios else 0.0,
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _print_layer_summary(layer_idx: int, summary: dict, magnitudes, K):
    print(f"\n  === LAYER {layer_idx} SUMMARY (flip-to-incorrect rates) ===")
    print(f"  {'mag':>5s}  {'sing_max':>8s}  {'coord':>8s}  "
          f"{'within':>8s}  {'full_R':>8s}  {'sing/within':>11s}  {'coord/within':>12s}")
    for rec in summary["per_magnitude"]:
        print(f"  {rec['mag_mult']:5.1f}  "
              f"{rec['singles_max_fti_rate']:8.3f}  "
              f"{rec['coordinated']['fti_rate']:8.3f}  "
              f"{rec['within_span']['fti_rate']:8.3f}  "
              f"{rec['full_random']['fti_rate']:8.3f}  "
              f"{rec['singles_max_vs_within']:11.2f}x  "
              f"{rec['coordinated_vs_within']:12.2f}x")


# ── Decision tree ───────────────────────────────────────────────────────────

def apply_decision_tree(layer_results: dict, args) -> dict:
    """Apply the pre-registered decision tree.

    Metric: flipped_to_incorrect rate (the directional claim), compared to
    the within-span baseline of the same (layer, magnitude) cell. Because
    the "best" cell is selected post hoc across all cells, a verdict also
    requires the Holm-adjusted Fisher p-value of that cell to clear 0.05 —
    otherwise the verdict downgrades to AMBIGUOUS with an explanation.

    Audit fix: REDUNDANT_PATHWAY no longer requires pairwise superlinearity
    (which previously could only be measured when a *single* component
    passed — making the verdict unreachable in exactly the scenario it
    describes). Pairwise superlinearity is reported as supporting evidence
    when available.
    """
    best_single = {"ratio": 0.0, "rate": 0.0, "layer": None, "mag_mult": None,
                   "k": None, "p_holm": None}
    best_coord = {"ratio": 0.0, "rate": 0.0, "layer": None, "mag_mult": None,
                  "p_holm": None}
    pairwise_super = None

    for layer_key, summary in layer_results.items():
        for rec in summary["per_magnitude"]:
            ws = rec["within_span"]["fti_rate"]
            for c in rec["singles"]:
                ratio = c["fti_rate"] / max(ws, 1e-6)
                if ratio > best_single["ratio"] and c["fti_rate"] > 0:
                    best_single = {
                        "ratio": ratio, "rate": c["fti_rate"], "layer": layer_key,
                        "mag_mult": rec["mag_mult"], "k": c["k"],
                        "within_span_rate": ws,
                        "p_raw": c.get("p_raw"), "p_holm": c.get("p_holm"),
                        "rate_confident": c.get("fti_rate_confident"),
                        "ci95": c.get("fti_ci95"),
                    }
            co = rec["coordinated"]
            r = co["fti_rate"] / max(ws, 1e-6)
            if r > best_coord["ratio"] and co["fti_rate"] > 0:
                best_coord = {
                    "ratio": r, "rate": co["fti_rate"], "layer": layer_key,
                    "mag_mult": rec["mag_mult"], "within_span_rate": ws,
                    "p_raw": co.get("p_raw"), "p_holm": co.get("p_holm"),
                    "rate_confident": co.get("fti_rate_confident"),
                    "ci95": co.get("fti_ci95"),
                }
        ps = summary.get("pairwise_summary")
        if ps:
            if pairwise_super is None or ps["median_superlinearity"] > pairwise_super["median"]:
                pairwise_super = {"layer": layer_key, "median": ps["median_superlinearity"],
                                  "mean": ps["mean_superlinearity"]}

    def _sig(best):
        return best["p_holm"] is not None and best["p_holm"] < 0.05

    single_effect = best_single["ratio"] >= 2.0 and best_single["rate"] >= 0.10
    coord_effect = best_coord["ratio"] >= 2.0 and best_coord["rate"] >= 0.15

    circuits = single_effect and _sig(best_single)
    redundant = (not circuits) and coord_effect and _sig(best_coord)
    wave = (not circuits and not redundant
            and best_single["ratio"] < 2.0
            and best_coord["ratio"] < 2.0)

    if circuits:
        verdict = "CIRCUITS_MESOSCALE"
        next_step = ("Phase 6: bridge to GemmaScope SAE; decompose component "
                     f"{best_single['k']} in SAE basis.")
    elif redundant:
        verdict = "REDUNDANT_PATHWAY"
        next_step = "Phase 6: scale K, search minimal sufficient coordinated set."
        if pairwise_super is not None and pairwise_super["median"] > 1.3:
            next_step += (" Pairwise KL superlinearity supports constructive "
                          "interference.")
    elif wave:
        verdict = "WAVE_SUPERPOSITION"
        next_step = "Phase 6: abandon linear interventions; pivot to nonlinear/geometric."
    else:
        verdict = "AMBIGUOUS"
        if (single_effect and not _sig(best_single)) or (coord_effect and not _sig(best_coord)):
            next_step = ("Effect-size threshold met but not significant after "
                         "Holm correction — increase n_prompts (e.g. 160) and "
                         "re-run before deciding.")
        else:
            next_step = "Increase n_prompts to 160 before re-deciding."

    return {
        "verdict": verdict,
        "next_step": next_step,
        "metric": "flipped_to_incorrect",
        "best_single": best_single,
        "best_coord": best_coord,
        "pairwise_superlinearity": pairwise_super,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    t_start = time.time()

    print("=" * 70)
    print("SDQ PHASE 5: MESOSCALE CAUSAL INTERVENTION")
    print("=" * 70)

    # Load commitment direction artifact
    dir_path = Path(args.commitment_direction)
    if not dir_path.exists():
        print(f"ERROR: commitment direction not found: {dir_path}")
        sys.exit(1)
    dir_data = torch.load(dir_path, map_location="cpu", weights_only=True)
    direction_std = dir_data["direction"]
    scaler_mu = dir_data["scaler_mu"]
    scaler_sd = dir_data["scaler_sd"]
    hidden_dim = int(dir_data["hidden_dim"])
    weight_norm = float(dir_data["weight_norm"])
    bias = float(dir_data.get("bias", 0.0))
    print(f"Loaded commitment direction (dim={hidden_dim})")

    # The Phase 3 artifact records which h_0 definition it was trained on
    # (older artifacts predate the field and were implicitly first_gen).
    artifact_source = dir_data.get("h0_source", "first_gen (pre-audit artifact)")
    if str(artifact_source) != args.h0_source:
        print(f"  WARNING: commitment_direction.pt was trained on "
              f"h0_source='{artifact_source}' but this run uses "
              f"'{args.h0_source}'. Re-run Phase 3 with the matching source "
              f"for a coherent pipeline.")

    # Load benchmark + h_0
    examples = load_benchmark(args.benchmark)
    runs_dir = Path(args.runs_dir)
    print(f"Loading h_0 from {runs_dir} (source={args.h0_source}) ...")
    all_examples = collect_h0_and_labels(examples, runs_dir, layer=-1,
                                         source=args.h0_source)
    H0 = torch.stack([e["_h_0"] for e in all_examples])
    y = torch.tensor([0 if e["_correct"] else 1 for e in all_examples], dtype=torch.long)
    n_correct = int((y == 0).sum().item())
    print(f"Loaded {len(all_examples)} examples: {n_correct} correct, "
          f"{len(all_examples) - n_correct} incorrect")

    # Fit mesoscale basis on the *full* set (correct + incorrect, matching probe training)
    print(f"\nFitting K={args.K} supervised+Varimax mesoscale basis "
          f"(epochs={args.basis_epochs})...")
    basis = fit_supervised_mesoscale_basis(
        H0, y, K=args.K,
        scaler_mu=scaler_mu, scaler_sd=scaler_sd,
        epochs=args.basis_epochs,
    )
    gram = basis.basis_raw.T @ basis.basis_raw
    print(f"  basis_raw orthonormality: max|gram - I| = "
          f"{(gram - torch.eye(args.K)).abs().max().item():.2e}")
    print(f"  signed_scores: {[f'{s:.3f}' for s in basis.signed_scores.tolist()]}")

    # Probe scores for selection
    probe_scores = compute_probe_scores(
        H0, direction_std, weight_norm, bias, scaler_mu, scaler_sd,
    )
    lo, hi = args.boundary_range
    selected = select_examples(all_examples, probe_scores, args.n_prompts, lo, hi)
    if not selected:
        print("ERROR: No examples selected.")
        sys.exit(1)
    n_boundary = sum(1 for e in selected if lo <= e["_probe_score"] <= hi)
    print(f"\nSelected {len(selected)} correct examples ({n_boundary} near-boundary)")

    fam_counts: dict[str, int] = {}
    for ex in selected:
        fam_counts[ex["task_family"]] = fam_counts.get(ex["task_family"], 0) + 1
    for fam, c in sorted(fam_counts.items()):
        print(f"  {fam:25s}  {c}")

    # Load model
    print("\nLoading model...", flush=True)
    from sdq.instrumentation.model_loader import load_model
    bundle = load_model(args.config, model_path_override=args.model_path)
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device
    num_layers = model.config.num_hidden_layers

    layer_indices = [l if l >= 0 else num_layers + l for l in args.layers]
    layer_indices = [max(0, min(l, num_layers - 1)) for l in layer_indices]
    print(f"Layers: {layer_indices} (of {num_layers})")

    components = (args.components if args.components is not None
                  else list(range(args.K)))
    bad = [k for k in components if not 0 <= k < args.K]
    if bad:
        print(f"ERROR: --components {bad} out of range for K={args.K}")
        sys.exit(1)
    print(f"Components: {components} (of K={args.K})  "
          f"coordinated: {'skipped' if args.skip_coordinated else 'included'}  "
          f"prefill surface: {args.prefill_positions}")

    # Per-layer prompt-final states of the FULL population, for sigma
    # calibration at the layers actually being perturbed (audit fix: sigmas
    # were previously last-layer for singles and selected-subset for the
    # baselines — neither matched the intervention site).
    print("Loading per-layer prompt-final states for calibration ...")
    h0_by_layer = collect_h0_multi_layer(all_examples, layer_indices)

    n_per_prompt_per_mag = (len(components)
                            + (0 if args.skip_coordinated else 1)
                            + args.n_within_span + args.n_full_random)
    n_gens = (
        len(selected)  # baselines, generated once
        + len(selected) * len(layer_indices)
        * len(args.magnitudes) * n_per_prompt_per_mag
    )
    print(f"Estimated generations (excluding pairwise): {n_gens}")

    # Checkpoint: resume if a compatible one exists
    ckpt_path = _checkpoint_path(args)
    fingerprint = _checkpoint_fingerprint(args, layer_indices, selected,
                                          components)
    ckpt = _load_checkpoint(ckpt_path, fingerprint, args.fresh)

    # Run sweep
    layer_results = run_sweep(
        model, tokenizer, device, selected, basis, h0_by_layer,
        args, layer_indices, components, t_start,
        ckpt, ckpt_path,
    )

    # Significance: Fisher exact vs within-span, Holm-corrected across every
    # cell the decision tree selects over.
    add_significance(layer_results)

    # Pooled per-(layer, component) tests across magnitudes — the primary
    # pre-registered test for targeted confirmation runs.
    add_pooled_component_tests(layer_results)
    print("\n  === POOLED PER-COMPONENT TESTS (across all magnitudes) ===")
    print(f"  {'layer':>8s} {'k':>3s} {'fti_rate':>9s} {'within':>7s} "
          f"{'ratio':>6s} {'p_raw':>8s} {'p_holm':>7s}")
    for summary in layer_results.values():
        for row in summary.get("pooled_components", []):
            print(f"  {row['layer']:>8s} {row['k']:>3d} "
                  f"{row['fti_rate']:>9.3f} {row['within_span_rate']:>7.3f} "
                  f"{row['ratio']:>6.2f} {row['p_raw']:>8.4f} "
                  f"{row['p_holm']:>7.4f}")

    # Decision tree
    decision = apply_decision_tree(layer_results, args)

    print("\n" + "=" * 70)
    print(f"VERDICT: {decision['verdict']}")
    print("=" * 70)
    bs, bc = decision["best_single"], decision["best_coord"]
    p_bs = f"{bs['p_holm']:.4f}" if bs.get("p_holm") is not None else "n/a"
    p_bc = f"{bc['p_holm']:.4f}" if bc.get("p_holm") is not None else "n/a"
    print(f"Best single component: ratio {bs['ratio']:.2f}x at "
          f"{bs['layer']} mag_mult={bs['mag_mult']} k={bs['k']} "
          f"(fti rate={bs['rate']:.3f}, Holm p={p_bs})")
    print(f"Best coordinated:      ratio {bc['ratio']:.2f}x at "
          f"{bc['layer']} mag_mult={bc['mag_mult']} "
          f"(fti rate={bc['rate']:.3f}, Holm p={p_bc})")
    if decision["pairwise_superlinearity"] is not None:
        print(f"Pairwise superlinearity: median "
              f"{decision['pairwise_superlinearity']['median']:.2f} at "
              f"{decision['pairwise_superlinearity']['layer']}")
    print(f"\nNext step: {decision['next_step']}")

    # Save results
    results = {
        "summary": {
            "K": args.K,
            "n_examples": len(selected),
            "n_boundary": n_boundary,
            "boundary_range": [lo, hi],
            "magnitudes": args.magnitudes,
            "layers": layer_indices,
            "n_within_span": args.n_within_span,
            "n_full_random": args.n_full_random,
            "include_pairwise": args.include_pairwise,
            "components": components,
            "skip_coordinated": args.skip_coordinated,
            "prefill_positions": args.prefill_positions,
            "h0_source": args.h0_source,
            "family_counts": fam_counts,
            "component_stds_last_layer": basis.component_stds.tolist(),
            "signed_scores": basis.signed_scores.tolist(),
            "primary_metric": "flipped_to_incorrect",
        },
        "decision": decision,
        "per_layer": layer_results,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "elapsed_seconds": time.time() - t_start,
    }
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"Checkpoint {ckpt_path} removed (raw trials are in {out_path}).")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
