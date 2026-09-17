#!/usr/bin/env python3
"""SDQ Phase 3: Commitment geometry probes.

Reframes SDQ from "reasoning dynamics" to "answer commitment geometry."

Working thesis under test:
    Transformer prompt encodings contain a low-dimensional "commitment
    subspace" that predicts final answer correctness before any generation
    occurs. This subspace is partially family-conditional and partially
    general.

This script runs the full Phase 3 evaluation:
  1. Trains commitment probes (h_0 linear, z_0 linear, low-rank variants)
  2. Compares against logit-confidence baseline and family prior
  3. Evaluates per-family (stratified) — the only honest metric
  4. Runs leave-one-family-out cross-family transfer
  5. Estimates commitment subspace dimensionality via PCA rank sweep
  6. Extracts the commitment direction for Phase 4 causal intervention

Stopping rule (from plan):
    Within-family AUROC for h_0 probe must exceed BOTH the family prior
    AND logit-confidence baseline by >= 0.03 on held-out data.

Usage:
    python run_sdq_commitment.py --runs-dir data/runs/gemma-2-2b/gemma-2-2b
    python run_sdq_commitment.py --runs-dir data/runs/gemma-2-2b/gemma-2-2b \\
        --encoder-checkpoint ews_model2.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from sdq.eval.commitment_probe import (
    CommitmentDirection,
    LinearCommitmentProbe,
    LowRankCommitmentProbe,
    ProbeResult,
    SubspaceResult,
    TransferResult,
    compute_auroc,
    cross_family_transfer,
    extract_commitment_direction,
    extract_logit_features,
    format_probe_result,
    format_subspace_result,
    format_transfer_results,
    mean_within_family_auroc,
    per_family_auroc,
    subspace_rank_sweep,
    train_probe,
)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDQ Phase 3: Commitment geometry probes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--runs-dir", required=True,
                   help="Directory containing captured generation runs")
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--layer", type=int, default=-1,
                   help="Transformer layer (0-indexed, -1=last)")
    p.add_argument("--h0-source", choices=["prompt_final", "first_gen"],
                   default="prompt_final",
                   help="Which state to use as h_0. 'prompt_final' is the "
                        "last-prompt-token state from activations.pt (truly "
                        "before generation). 'first_gen' is the legacy "
                        "definition: gen_activations[0], the state at the "
                        "first generated token (after it was emitted).")
    p.add_argument("--skip-leakage-check", action="store_true",
                   help="Skip the side-by-side AUROC comparison between the "
                        "two h_0 definitions.")
    p.add_argument("--min-gen-tokens", type=int, default=4)
    p.add_argument("--max-gen-tokens", type=int, default=64)
    p.add_argument("--test-fraction", type=float, default=0.2)

    # Encoder (optional, for z_0 probe)
    p.add_argument("--encoder-checkpoint", default=None,
                   help="Path to pretrained MultiScaleConvEncoder or EWS model "
                        "checkpoint containing encoder state dict")
    p.add_argument("--latent-dim", type=int, default=128)

    # Probe training
    p.add_argument("--probe-epochs", type=int, default=400)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--probe-weight-decay", type=float, default=1e-3)

    # Subspace analysis
    p.add_argument("--max-rank", type=int, default=256,
                   help="Maximum PCA rank to test in subspace sweep")
    p.add_argument("--skip-subspace", action="store_true",
                   help="Skip the (slower) subspace rank sweep")
    p.add_argument("--skip-transfer", action="store_true",
                   help="Skip cross-family transfer evaluation")

    # Output
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="commitment_results.json")
    p.add_argument("--save-direction", default="commitment_direction.pt",
                   help="Save the commitment direction tensor for Phase 4")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_commitment_dataset(
    runs_dir: Path,
    benchmark: dict[str, dict],
    layer_idx: int,
    min_gen: int,
    max_gen: int,
    device: str,
    h0_source: str = "prompt_final",
) -> list[dict]:
    """Load h_0 and logit features for each captured run.

    Two h_0 definitions are loaded side by side (audit 2026-06):
      h_0_prompt_final — activations.pt[layer, -1, :], the last-prompt-token
          state. Truly "before generation begins". This is the default.
      h_0_first_gen — gen_activations[0, layer, :], the state at the first
          generated token (legacy definition; the model has already emitted
          one token). Kept for the leakage comparison.

    The primary `h_0` key points at the source selected via `h0_source`.
    Runs missing the file for the selected source are skipped — the dataset
    never mixes definitions.
    """
    from sdq.labels.outcome_labeler import label_run

    dataset: list[dict] = []
    n_missing_primary = 0
    run_dirs = sorted(d for d in runs_dir.iterdir()
                      if d.is_dir() and (d / "gen_activations.pt").exists())
    if not run_dirs:
        print(f"ERROR: No generation runs in {runs_dir}")
        sys.exit(1)

    for rd in run_dirs:
        with open(rd / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)

        prompt_id = meta.get("prompt", {}).get("id", rd.name)
        if prompt_id not in benchmark:
            continue
        bm = benchmark[prompt_id]
        new_tokens = meta.get("output", {}).get("new_tokens", "")

        gen_act = torch.load(rd / "gen_activations.pt",
                             map_location=device, weights_only=True)
        if gen_act.shape[0] < min_gen:
            continue
        gen_act = gen_act[:max_gen]

        actual_layer = layer_idx if layer_idx >= 0 else gen_act.shape[1] + layer_idx
        actual_layer = max(0, min(actual_layer, gen_act.shape[1] - 1))
        h_all = gen_act[:, actual_layer, :]  # [T, D]
        h_0_first_gen = h_all[0].float().cpu()

        # Prompt-final state: activations.pt[layer, -1, :]
        h_0_prompt_final = None
        act_path = rd / "activations.pt"
        if act_path.exists():
            act = torch.load(act_path, map_location="cpu", weights_only=True)
            pf_layer = layer_idx if layer_idx >= 0 else act.shape[0] + layer_idx
            pf_layer = max(0, min(pf_layer, act.shape[0] - 1))
            h_0_prompt_final = act[pf_layer, -1, :].float()

        h_0_primary = (h_0_prompt_final if h0_source == "prompt_final"
                       else h_0_first_gen)
        if h_0_primary is None:
            n_missing_primary += 1
            continue

        # The first generation-step score is the distribution that produced
        # the first generated token.  Older captures did not save
        # ``gen_logits.pt``, but they did save the equivalent prompt-final
        # distribution in ``logits.pt``.  Prefer the former and fall back to
        # the latter so the declared logit baseline is not silently absent.
        logit_0 = None
        gl_path = rd / "gen_logits.pt"
        if gl_path.exists():
            gl = torch.load(gl_path, map_location=device, weights_only=True)
            if gl.shape[0] > 0:
                logit_0 = gl[0].float()
        elif (rd / "logits.pt").exists():
            prompt_logits = torch.load(
                rd / "logits.pt", map_location=device, weights_only=True
            )
            logit_0 = prompt_logits[0, -1].float()

        outcome = label_run(new_tokens, bm["answer_id"], bm["task_family"])

        dataset.append({
            "h_all": h_all,
            "h_0": h_0_primary,
            "h_0_prompt_final": h_0_prompt_final,
            "h_0_first_gen": h_0_first_gen,
            "logit_0": logit_0,
            "correct": outcome.correct,
            "task_family": bm["task_family"],
            "example_id": prompt_id,
        })

    if n_missing_primary:
        print(f"  Skipped {n_missing_primary} runs missing the file for "
              f"h0-source='{h0_source}' (definitions are never mixed)")
    return dataset


# ── Encoder z_0 extraction ──────────────────────────────────────────────────

def extract_z0_features(
    dataset: list[dict],
    encoder_checkpoint: str,
    latent_dim: int,
    device: str,
) -> Tensor | None:
    """Load encoder and compute z_0 for each example."""
    ckpt_path = Path(encoder_checkpoint)
    if not ckpt_path.exists():
        print(f"  Encoder checkpoint not found: {ckpt_path}")
        return None

    from sdq.latent.encoder import MultiScaleConvEncoder

    hidden_dim = dataset[0]["h_all"].shape[1]
    encoder = MultiScaleConvEncoder(
        hidden_dim=hidden_dim, latent_dim=latent_dim,
    ).to(device)

    # Try loading as standalone encoder dict, else as EWS model dict
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "encoder" in state:
        encoder.load_state_dict(state["encoder"])
    elif any(k.startswith("convs.") for k in state):
        encoder.load_state_dict(state)
    else:
        print(f"  Could not find encoder weights in {ckpt_path}")
        return None

    encoder.eval()
    z0_list = []
    with torch.no_grad():
        for d in dataset:
            z = encoder(d["h_all"].float())  # [T, latent_dim]
            z0_list.append(z[0].cpu())

    return torch.stack(z0_list)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    t_start = time.time()

    print("=" * 70)
    print("SDQ PHASE 3: COMMITMENT GEOMETRY PROBES")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    with open(args.benchmark, encoding="utf-8") as f:
        bm_raw = json.load(f)
    examples = bm_raw if isinstance(bm_raw, list) else bm_raw.get("examples", [])
    benchmark = {e["example_id"]: e for e in examples}
    print(f"Benchmark: {len(benchmark)} examples")

    runs_dir = Path(args.runs_dir)
    print(f"Loading from {runs_dir} ...")
    print(f"h_0 definition: {args.h0_source}"
          + (" (last prompt token, before generation)"
             if args.h0_source == "prompt_final"
             else " (LEGACY: first generated token — after one token emitted)"))
    dataset = load_commitment_dataset(
        runs_dir, benchmark, args.layer,
        args.min_gen_tokens, args.max_gen_tokens, device,
        h0_source=args.h0_source,
    )
    n_total = len(dataset)
    n_correct = sum(1 for d in dataset if d["correct"])
    n_incorrect = n_total - n_correct
    print(f"Loaded {n_total} runs: {n_correct} correct, {n_incorrect} incorrect")

    if not dataset:
        sys.exit(1)

    hidden_dim = dataset[0]["h_0"].shape[0]
    has_logits = all(d["logit_0"] is not None for d in dataset)
    print(f"Hidden dim: {hidden_dim}, logits available: {has_logits}")

    # ── Train/test split (same seed/convention as EWS for comparability) ──
    torch.manual_seed(args.seed)
    perm = torch.randperm(n_total)
    n_test = max(1, int(n_total * args.test_fraction))
    test_idx = set(perm[:n_test].tolist())

    train_data = [dataset[i] for i in range(n_total) if i not in test_idx]
    test_data = [dataset[i] for i in range(n_total) if i in test_idx]
    print(f"Train: {len(train_data)}  Test: {len(test_data)}")

    # Label vectors (1 = incorrect, matching EWS convention)
    y_train = torch.tensor([0 if d["correct"] else 1 for d in train_data])
    y_test = torch.tensor([0 if d["correct"] else 1 for d in test_data])
    train_families = [d["task_family"] for d in train_data]
    test_families = [d["task_family"] for d in test_data]
    all_families = [d["task_family"] for d in dataset]
    y_all = torch.tensor([0 if d["correct"] else 1 for d in dataset])

    # Family breakdown
    fam_breakdown: dict[str, dict[str, int]] = {}
    for d in dataset:
        f = d["task_family"]
        if f not in fam_breakdown:
            fam_breakdown[f] = {"correct": 0, "incorrect": 0}
        fam_breakdown[f]["correct" if d["correct"] else "incorrect"] += 1
    print("\nFamily breakdown:")
    for f, counts in sorted(fam_breakdown.items()):
        total = counts["correct"] + counts["incorrect"]
        pct = counts["correct"] / total * 100
        print(f"  {f:25s}  {counts['correct']:3d}/{total:3d} correct ({pct:.0f}%)")

    # ── Feature matrices ─────────────────────────────────────────────────
    X_train_h = torch.stack([d["h_0"] for d in train_data])
    X_test_h = torch.stack([d["h_0"] for d in test_data])
    X_all_h = torch.stack([d["h_0"] for d in dataset])

    results: dict = {
        "meta": {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "hidden_dim": hidden_dim,
            "family_breakdown": fam_breakdown,
        },
        "probes": {},
    }

    # ==================================================================
    # PROBE 1: h_0 linear (the primary probe)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PROBE 1: h_0 linear (raw hidden state at t=0)")
    print("=" * 70)

    h0_probe = LinearCommitmentProbe(hidden_dim)
    h0_result = train_probe(
        h0_probe, X_train_h, y_train, X_test_h, y_test, test_families,
        name="h_0_linear",
        epochs=args.probe_epochs, lr=args.probe_lr,
        weight_decay=args.probe_weight_decay, device=device,
    )
    print(format_probe_result(h0_result))
    results["probes"]["h_0_linear"] = {
        "pooled_auroc": h0_result.pooled_auroc,
        "mean_within_family_auroc": h0_result.mean_within_family_auroc,
        "family_aurocs": h0_result.family_aurocs,
        "h0_source": args.h0_source,
    }

    # ==================================================================
    # LEAKAGE CHECK: prompt_final vs first_gen h_0
    # ==================================================================
    # The legacy h_0 (gen_activations[0]) is the state at the first
    # *generated* token — the model has already emitted one token, which can
    # leak answer information into the probe. Training the same probe on the
    # alternate definition quantifies that leakage: a large AUROC gap means
    # the "predicts before generation" number was inflated.
    if not args.skip_leakage_check:
        alt_key = ("h_0_first_gen" if args.h0_source == "prompt_final"
                   else "h_0_prompt_final")
        has_alt = all(d[alt_key] is not None for d in dataset)
        if has_alt:
            print("\n" + "=" * 70)
            print(f"LEAKAGE CHECK: probe on alternate h_0 ({alt_key})")
            print("=" * 70)

            X_train_alt = torch.stack([d[alt_key] for d in train_data])
            X_test_alt = torch.stack([d[alt_key] for d in test_data])
            alt_result = train_probe(
                LinearCommitmentProbe(hidden_dim),
                X_train_alt, y_train, X_test_alt, y_test, test_families,
                name=f"h_0_linear[{alt_key}]",
                epochs=args.probe_epochs, lr=args.probe_lr,
                weight_decay=args.probe_weight_decay, device=device,
            )
            print(format_probe_result(alt_result))

            X_all_alt = torch.stack([d[alt_key] for d in dataset])
            X_all_pri = torch.stack([d["h_0"] for d in dataset]).cpu()
            cos = F.cosine_similarity(X_all_pri, X_all_alt.cpu(), dim=1)
            gap = h0_result.pooled_auroc - alt_result.pooled_auroc
            print(f"\n  Mean cosine(primary h_0, alternate h_0): "
                  f"{cos.mean().item():.4f} (min {cos.min().item():.4f})")
            print(f"  Pooled AUROC  {args.h0_source}={h0_result.pooled_auroc:.4f}  "
                  f"vs alternate={alt_result.pooled_auroc:.4f}  "
                  f"(gap {gap:+.4f})")
            if args.h0_source == "prompt_final" and gap < -0.02:
                print("  NOTE: first_gen probe is stronger — part of the legacy "
                      "signal was first-token leakage, not prompt encoding.")

            results["leakage_check"] = {
                "alternate_source": alt_key,
                "alt_pooled_auroc": alt_result.pooled_auroc,
                "alt_mean_within_family_auroc": alt_result.mean_within_family_auroc,
                "alt_family_aurocs": alt_result.family_aurocs,
                "primary_pooled_auroc": h0_result.pooled_auroc,
                "pooled_auroc_gap_primary_minus_alt": gap,
                "mean_cosine_between_sources": cos.mean().item(),
            }
        else:
            print("\n[Leakage check skipped — alternate h_0 source not "
                  "available for all runs]")

    # ==================================================================
    # PROBE 2: z_0 linear (encoder latent, optional)
    # ==================================================================
    z0_result: ProbeResult | None = None
    if args.encoder_checkpoint:
        print("\n" + "=" * 70)
        print(f"PROBE 2: z_0 linear (encoder from {args.encoder_checkpoint})")
        print("=" * 70)

        z0_all = extract_z0_features(dataset, args.encoder_checkpoint,
                                     args.latent_dim, device)
        if z0_all is not None:
            X_train_z = z0_all[[i for i in range(n_total) if i not in test_idx]]
            X_test_z = z0_all[[i for i in range(n_total) if i in test_idx]]

            z0_probe = LinearCommitmentProbe(args.latent_dim)
            z0_result = train_probe(
                z0_probe, X_train_z, y_train, X_test_z, y_test, test_families,
                name="z_0_linear",
                epochs=args.probe_epochs, lr=args.probe_lr,
                weight_decay=args.probe_weight_decay, device=device,
            )
            print(format_probe_result(z0_result))
            results["probes"]["z_0_linear"] = {
                "pooled_auroc": z0_result.pooled_auroc,
                "mean_within_family_auroc": z0_result.mean_within_family_auroc,
                "family_aurocs": z0_result.family_aurocs,
            }

    # ==================================================================
    # BASELINE 1: Logit confidence (entropy, margin, top-k)
    # ==================================================================
    logit_result: ProbeResult | None = None
    if has_logits:
        print("\n" + "=" * 70)
        print("BASELINE: Logit confidence features at t=0")
        print("=" * 70)

        X_train_logit = torch.stack([
            extract_logit_features(d["logit_0"]) for d in train_data
        ])
        X_test_logit = torch.stack([
            extract_logit_features(d["logit_0"]) for d in test_data
        ])

        logit_probe = LinearCommitmentProbe(5)  # 5 logit features
        logit_result = train_probe(
            logit_probe, X_train_logit, y_train, X_test_logit, y_test,
            test_families, name="logit_confidence",
            epochs=args.probe_epochs, lr=args.probe_lr,
            weight_decay=args.probe_weight_decay, device=device,
        )
        print(format_probe_result(logit_result))
        results["probes"]["logit_confidence"] = {
            "pooled_auroc": logit_result.pooled_auroc,
            "mean_within_family_auroc": logit_result.mean_within_family_auroc,
            "family_aurocs": logit_result.family_aurocs,
        }
    else:
        print("\n[Logit baseline skipped — no gen_logits captured]")

    # ==================================================================
    # BASELINE 2: Family prior
    # ==================================================================
    print("\n" + "=" * 70)
    print("BASELINE: Family prior (train-set failure rate)")
    print("=" * 70)

    fam_rate: dict[str, float] = {}
    fam_counts: dict[str, int] = {}
    for f, y in zip(train_families, y_train.tolist()):
        fam_rate[f] = fam_rate.get(f, 0.0) + y
        fam_counts[f] = fam_counts.get(f, 0) + 1
    for f in fam_rate:
        fam_rate[f] /= fam_counts[f]

    prior_scores = torch.tensor([fam_rate.get(f, 0.5) for f in test_families])
    prior_pooled = compute_auroc(prior_scores, y_test)
    prior_fam = per_family_auroc(prior_scores, y_test, test_families)
    prior_wf = mean_within_family_auroc(prior_scores, y_test, test_families)

    print(f"  Pooled AUROC:              {prior_pooled:.4f}")
    print(f"  Mean within-family AUROC:  {prior_wf:.4f}  (trivially 0.5)")
    results["probes"]["family_prior"] = {
        "pooled_auroc": prior_pooled,
        "mean_within_family_auroc": prior_wf,
        "family_aurocs": prior_fam,
        "family_failure_rates_train": fam_rate,
    }

    # ==================================================================
    # COMPARISON TABLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    comparison_rows: list[dict] = []
    for name, pr in [
        ("h_0_linear", h0_result),
        ("z_0_linear", z0_result),
        ("logit_confidence", logit_result),
    ]:
        if pr is not None:
            comparison_rows.append({
                "name": name,
                "pooled": pr.pooled_auroc,
                "within_family": pr.mean_within_family_auroc,
            })
    comparison_rows.append({
        "name": "family_prior",
        "pooled": prior_pooled,
        "within_family": prior_wf,
    })

    print(f"  {'Probe':22s}  {'Pooled':>8s}  {'Within-fam':>10s}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*10}")
    for row in comparison_rows:
        print(f"  {row['name']:22s}  {row['pooled']:8.4f}  {row['within_family']:10.4f}")
    results["comparison"] = comparison_rows

    # ── Stopping rule evaluation ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("STOPPING RULE EVALUATION (Phase 3)")
    print("-" * 70)

    h0_wf = h0_result.mean_within_family_auroc
    margin = 0.03

    beats_family_prior = h0_wf > prior_wf + margin
    beats_logit = None
    logit_wf = None
    if logit_result is not None:
        logit_wf = logit_result.mean_within_family_auroc
        beats_logit = h0_wf > logit_wf + margin

    # A missing required baseline makes the stopping rule inconclusive; it is
    # not evidence that the hidden-state probe beat that baseline.
    phase3_pass = beats_family_prior and beats_logit is True

    print(f"  h_0 within-family AUROC:     {h0_wf:.4f}")
    print(f"  Family prior within-family:  {prior_wf:.4f}")
    if logit_wf is not None:
        print(f"  Logit confidence within-fam: {logit_wf:.4f}")
    print(f"  Required margin:             {margin}")
    print(f"  Beats family prior (+{margin}):  {beats_family_prior}")
    print(f"  Beats logit confidence (+{margin}): {beats_logit}")
    print(f"  >>> PHASE 3 PASS: {phase3_pass}")

    results["stopping_rule"] = {
        "h0_within_family_auroc": h0_wf,
        "family_prior_within_family_auroc": prior_wf,
        "logit_confidence_within_family_auroc": logit_wf,
        "margin": margin,
        "beats_family_prior": beats_family_prior,
        "beats_logit_confidence": beats_logit,
        "phase3_pass": phase3_pass,
    }

    if not phase3_pass:
        print("\n  Phase 3 FAILED the stopping rule.")
        print("  The commitment thesis is not supported: hidden states do not")
        print("  contain more within-family correctness information than the")
        print("  baselines for this model/benchmark combination.")

    # ==================================================================
    # CROSS-FAMILY TRANSFER (3.5)
    # ==================================================================
    if not args.skip_transfer:
        print("\n" + "=" * 70)
        print("CROSS-FAMILY TRANSFER (leave-one-out)")
        print("=" * 70)

        transfer_results = cross_family_transfer(
            X_all_h, y_all, all_families, hidden_dim,
            epochs=args.probe_epochs, lr=args.probe_lr,
            weight_decay=args.probe_weight_decay, device=device,
        )
        print(format_transfer_results(transfer_results))

        results["cross_family_transfer"] = [
            {
                "held_out_family": r.held_out_family,
                "auroc": r.auroc if r.auroc == r.auroc else None,
                "n_test": r.n_test,
                "n_pos": r.n_pos,
                "n_neg": r.n_neg,
            }
            for r in transfer_results
        ]

        valid_transfers = [r.auroc for r in transfer_results if r.auroc == r.auroc]
        if valid_transfers:
            results["cross_family_transfer_summary"] = {
                "mean_auroc": sum(valid_transfers) / len(valid_transfers),
                "n_above_chance": sum(1 for v in valid_transfers if v > 0.55),
                "n_families": len(valid_transfers),
            }

    # ==================================================================
    # SUBSPACE RANK SWEEP (3.4)
    # ==================================================================
    if not args.skip_subspace:
        print("\n" + "=" * 70)
        print("SUBSPACE RANK SWEEP")
        print("=" * 70)

        subspace = subspace_rank_sweep(
            X_train_h, y_train, X_test_h, y_test, test_families,
            full_auroc=h0_result.pooled_auroc,
            max_rank=min(args.max_rank, hidden_dim),
            epochs=args.probe_epochs, lr=args.probe_lr,
            weight_decay=args.probe_weight_decay, device=device,
        )
        print(format_subspace_result(subspace))

        results["subspace"] = {
            "rank_aurocs": {str(k): v for k, v in subspace.rank_aurocs.items()},
            "rank_within_family_aurocs": {
                str(k): v for k, v in subspace.rank_within_family_aurocs.items()
            },
            "full_auroc": subspace.full_auroc,
            "rank_for_90pct": subspace.rank_for_90pct,
            "rank_for_95pct": subspace.rank_for_95pct,
            "pca_explained_variance_top20": subspace.pca_explained_variance[:20],
        }

    # ==================================================================
    # COMMITMENT DIRECTION EXTRACTION
    # ==================================================================
    print("\n" + "=" * 70)
    print("COMMITMENT DIRECTION")
    print("=" * 70)

    # Retrain a final probe on ALL data for maximum power.
    # NOTE: this probe is for *causal* downstream use (Phases 4/5), not for
    # reporting AUROC — it has seen the test split.
    from sdq.eval.commitment_probe import Standardizer, fit_probe
    scaler = Standardizer.fit(X_all_h)
    final_probe = fit_probe(
        LinearCommitmentProbe(hidden_dim),
        scaler.transform(X_all_h), y_all,
        epochs=args.probe_epochs, lr=args.probe_lr,
        weight_decay=args.probe_weight_decay, device=device,
    )

    direction = extract_commitment_direction(final_probe, top_k=20)

    print(f"  Weight vector norm: {direction.weight_norm:.4f}")
    print(f"  Bias:               {direction.bias:.4f}")
    print(f"  Top 20 contributing dimensions (of {hidden_dim}):")
    for idx, w in zip(direction.top_components_idx[:10],
                      direction.top_components_weight[:10]):
        print(f"    dim {idx:5d}  |w|={w:.4f}")
    print(f"    ... ({len(direction.top_components_idx)} total)")

    results["commitment_direction"] = {
        "weight_norm": direction.weight_norm,
        "bias": direction.bias,
        "top_20_dims": direction.top_components_idx,
        "top_20_weights": direction.top_components_weight,
    }

    # Save direction tensor for Phase 4 intervention
    # Compute raw-space direction: d(score)/d(h_i) = w_i / sd_i
    from sdq.eval.commitment_intervention import direction_to_raw_space
    dir_raw = direction_to_raw_space(direction.direction.cpu(), scaler.sd.cpu())

    direction_save = {
        "direction": direction.direction.cpu(),
        "direction_raw": dir_raw,
        "weight_norm": direction.weight_norm,
        "bias": direction.bias,
        "scaler_mu": scaler.mu.cpu(),
        "scaler_sd": scaler.sd.cpu(),
        "hidden_dim": hidden_dim,
        "trained_on_n": n_total,
        "h0_source": args.h0_source,
        "layer": args.layer,
    }
    torch.save(direction_save, args.save_direction)
    cos_sim = torch.dot(direction.direction.cpu(), dir_raw).item()
    print(f"  Direction saved to {args.save_direction}")
    print(f"  Raw-space direction included (cosine with std-space: {cos_sim:.4f})")

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================
    elapsed = time.time() - t_start
    results["args"] = {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items()}
    results["elapsed_seconds"] = elapsed

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Total time: {elapsed:.0f}s")

    # ── Final verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if phase3_pass:
        logit_wf_str = f"{logit_wf:.4f}" if logit_wf is not None else "N/A"
        print("VERDICT: Commitment geometry thesis SUPPORTED.")
        print(f"  h_0 linear probe within-family AUROC ({h0_wf:.4f}) exceeds")
        print(f"  family prior ({prior_wf:.4f}) and logit confidence "
              f"({logit_wf_str}) by >= {margin}.")
        print("  Proceed to Phase 4: causal intervention on the commitment subspace.")
    else:
        print("VERDICT: Commitment geometry thesis NOT SUPPORTED.")
        print("  See stopping rule details above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
