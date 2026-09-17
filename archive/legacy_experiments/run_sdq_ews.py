#!/usr/bin/env python3
"""SDQ Early-Warning System — main training pipeline.

End-to-end training for the proxy-drift early-warning monitor:
  Phase 1: Load or train semantic encoder (MultiScaleConvEncoder)
  Phase 2: Train drift encoder + temporal risk model jointly
  Phase 3: Evaluate with AUROC, AUPRC, lead-time metrics

Usage:
    python run_sdq_ews.py --runs-dir data/runs/gemma-2-2b
    python run_sdq_ews.py --runs-dir data/runs/gemma-2-2b --encoder-checkpoint ews_encoder.pt
    python run_sdq_ews.py --runs-dir data/runs/gemma-2-2b --output ews_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDQ Early-Warning System training")

    # Data
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--layer", type=int, default=-1,
                   help="Transformer layer to use (0-indexed, -1=last)")
    p.add_argument("--min-gen-tokens", type=int, default=4)
    p.add_argument("--max-gen-tokens", type=int, default=64)
    p.add_argument("--test-fraction", type=float, default=0.2)

    # Encoder
    p.add_argument("--encoder-checkpoint", default=None,
                   help="Path to pretrained encoder weights (skip encoder training)")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--encoder-lr", type=float, default=1e-3)
    p.add_argument("--encoder-epochs", type=int, default=30)

    # Drift encoder
    p.add_argument("--drift-dim", type=int, default=64)
    p.add_argument("--drift-dropout", type=float, default=0.1)

    # Risk model
    p.add_argument("--risk-hidden-dim", type=int, default=128)
    p.add_argument("--risk-num-layers", type=int, default=2)
    p.add_argument("--risk-dropout", type=float, default=0.1)
    p.add_argument("--use-attention", action="store_true", default=False)

    # Training
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--risk-horizon", type=int, default=5,
                   help="Future window k for risk labels")
    p.add_argument("--recover-weight", type=float, default=0.5,
                   help="Weight for recoverability loss relative to risk loss")

    # Output
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="ews_results.json")
    p.add_argument("--save-checkpoint", default="ews_model.pt")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_benchmark_lookup(path: str | Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    examples = data if isinstance(data, list) else data.get("examples", [])
    return {e["example_id"]: e for e in examples}


def load_ews_dataset(
    runs_dir: Path,
    benchmark: dict[str, dict],
    layer_idx: int,
    min_gen: int,
    max_gen: int,
    device: str,
) -> list[dict]:
    """Load all generation runs and prepare training data.

    Returns list of dicts with keys:
      - gen_h: [T, D] generation hidden states for selected layer
      - correct: bool
      - task_family: str
      - example_id: str
    """
    from sdq.labels.outcome_labeler import label_run

    dataset: list[dict] = []
    run_dirs = sorted(d for d in runs_dir.iterdir()
                      if d.is_dir() and (d / "gen_activations.pt").exists())

    if not run_dirs:
        print(f"ERROR: No generation runs found in {runs_dir}")
        print("Run capture_runs_ews.py first.")
        sys.exit(1)

    for rd in run_dirs:
        with open(rd / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)

        prompt_id = meta.get("prompt", {}).get("id", rd.name)
        if prompt_id not in benchmark:
            continue

        bm = benchmark[prompt_id]
        new_tokens = meta.get("output", {}).get("new_tokens", "")

        gen_act = torch.load(rd / "gen_activations.pt", map_location=device, weights_only=True)
        if gen_act.shape[0] < min_gen:
            continue
        gen_act = gen_act[:max_gen]

        actual_layer = layer_idx if layer_idx >= 0 else gen_act.shape[1] + layer_idx
        actual_layer = max(0, min(actual_layer, gen_act.shape[1] - 1))

        h = gen_act[:, actual_layer, :]  # [T, D]

        outcome = label_run(new_tokens, bm["answer_id"], bm["task_family"])

        # Also load gen_logits if available (for baseline comparison)
        gen_logits = None
        gen_logits_path = rd / "gen_logits.pt"
        if gen_logits_path.exists():
            gen_logits = torch.load(gen_logits_path, map_location=device, weights_only=True)
            gen_logits = gen_logits[:max_gen]

        dataset.append({
            "gen_h": h,
            "correct": outcome.correct,
            "task_family": bm["task_family"],
            "example_id": prompt_id,
            "gen_logits": gen_logits,
        })

    return dataset


# ── Training utilities ───────────────────────────────────────────────────────

def compute_auroc(scores: Tensor, labels: Tensor) -> float:
    """AUROC without sklearn."""
    if labels.unique().numel() < 2:
        return 0.5
    pos = (labels == 1)
    neg = (labels == 0)
    n_pos = pos.sum().item()
    n_neg = neg.sum().item()
    if n_pos == 0 or n_neg == 0:
        return 0.5

    sorted_indices = scores.argsort(descending=True)
    sorted_labels = labels[sorted_indices]

    tp = 0.0
    auc = 0.0
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            auc += tp

    return auc / (n_pos * n_neg)


def make_batches(
    dataset: list[dict],
    batch_size: int,
    shuffle: bool = True,
) -> list[list[dict]]:
    """Split dataset into batches."""
    if shuffle:
        perm = torch.randperm(len(dataset)).tolist()
        dataset = [dataset[i] for i in perm]

    batches = []
    for i in range(0, len(dataset), batch_size):
        batches.append(dataset[i:i + batch_size])
    return batches


# ── Main training pipeline ───────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    print("=" * 70)
    print("SDQ EARLY-WARNING SYSTEM — TRAINING")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    benchmark = load_benchmark_lookup(args.benchmark)
    print(f"Benchmark: {len(benchmark)} examples")

    runs_dir = Path(args.runs_dir)
    print(f"Loading generation runs from {runs_dir} ...")
    dataset = load_ews_dataset(
        runs_dir, benchmark, args.layer,
        args.min_gen_tokens, args.max_gen_tokens, device,
    )

    n_correct = sum(1 for d in dataset if d["correct"])
    n_incorrect = len(dataset) - n_correct
    print(f"Loaded {len(dataset)} runs: {n_correct} correct, {n_incorrect} incorrect")

    if len(dataset) == 0:
        print("ERROR: No valid runs.")
        sys.exit(1)

    hidden_dim = dataset[0]["gen_h"].shape[1]
    print(f"Hidden dim: {hidden_dim}")

    # ── Train/test split ─────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    perm = torch.randperm(len(dataset))
    n_test = max(1, int(len(dataset) * args.test_fraction))
    test_idx = set(perm[:n_test].tolist())

    train_data = [dataset[i] for i in range(len(dataset)) if i not in test_idx]
    test_data = [dataset[i] for i in range(len(dataset)) if i in test_idx]
    print(f"Train: {len(train_data)}  Test: {len(test_data)}")

    # ── Phase 1: Semantic encoder ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: SEMANTIC ENCODER")
    print("=" * 70)

    from sdq.latent.encoder import MultiScaleConvEncoder

    encoder = MultiScaleConvEncoder(
        hidden_dim=hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    if args.encoder_checkpoint and Path(args.encoder_checkpoint).exists():
        print(f"Loading encoder from {args.encoder_checkpoint}")
        state = torch.load(args.encoder_checkpoint, map_location=device, weights_only=True)
        encoder.load_state_dict(state)
    else:
        print(f"Training encoder ({args.encoder_epochs} epochs)...")
        enc_optimizer = AdamW(encoder.parameters(), lr=args.encoder_lr)
        enc_scheduler = CosineAnnealingLR(enc_optimizer, T_max=args.encoder_epochs)

        encoder.train()
        for epoch in range(args.encoder_epochs):
            epoch_loss = 0.0
            n_samples = 0

            batches = make_batches(train_data, args.batch_size)
            for batch in batches:
                loss_batch = torch.tensor(0.0, device=device)

                for sample in batch:
                    h = sample["gen_h"]  # [T, D]
                    z = encoder(h.float())       # [T, latent_dim]

                    # Reconstruction objective: z should preserve information from h
                    # Use a contrastive-style temporal coherence loss:
                    # nearby timesteps should have similar z, distant ones less so
                    if z.shape[0] >= 4:
                        z_norm = F.normalize(z, dim=-1)
                        # Positive pairs: adjacent steps
                        pos_sim = (z_norm[:-1] * z_norm[1:]).sum(dim=-1)
                        # Negative pairs: random distant steps
                        neg_idx = torch.randperm(z.shape[0], device=device)
                        neg_sim = (z_norm * z_norm[neg_idx]).sum(dim=-1)
                        contrastive = -pos_sim.mean() + F.relu(neg_sim.mean() - 0.1 + 0.5)
                        loss_batch = loss_batch + contrastive

                    n_samples += 1

                if n_samples > 0:
                    loss_batch = loss_batch / n_samples
                    enc_optimizer.zero_grad()
                    loss_batch.backward()
                    enc_optimizer.step()
                    epoch_loss += loss_batch.item()

            enc_scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{args.encoder_epochs}  loss={epoch_loss:.4f}")

        if args.encoder_checkpoint:
            torch.save(encoder.state_dict(), args.encoder_checkpoint)
            print(f"  Encoder saved to {args.encoder_checkpoint}")

    encoder.eval()

    # ── Phase 2: Drift encoder + risk model training ─────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: DRIFT ENCODER + TEMPORAL RISK MODEL")
    print("=" * 70)

    from sdq.features.drift_encoder import DriftEncoder, extract_raw_drift_features
    from sdq.models.temporal_risk_model import TemporalRiskModel

    drift_encoder = DriftEncoder(
        latent_dim=args.latent_dim,
        drift_dim=args.drift_dim,
        dropout=args.drift_dropout,
    ).to(device)

    risk_model = TemporalRiskModel(
        semantic_dim=args.latent_dim,
        drift_dim=args.drift_dim,
        hidden_dim=args.risk_hidden_dim,
        num_layers=args.risk_num_layers,
        dropout=args.risk_dropout,
        use_attention=args.use_attention,
    ).to(device)

    all_params = list(drift_encoder.parameters()) + list(risk_model.parameters())
    optimizer = AdamW(all_params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Class balance for loss weighting
    train_n_pos = sum(1 for d in train_data if not d["correct"])
    train_n_neg = sum(1 for d in train_data if d["correct"])
    if train_n_pos > 0 and train_n_neg > 0:
        pos_weight = torch.tensor([train_n_neg / train_n_pos], device=device)
    else:
        pos_weight = torch.ones(1, device=device)

    best_test_auroc = 0.0
    history: list[dict] = []

    print(f"Training ({args.epochs} epochs, lr={args.lr})...")
    t0 = time.time()

    for epoch in range(args.epochs):
        drift_encoder.train()
        risk_model.train()

        epoch_risk_loss = 0.0
        epoch_recover_loss = 0.0
        n_samples = 0

        batches = make_batches(train_data, args.batch_size)
        for batch in batches:
            batch_risk_loss = torch.tensor(0.0, device=device)
            batch_recover_loss = torch.tensor(0.0, device=device)

            for sample in batch:
                h = sample["gen_h"]  # [T, D]
                correct = sample["correct"]

                with torch.no_grad():
                    z = encoder(h.float())  # [T, latent_dim]

                raw_feats = extract_raw_drift_features(h)  # [T, 5]
                p = drift_encoder(z, raw_feats)  # [T, drift_dim]
                output, _ = risk_model(z.unsqueeze(0), p.unsqueeze(0))

                # Risk labels: all timesteps in a failing sequence labeled 1
                risk_label = torch.full(
                    (output.risk_logits.shape[-1],),
                    0.0 if correct else 1.0,
                    device=device,
                )

                L_risk = F.binary_cross_entropy_with_logits(
                    output.risk_logits.squeeze(0), risk_label, pos_weight=pos_weight,
                )

                # Recoverability: for incorrect sequences, recovery=0 (doomed).
                # For correct sequences, recovery=1 (stable).
                recover_label = torch.full_like(risk_label, 1.0 if correct else 0.0)
                L_recover = F.binary_cross_entropy_with_logits(
                    output.recover_logits.squeeze(0), recover_label,
                )

                batch_risk_loss = batch_risk_loss + L_risk
                batch_recover_loss = batch_recover_loss + L_recover
                n_samples += 1

            if n_samples > 0:
                loss = (batch_risk_loss + args.recover_weight * batch_recover_loss) / n_samples
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                epoch_risk_loss += batch_risk_loss.item()
                epoch_recover_loss += batch_recover_loss.item()

        scheduler.step()

        # Evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            drift_encoder.eval()
            risk_model.eval()

            test_scores = []
            test_labels = []

            with torch.no_grad():
                for sample in test_data:
                    h = sample["gen_h"]
                    z = encoder(h.float())
                    raw_feats = extract_raw_drift_features(h)
                    p = drift_encoder(z, raw_feats)
                    output, _ = risk_model(z.unsqueeze(0), p.unsqueeze(0))

                    # Take mean risk over the sequence as the sequence-level score
                    seq_risk = output.risk.squeeze(0).mean().item()
                    test_scores.append(seq_risk)
                    test_labels.append(0 if sample["correct"] else 1)

            test_auroc = compute_auroc(
                torch.tensor(test_scores), torch.tensor(test_labels),
            )

            if test_auroc > best_test_auroc:
                best_test_auroc = test_auroc
                torch.save({
                    "drift_encoder": drift_encoder.state_dict(),
                    "risk_model": risk_model.state_dict(),
                    "encoder": encoder.state_dict(),
                    "epoch": epoch,
                    "auroc": test_auroc,
                }, args.save_checkpoint)

            avg_risk = epoch_risk_loss / max(n_samples, 1)
            avg_rec = epoch_recover_loss / max(n_samples, 1)
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
                  f"risk_loss={avg_risk:.4f}  recover_loss={avg_rec:.4f}  "
                  f"test_auroc={test_auroc:.4f}  best={best_test_auroc:.4f}  "
                  f"[{elapsed:.0f}s]")

            history.append({
                "epoch": epoch + 1,
                "risk_loss": avg_risk,
                "recover_loss": avg_rec,
                "test_auroc": test_auroc,
            })

    # ── Phase 3: Detailed evaluation ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: EVALUATION")
    print("=" * 70)

    # Load best checkpoint
    if Path(args.save_checkpoint).exists():
        ckpt = torch.load(args.save_checkpoint, map_location=device, weights_only=True)
        drift_encoder.load_state_dict(ckpt["drift_encoder"])
        risk_model.load_state_dict(ckpt["risk_model"])
        print(f"Loaded best checkpoint (epoch {ckpt['epoch']+1}, auroc={ckpt['auroc']:.4f})")

    drift_encoder.eval()
    risk_model.eval()

    # Per-timestep AUROC on test set
    max_t = max(d["gen_h"].shape[0] for d in test_data)
    timestep_aurocs: dict[int, float] = {}

    with torch.no_grad():
        # Precompute all risk scores
        all_risk_curves: list[Tensor] = []
        all_labels: list[int] = []

        for sample in test_data:
            h = sample["gen_h"]
            z = encoder(h.float())
            raw_feats = extract_raw_drift_features(h)
            p = drift_encoder(z, raw_feats)
            output, _ = risk_model(z.unsqueeze(0), p.unsqueeze(0))

            all_risk_curves.append(output.risk.squeeze(0).cpu())
            all_labels.append(0 if sample["correct"] else 1)

        for t in range(max_t):
            scores_t = []
            labs_t = []
            for risk_curve, lab in zip(all_risk_curves, all_labels):
                if risk_curve.shape[0] > t:
                    scores_t.append(risk_curve[t].item())
                    labs_t.append(lab)

            if len(set(labs_t)) >= 2:
                timestep_aurocs[t] = compute_auroc(
                    torch.tensor(scores_t), torch.tensor(labs_t),
                )
            else:
                timestep_aurocs[t] = 0.5

    # Sequence-level AUROC
    seq_scores = []
    seq_labels = []
    for risk_curve, lab in zip(all_risk_curves, all_labels):
        seq_scores.append(risk_curve.mean().item())
        seq_labels.append(lab)
    seq_auroc = compute_auroc(torch.tensor(seq_scores), torch.tensor(seq_labels))

    # Per-family AUROC
    family_aurocs: dict[str, float] = {}
    family_map: dict[str, list[int]] = {}
    for i, sample in enumerate(test_data):
        fam = sample["task_family"]
        if fam not in family_map:
            family_map[fam] = []
        family_map[fam].append(i)

    for fam, indices in family_map.items():
        fam_scores = [seq_scores[i] for i in indices]
        fam_labs = [seq_labels[i] for i in indices]
        if len(set(fam_labs)) >= 2:
            family_aurocs[fam] = compute_auroc(
                torch.tensor(fam_scores), torch.tensor(fam_labs),
            )
        else:
            family_aurocs[fam] = 0.5

    # Lead time analysis
    first_above_55 = None
    first_above_65 = None
    first_above_75 = None
    for t in sorted(timestep_aurocs.keys()):
        auc = timestep_aurocs[t]
        if first_above_55 is None and auc > 0.55:
            first_above_55 = t
        if first_above_65 is None and auc > 0.65:
            first_above_65 = t
        if first_above_75 is None and auc > 0.75:
            first_above_75 = t

    # Threshold-based lead time, calibration (ECE), and recoverer FP.
    # These are computed by early_warning_metrics but were previously dropped.
    from sdq.eval.early_warning_metrics import (
        average_lead_time as _average_lead_time,
        calibration_analysis as _calibration_analysis,
        false_positive_rate_on_recoverers as _fp_on_recoverers,
    )

    avg_lead_time = _average_lead_time(all_risk_curves, all_labels, threshold=0.5)

    all_step_scores = torch.cat([rc for rc in all_risk_curves])
    all_step_labels = torch.cat([
        torch.full((rc.shape[0],), lab, dtype=torch.long)
        for rc, lab in zip(all_risk_curves, all_labels)
    ])
    calibration_bins, ece = _calibration_analysis(all_step_scores, all_step_labels)

    # Recoverer = correct sequence whose risk curve transiently exceeded 0.5.
    recoverer_mask = [
        (lab == 0) and bool((rc > 0.5).any().item())
        for rc, lab in zip(all_risk_curves, all_labels)
    ]
    fp_recoverers = _fp_on_recoverers(
        all_risk_curves, all_labels, recoverer_mask, threshold=0.5,
    )
    n_recoverers = int(sum(recoverer_mask))

    # Print report
    print(f"\nSequence-level AUROC: {seq_auroc:.4f}")
    print(f"\nPer-family AUROC:")
    for fam, auc in sorted(family_aurocs.items()):
        print(f"  {fam:25s}  {auc:.4f}")

    print(f"\nPer-timestep AUROC curve:")
    for t in sorted(timestep_aurocs.keys()):
        auc = timestep_aurocs[t]
        bar = "#" * int(auc * 40)
        print(f"  t={t:3d}  {auc:.4f}  {bar}")

    print(f"\nLead time analysis:")
    print(f"  First t > 0.55 AUROC: {first_above_55}")
    print(f"  First t > 0.65 AUROC: {first_above_65}")
    print(f"  First t > 0.75 AUROC: {first_above_75}")
    print(f"  Average lead time (tokens before end, risk>0.5): {avg_lead_time}")
    print(f"  ECE (calibration error): {ece:.4f}")
    print(f"  Recoverers (correct seqs w/ transient risk>0.5): {n_recoverers}")
    print(f"  FP rate on recoverers: {fp_recoverers}")

    # ── Baseline comparison ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    from sdq.eval.ews_baselines import run_all_baselines, format_baseline_comparison
    from sdq.eval.early_warning_metrics import compute_auprc

    test_gen_h = [d["gen_h"] for d in test_data]
    test_gen_logits = [d.get("gen_logits") for d in test_data]
    test_families = [d["task_family"] for d in test_data]

    baselines = run_all_baselines(
        gen_h_list=test_gen_h,
        labels=(seq_labels if isinstance(seq_labels, list) else seq_labels.tolist()),
        task_families=test_families,
        gen_logits_list=test_gen_logits,
    )

    seq_auprc = compute_auprc(
        torch.tensor(seq_scores), torch.tensor((seq_labels if isinstance(seq_labels, list) else seq_labels.tolist())),
    )
    print(format_baseline_comparison(seq_auroc, seq_auprc, baselines))

    baseline_results = {
        b.name: {"auroc": b.sequence_auroc, "auprc": b.sequence_auprc}
        for b in baselines
    }

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "summary": {
            "sequence_auroc": seq_auroc,
            "sequence_auprc": seq_auprc,
            "best_test_auroc": best_test_auroc,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "first_above_55_t": first_above_55,
            "first_above_65_t": first_above_65,
            "first_above_75_t": first_above_75,
            "average_lead_time": avg_lead_time,
            "expected_calibration_error": ece,
            "n_recoverers": n_recoverers,
            "false_positive_rate_on_recoverers": fp_recoverers,
        },
        "baselines": baseline_results,
        "calibration_bins": calibration_bins,
        "timestep_aurocs": {str(k): v for k, v in timestep_aurocs.items()},
        "family_aurocs": family_aurocs,
        "training_history": history,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Checkpoint saved to {args.save_checkpoint}")
    print("Done.")


if __name__ == "__main__":
    main()
