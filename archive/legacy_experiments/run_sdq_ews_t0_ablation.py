#!/usr/bin/env python3
"""t=0 ablation for the SDQ early-warning system.

Plan reference: Phase 2 of groovy-seeking-waterfall.md.

The goal is to answer a single question: how much of the full EWS AUROC
comes from the very first generation timestep? If a linear probe on the
t=0 hidden state (or the encoder's z_0) is within a few points of the
full temporal pipeline, then the GRU is inert and the signal is static
answer commitment at prompt encoding — not any kind of dynamic drift.

Probes compared (all stratified per-family plus pooled):
  - h_0        : raw hidden state at generation timestep 0 (last layer)
  - z_0        : semantic encoder output at t=0 (if --encoder-checkpoint given)
  - logit_ent  : Shannon entropy of the t=0 logit distribution
  - family_prior: each example labeled with its family's training failure rate

Outputs a compact JSON summary next to the run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDQ EWS t=0 ablation")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--min-gen-tokens", type=int, default=4)
    p.add_argument("--max-gen-tokens", type=int, default=64)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--encoder-checkpoint", default=None,
                   help="If given, evaluate z_0 probe using this encoder.")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--probe-epochs", type=int, default=400)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--probe-weight-decay", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="ews_t0_ablation_results.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Shared utilities (compute_auroc lifted from run_sdq_ews.py) ──────────────

def compute_auroc(scores: Tensor, labels: Tensor) -> float:
    if labels.unique().numel() < 2:
        return 0.5
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = scores.argsort(descending=True)
    sl = labels[order]
    tp = 0.0
    auc = 0.0
    for lab in sl:
        if lab == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


# ── Data loading (t=0 only) ──────────────────────────────────────────────────

def load_t0_dataset(
    runs_dir: Path,
    benchmark: dict[str, dict],
    layer_idx: int,
    min_gen: int,
    max_gen: int,
    device: str,
) -> list[dict]:
    """Load just the first generation-step hidden state and logits per run."""
    from sdq.labels.outcome_labeler import label_run

    dataset: list[dict] = []
    run_dirs = sorted(d for d in runs_dir.iterdir()
                      if d.is_dir() and (d / "gen_activations.pt").exists())
    if not run_dirs:
        print(f"ERROR: No runs found in {runs_dir}")
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

        gen_logits = None
        gl_path = rd / "gen_logits.pt"
        if gl_path.exists():
            gen_logits = torch.load(gl_path, map_location=device, weights_only=True)
            gen_logits = gen_logits[:max_gen]

        outcome = label_run(new_tokens, bm["answer_id"], bm["task_family"])

        dataset.append({
            "h_all": h_all,                     # kept so we can run encoder on full seq
            "h_0": h_all[0].float(),            # [D]
            "logit_0": gen_logits[0].float() if gen_logits is not None else None,
            "correct": outcome.correct,
            "task_family": bm["task_family"],
            "example_id": prompt_id,
        })
    return dataset


# ── Linear probe ─────────────────────────────────────────────────────────────

def train_linear_probe(
    X_train: Tensor,
    y_train: Tensor,
    X_test: Tensor,
    y_test: Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> tuple[Tensor, float]:
    """Train a 1-layer logistic probe. Returns (test_scores, test_auroc)."""
    X_train = X_train.to(device)
    y_train = y_train.to(device).float()
    X_test = X_test.to(device)
    y_test_cpu = y_test.cpu()

    # Standardize by train stats for stable optimization.
    mu = X_train.mean(dim=0, keepdim=True)
    sd = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xtr = (X_train - mu) / sd
    Xte = (X_test - mu) / sd

    probe = nn.Linear(Xtr.shape[1], 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(Xtr).squeeze(-1)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        scores = torch.sigmoid(probe(Xte).squeeze(-1)).cpu()
    auroc = compute_auroc(scores, y_test_cpu)
    return scores, auroc


def per_family_auroc(
    scores: Tensor,
    labels: Tensor,
    families: list[str],
) -> dict[str, float]:
    fam_idx: dict[str, list[int]] = {}
    for i, f in enumerate(families):
        fam_idx.setdefault(f, []).append(i)
    out: dict[str, float] = {}
    for f, idx in fam_idx.items():
        s = scores[idx]
        y = labels[idx]
        if y.unique().numel() < 2:
            out[f] = float("nan")
        else:
            out[f] = compute_auroc(s, y)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    print("=" * 70)
    print("SDQ EWS — t=0 ABLATION")
    print("=" * 70)

    # Benchmark
    with open(args.benchmark, encoding="utf-8") as f:
        bm_raw = json.load(f)
    examples = bm_raw if isinstance(bm_raw, list) else bm_raw.get("examples", [])
    benchmark = {e["example_id"]: e for e in examples}

    runs_dir = Path(args.runs_dir)
    print(f"Loading t=0 features from {runs_dir} ...")
    dataset = load_t0_dataset(
        runs_dir, benchmark, args.layer,
        args.min_gen_tokens, args.max_gen_tokens, device,
    )
    print(f"Loaded {len(dataset)} runs")
    if not dataset:
        sys.exit(1)

    n_correct = sum(1 for d in dataset if d["correct"])
    n_incorrect = len(dataset) - n_correct
    print(f"  correct={n_correct}  incorrect={n_incorrect}")

    # Train/test split — match run_sdq_ews.py's conventions for comparability.
    torch.manual_seed(args.seed)
    perm = torch.randperm(len(dataset))
    n_test = max(1, int(len(dataset) * args.test_fraction))
    test_idx = set(perm[:n_test].tolist())

    train_data = [dataset[i] for i in range(len(dataset)) if i not in test_idx]
    test_data = [dataset[i] for i in range(len(dataset)) if i in test_idx]
    print(f"  train={len(train_data)}  test={len(test_data)}")

    # Label vectors (1 = incorrect, matching EWS convention).
    y_train = torch.tensor([0 if d["correct"] else 1 for d in train_data])
    y_test = torch.tensor([0 if d["correct"] else 1 for d in test_data])
    test_families = [d["task_family"] for d in test_data]
    train_families = [d["task_family"] for d in train_data]

    results: dict[str, dict] = {}

    # ── Probe 1: raw h_0 ────────────────────────────────────────────────────
    print("\n[Probe] h_0 (raw hidden state at generation t=0)")
    Xtr_h = torch.stack([d["h_0"] for d in train_data])
    Xte_h = torch.stack([d["h_0"] for d in test_data])
    h_scores, h_auroc = train_linear_probe(
        Xtr_h, y_train, Xte_h, y_test,
        args.probe_epochs, args.probe_lr, args.probe_weight_decay, device,
    )
    h_fam_aurocs = per_family_auroc(h_scores, y_test, test_families)
    print(f"  pooled AUROC: {h_auroc:.4f}")
    for f, a in sorted(h_fam_aurocs.items()):
        print(f"    {f:25s} {a:.4f}")
    results["h_0_linear"] = {
        "pooled_auroc": h_auroc,
        "family_aurocs": h_fam_aurocs,
    }

    # ── Probe 2: z_0 from encoder (optional) ────────────────────────────────
    if args.encoder_checkpoint and Path(args.encoder_checkpoint).exists():
        print(f"\n[Probe] z_0 (encoder latent from {args.encoder_checkpoint})")
        from sdq.latent.encoder import MultiScaleConvEncoder
        hidden_dim = dataset[0]["h_all"].shape[1]
        encoder = MultiScaleConvEncoder(
            hidden_dim=hidden_dim, latent_dim=args.latent_dim,
        ).to(device)
        state = torch.load(args.encoder_checkpoint, map_location=device, weights_only=True)
        encoder.load_state_dict(state)
        encoder.eval()

        with torch.no_grad():
            def _z0(d):
                z = encoder(d["h_all"].float())  # [T, latent]
                return z[0].cpu()
            Xtr_z = torch.stack([_z0(d) for d in train_data])
            Xte_z = torch.stack([_z0(d) for d in test_data])

        z_scores, z_auroc = train_linear_probe(
            Xtr_z, y_train, Xte_z, y_test,
            args.probe_epochs, args.probe_lr, args.probe_weight_decay, device,
        )
        z_fam_aurocs = per_family_auroc(z_scores, y_test, test_families)
        print(f"  pooled AUROC: {z_auroc:.4f}")
        for f, a in sorted(z_fam_aurocs.items()):
            print(f"    {f:25s} {a:.4f}")
        results["z_0_linear"] = {
            "pooled_auroc": z_auroc,
            "family_aurocs": z_fam_aurocs,
        }
    else:
        print("\n[Probe] z_0 skipped (no --encoder-checkpoint given)")

    # ── Baseline: logit entropy at t=0 ───────────────────────────────────────
    if all(d["logit_0"] is not None for d in test_data):
        print("\n[Baseline] logit entropy at t=0")
        def _ent(logits: Tensor) -> float:
            p = F.softmax(logits, dim=-1)
            # add small epsilon; entropy in nats
            return float(-(p * (p.clamp_min(1e-12)).log()).sum().item())
        ent_scores = torch.tensor([_ent(d["logit_0"]) for d in test_data])
        ent_auroc = compute_auroc(ent_scores, y_test)
        ent_fam_aurocs = per_family_auroc(ent_scores, y_test, test_families)
        print(f"  pooled AUROC: {ent_auroc:.4f}")
        for f, a in sorted(ent_fam_aurocs.items()):
            print(f"    {f:25s} {a:.4f}")
        results["logit_entropy_t0"] = {
            "pooled_auroc": ent_auroc,
            "family_aurocs": ent_fam_aurocs,
        }
    else:
        print("\n[Baseline] logit entropy skipped (no gen_logits captured)")

    # ── Baseline: family prior (per-family train failure rate) ──────────────
    print("\n[Baseline] family prior (train-set failure rate per family)")
    fam_rate: dict[str, float] = {}
    fam_counts: dict[str, int] = {}
    for f, y in zip(train_families, y_train.tolist()):
        fam_rate[f] = fam_rate.get(f, 0.0) + y
        fam_counts[f] = fam_counts.get(f, 0) + 1
    for f in fam_rate:
        fam_rate[f] /= fam_counts[f]
    prior_scores = torch.tensor([fam_rate.get(f, 0.5) for f in test_families])
    prior_auroc = compute_auroc(prior_scores, y_test)
    prior_fam_aurocs = per_family_auroc(prior_scores, y_test, test_families)
    # Within-family AUROC is always 0.5 for a per-family constant — that's the point.
    print(f"  pooled AUROC: {prior_auroc:.4f}  (within-family is trivially 0.5)")
    results["family_prior"] = {
        "pooled_auroc": prior_auroc,
        "family_aurocs": prior_fam_aurocs,
        "family_failure_rates_train": fam_rate,
    }

    # ── Summary print ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY (pooled AUROC)")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:22s}  {r['pooled_auroc']:.4f}")

    # Decision note for the plan's stopping rule.
    h_auc = results["h_0_linear"]["pooled_auroc"]
    note_lines = []
    note_lines.append(
        "If h_0 (or z_0) within-family AUROC is close to the full EWS model's "
        "within-family AUROC, the temporal GRU is inert and the signal is "
        "static answer commitment at the prompt encoding (Phase 2 decision)."
    )
    note_lines.append(
        f"h_0 pooled AUROC = {h_auc:.4f}; compare to ews_results.json "
        "family_aurocs and sequence_auroc."
    )
    results["notes"] = note_lines

    # Argparse record
    results["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    results["n_train"] = len(train_data)
    results["n_test"] = len(test_data)
    results["n_correct"] = n_correct
    results["n_incorrect"] = n_incorrect

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
