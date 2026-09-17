#!/usr/bin/env python3
"""Signal validation probe for the early-warning system.

Loads generation-time hidden states captured by capture_runs_ews.py, extracts
simple per-timestep features, and trains a logistic regression / small MLP
to predict whether the final answer is correct or incorrect.

This is the go/no-go gate: if per-timestep hidden-state features have no
predictive power for final correctness, the full EWS pipeline will not work.

The key output is a per-timestep AUROC curve showing at which point during
generation the probe can first distinguish correct from incorrect trajectories.

Usage:
    python run_sdq_ews_probe.py --runs-dir data/runs/gemma-2-2b
    python run_sdq_ews_probe.py --runs-dir data/runs/gemma-2-2b --layer 20
    python run_sdq_ews_probe.py --runs-dir data/runs/gemma-2-2b --output ews_probe_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EWS signal validation probe")
    p.add_argument("--runs-dir", required=True,
                   help="Directory containing captured generation runs")
    p.add_argument("--benchmark", default="data/prompts/benchmark_v2.json",
                   help="Benchmark JSON for answer_id lookup")
    p.add_argument("--layer", type=int, default=-1,
                   help="Which transformer layer to probe (0-indexed, -1=last)")
    p.add_argument("--min-gen-tokens", type=int, default=4,
                   help="Skip runs with fewer generated tokens than this")
    p.add_argument("--max-gen-tokens", type=int, default=64,
                   help="Truncate generation sequences to this length")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--probe-type", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="ews_probe_results.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_timestep_features(
    gen_activations: Tensor,
    layer_idx: int,
) -> Tensor:
    """Extract per-timestep features from generation hidden states.

    Args:
        gen_activations: [num_gen_tokens, num_layers, hidden_dim]
        layer_idx: Which layer to use.

    Returns:
        [num_gen_tokens, num_features] feature tensor.
    """
    h = gen_activations[:, layer_idx, :]  # [T, D]
    T, D = h.shape

    features = []

    # 1. Hidden-state norm at each step
    h_norm = h.norm(dim=-1, keepdim=True)  # [T, 1]
    features.append(h_norm)

    # 2. Delta norm (||h_t - h_{t-1}||)
    delta = h[1:] - h[:-1]  # [T-1, D]
    delta_norm = delta.norm(dim=-1, keepdim=True)  # [T-1, 1]
    delta_norm = torch.cat([torch.zeros(1, 1, device=h.device), delta_norm], dim=0)  # [T, 1]
    features.append(delta_norm)

    # 3. Cosine similarity to previous step
    cos_sim = F.cosine_similarity(h[1:], h[:-1], dim=-1).unsqueeze(-1)  # [T-1, 1]
    cos_sim = torch.cat([torch.ones(1, 1, device=h.device), cos_sim], dim=0)  # [T, 1]
    features.append(cos_sim)

    # 4. Norm growth ratio ||h_t|| / ||h_{t-1}||
    h_norm_safe = h_norm.clamp(min=1e-8)
    growth = h_norm_safe[1:] / h_norm_safe[:-1]  # [T-1, 1]
    growth = torch.cat([torch.ones(1, 1, device=h.device), growth], dim=0)  # [T, 1]
    features.append(growth)

    # 5. Delta direction change (cosine between consecutive deltas = curvature proxy)
    if T >= 3:
        cos_delta = F.cosine_similarity(delta[1:], delta[:-1], dim=-1).unsqueeze(-1)
        cos_delta = torch.cat([
            torch.zeros(2, 1, device=h.device),
            cos_delta,
        ], dim=0)
    else:
        cos_delta = torch.zeros(T, 1, device=h.device)
    features.append(cos_delta)

    # 6. Rolling mean delta norm (window=3)
    if T >= 3:
        roll_delta = delta_norm.unfold(0, min(3, T), 1).mean(dim=-1)  # [T - 2, 1]
        pad_len = T - roll_delta.shape[0]
        roll_delta = torch.cat([
            torch.zeros(pad_len, 1, device=h.device),
            roll_delta,
        ], dim=0)
    else:
        roll_delta = delta_norm.clone()
    features.append(roll_delta)

    # 7. Distance from trajectory centroid
    centroid = h.mean(dim=0, keepdim=True)  # [1, D]
    dist_centroid = (h - centroid).norm(dim=-1, keepdim=True)  # [T, 1]
    features.append(dist_centroid)

    return torch.cat(features, dim=-1)  # [T, 7]


NUM_FEATURES = 7  # must match extract_timestep_features output


# ── Probe models ─────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_benchmark_lookup(path: str | Path) -> dict[str, dict]:
    """Load benchmark and return {example_id: example_dict}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    examples = data if isinstance(data, list) else data.get("examples", [])
    return {e["example_id"]: e for e in examples}


def load_generation_runs(
    runs_dir: Path,
    benchmark: dict[str, dict],
    layer_idx: int,
    min_gen: int,
    max_gen: int,
    device: str,
) -> tuple[list[Tensor], list[int], list[str], list[str]]:
    """Load all generation runs and extract features.

    Returns:
        features_list: list of [T_i, num_features] tensors
        labels: list of 0 (correct) or 1 (incorrect)
        example_ids: corresponding example ids
        task_families: corresponding task families
    """
    from sdq.labels.outcome_labeler import label_run

    features_list: list[Tensor] = []
    labels: list[int] = []
    example_ids: list[str] = []
    task_families: list[str] = []

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
        if actual_layer < 0 or actual_layer >= gen_act.shape[1]:
            actual_layer = gen_act.shape[1] - 1

        feat = extract_timestep_features(gen_act, actual_layer)
        outcome = label_run(new_tokens, bm["answer_id"], bm["task_family"])

        features_list.append(feat)
        labels.append(0 if outcome.correct else 1)
        example_ids.append(prompt_id)
        task_families.append(bm["task_family"])

    return features_list, labels, example_ids, task_families


# ── Training ─────────────────────────────────────────────────────────────────

def compute_auroc(scores: Tensor, labels: Tensor) -> float:
    """Simple AUROC computation without sklearn dependency."""
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
    fp = 0.0
    auc = 0.0
    prev_fp = 0.0

    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
            auc += tp

    return auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5


def train_probe(
    train_features: list[Tensor],
    train_labels: list[int],
    probe_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> nn.Module:
    """Train a probe on per-timestep features to predict sequence-level outcome."""
    # Flatten: each (timestep, features) pair gets the sequence label
    all_feats = []
    all_labs = []
    for feat, lab in zip(train_features, train_labels):
        all_feats.append(feat)
        all_labs.extend([lab] * feat.shape[0])

    X = torch.cat(all_feats, dim=0).to(device)  # [total_steps, num_features]
    y = torch.tensor(all_labs, dtype=torch.float32, device=device)

    if probe_type == "linear":
        model = LinearProbe(X.shape[1]).to(device)
    else:
        model = MLPProbe(X.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Handle class imbalance
    n_pos = y.sum().item()
    n_neg = len(y) - n_pos
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
    else:
        pos_weight = torch.ones(1, device=device)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(X.shape[0], device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, X.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            logits = model(X[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}")

    model.eval()
    return model


def evaluate_per_timestep(
    model: nn.Module,
    features_list: list[Tensor],
    labels: list[int],
    max_t: int,
    device: str,
) -> dict[int, float]:
    """Evaluate probe AUROC at each relative timestep.

    For each timestep t (0-indexed), collects the probe's score for all
    sequences that are at least t+1 tokens long, and computes AUROC.
    """
    model.eval()
    results: dict[int, float] = {}

    with torch.no_grad():
        for t in range(max_t):
            scores = []
            labs = []
            for feat, lab in zip(features_list, labels):
                if feat.shape[0] <= t:
                    continue
                f_t = feat[t:t+1].to(device)
                score = torch.sigmoid(model(f_t)).item()
                scores.append(score)
                labs.append(lab)

            if len(set(labs)) < 2:
                results[t] = 0.5
                continue

            scores_t = torch.tensor(scores)
            labs_t = torch.tensor(labs)
            results[t] = compute_auroc(scores_t, labs_t)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("EWS SIGNAL VALIDATION PROBE")
    print("=" * 60)

    runs_dir = Path(args.runs_dir)
    benchmark = load_benchmark_lookup(args.benchmark)
    print(f"Benchmark: {len(benchmark)} examples loaded")

    print(f"Loading generation runs from {runs_dir} ...")
    features_list, labels, example_ids, task_families = load_generation_runs(
        runs_dir, benchmark, args.layer, args.min_gen_tokens, args.max_gen_tokens, args.device,
    )

    n_correct = sum(1 for l in labels if l == 0)
    n_incorrect = sum(1 for l in labels if l == 1)
    total = len(labels)
    print(f"Loaded {total} runs: {n_correct} correct, {n_incorrect} incorrect")
    if total == 0:
        print("ERROR: No valid runs found.")
        sys.exit(1)

    gen_lengths = [f.shape[0] for f in features_list]
    print(f"Generation lengths: min={min(gen_lengths)}, max={max(gen_lengths)}, "
          f"mean={sum(gen_lengths)/len(gen_lengths):.1f}")

    # Family breakdown
    family_counts: dict[str, dict[str, int]] = {}
    for fam, lab in zip(task_families, labels):
        if fam not in family_counts:
            family_counts[fam] = {"correct": 0, "incorrect": 0}
        family_counts[fam]["correct" if lab == 0 else "incorrect"] += 1
    print("\nPer-family breakdown:")
    for fam, counts in sorted(family_counts.items()):
        print(f"  {fam:25s}  correct={counts['correct']:4d}  incorrect={counts['incorrect']:4d}")

    # Train/test split
    torch.manual_seed(args.seed)
    perm = torch.randperm(total)
    n_test = max(1, int(total * args.test_fraction))
    test_idx = set(perm[:n_test].tolist())

    train_feats = [features_list[i] for i in range(total) if i not in test_idx]
    train_labs = [labels[i] for i in range(total) if i not in test_idx]
    test_feats = [features_list[i] for i in range(total) if i in test_idx]
    test_labs = [labels[i] for i in range(total) if i in test_idx]

    print(f"\nTrain: {len(train_feats)}  Test: {len(test_feats)}")

    # Check if we have both classes in train/test
    if len(set(train_labs)) < 2:
        print("WARNING: Train set has only one class. Probe will be trivial.")
    if len(set(test_labs)) < 2:
        print("WARNING: Test set has only one class. AUROC will be undefined.")

    # Train
    print(f"\nTraining {args.probe_type} probe ({args.epochs} epochs)...")
    model = train_probe(
        train_feats, train_labs,
        probe_type=args.probe_type,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Evaluate per timestep
    max_t = max(f.shape[0] for f in test_feats) if test_feats else 1
    print(f"\nEvaluating per-timestep AUROC (up to t={max_t-1})...")
    auroc_curve = evaluate_per_timestep(model, test_feats, test_labs, max_t, args.device)

    # Pooled-timestep AUROC (as before) and sequence-level AUROC
    with torch.no_grad():
        all_scores, all_labs_flat = [], []
        seq_scores: list[float] = []
        for feat, lab in zip(test_feats, test_labs):
            s = torch.sigmoid(model(feat.to(args.device)))
            all_scores.append(s.cpu())
            all_labs_flat.extend([lab] * feat.shape[0])
            seq_scores.append(s.mean().item())
        overall_auroc = compute_auroc(
            torch.cat(all_scores),
            torch.tensor(all_labs_flat),
        )
        sequence_auroc = compute_auroc(
            torch.tensor(seq_scores),
            torch.tensor(test_labs),
        )

    # Family-prior baseline: score each test example by the training-set
    # failure rate of its family. This is the honest null hypothesis — it
    # asks "can you beat knowing what family the prompt belongs to?"
    train_fams = [task_families[i] for i in range(total) if i not in test_idx]
    test_fams = [task_families[i] for i in range(total) if i in test_idx]

    fam_rates: dict[str, float] = {}
    for fam in set(train_fams):
        fam_labs_train = [l for f, l in zip(train_fams, train_labs) if f == fam]
        if fam_labs_train:
            fam_rates[fam] = sum(fam_labs_train) / len(fam_labs_train)
        else:
            fam_rates[fam] = 0.5
    fam_prior_scores = torch.tensor(
        [fam_rates.get(f, 0.5) for f in test_fams],
        dtype=torch.float32,
    )
    family_prior_auroc = compute_auroc(fam_prior_scores, torch.tensor(test_labs))

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Pooled-timestep AUROC:  {overall_auroc:.4f}")
    print(f"Sequence-level AUROC:   {sequence_auroc:.4f}")
    print(f"Family-prior baseline:  {family_prior_auroc:.4f}  (the null — "
          f"how well family identity alone predicts correctness)")
    print(f"\nPer-timestep AUROC curve (test set):")

    first_above_chance = None
    first_strong = None
    for t in sorted(auroc_curve.keys()):
        auc = auroc_curve[t]
        bar = "#" * int(auc * 40)
        print(f"  t={t:3d}  AUROC={auc:.4f}  {bar}")
        if first_above_chance is None and auc > 0.55:
            first_above_chance = t
        if first_strong is None and auc > 0.65:
            first_strong = t

    print()
    if first_above_chance is not None:
        print(f"First timestep > 0.55 AUROC: t={first_above_chance}")
    else:
        print("No timestep reached > 0.55 AUROC")

    if first_strong is not None:
        print(f"First timestep > 0.65 AUROC: t={first_strong}")

    # Signal gate: beat the family prior by at least `margin`. This is the
    # meaningful null — beating 0.55 only tells you that family identity has
    # signal, which we already know.
    margin = 0.03
    signal_present = sequence_auroc > family_prior_auroc + margin
    beats_chance = sequence_auroc > 0.55
    print(f"\nFamily prior: {family_prior_auroc:.4f}  "
          f"Sequence AUROC: {sequence_auroc:.4f}  "
          f"Margin: +{sequence_auroc - family_prior_auroc:.4f}")
    print(f"Signal above family prior (+{margin}): "
          f"{'YES' if signal_present else 'NO'}")
    if signal_present:
        print("  -> Hidden-state dynamics carry correctness info beyond "
              "family identity. Proceed with full EWS pipeline.")
    elif beats_chance:
        print("  -> Probe beats chance but NOT family prior. Any 'signal' "
              "is probably family identity leakage. Revisit framing.")
    else:
        print("  -> Probe does not exceed chance. Dynamics are inert on "
              "this setup.")

    # Save results
    results = {
        "overall_auroc": overall_auroc,
        "sequence_auroc": sequence_auroc,
        "family_prior_auroc": family_prior_auroc,
        "signal_margin": sequence_auroc - family_prior_auroc,
        "auroc_curve": {str(k): v for k, v in auroc_curve.items()},
        "first_above_chance_t": first_above_chance,
        "first_strong_t": first_strong,
        "signal_present": signal_present,
        "signal_gate": {
            "null": "family_prior",
            "margin": margin,
            "beats_chance": beats_chance,
        },
        "family_failure_rates_train": fam_rates,
        "n_train": len(train_feats),
        "n_test": len(test_feats),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "family_breakdown": family_counts,
        "probe_type": args.probe_type,
        "layer": args.layer,
        "args": vars(args),
    }

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
