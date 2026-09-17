#!/usr/bin/env python3
"""Diagnostic script for triplet loss debugging.

Loads the benchmark data and a fresh encoder, computes raw distances,
and reports exactly what the triplet pipeline sees.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from sdq.instrumentation import collect_runs, set_seed
from sdq.trajectories import extract_trajectory, normalize_trajectory, Trajectory
from sdq.latent import TemporalConvEncoder
from sdq.losses.triplet import semi_hard_triplet_loss


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load benchmark metadata ---
    with open("data/prompts/benchmark_v1.json", encoding="utf-8") as f:
        bm = json.load(f)
    with open("data/prompts/splits/split_semantic_family.json", encoding="utf-8") as f:
        split = json.load(f)

    pid_meta = {e["example_id"]: e for e in bm["examples"]}
    sg: dict[str, list[str]] = defaultdict(list)
    for ex in bm["examples"]:
        sg[ex["semantic_task_id"]].append(ex["example_id"])

    train_ids = set(split["train_ids"])

    # --- Load subset of runs (first 60 for speed) ---
    print("Loading runs...")
    runs = collect_runs("data/runs", prefix="bench_", device="cpu")
    print(f"Loaded {len(runs)} runs total")

    # Build groups, take 20 train groups
    groups: dict[str, list[str]] = {}
    for stid, pids in sg.items():
        found = [p for p in pids if p in runs and p in train_ids]
        if len(found) >= 2:
            groups[stid] = found
    group_names = list(groups.keys())[:20]
    groups = {k: groups[k] for k in group_names}
    all_pids = [p for pids in groups.values() for p in pids]
    print(f"Using {len(groups)} groups, {len(all_pids)} PIDs")

    # --- Extract trajectories and encode ---
    print("Extracting trajectories...")
    trajectories: dict[str, Trajectory] = {}
    for pid in all_pids:
        traj = extract_trajectory(runs[pid], layer=-1, to_float=True)
        normed = normalize_trajectory(traj.states, mode="center_scale")
        trajectories[pid] = Trajectory(
            states=normed, prompt_id=pid, layer=traj.layer,
            token_ids=traj.token_ids, token_strings=traj.token_strings,
            metadata=traj.metadata,
        )

    hidden_dim = next(iter(trajectories.values())).D
    encoder = TemporalConvEncoder(hidden_dim, 64, window_size=3).to(device)

    print("Encoding latents...")
    latents: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for pid in all_pids:
            h = trajectories[pid].states.to(device)
            latents[pid] = encoder(h)

    # ===================================================================
    # CHECK 1: Raw distance distributions
    # ===================================================================
    print("\n" + "=" * 60)
    print("CHECK 1: Raw distance distributions")
    print("=" * 60)

    pos_dists_sq = []  # squared L2 (what code uses)
    pos_dists_l2 = []  # L2
    neg_dists_sq = []
    neg_dists_l2 = []

    fam_names = list(groups.keys())
    for fi, fam in enumerate(fam_names):
        pids = groups[fam]
        neg_pool = [p for f in fam_names if f != fam for p in groups[f]]
        for i in range(len(pids)):
            for j in range(len(pids)):
                if i == j:
                    continue
                a_mean = latents[pids[i]].mean(dim=0)
                p_mean = latents[pids[j]].mean(dim=0)
                d_sq = (a_mean - p_mean).pow(2).sum().item()
                d_l2 = (a_mean - p_mean).pow(2).sum().sqrt().item()
                pos_dists_sq.append(d_sq)
                pos_dists_l2.append(d_l2)

            for np_ in random.sample(neg_pool, min(10, len(neg_pool))):
                a_mean = latents[pids[i]].mean(dim=0)
                n_mean = latents[np_].mean(dim=0)
                d_sq = (a_mean - n_mean).pow(2).sum().item()
                d_l2 = (a_mean - n_mean).pow(2).sum().sqrt().item()
                neg_dists_sq.append(d_sq)
                neg_dists_l2.append(d_l2)

    print(f"\nPositive distances (same semantic group):")
    print(f"  Squared L2:  mean={sum(pos_dists_sq)/len(pos_dists_sq):.2f}  "
          f"min={min(pos_dists_sq):.2f}  max={max(pos_dists_sq):.2f}")
    print(f"  L2:          mean={sum(pos_dists_l2)/len(pos_dists_l2):.2f}  "
          f"min={min(pos_dists_l2):.2f}  max={max(pos_dists_l2):.2f}")

    print(f"\nNegative distances (different semantic group):")
    print(f"  Squared L2:  mean={sum(neg_dists_sq)/len(neg_dists_sq):.2f}  "
          f"min={min(neg_dists_sq):.2f}  max={max(neg_dists_sq):.2f}")
    print(f"  L2:          mean={sum(neg_dists_l2)/len(neg_dists_l2):.2f}  "
          f"min={min(neg_dists_l2):.2f}  max={max(neg_dists_l2):.2f}")

    d_pos_mean = sum(pos_dists_sq) / len(pos_dists_sq)
    d_neg_mean = sum(neg_dists_sq) / len(neg_dists_sq)
    print(f"\n  d_pos_sq - d_neg_sq (mean): {d_pos_mean - d_neg_mean:.2f}")
    print(f"  Margin (2.0) as fraction of mean distance: {2.0 / d_pos_mean:.4f}")
    print(f"  >>> If margin << distances, loss ≈ margin always")

    # ===================================================================
    # CHECK 2: What semi-hard mining actually selects
    # ===================================================================
    print("\n" + "=" * 60)
    print("CHECK 2: Semi-hard mining behavior")
    print("=" * 60)

    selected_d_pos = []
    selected_d_neg = []
    loss_values = []
    active_count = 0
    total_count = 0

    for fam in fam_names[:10]:
        pids = [p for p in groups[fam] if p in latents]
        neg_pids = [p for f in fam_names if f != fam for p in groups[f] if p in latents]
        if len(pids) < 2 or not neg_pids:
            continue

        for i in range(len(pids)):
            for j in range(len(pids)):
                if i == j:
                    continue
                anchor_z = latents[pids[i]]
                pos_z = latents[pids[j]]

                # Replicate exact code from compute_triplet_loss
                anchor_mean = anchor_z.mean(dim=0).detach()
                best_dist = float("inf")
                best_neg_pid = None
                for np_ in neg_pids:
                    d = (anchor_mean - latents[np_].mean(dim=0).detach()).pow(2).sum().item()
                    if d < best_dist:
                        best_dist = d
                        best_neg_pid = np_

                best_neg = latents[best_neg_pid]

                # Compute what semi_hard_triplet_loss computes
                a_m = anchor_z.mean(dim=0)
                p_m = pos_z.mean(dim=0)
                n_m = best_neg.mean(dim=0)
                d_pos_avg = (a_m - p_m).pow(2).sum().item()
                d_neg_avg = (a_m - n_m).pow(2).sum().item()

                T_ap = min(anchor_z.shape[0], pos_z.shape[0])
                T_an = min(anchor_z.shape[0], best_neg.shape[0])
                d_pos_traj = (anchor_z[:T_ap] - pos_z[:T_ap]).pow(2).sum(dim=-1).mean().item()
                d_neg_traj = (anchor_z[:T_an] - best_neg[:T_an]).pow(2).sum(dim=-1).mean().item()

                d_pos = (d_pos_avg + d_pos_traj) / 2
                d_neg = (d_neg_avg + d_neg_traj) / 2

                loss_val = max(0.0, d_pos - d_neg + 2.0)

                selected_d_pos.append(d_pos)
                selected_d_neg.append(d_neg)
                loss_values.append(loss_val)
                total_count += 1
                if loss_val > 0:
                    active_count += 1

    print(f"\nTotal triplets: {total_count}")
    print(f"Active (loss > 0): {active_count} ({active_count/max(total_count,1)*100:.1f}%)")
    print(f"\nSelected d_pos:  mean={sum(selected_d_pos)/len(selected_d_pos):.2f}  "
          f"min={min(selected_d_pos):.2f}  max={max(selected_d_pos):.2f}")
    print(f"Selected d_neg:  mean={sum(selected_d_neg)/len(selected_d_neg):.2f}  "
          f"min={min(selected_d_neg):.2f}  max={max(selected_d_neg):.2f}")
    avg_gap = sum(d_p - d_n for d_p, d_n in zip(selected_d_pos, selected_d_neg)) / len(selected_d_pos)
    print(f"d_pos - d_neg:   mean={avg_gap:.4f}")
    print(f"Loss values:     mean={sum(loss_values)/len(loss_values):.4f}  "
          f"min={min(loss_values):.4f}  max={max(loss_values):.4f}")
    print(f"\n>>> margin=2.0, avg(d_pos-d_neg)={avg_gap:.4f}")
    print(f">>> Expected loss ≈ {avg_gap + 2.0:.4f}  (matches ~2.00 plateau!)")

    # ===================================================================
    # CHECK 3: Verify positive/negative labels
    # ===================================================================
    print("\n" + "=" * 60)
    print("CHECK 3: Positive/negative label verification")
    print("=" * 60)

    for fam in fam_names[:5]:
        pids = groups[fam]
        print(f"\n  Group '{fam}': {len(pids)} members")
        for pid in pids:
            meta = pid_meta.get(pid, {})
            print(f"    {pid}: task_family={meta.get('task_family', '?')}  "
                  f"semantic_task={meta.get('semantic_task_id', '?')}  "
                  f"surface={meta.get('surface_template_id', '?')}  "
                  f"answer={meta.get('answer_id', '?')}")

    # ===================================================================
    # CHECK 4: Gradient contribution comparison
    # ===================================================================
    print("\n" + "=" * 60)
    print("CHECK 4: Loss magnitude comparison")
    print("=" * 60)
    print("  (This shows raw unweighted loss values)")
    print(f"  Triplet margin: 2.0")
    print(f"  Typical d_pos: {sum(selected_d_pos)/len(selected_d_pos):.2f}")
    print(f"  Typical d_neg: {sum(selected_d_neg)/len(selected_d_neg):.2f}")
    print(f"  Triplet loss:  {sum(loss_values)/len(loss_values):.4f}")
    print(f"  Triplet weight: 2.0 → weighted = {2.0 * sum(loss_values)/len(loss_values):.4f}")
    print(f"\n  The triplet loss is ~2.0 because:")
    print(f"    d_pos ≈ d_neg (hard mining picks closest negative)")
    print(f"    so loss ≈ 0 + margin = 2.0")
    print(f"    margin is negligible vs distances ({2.0:.1f} vs {sum(selected_d_pos)/len(selected_d_pos):.1f})")
    print(f"    gradients from margin won't budge anything")

    # ===================================================================
    # DIAGNOSIS SUMMARY
    # ===================================================================
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    avg_d = (sum(selected_d_pos) + sum(selected_d_neg)) / (len(selected_d_pos) + len(selected_d_neg))
    margin_ratio = 2.0 / avg_d
    print(f"\n  1. SCALE MISMATCH: margin=2.0, avg_distance={avg_d:.1f}")
    print(f"     margin/distance ratio = {margin_ratio:.4f}")
    if margin_ratio < 0.1:
        print(f"     >>> CRITICAL: margin is {1/margin_ratio:.0f}x smaller than distances")
        print(f"     >>> Fix: use L2 distance (not squared L2), or scale margin up")
    print(f"\n  2. MINING: code picks CLOSEST negative (hard mining)")
    print(f"     With {len(neg_dists_sq)} negatives, closest ≈ furthest positive")
    print(f"     >>> Fix: use true semi-hard mining (d_neg > d_pos, d_neg < d_pos+margin)")
    print(f"             or switch to InfoNCE")
    print(f"\n  3. Active triplets: {active_count}/{total_count} = 100%")
    if active_count == total_count:
        print(f"     >>> ALL triplets active but none improving — stuck regime")

    # ===================================================================
    # CHECK 5: InfoNCE with projection head + variance reg
    # ===================================================================
    print("\n" + "=" * 60)
    print("CHECK 5: InfoNCE with projection head + anti-collapse")
    print("=" * 60)

    from run_sdq_v35 import compute_infonce_loss, ContrastiveHead

    # --- 5a: Without projection head (vanilla, shows collapse) ---
    infonce_loss_v, infonce_diag_v = compute_infonce_loss(
        latents, groups, temperature=0.5,
    )
    print(f"\n  Vanilla (no proj head, temp=0.5):")
    print(f"    Loss:       {infonce_loss_v.item():.4f}")
    print(f"    Pos cos:    {infonce_diag_v.get('avg_pos_cosine', 0):.4f}")
    print(f"    Neg cos:    {infonce_diag_v.get('avg_neg_cosine', 0):.4f}")
    print(f"    Cos gap:    {infonce_diag_v.get('cos_gap', 0):.4f}")
    print(f"    Embed std:  {infonce_diag_v.get('embed_std', 0):.6f}")
    print(f"    Var loss:   {infonce_diag_v.get('var_loss', 0):.4f}")

    # --- 5b: With projection head ---
    proj_head = ContrastiveHead(64, proj_dim=64).to(device)
    infonce_loss_p, infonce_diag_p = compute_infonce_loss(
        latents, groups, temperature=0.5, proj_head=proj_head,
    )
    print(f"\n  With proj head (temp=0.5):")
    print(f"    Loss:       {infonce_loss_p.item():.4f}")
    print(f"    Pos cos:    {infonce_diag_p.get('avg_pos_cosine', 0):.4f}")
    print(f"    Neg cos:    {infonce_diag_p.get('avg_neg_cosine', 0):.4f}")
    print(f"    Cos gap:    {infonce_diag_p.get('cos_gap', 0):.4f}")
    print(f"    Embed std:  {infonce_diag_p.get('embed_std', 0):.6f}")
    print(f"    Var loss:   {infonce_diag_p.get('var_loss', 0):.4f}")

    # --- 5c: Gradient check with proj head ---
    encoder2 = TemporalConvEncoder(hidden_dim, 64, window_size=3).to(device)
    proj2 = ContrastiveHead(64, proj_dim=64).to(device)
    latents2: dict[str, torch.Tensor] = {}
    for pid in all_pids[:30]:
        h = trajectories[pid].states.to(device)
        latents2[pid] = encoder2(h)

    small_groups = {k: [p for p in v if p in latents2] for k, v in groups.items()}
    small_groups = {k: v for k, v in small_groups.items() if len(v) >= 2}

    loss2, diag2 = compute_infonce_loss(
        latents2, small_groups, temperature=0.5, proj_head=proj2,
    )
    loss2.backward()

    enc_grads = [p.grad.norm().item() for p in encoder2.parameters() if p.grad is not None]
    head_grads = [p.grad.norm().item() for p in proj2.parameters() if p.grad is not None]
    print(f"\n  Gradient check ({len(latents2)} PIDs, {len(small_groups)} groups):")
    print(f"    Loss = {loss2.item():.4f} (nce={diag2.get('nce_loss',0):.4f} var={diag2.get('var_loss',0):.4f})")
    print(f"    Encoder grads: {len(enc_grads)} params, "
          f"min={min(enc_grads):.6f} max={max(enc_grads):.6f}")
    print(f"    Head grads:    {len(head_grads)} params, "
          f"min={min(head_grads):.6f} max={max(head_grads):.6f}")
    print(f"    Embed std:     {diag2.get('embed_std', 0):.6f}")
    print(f"    Cos gap:       {diag2.get('cos_gap', 0):.4f}")

    all_nonzero = all(g > 0 for g in enc_grads + head_grads)
    print(f"    All nonzero:   {all_nonzero}")
    if all_nonzero:
        print("    >>> PASS: Gradients flow through proj head to encoder")
    else:
        print("    >>> WARN: Some zero gradients")


if __name__ == "__main__":
    main()
