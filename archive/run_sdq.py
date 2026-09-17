#!/usr/bin/env python3
"""Run SDQ training on captured activation data and report results.

Uses the syllogistic (cat/bird) and arithmetic (add5/add15) families
as semantic equivalence groups. Each family has 4 surface-form variants
of the same underlying reasoning, making them ideal for SDQ pairwise
training.

Usage:
    python run_sdq.py [--epochs 200] [--lr 1e-3] [--layer -1] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import torch
import torch.nn as nn

from sdq.instrumentation import collect_runs, set_seed
from sdq.trajectories import extract_trajectory, Trajectory, normalize_trajectory, velocity_vectors
from sdq.alignment import MonotoneAligner, soft_dtw_alignment
from sdq.transport import LocalTransportField, TransportResult, cycle_consistency_metrics
from sdq.latent import (
    TemporalConvEncoder,
    ResidualDecoder,
    TemporalGaugeEncoder,
    LatentODE,
)
from sdq.losses import (
    reconstruction_loss,
    semantic_consistency_loss,
    transport_consistency_loss,
    cycle_loss,
    gauge_regularization_loss,
    SDQLoss,
)
from sdq.losses.cycle import lowrank_cycle_loss
from sdq.losses.gauge_regularization import lowrank_gauge_regularization_loss
from sdq.losses.total import LossWeights, LossBreakdown
from sdq.eval import rank_analysis, evaluate_holdout


# ---------------------------------------------------------------------------
# Semantic equivalence groups: each list of prompt_ids shares the same
# reasoning but uses different surface forms.
# ---------------------------------------------------------------------------
FAMILIES = {
    "cat": ["cat_therefore", "cat_reorder", "cat_since", "cat_given"],
    "bird": ["bird_therefore", "bird_reorder", "bird_since", "bird_given"],
    "add5": ["add5_base", "add5_reorder", "add5_worded", "add5_worded2"],
    "add15": ["add15_base", "add15_reorder", "add15_worded", "add15_worded2"],
}

# Train on cat + add5, hold out bird + add15 for transfer evaluation
TRAIN_FAMILIES = ["cat", "add5"]
HOLDOUT_FAMILIES = ["bird", "add15"]


@dataclass
class SDQModel:
    """All trainable SDQ components bundled together."""

    encoder: TemporalConvEncoder
    decoder: ResidualDecoder
    gauge_encoder: TemporalGaugeEncoder
    aligner: MonotoneAligner
    transport: LocalTransportField
    dynamics: LatentODE

    def parameters(self):
        for module in [
            self.encoder,
            self.decoder,
            self.gauge_encoder,
            self.aligner,
            self.transport,
            self.dynamics,
        ]:
            yield from module.parameters()

    def train(self):
        for module in [
            self.encoder,
            self.decoder,
            self.gauge_encoder,
            self.aligner,
            self.transport,
            self.dynamics,
        ]:
            module.train()

    def eval(self):
        for module in [
            self.encoder,
            self.decoder,
            self.gauge_encoder,
            self.aligner,
            self.transport,
            self.dynamics,
        ]:
            module.eval()

    def to(self, device):
        for module in [
            self.encoder,
            self.decoder,
            self.gauge_encoder,
            self.aligner,
            self.transport,
            self.dynamics,
        ]:
            module.to(device)
        return self


# ---------------------------------------------------------------------------
# Training step: process one pair from a semantic equivalence group
# ---------------------------------------------------------------------------
def train_pair(
    model: SDQModel,
    h_i: torch.Tensor,  # [T_i, D]
    h_j: torch.Tensor,  # [T_j, D]
    loss_fn: SDQLoss,
    compute_cycle: bool = True,
) -> LossBreakdown:
    """One forward pass for a (source, target) pair.

    Returns the LossBreakdown (call .total.backward() after).
    """
    # 1. Align target to source time
    h_j_aligned = model.aligner.align_trajectory(h_i, h_j)  # [T_i, D]

    # 2. Encode both into latent space
    z_i = model.encoder(h_i)  # [T_i, D_z]
    z_j = model.encoder(h_j_aligned)  # [T_i, D_z]

    # 3. Gauge: infer surface-form context
    u_i = model.gauge_encoder(h_i)  # [T_i, D_u]
    u_j = model.gauge_encoder(h_j_aligned)  # [T_i, D_u]

    # 4. Decode (reconstruct observed)
    h_i_hat = model.decoder(z_i, u_i)  # [T_i, D]
    h_j_hat = model.decoder(z_j, u_j)  # [T_i, D]

    # 5. Transport: learn local operator G_t^{i->j}
    transport_result = model.transport(h_i, h_j_aligned, return_operators=False)

    # 6. Compute individual losses
    L_rec = (
        reconstruction_loss(h_i, h_i_hat)
        + reconstruction_loss(h_j_aligned, h_j_hat)
    ) / 2.0

    L_sem = semantic_consistency_loss(z_i, z_j)
    L_trans = transport_consistency_loss(transport_result)

    # Cycle loss and gauge regularization from low-rank factors (efficient)
    L_cycle = torch.tensor(0.0, device=h_i.device)
    L_gauge = torch.tensor(0.0, device=h_i.device)
    if compute_cycle and transport_result.U is not None:
        transport_result_ji = model.transport(
            h_j_aligned, h_i, return_operators=False
        )
        if transport_result_ji.U is not None:
            T_min = min(transport_result.U.shape[0], transport_result_ji.U.shape[0])
            L_cycle = lowrank_cycle_loss(
                transport_result.U[:T_min], transport_result.V[:T_min],
                transport_result_ji.U[:T_min], transport_result_ji.V[:T_min],
            )

        # Gauge regularization from low-rank factors
        L_gauge = lowrank_gauge_regularization_loss(
            transport_result.U, transport_result.V
        )

    # Dynamics consistency: z should evolve smoothly
    L_dyn = model.dynamics.dynamics_consistency_loss(z_i)

    # Combine
    breakdown = loss_fn(
        L_rec=L_rec,
        L_sem=L_sem,
        L_trans=L_trans,
        L_cycle=L_cycle,
        L_gauge=L_gauge,
        L_event=L_dyn,  # re-use event slot for dynamics
    )
    return breakdown


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_training(args: argparse.Namespace) -> dict:
    """Full SDQ training pipeline. Returns results dict."""
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load runs and extract trajectories
    # ------------------------------------------------------------------
    print("\n=== Loading runs ===")
    all_prompt_ids = []
    for fam in list(FAMILIES.values()):
        all_prompt_ids.extend(fam)

    runs = collect_runs("data/runs", prompt_ids=all_prompt_ids, device="cpu")
    print(f"Loaded {len(runs)} runs (requested {len(all_prompt_ids)} prompt IDs)")

    # Check which families are fully available
    available_families = {}
    for fam_name, pids in FAMILIES.items():
        found = [pid for pid in pids if pid in runs]
        if found:
            available_families[fam_name] = found
        print(f"  {fam_name}: {len(found)}/{len(pids)} runs")

    # Extract trajectories from the chosen layer and normalize
    print(f"\nExtracting trajectories from layer {args.layer}...")
    trajectories: dict[str, Trajectory] = {}
    raw_norms = []
    for pid, run in runs.items():
        traj = extract_trajectory(run, layer=args.layer, to_float=True)
        raw_norm = traj.states.norm(dim=-1).mean().item()
        raw_norms.append(raw_norm)
        # Normalize to unit norm: makes transport loss scale-invariant
        normed = normalize_trajectory(traj.states, mode="unit_norm")
        traj = Trajectory(
            states=normed,
            prompt_id=traj.prompt_id,
            layer=traj.layer,
            token_ids=traj.token_ids,
            token_strings=traj.token_strings,
            metadata=traj.metadata,
        )
        trajectories[pid] = traj
        print(f"  {pid}: T={traj.T}, D={traj.D}, raw_norm={raw_norm:.1f}, normed_norm={normed.norm(dim=-1).mean():.3f}")

    hidden_dim = next(iter(trajectories.values())).D
    avg_raw_norm = sum(raw_norms) / len(raw_norms)
    print(f"\nHidden dim: {hidden_dim}, avg raw norm: {avg_raw_norm:.1f}")
    
    # Show velocity scales after normalization
    for pid in list(trajectories.keys())[:2]:
        vel = velocity_vectors(trajectories[pid].states)
        print(f"  {pid} velocity norm: {vel.norm(dim=-1).mean():.4f}")

    # ------------------------------------------------------------------
    # 2. Build SDQ model
    # ------------------------------------------------------------------
    latent_dim = args.latent_dim
    gauge_dim = args.gauge_dim
    rank = args.rank

    print(f"\n=== Building SDQ model ===")
    print(f"  latent_dim={latent_dim}, gauge_dim={gauge_dim}, rank={rank}")

    model = SDQModel(
        encoder=TemporalConvEncoder(hidden_dim, latent_dim, window_size=3),
        decoder=ResidualDecoder(latent_dim, gauge_dim, hidden_dim),
        gauge_encoder=TemporalGaugeEncoder(hidden_dim, gauge_dim, window_size=3),
        aligner=MonotoneAligner(hidden_dim, align_dim=min(128, hidden_dim)),
        transport=LocalTransportField(hidden_dim, rank=rank, context_dim=min(128, hidden_dim)),
        dynamics=LatentODE(latent_dim),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {param_count:,}")

    loss_fn = SDQLoss(
        weights=LossWeights(
            reconstruction=1.0,
            semantic=2.0,
            transport=2.0,
            cycle=1.0,
            gauge=0.1,
            event=0.1,  # dynamics
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ------------------------------------------------------------------
    # 3. Prepare training pairs
    # ------------------------------------------------------------------
    train_pairs: list[tuple[str, str, str]] = []  # (family, pid_i, pid_j)
    for fam_name in TRAIN_FAMILIES:
        if fam_name not in available_families:
            continue
        pids = available_families[fam_name]
        for p_i, p_j in combinations(pids, 2):
            train_pairs.append((fam_name, p_i, p_j))

    holdout_pairs: list[tuple[str, str, str]] = []
    for fam_name in HOLDOUT_FAMILIES:
        if fam_name not in available_families:
            continue
        pids = available_families[fam_name]
        for p_i, p_j in combinations(pids, 2):
            holdout_pairs.append((fam_name, p_i, p_j))

    print(f"\nTraining pairs: {len(train_pairs)}")
    print(f"Holdout pairs:  {len(holdout_pairs)}")

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(f"\n=== Training for {args.epochs} epochs ===")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {
            "total": 0.0, "reconstruction": 0.0, "semantic": 0.0,
            "transport": 0.0, "cycle": 0.0, "gauge": 0.0, "dynamics": 0.0,
        }
        pair_count = 0

        for fam_name, pid_i, pid_j in train_pairs:
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)

            optimizer.zero_grad()
            breakdown = train_pair(
                model, h_i, h_j, loss_fn,
                compute_cycle=True,  # always on — low-rank is efficient
            )
            breakdown.total.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bd = breakdown.to_dict()
            for k in epoch_losses:
                if k == "dynamics":
                    epoch_losses[k] += bd.get("event", 0.0)
                else:
                    epoch_losses[k] += bd.get(k, 0.0)
            pair_count += 1

        scheduler.step()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(pair_count, 1)
        epoch_losses["lr"] = scheduler.get_last_lr()[0]
        history.append(epoch_losses)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d} | "
                f"total={epoch_losses['total']:.4f} "
                f"rec={epoch_losses['reconstruction']:.4f} "
                f"sem={epoch_losses['semantic']:.4f} "
                f"trans={epoch_losses['transport']:.4f} "
                f"cycle={epoch_losses['cycle']:.4f} "
                f"gauge={epoch_losses['gauge']:.4f} "
                f"dyn={epoch_losses['dynamics']:.4f} "
                f"lr={epoch_losses['lr']:.6f}"
            )

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    print("\n=== Evaluation ===")
    model.eval()
    results: dict = {"training_history": history}

    # 5a. Transport rank analysis on training pairs
    print("\n--- Transport Rank Analysis ---")
    rank_results = {}
    with torch.no_grad():
        for fam_name, pid_i, pid_j in train_pairs[:6]:  # analyze first 6
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)
            h_j_aligned = model.aligner.align_trajectory(h_i, h_j)
            tr = model.transport(h_i, h_j_aligned, return_operators=True)
            if tr.G_t is not None:
                # Average transport over time
                G_mean = tr.G_t.mean(dim=0).cpu()
                ra = rank_analysis(G_mean)
                pair_key = f"{pid_i}->{pid_j}"
                rank_results[pair_key] = {
                    "effective_rank": ra.effective_rank,
                    "nuclear_norm": ra.nuclear_norm,
                    "frobenius_deviation": ra.frobenius_deviation,
                    "spectral_norm": ra.spectral_norm,
                    "top_k_explained": {str(k): v for k, v in ra.top_k_explained.items()},
                }
                print(
                    f"  {pair_key}: eff_rank={ra.effective_rank:.2f}, "
                    f"nuc_norm={ra.nuclear_norm:.4f}, "
                    f"frob_dev={ra.frobenius_deviation:.4f}"
                )
    results["transport_rank"] = rank_results

    # 5b. Latent semantic consistency: within-group vs cross-group
    print("\n--- Latent Semantic Consistency ---")
    latent_states: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for pid, traj in trajectories.items():
            h = traj.states.to(device)
            z = model.encoder(h)
            latent_states[pid] = z.cpu()

    within_distances = []
    cross_distances = []
    for fam_i, pids_i in available_families.items():
        for a, b in combinations(pids_i, 2):
            if a in latent_states and b in latent_states:
                T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                within_distances.append(d)

    fam_names = list(available_families.keys())
    for i in range(len(fam_names)):
        for j in range(i + 1, len(fam_names)):
            pids_i = available_families[fam_names[i]]
            pids_j = available_families[fam_names[j]]
            for a in pids_i[:2]:
                for b in pids_j[:2]:
                    if a in latent_states and b in latent_states:
                        T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                        d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                        cross_distances.append(d)

    avg_within = sum(within_distances) / max(len(within_distances), 1)
    avg_cross = sum(cross_distances) / max(len(cross_distances), 1)
    separation = avg_cross / max(avg_within, 1e-8)

    print(f"  Within-group latent distance:  {avg_within:.4f}")
    print(f"  Cross-group latent distance:   {avg_cross:.4f}")
    print(f"  Separation ratio (cross/within): {separation:.2f}")

    results["latent_consistency"] = {
        "within_group_avg_distance": avg_within,
        "cross_group_avg_distance": avg_cross,
        "separation_ratio": separation,
    }

    # 5c. Reconstruction quality on training data
    print("\n--- Reconstruction Quality ---")
    rec_errors = {}
    with torch.no_grad():
        for pid, traj in trajectories.items():
            h = traj.states.to(device)
            z = model.encoder(h)
            u = model.gauge_encoder(h)
            h_hat = model.decoder(z, u)
            err = reconstruction_loss(h, h_hat).item()
            rec_errors[pid] = err

    avg_rec = sum(rec_errors.values()) / len(rec_errors)
    print(f"  Average reconstruction MSE: {avg_rec:.6f}")
    for pid, err in sorted(rec_errors.items()):
        family = next((f for f, pids in FAMILIES.items() if pid in pids), "?")
        print(f"    {pid} ({family}): {err:.6f}")
    results["reconstruction"] = rec_errors

    # 5d. Transport consistency: mean residual on training vs holdout
    print("\n--- Transport Consistency ---")
    train_residuals = []
    holdout_residuals = []

    with torch.no_grad():
        for pairs, residual_list, label in [
            (train_pairs, train_residuals, "train"),
            (holdout_pairs, holdout_residuals, "holdout"),
        ]:
            for fam_name, pid_i, pid_j in pairs:
                h_i = trajectories[pid_i].states.to(device)
                h_j = trajectories[pid_j].states.to(device)
                h_j_aligned = model.aligner.align_trajectory(h_i, h_j)
                tr = model.transport(h_i, h_j_aligned)
                res = tr.residual.norm(dim=-1).mean().item()
                residual_list.append(res)

    avg_train_res = sum(train_residuals) / max(len(train_residuals), 1)
    avg_holdout_res = sum(holdout_residuals) / max(len(holdout_residuals), 1)
    print(f"  Train transport residual:   {avg_train_res:.6f}")
    print(f"  Holdout transport residual: {avg_holdout_res:.6f}")
    results["transport_consistency"] = {
        "train_residual": avg_train_res,
        "holdout_residual": avg_holdout_res,
    }

    # 5e. Cycle consistency on training pairs
    print("\n--- Cycle Consistency ---")
    cycle_errors = []
    with torch.no_grad():
        for fam_name, pid_i, pid_j in train_pairs[:6]:
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)
            h_j_aligned = model.aligner.align_trajectory(h_i, h_j)
            tr_ij = model.transport(h_i, h_j_aligned, return_operators=True)
            tr_ji = model.transport(h_j_aligned, h_i, return_operators=True)
            if tr_ij.G_t is not None and tr_ji.G_t is not None:
                T_min = min(tr_ij.G_t.shape[0], tr_ji.G_t.shape[0])
                metrics = cycle_consistency_metrics(
                    tr_ij.G_t[:T_min].cpu(),
                    tr_ji.G_t[:T_min].cpu(),
                )
                cycle_errors.append(metrics)
                print(f"  {pid_i}<->{pid_j}: {metrics}")

    results["cycle_consistency"] = cycle_errors

    # 5f. Soft-DTW alignment quality
    print("\n--- Alignment Quality ---")
    dtw_distances = {}
    with torch.no_grad():
        for fam_name, pids in available_families.items():
            for a, b in combinations(pids, 2):
                h_a = trajectories[a].states.to(device)
                h_b = trajectories[b].states.to(device)
                dist, path = soft_dtw_alignment(h_a, h_b, gamma=1.0)
                key = f"{a}<->{b}"
                dtw_distances[key] = dist.item()
    results["alignment_dtw"] = dtw_distances
    for key, dist in sorted(dtw_distances.items()):
        print(f"  {key}: DTW={dist:.4f}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SDQ TRAINING SUMMARY")
    print("=" * 60)
    print(f"Epochs:              {args.epochs}")
    print(f"Training pairs:      {len(train_pairs)}")
    print(f"Holdout pairs:       {len(holdout_pairs)}")
    print(f"Final total loss:    {history[-1]['total']:.6f}")
    print(f"Final recon loss:    {history[-1]['reconstruction']:.6f}")
    print(f"Final semantic loss: {history[-1]['semantic']:.6f}")
    print(f"Final transport loss:{history[-1]['transport']:.6f}")
    print(f"Avg recon MSE:       {avg_rec:.6f}")
    print(f"Latent separation:   {separation:.2f} (cross/within)")
    print(f"Train transport res: {avg_train_res:.6f}")
    print(f"Holdout transport res:{avg_holdout_res:.6f}")
    if rank_results:
        avg_eff_rank = sum(r["effective_rank"] for r in rank_results.values()) / len(rank_results)
        print(f"Avg effective rank:  {avg_eff_rank:.2f}")
    print("=" * 60)

    results["summary"] = {
        "epochs": args.epochs,
        "train_pairs": len(train_pairs),
        "holdout_pairs": len(holdout_pairs),
        "final_loss": history[-1],
        "avg_reconstruction_mse": avg_rec,
        "latent_separation_ratio": separation,
        "train_transport_residual": avg_train_res,
        "holdout_transport_residual": avg_holdout_res,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run SDQ training pipeline")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--layer", type=int, default=-1, help="Layer to extract (-1 = last)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--gauge-dim", type=int, default=32, help="Gauge dimension")
    parser.add_argument("--rank", type=int, default=16, help="Transport rank")
    parser.add_argument("--output", type=str, default="artifacts/sdq_results.json", help="Output path")
    args = parser.parse_args()

    start = time.time()
    results = run_training(args)
    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed
    print(f"\nCompleted in {elapsed:.1f}s")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
