#!/usr/bin/env python3
"""SDQ v2 training: improved transport, semantic separation, and evaluation.

Key improvements over v1:
  1. Transport conditioned on transform-type (shared across families)
  2. Triplet loss + hard negatives for latent semantic separation
  3. Latent velocity matching loss
  4. center_scale normalization for cross-family comparability
  5. Lower rank (4) forcing simpler transport
  6. Triple composition evaluation
  7. Identity-baseline and shuffle controls
  8. Hold out by semantic family (train cat+add5, test bird+add15)
     with shared transform types

Usage:
    python run_sdq_v2.py [--epochs 300] [--device cuda]
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
from sdq.trajectories import (
    extract_trajectory, Trajectory, normalize_trajectory, velocity_vectors,
)
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
    SDQLoss,
)
from sdq.losses.cycle import lowrank_cycle_loss
from sdq.losses.gauge_regularization import lowrank_gauge_regularization_loss
from sdq.losses.triplet import triplet_semantic_loss
from sdq.losses.velocity import latent_velocity_loss
from sdq.losses.total import LossWeights, LossBreakdown
from sdq.eval import rank_analysis
from sdq.eval.composition import lowrank_composition_error, batch_composition_test


# ---------------------------------------------------------------------------
# Semantic equivalence groups
# ---------------------------------------------------------------------------
FAMILIES = {
    "cat": ["cat_therefore", "cat_reorder", "cat_since", "cat_given"],
    "bird": ["bird_therefore", "bird_reorder", "bird_since", "bird_given"],
    "add5": ["add5_base", "add5_reorder", "add5_worded", "add5_worded2"],
    "add15": ["add15_base", "add15_reorder", "add15_worded", "add15_worded2"],
}

TRAIN_FAMILIES = ["cat", "add5"]
HOLDOUT_FAMILIES = ["bird", "add15"]

# Map each prompt_id to its family
PID_TO_FAMILY = {pid: fam for fam, pids in FAMILIES.items() for pid in pids}

# ---------------------------------------------------------------------------
# Transform types: encode the *surface-form change* not the semantic content.
# e.g. "therefore→since" is the same transform whether applied to cat or bird.
# We extract the suffix after the family prefix as the "surface variant".
# ---------------------------------------------------------------------------
SURFACE_VARIANTS = {
    "cat": ["therefore", "reorder", "since", "given"],
    "bird": ["therefore", "reorder", "since", "given"],
    "add5": ["base", "reorder", "worded", "worded2"],
    "add15": ["base", "reorder", "worded", "worded2"],
}


def get_variant(pid: str) -> str:
    """Extract surface variant from prompt_id."""
    for fam, pids in FAMILIES.items():
        if pid in pids:
            return pid[len(fam) + 1:]  # strip family prefix + underscore
    return pid


def get_transform_type(pid_i: str, pid_j: str) -> str:
    """Get a canonical transform type name for a (source, target) pair."""
    v_i = get_variant(pid_i)
    v_j = get_variant(pid_j)
    return f"{v_i}->{v_j}"


# Build a global transform-type vocabulary
def build_transform_vocab(families: dict[str, list[str]]) -> dict[str, int]:
    """Create integer IDs for all transform types across families."""
    all_variants = set()
    for fam, pids in families.items():
        variants = [get_variant(pid) for pid in pids]
        for v_i in variants:
            for v_j in variants:
                if v_i != v_j:
                    all_variants.add(f"{v_i}->{v_j}")
    return {tt: i for i, tt in enumerate(sorted(all_variants))}


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------
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
        for module in [self.encoder, self.decoder, self.gauge_encoder,
                       self.aligner, self.transport, self.dynamics]:
            yield from module.parameters()

    def train(self):
        for m in [self.encoder, self.decoder, self.gauge_encoder,
                  self.aligner, self.transport, self.dynamics]:
            m.train()

    def eval(self):
        for m in [self.encoder, self.decoder, self.gauge_encoder,
                  self.aligner, self.transport, self.dynamics]:
            m.eval()

    def to(self, device):
        for m in [self.encoder, self.decoder, self.gauge_encoder,
                  self.aligner, self.transport, self.dynamics]:
            m.to(device)
        return self


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_pair(
    model: SDQModel,
    h_i: torch.Tensor,
    h_j: torch.Tensor,
    loss_fn: SDQLoss,
    transform_type_id: int | None = None,
) -> LossBreakdown:
    """One forward pass for a (source, target) pair."""
    # 1. Align
    h_j_aligned = model.aligner.align_trajectory(h_i, h_j)

    # 2. Encode
    z_i = model.encoder(h_i)
    z_j = model.encoder(h_j_aligned)

    # 3. Gauge
    u_i = model.gauge_encoder(h_i)
    u_j = model.gauge_encoder(h_j_aligned)

    # 4. Decode
    h_i_hat = model.decoder(z_i, u_i)
    h_j_hat = model.decoder(z_j, u_j)

    # 5. Transport with transform-type conditioning
    transport_ij = model.transport(
        h_i, h_j_aligned, return_operators=False,
        transform_type_id=transform_type_id,
    )

    # 6. Losses
    L_rec = (reconstruction_loss(h_i, h_i_hat) +
             reconstruction_loss(h_j_aligned, h_j_hat)) / 2.0
    L_sem = semantic_consistency_loss(z_i, z_j)
    L_trans = transport_consistency_loss(transport_ij)

    # Velocity matching in latent space
    L_vel = latent_velocity_loss(z_i, z_j)

    # Cycle + gauge from low-rank factors
    L_cycle = torch.tensor(0.0, device=h_i.device)
    L_gauge = torch.tensor(0.0, device=h_i.device)
    if transport_ij.U is not None:
        transport_ji = model.transport(
            h_j_aligned, h_i, return_operators=False,
            transform_type_id=transform_type_id,
        )
        if transport_ji.U is not None:
            T_min = min(transport_ij.U.shape[0], transport_ji.U.shape[0])
            L_cycle = lowrank_cycle_loss(
                transport_ij.U[:T_min], transport_ij.V[:T_min],
                transport_ji.U[:T_min], transport_ji.V[:T_min],
            )
        L_gauge = lowrank_gauge_regularization_loss(
            transport_ij.U, transport_ij.V,
        )

    # Dynamics
    L_dyn = model.dynamics.dynamics_consistency_loss(z_i)

    return loss_fn(
        L_rec=L_rec, L_sem=L_sem, L_trans=L_trans,
        L_cycle=L_cycle, L_gauge=L_gauge, L_event=L_dyn,
        L_velocity=L_vel,
    )


def compute_epoch_triplet(
    model: SDQModel,
    trajectories: dict[str, Trajectory],
    train_pids: list[str],
    pid_to_family: dict[str, str],
    device: torch.device,
    margin: float = 1.0,
) -> torch.Tensor:
    """Compute triplet loss across all training PIDs."""
    # Encode all training trajectories
    latents: dict[str, torch.Tensor] = {}
    for pid in train_pids:
        h = trajectories[pid].states.to(device)
        z = model.encoder(h)
        latents[pid] = z

    # Group by family
    families: dict[str, list[str]] = {}
    for pid in train_pids:
        fam = pid_to_family[pid]
        families.setdefault(fam, []).append(pid)

    fam_names = list(families.keys())
    if len(fam_names) < 2:
        return torch.tensor(0.0, device=device)

    total = torch.tensor(0.0, device=device)
    count = 0

    for fam in fam_names:
        pids = families[fam]
        neg_pids = [p for f in fam_names if f != fam for p in families[f]]
        if len(pids) < 2 or not neg_pids:
            continue

        for i in range(len(pids)):
            for j in range(len(pids)):
                if i == j:
                    continue
                anchor_z = latents[pids[i]]
                pos_z = latents[pids[j]]
                # Hard negative: closest from other families
                anchor_mean = anchor_z.mean(dim=0).detach()
                best_dist = float("inf")
                best_neg = None
                for np_ in neg_pids:
                    d = (anchor_mean - latents[np_].mean(dim=0).detach()).pow(2).sum().item()
                    if d < best_dist:
                        best_dist = d
                        best_neg = latents[np_]

                total = total + triplet_semantic_loss(
                    anchor_z, pos_z, best_neg, margin=margin,
                )
                count += 1

    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_training(args: argparse.Namespace) -> dict:
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load and normalize
    # ------------------------------------------------------------------
    print("\n=== Loading runs ===")
    all_pids = [pid for fam in FAMILIES.values() for pid in fam]
    runs = collect_runs("data/runs", prompt_ids=all_pids, device="cpu")
    print(f"Loaded {len(runs)} runs")

    available_families: dict[str, list[str]] = {}
    for fam_name, pids in FAMILIES.items():
        found = [pid for pid in pids if pid in runs]
        if found:
            available_families[fam_name] = found
        print(f"  {fam_name}: {len(found)}/{len(pids)} runs")

    # Extract + normalize
    norm_mode = args.norm_mode
    print(f"\nNormalization mode: {norm_mode}")
    trajectories: dict[str, Trajectory] = {}
    for pid, run in runs.items():
        traj = extract_trajectory(run, layer=args.layer, to_float=True)
        normed = normalize_trajectory(traj.states, mode=norm_mode)
        trajectories[pid] = Trajectory(
            states=normed, prompt_id=traj.prompt_id, layer=traj.layer,
            token_ids=traj.token_ids, token_strings=traj.token_strings,
            metadata=traj.metadata,
        )

    hidden_dim = next(iter(trajectories.values())).D

    # Print per-family velocity statistics
    print("\nPer-family velocity statistics:")
    for fam, pids in available_families.items():
        vel_norms = []
        for pid in pids:
            vel = velocity_vectors(trajectories[pid].states)
            vel_norms.append(vel.norm(dim=-1).mean().item())
        avg_vel = sum(vel_norms) / len(vel_norms)
        print(f"  {fam}: avg |vel| = {avg_vel:.4f}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    rank = args.rank
    transform_vocab = build_transform_vocab(FAMILIES)
    num_tt = len(transform_vocab)
    print(f"\nTransform types: {num_tt}")
    for tt, idx in sorted(transform_vocab.items(), key=lambda x: x[1]):
        print(f"  [{idx}] {tt}")

    model = SDQModel(
        encoder=TemporalConvEncoder(hidden_dim, args.latent_dim, window_size=3),
        decoder=ResidualDecoder(args.latent_dim, args.gauge_dim, hidden_dim),
        gauge_encoder=TemporalGaugeEncoder(hidden_dim, args.gauge_dim, window_size=3),
        aligner=MonotoneAligner(hidden_dim, align_dim=min(128, hidden_dim)),
        transport=LocalTransportField(
            hidden_dim, rank=rank, context_dim=min(128, hidden_dim),
            num_transform_types=num_tt, transform_embed_dim=16,
        ),
        dynamics=LatentODE(args.latent_dim),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count:,}")
    print(f"Transport rank: {rank}")

    loss_fn = SDQLoss(weights=LossWeights(
        reconstruction=1.0,
        semantic=2.0,
        transport=2.0,
        cycle=1.0,
        gauge=0.1,
        event=0.1,
        triplet=1.0,
        velocity=1.0,
    ))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ------------------------------------------------------------------
    # 3. Prepare pairs
    # ------------------------------------------------------------------
    train_pids: list[str] = []
    for fam in TRAIN_FAMILIES:
        if fam in available_families:
            train_pids.extend(available_families[fam])

    holdout_pids: list[str] = []
    for fam in HOLDOUT_FAMILIES:
        if fam in available_families:
            holdout_pids.extend(available_families[fam])

    train_pairs: list[tuple[str, str, str]] = []
    for fam_name in TRAIN_FAMILIES:
        if fam_name not in available_families:
            continue
        for p_i, p_j in combinations(available_families[fam_name], 2):
            train_pairs.append((fam_name, p_i, p_j))

    holdout_pairs: list[tuple[str, str, str]] = []
    for fam_name in HOLDOUT_FAMILIES:
        if fam_name not in available_families:
            continue
        for p_i, p_j in combinations(available_families[fam_name], 2):
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
            "transport": 0.0, "cycle": 0.0, "gauge": 0.0,
            "dynamics": 0.0, "triplet": 0.0, "velocity": 0.0,
        }
        pair_count = 0

        # --- Pairwise losses ---
        for fam_name, pid_i, pid_j in train_pairs:
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)

            tt = get_transform_type(pid_i, pid_j)
            tt_id = transform_vocab.get(tt)

            optimizer.zero_grad()
            breakdown = train_pair(model, h_i, h_j, loss_fn, transform_type_id=tt_id)

            # Add triplet loss (computed over all train PIDs)
            if loss_fn.weights.triplet > 0:
                L_trip = compute_epoch_triplet(
                    model, trajectories, train_pids, PID_TO_FAMILY,
                    device, margin=args.triplet_margin,
                )
                triplet_weighted = loss_fn.weights.triplet * L_trip
                total_with_triplet = breakdown.total + triplet_weighted
            else:
                L_trip = torch.tensor(0.0, device=device)
                total_with_triplet = breakdown.total

            total_with_triplet.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bd = breakdown.to_dict()
            for k in epoch_losses:
                if k == "dynamics":
                    epoch_losses[k] += bd.get("event", 0.0)
                elif k == "triplet":
                    epoch_losses[k] += L_trip.item()
                else:
                    epoch_losses[k] += bd.get(k, 0.0)
            pair_count += 1

        scheduler.step()

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
                f"cyc={epoch_losses['cycle']:.4f} "
                f"trip={epoch_losses['triplet']:.4f} "
                f"vel={epoch_losses['velocity']:.4f} "
                f"lr={epoch_losses['lr']:.6f}"
            )

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    print("\n=== Evaluation ===")
    model.eval()
    results: dict = {"training_history": history}

    # 5a. Transport rank analysis
    print("\n--- Transport Rank Analysis ---")
    rank_results = {}
    with torch.no_grad():
        for fam_name, pid_i, pid_j in train_pairs[:6]:
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)
            h_j_aligned = model.aligner.align_trajectory(h_i, h_j)
            tt = get_transform_type(pid_i, pid_j)
            tt_id = transform_vocab.get(tt)
            tr = model.transport(h_i, h_j_aligned, return_operators=True,
                                 transform_type_id=tt_id)
            G_mean = tr.G_t.mean(dim=0).cpu()
            ra = rank_analysis(G_mean)
            pair_key = f"{pid_i}->{pid_j}"
            rank_results[pair_key] = {
                "effective_rank": ra.effective_rank,
                "nuclear_norm": ra.nuclear_norm,
                "frobenius_deviation": ra.frobenius_deviation,
                "top_k_explained": {str(k): v for k, v in ra.top_k_explained.items()},
            }
            print(f"  {pair_key}: eff_rank={ra.effective_rank:.2f}, "
                  f"nuc={ra.nuclear_norm:.4f}, frob={ra.frobenius_deviation:.4f}")
    results["transport_rank"] = rank_results

    # 5b. Latent semantic consistency
    print("\n--- Latent Semantic Consistency ---")
    latent_states: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for pid, traj in trajectories.items():
            h = traj.states.to(device)
            latent_states[pid] = model.encoder(h).cpu()

    within_dists, cross_dists = [], []
    for fam, pids in available_families.items():
        for a, b in combinations(pids, 2):
            if a in latent_states and b in latent_states:
                T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                within_dists.append(d)

    fam_names = list(available_families.keys())
    for i in range(len(fam_names)):
        for j in range(i + 1, len(fam_names)):
            for a in available_families[fam_names[i]][:2]:
                for b in available_families[fam_names[j]][:2]:
                    if a in latent_states and b in latent_states:
                        T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                        d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                        cross_dists.append(d)

    avg_within = sum(within_dists) / max(len(within_dists), 1)
    avg_cross = sum(cross_dists) / max(len(cross_dists), 1)
    separation = avg_cross / max(avg_within, 1e-8)
    print(f"  Within-group:  {avg_within:.4f}")
    print(f"  Cross-group:   {avg_cross:.4f}")
    print(f"  Separation:    {separation:.2f}")
    results["latent_consistency"] = {
        "within": avg_within, "cross": avg_cross, "separation": separation,
    }

    # 5c. Reconstruction quality
    print("\n--- Reconstruction Quality ---")
    rec_errors = {}
    with torch.no_grad():
        for pid, traj in trajectories.items():
            h = traj.states.to(device)
            z = model.encoder(h)
            u = model.gauge_encoder(h)
            h_hat = model.decoder(z, u)
            rec_errors[pid] = reconstruction_loss(h, h_hat).item()
    avg_rec = sum(rec_errors.values()) / len(rec_errors)
    print(f"  Average MSE: {avg_rec:.6f}")
    for pid, err in sorted(rec_errors.items()):
        print(f"    {pid}: {err:.6f}")
    results["reconstruction"] = rec_errors

    # 5d. Transport residuals (train vs holdout)
    print("\n--- Transport Consistency ---")
    train_res, holdout_res = [], []
    with torch.no_grad():
        for pairs, res_list in [(train_pairs, train_res), (holdout_pairs, holdout_res)]:
            for fam, pid_i, pid_j in pairs:
                h_i = trajectories[pid_i].states.to(device)
                h_j = trajectories[pid_j].states.to(device)
                h_j_a = model.aligner.align_trajectory(h_i, h_j)
                tt = get_transform_type(pid_i, pid_j)
                tt_id = transform_vocab.get(tt)
                tr = model.transport(h_i, h_j_a, transform_type_id=tt_id)
                res_list.append(tr.residual.norm(dim=-1).mean().item())

    avg_train_res = sum(train_res) / max(len(train_res), 1)
    avg_holdout_res = sum(holdout_res) / max(len(holdout_res), 1)
    print(f"  Train:   {avg_train_res:.6f}")
    print(f"  Holdout: {avg_holdout_res:.6f}")
    results["transport_residuals"] = {
        "train": avg_train_res, "holdout": avg_holdout_res,
    }

    # 5e. Cycle consistency
    print("\n--- Cycle Consistency ---")
    cycle_errors = []
    with torch.no_grad():
        for fam, pid_i, pid_j in train_pairs[:6]:
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)
            h_j_a = model.aligner.align_trajectory(h_i, h_j)
            tt = get_transform_type(pid_i, pid_j)
            tt_id = transform_vocab.get(tt)
            tr_ij = model.transport(h_i, h_j_a, return_operators=True, transform_type_id=tt_id)
            tr_ji = model.transport(h_j_a, h_i, return_operators=True, transform_type_id=tt_id)
            T_min = min(tr_ij.G_t.shape[0], tr_ji.G_t.shape[0])
            metrics = cycle_consistency_metrics(
                tr_ij.G_t[:T_min].cpu(), tr_ji.G_t[:T_min].cpu(),
            )
            cycle_errors.append(metrics)
            print(f"  {pid_i}<->{pid_j}: {metrics}")
    results["cycle_consistency"] = cycle_errors

    # 5f. Triple composition test (THE KEY NEW TEST)
    print("\n--- Triple Composition Test ---")
    comp_results = []
    with torch.no_grad():
        for fam_name in TRAIN_FAMILIES:
            if fam_name not in available_families:
                continue
            pids = available_families[fam_name]
            if len(pids) < 3:
                continue

            # Compute transport factors for all pairs in this family
            factors: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
            for a, b in combinations(pids, 2):
                h_a = trajectories[a].states.to(device)
                h_b = trajectories[b].states.to(device)
                h_b_a = model.aligner.align_trajectory(h_a, h_b)
                tt = get_transform_type(a, b)
                tt_id = transform_vocab.get(tt)
                tr = model.transport(h_a, h_b_a, return_operators=False, transform_type_id=tt_id)
                factors[(a, b)] = (tr.U.cpu(), tr.V.cpu())
                # Also compute reverse
                tr_rev = model.transport(h_b_a, h_a, return_operators=False, transform_type_id=tt_id)
                factors[(b, a)] = (tr_rev.U.cpu(), tr_rev.V.cpu())

            # Test all triples
            for a in pids:
                for b in pids:
                    if b == a:
                        continue
                    for c in pids:
                        if c == a or c == b:
                            continue
                        if (a, b) in factors and (b, c) in factors and (a, c) in factors:
                            err = lowrank_composition_error(
                                factors[(a, b)][0], factors[(a, b)][1],
                                factors[(b, c)][0], factors[(b, c)][1],
                                factors[(a, c)][0], factors[(a, c)][1],
                            )
                            comp_results.append({
                                "family": fam_name,
                                "triple": f"{a}->{b}->{c} vs {a}->{c}",
                                "composition_error": err,
                            })

    if comp_results:
        avg_comp = sum(r["composition_error"] for r in comp_results) / len(comp_results)
        print(f"  Average composition error: {avg_comp:.6f} ({len(comp_results)} triples)")
        for r in comp_results[:8]:
            print(f"    {r['triple']}: {r['composition_error']:.6f}")
    results["composition"] = comp_results

    # 5g. Transport sharing test: same transform type, different families
    print("\n--- Transport Sharing (same transform type, different families) ---")
    sharing_results = {}
    with torch.no_grad():
        # For cat and bird (same variant set), compare transports
        for fam_train, fam_holdout in [("cat", "bird")]:
            if fam_train not in available_families or fam_holdout not in available_families:
                continue
            train_v = [(pid, get_variant(pid)) for pid in available_families[fam_train]]
            holdout_v = [(pid, get_variant(pid)) for pid in available_families[fam_holdout]]

            for (pid_t1, v1), (pid_t2, v2) in combinations(train_v, 2):
                # Find matching holdout pair
                pid_h1 = next((p for p, v in holdout_v if v == v1), None)
                pid_h2 = next((p for p, v in holdout_v if v == v2), None)
                if pid_h1 is None or pid_h2 is None:
                    continue

                # Compute U,V for train pair
                h_t1 = trajectories[pid_t1].states.to(device)
                h_t2 = trajectories[pid_t2].states.to(device)
                h_t2_a = model.aligner.align_trajectory(h_t1, h_t2)
                tt = get_transform_type(pid_t1, pid_t2)
                tt_id = transform_vocab.get(tt)
                tr_train = model.transport(h_t1, h_t2_a, return_operators=False, transform_type_id=tt_id)

                # Compute for holdout pair
                h_h1 = trajectories[pid_h1].states.to(device)
                h_h2 = trajectories[pid_h2].states.to(device)
                h_h2_a = model.aligner.align_trajectory(h_h1, h_h2)
                tr_hold = model.transport(h_h1, h_h2_a, return_operators=False, transform_type_id=tt_id)

                # Compare transport residuals
                train_r = tr_train.residual.norm(dim=-1).mean().item()
                hold_r = tr_hold.residual.norm(dim=-1).mean().item()
                key = f"{v1}->{v2}"
                sharing_results[key] = {
                    "train_pair": f"{pid_t1}->{pid_t2}",
                    "holdout_pair": f"{pid_h1}->{pid_h2}",
                    "train_residual": train_r,
                    "holdout_residual": hold_r,
                }
                print(f"  {key}: train={train_r:.4f} holdout={hold_r:.4f}")

    results["transport_sharing"] = sharing_results

    # 5h. Identity baseline control
    print("\n--- Identity Baseline Control ---")
    identity_res = []
    with torch.no_grad():
        for fam, pid_i, pid_j in train_pairs:
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)
            h_j_a = model.aligner.align_trajectory(h_i, h_j)
            vel_i = h_i[1:] - h_i[:-1]
            vel_j = h_j_a[1:] - h_j_a[:-1]
            T_min = min(vel_i.shape[0], vel_j.shape[0])
            identity_residual = (vel_j[:T_min] - vel_i[:T_min]).norm(dim=-1).mean().item()
            identity_res.append(identity_residual)

    avg_identity = sum(identity_res) / max(len(identity_res), 1)
    improvement = 1 - (avg_train_res / max(avg_identity, 1e-8))
    print(f"  Identity baseline residual: {avg_identity:.6f}")
    print(f"  Learned transport residual: {avg_train_res:.6f}")
    print(f"  Improvement: {improvement:.1%}")
    results["identity_baseline"] = {
        "identity_residual": avg_identity,
        "learned_residual": avg_train_res,
        "improvement_pct": improvement,
    }

    # 5i. Shuffle control: randomly permute family assignments
    print("\n--- Shuffle Control ---")
    import random
    rng = random.Random(999)
    all_train = [pid for fam in TRAIN_FAMILIES if fam in available_families
                 for pid in available_families[fam]]
    shuffled = all_train.copy()
    rng.shuffle(shuffled)
    shuffle_within, shuffle_cross = [], []
    with torch.no_grad():
        # Compute latent distances for shuffled pairings
        for i in range(0, len(shuffled) - 1, 2):
            a, b = shuffled[i], shuffled[i + 1]
            if a in latent_states and b in latent_states:
                T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                if PID_TO_FAMILY.get(a) == PID_TO_FAMILY.get(b):
                    shuffle_within.append(d)
                else:
                    shuffle_cross.append(d)

    if shuffle_within and shuffle_cross:
        sw = sum(shuffle_within) / len(shuffle_within)
        sc = sum(shuffle_cross) / len(shuffle_cross)
        print(f"  Shuffle within:  {sw:.4f}")
        print(f"  Shuffle cross:   {sc:.4f}")
        print(f"  True within:     {avg_within:.4f}")
        print(f"  True cross:      {avg_cross:.4f}")
        results["shuffle_control"] = {
            "shuffle_within": sw, "shuffle_cross": sc,
            "true_within": avg_within, "true_cross": avg_cross,
        }

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SDQ v2 TRAINING SUMMARY")
    print("=" * 60)
    print(f"Epochs:               {args.epochs}")
    print(f"Rank:                 {rank}")
    print(f"Normalization:        {norm_mode}")
    print(f"Transform types:      {num_tt}")
    print(f"Training pairs:       {len(train_pairs)}")
    print(f"Holdout pairs:        {len(holdout_pairs)}")
    print(f"Final total loss:     {history[-1]['total']:.6f}")
    print(f"Final recon:          {history[-1]['reconstruction']:.6f}")
    print(f"Final transport:      {history[-1]['transport']:.6f}")
    print(f"Final triplet:        {history[-1]['triplet']:.6f}")
    print(f"Final velocity:       {history[-1]['velocity']:.6f}")
    print(f"Avg recon MSE:        {avg_rec:.6f}")
    print(f"Latent separation:    {separation:.2f}")
    print(f"Train transport res:  {avg_train_res:.6f}")
    print(f"Holdout transport res:{avg_holdout_res:.6f}")
    print(f"Identity baseline:    {avg_identity:.6f}")
    print(f"Transport improvement:{improvement:.1%}")
    if rank_results:
        avg_eff_rank = sum(r["effective_rank"] for r in rank_results.values()) / len(rank_results)
        print(f"Avg effective rank:   {avg_eff_rank:.2f}")
    if comp_results:
        avg_comp = sum(r["composition_error"] for r in comp_results) / len(comp_results)
        print(f"Avg composition err:  {avg_comp:.6f}")
    print("=" * 60)

    results["summary"] = {
        "epochs": args.epochs,
        "rank": rank,
        "norm_mode": norm_mode,
        "num_transform_types": num_tt,
        "final_loss": history[-1],
        "avg_reconstruction_mse": avg_rec,
        "latent_separation_ratio": separation,
        "train_transport_residual": avg_train_res,
        "holdout_transport_residual": avg_holdout_res,
        "identity_baseline_residual": avg_identity,
        "transport_improvement_pct": improvement,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="SDQ v2 training pipeline")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--gauge-dim", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--norm-mode", type=str, default="center_scale",
                        choices=["unit_norm", "center_scale", "standardize"])
    parser.add_argument("--triplet-margin", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="artifacts/sdq_v2_results.json")
    args = parser.parse_args()

    start = time.time()
    results = run_training(args)
    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed
    print(f"\nCompleted in {elapsed:.1f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
