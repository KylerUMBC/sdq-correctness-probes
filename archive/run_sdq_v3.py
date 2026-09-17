#!/usr/bin/env python3
"""SDQ v3 training: semantic-focused improvements.

Key changes over v2 (per improvements.md):
  1. Semi-hard triplet loss with margin=2.0 (prevents trivial satisfaction)
  2. Velocity L2 + cosine + curvature losses (motion-aware semantics)
  3. Proper shuffle control (all-vs-all pairs, not adjacent only)
  4. Retrieval accuracy evaluation
  5. Latent motion similarity evaluation (velocity cosine within vs cross)
  6. Hard-negative resistance test
  7. Per-family balanced diagnostic reporting
  8. Identity baseline for transport
  9. Transport sharing across families (same transform type)
  10. Triple composition test

Usage:
    python run_sdq_v3.py [--epochs 300] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
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
from sdq.losses.triplet import semi_hard_triplet_loss
from sdq.losses.velocity import latent_velocity_loss, latent_curvature_loss, latent_velocity_cosine_loss
from sdq.losses.total import LossWeights, LossBreakdown
from sdq.eval import rank_analysis
from sdq.eval.composition import lowrank_composition_error
from sdq.eval.retrieval import retrieval_accuracy
from sdq.eval.motion_similarity import motion_similarity_eval
from sdq.eval.hard_negatives import hard_negative_test


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

PID_TO_FAMILY = {pid: fam for fam, pids in FAMILIES.items() for pid in pids}


def get_variant(pid: str) -> str:
    """Extract surface variant from prompt_id."""
    for fam, pids in FAMILIES.items():
        if pid in pids:
            return pid[len(fam) + 1:]
    return pid


def get_transform_type(pid_i: str, pid_j: str) -> str:
    return f"{get_variant(pid_i)}->{get_variant(pid_j)}"


def build_transform_vocab(families: dict[str, list[str]]) -> dict[str, int]:
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
class SDQModel:
    """All trainable SDQ components."""
    def __init__(self, encoder, decoder, gauge_encoder, aligner, transport, dynamics):
        self.encoder = encoder
        self.decoder = decoder
        self.gauge_encoder = gauge_encoder
        self.aligner = aligner
        self.transport = transport
        self.dynamics = dynamics
        self._modules = [encoder, decoder, gauge_encoder, aligner, transport, dynamics]

    def parameters(self):
        for m in self._modules:
            yield from m.parameters()

    def train(self):
        for m in self._modules:
            m.train()

    def eval(self):
        for m in self._modules:
            m.eval()

    def to(self, device):
        for m in self._modules:
            m.to(device)
        return self


# ---------------------------------------------------------------------------
# Training step: one (i, j) pair
# ---------------------------------------------------------------------------
def train_pair(
    model: SDQModel,
    h_i: torch.Tensor,
    h_j: torch.Tensor,
    loss_fn: SDQLoss,
    tt_id: int | None = None,
) -> tuple[LossBreakdown, torch.Tensor, torch.Tensor]:
    """Forward pass for one pair. Returns (breakdown, z_i, z_j)."""
    h_j_aligned = model.aligner.align_trajectory(h_i, h_j)

    z_i = model.encoder(h_i)
    z_j = model.encoder(h_j_aligned)

    u_i = model.gauge_encoder(h_i)
    u_j = model.gauge_encoder(h_j_aligned)

    h_i_hat = model.decoder(z_i, u_i)
    h_j_hat = model.decoder(z_j, u_j)

    tr_ij = model.transport(h_i, h_j_aligned, return_operators=False,
                            transform_type_id=tt_id)

    L_rec = (reconstruction_loss(h_i, h_i_hat) +
             reconstruction_loss(h_j_aligned, h_j_hat)) / 2.0
    L_sem = semantic_consistency_loss(z_i, z_j)
    L_trans = transport_consistency_loss(tr_ij)

    # Motion-aware semantic losses
    L_vel = latent_velocity_loss(z_i, z_j)
    L_vel_cos = latent_velocity_cosine_loss(z_i, z_j)
    L_curv = latent_curvature_loss(z_i, z_j)
    # Combine into the velocity slot
    L_velocity = L_vel + L_vel_cos + 0.5 * L_curv

    # Cycle + gauge from low-rank factors
    L_cycle = torch.tensor(0.0, device=h_i.device)
    L_gauge = torch.tensor(0.0, device=h_i.device)
    if tr_ij.U is not None:
        tr_ji = model.transport(h_j_aligned, h_i, return_operators=False,
                                transform_type_id=tt_id)
        if tr_ji.U is not None:
            T_min = min(tr_ij.U.shape[0], tr_ji.U.shape[0])
            L_cycle = lowrank_cycle_loss(
                tr_ij.U[:T_min], tr_ij.V[:T_min],
                tr_ji.U[:T_min], tr_ji.V[:T_min],
            )
        L_gauge = lowrank_gauge_regularization_loss(tr_ij.U, tr_ij.V)

    L_dyn = model.dynamics.dynamics_consistency_loss(z_i)

    breakdown = loss_fn(
        L_rec=L_rec, L_sem=L_sem, L_trans=L_trans,
        L_cycle=L_cycle, L_gauge=L_gauge, L_event=L_dyn,
        L_velocity=L_velocity,
    )
    return breakdown, z_i, z_j


def compute_triplet_loss(
    latents: dict[str, torch.Tensor],
    families: dict[str, list[str]],
    margin: float,
) -> torch.Tensor:
    """Hard-negative triplet loss across all training PIDs."""
    device = next(iter(latents.values())).device
    fam_names = list(families.keys())
    if len(fam_names) < 2:
        return torch.tensor(0.0, device=device)

    total = torch.tensor(0.0, device=device)
    count = 0

    for fam in fam_names:
        pids = [p for p in families[fam] if p in latents]
        neg_pids = [p for f in fam_names if f != fam
                    for p in families[f] if p in latents]
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

                total = total + semi_hard_triplet_loss(
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

    # --- Per-family diagnostics ---
    print("\n--- Per-Family Trajectory Diagnostics ---")
    family_diag: dict[str, dict] = {}
    for fam, pids in available_families.items():
        vel_norms, traj_lens, state_norms, state_vars = [], [], [], []
        for pid in pids:
            s = trajectories[pid].states
            vel = velocity_vectors(s)
            vel_norms.append(vel.norm(dim=-1).mean().item())
            traj_lens.append(s.shape[0])
            state_norms.append(s.norm(dim=-1).mean().item())
            state_vars.append(s.var(dim=0).mean().item())

        diag = {
            "avg_velocity_norm": sum(vel_norms) / len(vel_norms),
            "avg_traj_len": sum(traj_lens) / len(traj_lens),
            "avg_state_norm": sum(state_norms) / len(state_norms),
            "avg_state_var": sum(state_vars) / len(state_vars),
        }
        family_diag[fam] = diag
        print(f"  {fam}: |vel|={diag['avg_velocity_norm']:.4f} "
              f"T={diag['avg_traj_len']:.0f} "
              f"|h|={diag['avg_state_norm']:.4f} "
              f"var={diag['avg_state_var']:.6f}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    transform_vocab = build_transform_vocab(FAMILIES)
    num_tt = len(transform_vocab)
    print(f"\nTransform types: {num_tt}")

    model = SDQModel(
        encoder=TemporalConvEncoder(hidden_dim, args.latent_dim, window_size=3),
        decoder=ResidualDecoder(args.latent_dim, args.gauge_dim, hidden_dim),
        gauge_encoder=TemporalGaugeEncoder(hidden_dim, args.gauge_dim, window_size=3),
        aligner=MonotoneAligner(hidden_dim, align_dim=min(128, hidden_dim)),
        transport=LocalTransportField(
            hidden_dim, rank=args.rank, context_dim=min(128, hidden_dim),
            num_transform_types=num_tt, transform_embed_dim=16,
        ),
        dynamics=LatentODE(args.latent_dim),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}, rank={args.rank}")

    loss_fn = SDQLoss(weights=LossWeights(
        reconstruction=1.0,
        semantic=2.0,
        transport=2.0,
        cycle=1.0,
        gauge=0.1,
        event=0.1,
        triplet=2.0,   # strong triplet pressure
        velocity=1.5,   # strong motion-aware pressure
    ))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ------------------------------------------------------------------
    # 3. Prepare pairs
    # ------------------------------------------------------------------
    train_families_avail = {f: available_families[f]
                            for f in TRAIN_FAMILIES if f in available_families}
    holdout_families_avail = {f: available_families[f]
                              for f in HOLDOUT_FAMILIES if f in available_families}

    train_pairs: list[tuple[str, str, str]] = []
    for fam_name, pids in train_families_avail.items():
        for p_i, p_j in combinations(pids, 2):
            train_pairs.append((fam_name, p_i, p_j))

    holdout_pairs: list[tuple[str, str, str]] = []
    for fam_name, pids in holdout_families_avail.items():
        for p_i, p_j in combinations(pids, 2):
            holdout_pairs.append((fam_name, p_i, p_j))

    train_pids = [pid for fam in TRAIN_FAMILIES if fam in available_families
                  for pid in available_families[fam]]

    print(f"\nTraining pairs: {len(train_pairs)}, holdout: {len(holdout_pairs)}")

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(f"\n=== Training for {args.epochs} epochs ===")
    history: list[dict[str, float]] = []
    triplet_margin = args.triplet_margin

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {
            "total": 0.0, "reconstruction": 0.0, "semantic": 0.0,
            "transport": 0.0, "cycle": 0.0, "gauge": 0.0,
            "dynamics": 0.0, "triplet": 0.0, "velocity": 0.0,
        }
        pair_count = 0

        # Accumulate latents for epoch-level triplet
        epoch_latents: dict[str, torch.Tensor] = {}

        for fam_name, pid_i, pid_j in train_pairs:
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)

            tt = get_transform_type(pid_i, pid_j)
            tt_id = transform_vocab.get(tt)

            optimizer.zero_grad()
            breakdown, z_i, z_j = train_pair(model, h_i, h_j, loss_fn, tt_id=tt_id)

            # Cache latents for triplet (detach not needed — we want gradients)
            epoch_latents[pid_i] = z_i
            epoch_latents[pid_j] = z_j

            breakdown.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bd = breakdown.to_dict()
            for k in epoch_losses:
                if k == "dynamics":
                    epoch_losses[k] += bd.get("event", 0.0)
                elif k == "triplet":
                    pass  # computed separately below
                else:
                    epoch_losses[k] += bd.get(k, 0.0)
            pair_count += 1

        # --- Epoch-level triplet loss (separate backward) ---
        # Re-encode all training trajectories for fresh latents
        model.train()
        fresh_latents: dict[str, torch.Tensor] = {}
        for pid in train_pids:
            h = trajectories[pid].states.to(device)
            fresh_latents[pid] = model.encoder(h)

        optimizer.zero_grad()
        L_trip = compute_triplet_loss(
            fresh_latents, train_families_avail, margin=triplet_margin,
        )
        triplet_weighted = loss_fn.weights.triplet * L_trip
        triplet_weighted.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_losses["triplet"] = L_trip.item()

        scheduler.step()

        for k in epoch_losses:
            if k != "triplet":
                epoch_losses[k] /= max(pair_count, 1)
        epoch_losses["lr"] = scheduler.get_last_lr()[0]
        history.append(epoch_losses)

        if epoch % 25 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d} | "
                f"tot={epoch_losses['total']:.4f} "
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
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    model.eval()
    results: dict = {"training_history": history, "family_diagnostics": family_diag}

    # --- Encode all latents ---
    latent_states: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for pid, traj in trajectories.items():
            h = traj.states.to(device)
            latent_states[pid] = model.encoder(h).cpu()

    # --- 5a. Transport rank ---
    print("\n--- Transport Rank Analysis ---")
    rank_results = {}
    with torch.no_grad():
        for fam_name, pid_i, pid_j in train_pairs[:6]:
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)
            h_j_a = model.aligner.align_trajectory(h_i, h_j)
            tt_id = transform_vocab.get(get_transform_type(pid_i, pid_j))
            tr = model.transport(h_i, h_j_a, return_operators=True, transform_type_id=tt_id)
            G_mean = tr.G_t.mean(dim=0).cpu()
            ra = rank_analysis(G_mean)
            key = f"{pid_i}->{pid_j}"
            rank_results[key] = {
                "effective_rank": ra.effective_rank,
                "nuclear_norm": ra.nuclear_norm,
                "frobenius_deviation": ra.frobenius_deviation,
            }
            print(f"  {key}: eff_rank={ra.effective_rank:.2f} "
                  f"nuc={ra.nuclear_norm:.4f} frob={ra.frobenius_deviation:.4f}")
    results["transport_rank"] = rank_results

    # --- 5b. Latent semantic consistency ---
    print("\n--- Latent Semantic Consistency ---")
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
            for a in available_families[fam_names[i]]:
                for b in available_families[fam_names[j]]:
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

    # --- 5c. Retrieval accuracy ---
    print("\n--- Retrieval Accuracy ---")
    ret = retrieval_accuracy(latent_states, PID_TO_FAMILY)
    print(f"  Accuracy:  {ret.accuracy:.2%} ({ret.num_queries} queries)")
    print(f"  MRR:       {ret.mean_reciprocal_rank:.4f}")
    # Show misclassified
    for q in ret.per_query:
        if not q["correct"]:
            print(f"    MISS: {q['query']} ({q['family']}) -> {q['nearest']} ({q['nearest_family']})")
    results["retrieval"] = {
        "accuracy": ret.accuracy,
        "mrr": ret.mean_reciprocal_rank,
        "num_queries": ret.num_queries,
    }

    # --- 5d. Motion similarity ---
    print("\n--- Latent Motion Similarity ---")
    ms = motion_similarity_eval(latent_states, available_families)
    print(f"  Within velocity cosine:  {ms.within_velocity_cosine:.4f}")
    print(f"  Cross velocity cosine:   {ms.cross_velocity_cosine:.4f}")
    print(f"  Cosine separation:       {ms.velocity_cosine_separation:.2f} (within/cross, higher=better)")
    print(f"  Within velocity L2:      {ms.within_velocity_l2:.6f}")
    print(f"  Cross velocity L2:       {ms.cross_velocity_l2:.6f}")
    print(f"  L2 separation:           {ms.velocity_l2_separation:.2f} (cross/within, higher=better)")
    results["motion_similarity"] = asdict(ms)

    # --- 5e. Per-family reconstruction ---
    print("\n--- Per-Family Reconstruction ---")
    rec_errors: dict[str, float] = {}
    with torch.no_grad():
        for pid, traj in trajectories.items():
            h = traj.states.to(device)
            z = model.encoder(h)
            u = model.gauge_encoder(h)
            h_hat = model.decoder(z, u)
            rec_errors[pid] = reconstruction_loss(h, h_hat).item()

    fam_rec: dict[str, dict] = {}
    for fam, pids in available_families.items():
        errors = [rec_errors[p] for p in pids if p in rec_errors]
        avg_e = sum(errors) / max(len(errors), 1)
        fam_rec[fam] = {"avg_mse": avg_e, "per_pid": {p: rec_errors.get(p, 0) for p in pids}}
        print(f"  {fam}: avg MSE = {avg_e:.6f}")
        for p in pids:
            print(f"    {p}: {rec_errors.get(p, 0):.6f}")
    results["reconstruction"] = fam_rec

    # --- 5f. Per-family transport residuals ---
    print("\n--- Per-Family Transport Residuals ---")
    fam_transport: dict[str, dict] = {}
    with torch.no_grad():
        for fam, pids in available_families.items():
            residuals = []
            for a, b in combinations(pids, 2):
                h_a = trajectories[a].states.to(device)
                h_b = trajectories[b].states.to(device)
                h_b_a = model.aligner.align_trajectory(h_a, h_b)
                tt_id = transform_vocab.get(get_transform_type(a, b))
                tr = model.transport(h_a, h_b_a, transform_type_id=tt_id)
                residuals.append(tr.residual.norm(dim=-1).mean().item())
            avg_r = sum(residuals) / max(len(residuals), 1)
            is_train = fam in TRAIN_FAMILIES
            fam_transport[fam] = {"avg_residual": avg_r, "split": "train" if is_train else "holdout"}
            print(f"  {fam} ({'train' if is_train else 'holdout'}): {avg_r:.6f}")
    results["per_family_transport"] = fam_transport

    # --- 5g. Per-family latent distances ---
    print("\n--- Per-Family Latent Distances ---")
    fam_latent: dict[str, dict] = {}
    for fam, pids in available_families.items():
        within = []
        for a, b in combinations(pids, 2):
            if a in latent_states and b in latent_states:
                T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                within.append(d)
        cross_for_fam = []
        for other_fam in available_families:
            if other_fam == fam:
                continue
            for a in pids[:2]:
                for b in available_families[other_fam][:2]:
                    if a in latent_states and b in latent_states:
                        T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                        d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                        cross_for_fam.append(d)

        avg_w = sum(within) / max(len(within), 1)
        avg_c = sum(cross_for_fam) / max(len(cross_for_fam), 1)
        fam_latent[fam] = {"within": avg_w, "cross": avg_c,
                           "separation": avg_c / max(avg_w, 1e-8)}
        print(f"  {fam}: within={avg_w:.4f} cross={avg_c:.4f} "
              f"sep={avg_c / max(avg_w, 1e-8):.2f}")
    results["per_family_latent"] = fam_latent

    # --- 5h. Hard-negative resistance ---
    print("\n--- Hard-Negative Resistance ---")
    # Hard negatives: same surface variant, different semantic family
    # e.g., cat_therefore vs bird_therefore (same template "therefore", diff semantics)
    positive_pairs = []
    hardneg_pairs = []
    with torch.no_grad():
        # Positives: same family, different variant
        for fam, pids in available_families.items():
            for a, b in combinations(pids, 2):
                if a in latent_states and b in latent_states:
                    positive_pairs.append((latent_states[a], latent_states[b]))

        # Hard negatives: same variant, different family
        for fam_i in available_families:
            for fam_j in available_families:
                if fam_i >= fam_j:
                    continue
                for pid_i in available_families[fam_i]:
                    vi = get_variant(pid_i)
                    for pid_j in available_families[fam_j]:
                        vj = get_variant(pid_j)
                        if vi == vj and pid_i in latent_states and pid_j in latent_states:
                            hardneg_pairs.append((latent_states[pid_i], latent_states[pid_j]))

    if positive_pairs and hardneg_pairs:
        hn = hard_negative_test(positive_pairs, hardneg_pairs)
        print(f"  Positive distance:      {hn.positive_distance:.4f}")
        print(f"  Hard-negative distance: {hn.hard_negative_distance:.4f}")
        print(f"  Separation ratio:       {hn.separation_ratio:.2f}")
        print(f"  Passes:                 {hn.passes}")
        results["hard_negative"] = {
            "positive_distance": hn.positive_distance,
            "hard_negative_distance": hn.hard_negative_distance,
            "separation_ratio": hn.separation_ratio,
            "passes": hn.passes,
        }

    # --- 5i. Cycle consistency ---
    print("\n--- Cycle Consistency ---")
    cycle_errors = []
    with torch.no_grad():
        for fam, pid_i, pid_j in train_pairs[:6]:
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)
            h_j_a = model.aligner.align_trajectory(h_i, h_j)
            tt_id = transform_vocab.get(get_transform_type(pid_i, pid_j))
            tr_ij = model.transport(h_i, h_j_a, return_operators=True, transform_type_id=tt_id)
            tr_ji = model.transport(h_j_a, h_i, return_operators=True, transform_type_id=tt_id)
            T_min = min(tr_ij.G_t.shape[0], tr_ji.G_t.shape[0])
            metrics = cycle_consistency_metrics(
                tr_ij.G_t[:T_min].cpu(), tr_ji.G_t[:T_min].cpu(),
            )
            cycle_errors.append(metrics)
            print(f"  {pid_i}<->{pid_j}: {metrics}")
    results["cycle_consistency"] = cycle_errors

    # --- 5j. Triple composition ---
    print("\n--- Triple Composition ---")
    comp_results = []
    with torch.no_grad():
        for fam_name in TRAIN_FAMILIES:
            if fam_name not in available_families:
                continue
            pids = available_families[fam_name]
            if len(pids) < 3:
                continue
            factors: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
            for a, b in combinations(pids, 2):
                h_a = trajectories[a].states.to(device)
                h_b = trajectories[b].states.to(device)
                h_b_a = model.aligner.align_trajectory(h_a, h_b)
                tt_id = transform_vocab.get(get_transform_type(a, b))
                tr = model.transport(h_a, h_b_a, return_operators=False, transform_type_id=tt_id)
                factors[(a, b)] = (tr.U.cpu(), tr.V.cpu())
                tr_rev = model.transport(h_b_a, h_a, return_operators=False, transform_type_id=tt_id)
                factors[(b, a)] = (tr_rev.U.cpu(), tr_rev.V.cpu())

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
                                "error": err,
                            })

    if comp_results:
        avg_comp = sum(r["error"] for r in comp_results) / len(comp_results)
        print(f"  Avg composition error: {avg_comp:.6f} ({len(comp_results)} triples)")
        for r in comp_results[:6]:
            print(f"    {r['triple']}: {r['error']:.6f}")
    results["composition"] = comp_results

    # --- 5k. Transport sharing (same transform type, different families) ---
    print("\n--- Transport Sharing ---")
    sharing = {}
    with torch.no_grad():
        for fam_train, fam_holdout in [("cat", "bird")]:
            if fam_train not in available_families or fam_holdout not in available_families:
                continue
            train_v = [(pid, get_variant(pid)) for pid in available_families[fam_train]]
            holdout_v = [(pid, get_variant(pid)) for pid in available_families[fam_holdout]]

            for (pt1, v1), (pt2, v2) in combinations(train_v, 2):
                ph1 = next((p for p, v in holdout_v if v == v1), None)
                ph2 = next((p for p, v in holdout_v if v == v2), None)
                if ph1 is None or ph2 is None:
                    continue
                tt_id = transform_vocab.get(get_transform_type(pt1, pt2))

                h_t1 = trajectories[pt1].states.to(device)
                h_t2 = trajectories[pt2].states.to(device)
                h_t2_a = model.aligner.align_trajectory(h_t1, h_t2)
                tr_t = model.transport(h_t1, h_t2_a, transform_type_id=tt_id)

                h_h1 = trajectories[ph1].states.to(device)
                h_h2 = trajectories[ph2].states.to(device)
                h_h2_a = model.aligner.align_trajectory(h_h1, h_h2)
                tr_h = model.transport(h_h1, h_h2_a, transform_type_id=tt_id)

                key = f"{v1}->{v2}"
                sharing[key] = {
                    "train": tr_t.residual.norm(dim=-1).mean().item(),
                    "holdout": tr_h.residual.norm(dim=-1).mean().item(),
                }
                print(f"  {key}: train={sharing[key]['train']:.4f} holdout={sharing[key]['holdout']:.4f}")
    results["transport_sharing"] = sharing

    # --- 5l. Identity baseline ---
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
            identity_res.append((vel_j[:T_min] - vel_i[:T_min]).norm(dim=-1).mean().item())

    avg_identity = sum(identity_res) / max(len(identity_res), 1)
    avg_train_res = fam_transport.get("cat", {}).get("avg_residual", 0) if "cat" in fam_transport else 0
    # Recompute proper train avg
    train_avg_list = [fam_transport[f]["avg_residual"] for f in TRAIN_FAMILIES if f in fam_transport]
    holdout_avg_list = [fam_transport[f]["avg_residual"] for f in HOLDOUT_FAMILIES if f in fam_transport]
    avg_train_res = sum(train_avg_list) / max(len(train_avg_list), 1)
    avg_holdout_res = sum(holdout_avg_list) / max(len(holdout_avg_list), 1)
    improvement = 1 - (avg_train_res / max(avg_identity, 1e-8))
    print(f"  Identity baseline:     {avg_identity:.6f}")
    print(f"  Learned (train):       {avg_train_res:.6f}")
    print(f"  Learned (holdout):     {avg_holdout_res:.6f}")
    print(f"  Improvement over I:    {improvement:.1%}")
    results["identity_baseline"] = {
        "identity": avg_identity,
        "train": avg_train_res,
        "holdout": avg_holdout_res,
        "improvement": improvement,
    }

    # --- 5m. Shuffle control (FIXED: all-vs-all, not adjacent) ---
    print("\n--- Shuffle Control ---")
    all_pids_available = sorted(latent_states.keys())
    rng = random.Random(42)
    n_shuffle_trials = 200
    shuffle_within_dists, shuffle_cross_dists = [], []

    for _ in range(n_shuffle_trials):
        a, b = rng.sample(all_pids_available, 2)
        T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
        d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
        fam_a = PID_TO_FAMILY.get(a, "?")
        fam_b = PID_TO_FAMILY.get(b, "?")
        if fam_a == fam_b:
            shuffle_within_dists.append(d)
        else:
            shuffle_cross_dists.append(d)

    sw = sum(shuffle_within_dists) / max(len(shuffle_within_dists), 1)
    sc = sum(shuffle_cross_dists) / max(len(shuffle_cross_dists), 1)
    print(f"  Shuffle within ({len(shuffle_within_dists)} pairs):  {sw:.4f}")
    print(f"  Shuffle cross  ({len(shuffle_cross_dists)} pairs):   {sc:.4f}")
    print(f"  True within:   {avg_within:.4f}")
    print(f"  True cross:    {avg_cross:.4f}")
    print(f"  Shuffle separation: {sc / max(sw, 1e-8):.2f}")
    results["shuffle_control"] = {
        "shuffle_within": sw, "shuffle_cross": sc,
        "shuffle_n_within": len(shuffle_within_dists),
        "shuffle_n_cross": len(shuffle_cross_dists),
        "true_within": avg_within, "true_cross": avg_cross,
    }

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SDQ v3 SUMMARY")
    print("=" * 60)
    print(f"Epochs:                {args.epochs}")
    print(f"Rank:                  {args.rank}")
    print(f"Normalization:         {norm_mode}")
    print(f"Triplet margin:        {triplet_margin}")
    print(f"Final total loss:      {history[-1]['total']:.6f}")
    print(f"Final triplet:         {history[-1]['triplet']:.6f}")
    print(f"Final velocity:        {history[-1]['velocity']:.6f}")
    print(f"Latent separation:     {separation:.2f}")
    print(f"Retrieval accuracy:    {ret.accuracy:.2%}")
    print(f"Motion cos sep:        {ms.velocity_cosine_separation:.2f}")
    print(f"Train transport res:   {avg_train_res:.6f}")
    print(f"Holdout transport res: {avg_holdout_res:.6f}")
    print(f"Transport vs identity: {improvement:.1%}")
    if rank_results:
        avg_r = sum(r["effective_rank"] for r in rank_results.values()) / len(rank_results)
        print(f"Avg effective rank:    {avg_r:.2f}")
    if comp_results:
        print(f"Avg composition err:   {avg_comp:.6f}")
    if positive_pairs and hardneg_pairs:
        print(f"Hard-neg passes:       {hn.passes}")
        print(f"Hard-neg sep ratio:    {hn.separation_ratio:.2f}")
    print("=" * 60)

    results["summary"] = {
        "epochs": args.epochs,
        "rank": args.rank,
        "norm_mode": norm_mode,
        "triplet_margin": triplet_margin,
        "final_loss": history[-1],
        "latent_separation": separation,
        "retrieval_accuracy": ret.accuracy,
        "retrieval_mrr": ret.mean_reciprocal_rank,
        "motion_cosine_separation": ms.velocity_cosine_separation,
        "motion_l2_separation": ms.velocity_l2_separation,
        "train_transport_residual": avg_train_res,
        "holdout_transport_residual": avg_holdout_res,
        "identity_improvement": improvement,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="SDQ v3 training")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--gauge-dim", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--norm-mode", type=str, default="center_scale",
                        choices=["unit_norm", "center_scale", "standardize"])
    parser.add_argument("--triplet-margin", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="artifacts/sdq_v3_results.json")
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
