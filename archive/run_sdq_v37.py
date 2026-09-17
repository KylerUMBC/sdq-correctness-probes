#!/usr/bin/env python3
"""SDQ v3.7 training: structural improvements for 95+ scorecard.

Changes from v3.6 (structural, not hyperparameter):
  1. MultiScaleConvEncoder replaces TemporalConvEncoder
     Dilated causal conv stack: receptive field 15 tokens (was 3).
     Each z_t now sees h_{t-14:t} — can capture trajectory-level patterns.
  2. Mini-batched VICReg encoding
     Pad-stack-forward-unpack in chunks of 32, replaces sequential for-loop.
     Dramatically faster per epoch while preserving full signal.
  3. Transport gate_init=0.1 (was 0.0)
     Fixes dead-gradient trap in context-dependent correction.

Keeps from v3.6:
  - Transport weight 3.5, composition weight 0.30, composition freq 3
  - VICReg component weights (25/10/1/5/10), margins as CLI args
  - Contrastive freq 2 (matching v3.5)
  - All sdq/ loss and eval modules untouched

Targets:
  Scorecard:  >= 95/100
  Latent sep: ratio 5.0+ (was 2.72)
  Motion cos: ratio 5.0+ (was 1.55)
  Transport:  vs-identity > 65%

Usage:
    python run_sdq_v37.py [--epochs 300] [--device cuda]
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
from sdq.alignment import MonotoneAligner
from sdq.transport import (
    LocalTransportField, TransportResult, cycle_consistency_metrics,
    TransportPrototypes, prototype_regularization_loss,
)
from sdq.latent import (
    MultiScaleConvEncoder,
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
from sdq.losses.triplet import semi_hard_triplet_loss  # still used by downstream
from sdq.losses.velocity import (
    latent_velocity_loss, latent_curvature_loss, latent_velocity_cosine_loss,
)
from sdq.losses.composition import composition_loss
from sdq.losses.total import LossWeights, LossBreakdown
from sdq.eval import rank_analysis
from sdq.eval.composition import lowrank_composition_error
from sdq.eval.retrieval import retrieval_accuracy
from sdq.eval.motion_similarity import motion_similarity_eval
from sdq.eval.hard_negatives import hard_negative_test
from sdq.eval.bird_debug import family_debug
from sdq.eval.semantic_scorecard import compute_scorecard


# ---------------------------------------------------------------------------
# Semantic equivalence groups
# ---------------------------------------------------------------------------
# Benchmark metadata — populated by run_training()
_PID_META: dict[str, dict] = {}


def get_variant(pid: str) -> str:
    """Return surface template ID for this example."""
    meta = _PID_META.get(pid)
    if meta:
        return meta["surface_template_id"]
    # Fallback for legacy PIDs: strip family prefix
    parts = pid.split("_", 1)
    return parts[1] if len(parts) > 1 else pid


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
# Training step: one (i, j) pair — v3 style with soft prototype transport
# ---------------------------------------------------------------------------
def train_pair(
    model: SDQModel,
    h_i: torch.Tensor,
    h_j: torch.Tensor,
    loss_fn: SDQLoss,
    tt_id: int | None = None,
) -> tuple[LossBreakdown, torch.Tensor, torch.Tensor, dict]:
    """Forward pass for one pair. Returns (breakdown, z_i, z_j, transport_factors)."""
    h_j_aligned = model.aligner.align_trajectory(h_i, h_j)

    z_i = model.encoder(h_i)
    z_j = model.encoder(h_j_aligned)

    u_i = model.gauge_encoder(h_i)
    u_j = model.gauge_encoder(h_j_aligned)

    # Detach z from reconstruction so L_rec only trains gauge + decoder,
    # not the semantic encoder.  This prevents reconstruction from
    # collapsing same-family z's apart (each surface form is unique in h).
    h_i_hat = model.decoder(z_i.detach(), u_i)
    h_j_hat = model.decoder(z_j.detach(), u_j)

    # Transport via prototype model
    tr_ij = model.transport(
        h_i, h_j_aligned,
        transform_type_id=tt_id if tt_id is not None else 0,
    )

    L_rec = (reconstruction_loss(h_i, h_i_hat) +
             reconstruction_loss(h_j_aligned, h_j_hat)) / 2.0
    L_sem = semantic_consistency_loss(z_i, z_j)
    L_trans = tr_ij.residual.pow(2).sum(dim=-1).mean()

    # Motion-aware semantic losses (v3 core)
    L_vel = latent_velocity_loss(z_i, z_j)
    L_vel_cos = latent_velocity_cosine_loss(z_i, z_j)
    L_curv = latent_curvature_loss(z_i, z_j)
    L_velocity = L_vel + L_vel_cos + 0.5 * L_curv

    # Prototype regularization — soft penalty on ΔG (weaker than v4)
    L_proto_reg = prototype_regularization_loss(tr_ij)

    # Cycle + gauge from low-rank factors
    L_cycle = torch.tensor(0.0, device=h_i.device)
    L_gauge = torch.tensor(0.0, device=h_i.device)
    if tr_ij.U is not None:
        tr_ji = model.transport(
            h_j_aligned, h_i,
            transform_type_id=tt_id if tt_id is not None else 0,
        )
        if tr_ji.U is not None:
            T_min = min(tr_ij.U.shape[0], tr_ji.U.shape[0])
            L_cycle = lowrank_cycle_loss(
                tr_ij.U[:T_min], tr_ij.V[:T_min],
                tr_ji.U[:T_min], tr_ji.V[:T_min],
            )
        # Gauge reg + soft prototype penalty (combined into gauge slot)
        L_gauge = lowrank_gauge_regularization_loss(tr_ij.U, tr_ij.V) + 0.3 * L_proto_reg

    L_dyn = model.dynamics.dynamics_consistency_loss(z_i)

    breakdown = loss_fn(
        L_rec=L_rec, L_sem=L_sem, L_trans=L_trans,
        L_cycle=L_cycle, L_gauge=L_gauge, L_event=L_dyn,
        L_velocity=L_velocity,
    )

    # Collect factors for composition loss
    factors = {}
    if tr_ij.U is not None:
        factors["U"] = tr_ij.U
        factors["V"] = tr_ij.V

    return breakdown, z_i, z_j, factors


# ---------------------------------------------------------------------------
# Semantic contrastive loss — VICReg-style (variance + covariance + repulsion)
# ---------------------------------------------------------------------------
def compute_vicreg_loss(
    latents: dict[str, torch.Tensor],
    families: dict[str, list[str]],
    centroid_margin: float = 8.0,
    hn_margin: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """VICReg contrastive loss on raw (unnormalized) mean embeddings.

    Five components:
      - Invariance: MSE within-family → pulls same-family members together
      - Variance: per-dim std ≥ 1.0 across batch → prevents collapse
      - Covariance: decorrelates dimensions → uses full latent capacity
      - Centroid repulsion: L2 hinge → pushes family centroids apart
      - Hard-negative repulsion: extra push for same-variant cross-family pairs
    """
    device = next(iter(latents.values())).device
    fam_names = list(families.keys())
    if len(fam_names) < 2:
        return torch.tensor(0.0, device=device), {}

    all_pids: list[str] = []
    pid_to_fam: dict[str, str] = {}
    for fam, pids in families.items():
        for p in pids:
            if p in latents:
                all_pids.append(p)
                pid_to_fam[p] = fam

    if len(all_pids) < 4:
        return torch.tensor(0.0, device=device), {}

    # Position embeddings (time-averaged latent)
    emb_list = []
    for p in all_pids:
        z = latents[p]  # [T, D_z]
        emb_list.append(z.mean(dim=0))  # [D_z]
    embs = torch.stack(emb_list)  # [N, D_z]
    N, D = embs.shape

    # --- 1. Invariance: pull same-family members toward their centroid ---
    inv_loss = torch.tensor(0.0, device=device)
    inv_count = 0
    avg_within_spread = 0.0
    for fam in families:
        idxs = [i for i, p in enumerate(all_pids) if pid_to_fam[p] == fam]
        if len(idxs) >= 2:
            fam_embs = embs[idxs]  # [K_f, D]
            centroid = fam_embs.mean(dim=0)  # [D]
            inv_loss = inv_loss + (fam_embs - centroid).pow(2).mean()
            avg_within_spread += (fam_embs - centroid).pow(2).sum(dim=-1).mean().item()
            inv_count += 1
    if inv_count > 0:
        inv_loss = inv_loss / inv_count
        avg_within_spread /= inv_count

    # --- 2. Variance: force per-dimension std >= 1.0 across batch ---
    embed_std = embs.std(dim=0)  # [D]
    var_loss = torch.clamp(1.0 - embed_std, min=0.0).mean()

    # --- 3. Covariance: decorrelate dimensions ---
    centered = embs - embs.mean(dim=0)
    cov = (centered.T @ centered) / max(N - 1, 1)  # [D, D]
    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=device)
    cov_loss = cov[off_diag_mask].pow(2).mean()

    # --- 4. Centroid repulsion: push family centroids apart ---
    centroids: list[torch.Tensor] = []
    centroid_fams: list[str] = []
    for fam, pids in families.items():
        fam_indices = [i for i, p in enumerate(all_pids) if pid_to_fam[p] == fam]
        if fam_indices:
            centroids.append(embs[fam_indices].mean(dim=0))
            centroid_fams.append(fam)

    repulsion_loss = torch.tensor(0.0, device=device)
    avg_centroid_dist = 0.0
    if len(centroids) >= 2:
        ctensor = torch.stack(centroids)  # [K, D]
        K = ctensor.shape[0]
        dists = torch.cdist(ctensor.unsqueeze(0), ctensor.unsqueeze(0)).squeeze(0)
        triu_mask = torch.triu(torch.ones(K, K, dtype=torch.bool, device=device), diagonal=1)
        pairwise_dists = dists[triu_mask]
        repulsion_loss = torch.clamp(centroid_margin - pairwise_dists, min=0.0).mean()
        avg_centroid_dist = pairwise_dists.mean().item()

    # --- 5. Hard-negative repulsion: push same-variant cross-family pairs apart ---
    hn_loss = torch.tensor(0.0, device=device)
    hn_count = 0
    avg_hn_dist = 0.0
    pid_variants = [get_variant(p) for p in all_pids]
    # Group by variant for efficient pairing
    from collections import defaultdict
    variant_groups: dict[str, list[int]] = defaultdict(list)
    for idx, v in enumerate(pid_variants):
        variant_groups[v].append(idx)
    for v, idxs in variant_groups.items():
        if len(idxs) < 2:
            continue
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                i_idx, j_idx = idxs[ii], idxs[jj]
                if pid_to_fam[all_pids[i_idx]] != pid_to_fam[all_pids[j_idx]]:
                    # Same variant, different family → hard negative
                    d = (embs[i_idx] - embs[j_idx]).pow(2).sum().sqrt()
                    hn_loss = hn_loss + torch.clamp(hn_margin - d, min=0.0)
                    avg_hn_dist += d.item()
                    hn_count += 1
    if hn_count > 0:
        hn_loss = hn_loss / hn_count
        avg_hn_dist /= hn_count

    # Combined loss
    loss = (25.0 * inv_loss + 10.0 * var_loss + 1.0 * cov_loss
            + 5.0 * repulsion_loss + 10.0 * hn_loss)

    diag = {
        "inv_loss": inv_loss.item(),
        "var_loss": var_loss.item(),
        "cov_loss": cov_loss.item(),
        "repulsion_loss": repulsion_loss.item() if isinstance(repulsion_loss, torch.Tensor) else 0.0,
        "hn_loss": hn_loss.item() if isinstance(hn_loss, torch.Tensor) else 0.0,
        "embed_std": embed_std.mean().item(),
        "centroid_dist": avg_centroid_dist,
        "hn_dist": avg_hn_dist,
        "within_spread": avg_within_spread,
        "num_families": len(centroids),
        "num_embeddings": N,
        "hn_pairs": hn_count,
    }
    return loss, diag


# ---------------------------------------------------------------------------
# Composition loss (epoch-level, light regularizer)
# ---------------------------------------------------------------------------
def compute_composition_loss_epoch(
    model: SDQModel,
    trajectories: dict[str, Trajectory],
    families: dict[str, list[str]],
    transform_vocab: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Compute composition loss over all triples in the given families."""
    total = torch.tensor(0.0, device=device)
    count = 0

    for fam_name, pids in families.items():
        if len(pids) < 3:
            continue

        factors: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for a, b in combinations(pids, 2):
            if a not in trajectories or b not in trajectories:
                continue
            h_a = trajectories[a].states.to(device)
            h_b = trajectories[b].states.to(device)
            h_b_a = model.aligner.align_trajectory(h_a, h_b)
            tt_id = transform_vocab.get(get_transform_type(a, b), 0)
            tr = model.transport(h_a, h_b_a, transform_type_id=tt_id)
            if tr.U is not None:
                factors[(a, b)] = (tr.U, tr.V)
            tr_rev = model.transport(h_b_a, h_a, transform_type_id=tt_id)
            if tr_rev.U is not None:
                factors[(b, a)] = (tr_rev.U, tr_rev.V)

        for a in pids:
            for b in pids:
                if b == a:
                    continue
                for c in pids:
                    if c == a or c == b:
                        continue
                    if (a, b) in factors and (b, c) in factors and (a, c) in factors:
                        U_ab, V_ab = factors[(a, b)]
                        U_bc, V_bc = factors[(b, c)]
                        U_ac, V_ac = factors[(a, c)]
                        total = total + composition_loss(
                            U_ab, V_ab, U_bc, V_bc, U_ac, V_ac,
                        )
                        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total / count


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
    print("\n=== Loading benchmark data ===")
    benchmark_path = Path("data/prompts/benchmark_v1.json")
    split_path = Path("data/prompts/splits/split_semantic_family.json")
    with open(benchmark_path, encoding="utf-8") as f:
        benchmark_data = json.load(f)
    with open(split_path, encoding="utf-8") as f:
        split_data = json.load(f)

    # Populate global metadata lookup for get_variant()
    global _PID_META
    examples_list = benchmark_data["examples"]
    _PID_META = {e["example_id"]: e for e in examples_list}

    # Build semantic groups: semantic_task_id -> [example_ids]
    from collections import defaultdict, Counter
    _sg: dict[str, list[str]] = defaultdict(list)
    for ex in examples_list:
        _sg[ex["semantic_task_id"]].append(ex["example_id"])
    semantic_groups: dict[str, list[str]] = dict(_sg)

    train_id_set = set(split_data["train_ids"])
    holdout_id_set = set(split_data["test_ids"])
    print(f"Benchmark: {len(examples_list)} examples, "
          f"{len(semantic_groups)} semantic groups")
    print(f"Split: {len(train_id_set)} train, {len(holdout_id_set)} holdout")

    # Load all benchmark runs
    runs = collect_runs("data/runs", prefix="bench_", device="cpu")
    print(f"Loaded {len(runs)} benchmark runs")

    # Build available families (semantic groups with loaded runs)
    available_families: dict[str, list[str]] = {}
    PID_TO_FAMILY: dict[str, str] = {}
    for stid, pids in semantic_groups.items():
        found = [pid for pid in pids if pid in runs]
        if found:
            available_families[stid] = found
            for pid in found:
                PID_TO_FAMILY[pid] = stid

    # Determine train/holdout groups
    TRAIN_FAMILIES: list[str] = []
    HOLDOUT_FAMILIES: list[str] = []
    for stid, pids in available_families.items():
        if any(pid in train_id_set for pid in pids):
            TRAIN_FAMILIES.append(stid)
        elif any(pid in holdout_id_set for pid in pids):
            HOLDOUT_FAMILIES.append(stid)

    # Task family summary
    tf_counts = Counter(_PID_META[pid]["task_family"]
                        for pids in available_families.values()
                        for pid in pids)
    print(f"Available: {len(available_families)} groups "
          f"({len(TRAIN_FAMILIES)} train, {len(HOLDOUT_FAMILIES)} holdout)")
    for tf, count in sorted(tf_counts.items()):
        print(f"  {tf}: {count} runs")

    norm_mode = args.norm_mode
    print(f"\nNormalization mode: {norm_mode}")
    trajectories: dict[str, Trajectory] = {}
    for pid, run in runs.items():
        lr = tuple(args.layer_range) if args.layer_range else None
        traj = extract_trajectory(run, layer=args.layer, layer_range=lr, to_float=True)
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
    # 2. Build model — prototype transport (soft)
    # ------------------------------------------------------------------
    transform_vocab = build_transform_vocab(available_families)
    num_tt = len(transform_vocab)
    print(f"\nTransform types: {num_tt}")

    model = SDQModel(
        encoder=MultiScaleConvEncoder(hidden_dim, args.latent_dim),
        decoder=ResidualDecoder(args.latent_dim, args.gauge_dim, hidden_dim),
        gauge_encoder=TemporalGaugeEncoder(hidden_dim, args.gauge_dim, window_size=3),
        aligner=MonotoneAligner(hidden_dim, align_dim=min(128, hidden_dim)),
        transport=TransportPrototypes(
            hidden_dim, rank=args.rank, context_dim=min(128, hidden_dim),
            num_transform_types=num_tt, transform_embed_dim=16,
            gate_init=args.gate_init,
        ),
        dynamics=LatentODE(args.latent_dim),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}, rank={args.rank}")

    # v3 semantic weights as core, + v3.7 structural improvements
    loss_fn = SDQLoss(weights=LossWeights(
        reconstruction=1.0,
        semantic=3.0,
        transport=args.transport_weight,  # v3.7: 3.5 (kept from v3.6)
        cycle=1.0,
        gauge=0.1,       # includes soft prototype regularization
        event=0.1,
        triplet=5.0,     # v3 core: strong contrastive (VICReg)
        velocity=3.0,    # v3 core: motion-aware
        composition=args.composition_weight,  # v3.7: 0.30 (kept from v3.6)
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

    all_train_pairs: list[tuple[str, str, str]] = []
    for fam_name, pids in train_families_avail.items():
        for p_i, p_j in combinations(pids, 2):
            all_train_pairs.append((fam_name, p_i, p_j))

    holdout_pairs: list[tuple[str, str, str]] = []
    for fam_name, pids in holdout_families_avail.items():
        for p_i, p_j in combinations(pids, 2):
            holdout_pairs.append((fam_name, p_i, p_j))

    train_pids = [pid for fam in TRAIN_FAMILIES if fam in available_families
                  for pid in available_families[fam]]

    # Sample pairs per epoch for tractability with large benchmarks
    PAIRS_PER_EPOCH = min(100, len(all_train_pairs))
    print(f"\nTotal training pairs: {len(all_train_pairs)}, "
          f"sampling {PAIRS_PER_EPOCH}/epoch, holdout: {len(holdout_pairs)}")
    print(f"Composition weight: {args.composition_weight}, "
          f"composition freq: every {args.composition_freq} epochs")

    # ------------------------------------------------------------------
    # 4. Training loop — VICReg every CONTRASTIVE_FREQ pairs (all families)
    # ------------------------------------------------------------------
    CONTRASTIVE_FREQ = args.contrastive_freq

    best_loss = float("inf")
    patience_counter = 0

    print(f"\n=== Training for {args.epochs} epochs ===")
    print(f"Contrastive frequency: every {CONTRASTIVE_FREQ} pairs (all families)")
    print(f"Early stopping patience: {args.patience} epochs")
    print(f"Transport weight: {args.transport_weight} (v3.5 was 2.0)")
    history: list[dict[str, float]] = []
    triplet_margin = args.triplet_margin

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {
            "total": 0.0, "reconstruction": 0.0, "semantic": 0.0,
            "transport": 0.0, "cycle": 0.0, "gauge": 0.0,
            "dynamics": 0.0, "triplet": 0.0, "velocity": 0.0,
            "composition": 0.0,
        }
        pair_count = 0
        trip_count = 0
        trip_loss_accum = 0.0
        trip_diag: dict[str, float] = {}

        # --- Pair-level losses with interleaved VICReg ---
        train_pairs = random.sample(all_train_pairs, PAIRS_PER_EPOCH)
        for step_idx, (fam_name, pid_i, pid_j) in enumerate(train_pairs):
            if step_idx % 25 == 0:
                print(f"    step {step_idx}/{PAIRS_PER_EPOCH}", end="\r", flush=True)
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)

            tt = get_transform_type(pid_i, pid_j)
            tt_id = transform_vocab.get(tt, 0)

            optimizer.zero_grad()
            breakdown, z_i, z_j, factors = train_pair(
                model, h_i, h_j, loss_fn, tt_id=tt_id,
            )

            pair_loss = breakdown.total

            # Interleave VICReg every CONTRASTIVE_FREQ steps — all families
            if (step_idx + 1) % CONTRASTIVE_FREQ == 0:
                trip_families = train_families_avail
                trip_pids = [
                    pid for pids in trip_families.values() for pid in pids
                ]

                # Batched encoding in mini-batches to fit in GPU memory
                ENCODE_BATCH = 32
                fresh_latents: dict[str, torch.Tensor] = {}
                for chunk_start in range(0, len(trip_pids), ENCODE_BATCH):
                    chunk_pids = trip_pids[chunk_start:chunk_start + ENCODE_BATCH]
                    ch_list = []
                    ch_lengths = []
                    for pid in chunk_pids:
                        h = trajectories[pid].states.to(device)
                        ch_list.append(h)
                        ch_lengths.append(h.shape[0])
                    ch_T = max(ch_lengths)
                    ch_padded = torch.zeros(
                        len(ch_list), ch_T, ch_list[0].shape[1], device=device,
                    )
                    for ci, h in enumerate(ch_list):
                        ch_padded[ci, :ch_lengths[ci]] = h
                    ch_z = model.encoder(ch_padded)
                    for ci, pid in enumerate(chunk_pids):
                        fresh_latents[pid] = ch_z[ci, :ch_lengths[ci]]

                L_trip, trip_diag = compute_vicreg_loss(
                    fresh_latents, trip_families,
                    centroid_margin=args.vicreg_centroid_margin,
                    hn_margin=args.vicreg_hn_margin,
                )
                pair_loss = pair_loss + loss_fn.weights.triplet * L_trip
                trip_loss_accum += L_trip.item()
                trip_count += 1

            pair_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bd = breakdown.to_dict()
            for k in epoch_losses:
                if k == "dynamics":
                    epoch_losses[k] += bd.get("event", 0.0)
                elif k in ("triplet", "composition"):
                    pass  # tracked separately
                else:
                    epoch_losses[k] += bd.get(k, 0.0)
            pair_count += 1

        # --- Epoch-level: composition regularizer (every composition_freq epochs) ---
        L_comp = torch.tensor(0.0, device=device)
        if args.composition_weight > 0 and epoch % args.composition_freq == 0:
            n_comp = min(10, len(train_families_avail))
            comp_fams = {f: train_families_avail[f]
                         for f in random.sample(list(train_families_avail.keys()), n_comp)}
            optimizer.zero_grad()
            L_comp = compute_composition_loss_epoch(
                model, trajectories, comp_fams,
                transform_vocab, device,
            )
            (loss_fn.weights.composition * L_comp).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_losses["triplet"] = trip_loss_accum / max(trip_count, 1)
        epoch_losses["composition"] = L_comp.item()

        scheduler.step()

        for k in epoch_losses:
            if k not in ("triplet", "composition"):
                epoch_losses[k] /= max(pair_count, 1)
        epoch_losses["lr"] = scheduler.get_last_lr()[0]
        history.append(epoch_losses)

        # Early stopping — track total loss improvement
        current_loss = epoch_losses["total"]
        if current_loss < best_loss - 1e-4:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs, best={best_loss:.4f})")
                break

        if epoch % 25 == 0 or epoch == 1:
            trip_info = ""
            if trip_diag:
                trip_info = (
                    f" std={trip_diag.get('embed_std', 0):.4f}"
                    f" cdist={trip_diag.get('centroid_dist', 0):.3f}"
                    f" hnd={trip_diag.get('hn_dist', 0):.3f}"
                    f" inv={trip_diag.get('inv_loss', 0):.3f}"
                    f" var={trip_diag.get('var_loss', 0):.3f}"
                    f" hn={trip_diag.get('hn_loss', 0):.3f}"
                )
            print(
                f"  Epoch {epoch:4d} | "
                f"tot={epoch_losses['total']:.4f} "
                f"rec={epoch_losses['reconstruction']:.4f} "
                f"sem={epoch_losses['semantic']:.4f} "
                f"trans={epoch_losses['transport']:.4f} "
                f"cyc={epoch_losses['cycle']:.4f} "
                f"trip={epoch_losses['triplet']:.4f} "
                f"vel={epoch_losses['velocity']:.4f} "
                f"comp={epoch_losses['composition']:.4f} "
                f"lr={epoch_losses['lr']:.6f}"
                f"{trip_info}"
            )

    # ------------------------------------------------------------------
    # 5. Evaluation — v3 full suite + v4 diagnostics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    model.eval()
    results: dict = {"training_history": history, "family_diagnostics": family_diag}

    # --- Encode all latents (mini-batched) ---
    EVAL_BATCH = 64
    latent_states: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        all_eval_pids = list(trajectories.keys())
        for chunk_start in range(0, len(all_eval_pids), EVAL_BATCH):
            chunk_pids = all_eval_pids[chunk_start:chunk_start + EVAL_BATCH]
            ch_list = []
            ch_lengths = []
            for pid in chunk_pids:
                h = trajectories[pid].states.to(device)
                ch_list.append(h)
                ch_lengths.append(h.shape[0])
            ch_T = max(ch_lengths)
            ch_padded = torch.zeros(
                len(ch_list), ch_T, ch_list[0].shape[1], device=device,
            )
            for ci, h in enumerate(ch_list):
                ch_padded[ci, :ch_lengths[ci]] = h
            ch_z = model.encoder(ch_padded)
            for ci, pid in enumerate(chunk_pids):
                latent_states[pid] = ch_z[ci, :ch_lengths[ci]].cpu()

    # --- 5a. Transport rank ---
    print("\n--- Transport Rank Analysis ---")
    rank_results = {}
    with torch.no_grad():
        for fam_name, pid_i, pid_j in train_pairs[:6]:
            h_i = trajectories[pid_i].states.to(device)
            h_j = trajectories[pid_j].states.to(device)
            h_j_a = model.aligner.align_trajectory(h_i, h_j)
            tt_id = transform_vocab.get(get_transform_type(pid_i, pid_j), 0)
            tr = model.transport(h_i, h_j_a, transform_type_id=tt_id, return_operators=True)
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
    # Subsample cross-group pairs for speed (O(groups^2) otherwise)
    cross_sample_fams = fam_names[:50]
    for i in range(len(cross_sample_fams)):
        for j in range(i + 1, min(i + 11, len(cross_sample_fams))):
            for a in available_families[cross_sample_fams[i]][:2]:
                for b in available_families[cross_sample_fams[j]][:2]:
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
    for q in ret.per_query:
        if not q["correct"]:
            print(f"    MISS: {q['query']} ({q['family']}) -> {q['nearest']} ({q['nearest_family']})")
    results["retrieval"] = {
        "accuracy": ret.accuracy, "mrr": ret.mean_reciprocal_rank,
        "num_queries": ret.num_queries,
    }

    # --- 5d. Motion similarity ---
    print("\n--- Latent Motion Similarity ---")
    ms = motion_similarity_eval(latent_states, available_families)
    print(f"  Within velocity cosine:  {ms.within_velocity_cosine:.4f}")
    print(f"  Cross velocity cosine:   {ms.cross_velocity_cosine:.4f}")
    print(f"  Cosine separation:       {ms.velocity_cosine_separation:.2f}")
    print(f"  Within velocity L2:      {ms.within_velocity_l2:.6f}")
    print(f"  Cross velocity L2:       {ms.cross_velocity_l2:.6f}")
    print(f"  L2 separation:           {ms.velocity_l2_separation:.2f}")
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
    avg_rec = sum(v["avg_mse"] for v in fam_rec.values()) / max(len(fam_rec), 1)
    print(f"  {len(fam_rec)} groups, overall avg MSE = {avg_rec:.6f}")
    results["reconstruction"] = fam_rec

    # --- 5f. Per-family transport residuals ---
    print("\n--- Per-Family Transport Residuals ---")
    # Subsample groups for transport residuals (expensive: O(groups * pids^2))
    eval_fams = dict(list(available_families.items())[:40])
    fam_transport: dict[str, dict] = {}
    transport_residuals_map: dict[tuple[str, str], float] = {}
    with torch.no_grad():
        for fam, pids in eval_fams.items():
            residuals = []
            for a, b in combinations(pids, 2):
                h_a = trajectories[a].states.to(device)
                h_b = trajectories[b].states.to(device)
                h_b_a = model.aligner.align_trajectory(h_a, h_b)
                tt_id = transform_vocab.get(get_transform_type(a, b), 0)
                tr = model.transport(h_a, h_b_a, transform_type_id=tt_id)
                r = tr.residual.norm(dim=-1).mean().item()
                residuals.append(r)
                transport_residuals_map[(a, b)] = r
            avg_r = sum(residuals) / max(len(residuals), 1)
            is_train = fam in TRAIN_FAMILIES
            fam_transport[fam] = {"avg_residual": avg_r, "split": "train" if is_train else "holdout"}
    train_avg_res = [v["avg_residual"] for v in fam_transport.values() if v["split"] == "train"]
    holdout_avg_res = [v["avg_residual"] for v in fam_transport.values() if v["split"] == "holdout"]
    print(f"  Train ({len(train_avg_res)} groups):   avg={sum(train_avg_res)/max(len(train_avg_res),1):.6f}")
    print(f"  Holdout ({len(holdout_avg_res)} groups): avg={sum(holdout_avg_res)/max(len(holdout_avg_res),1):.6f}")
    results["per_family_transport"] = fam_transport

    # --- 5g. Per-family latent distances (subsample for speed) ---
    print("\n--- Per-Family Latent Distances ---")
    eval_fams_latent = dict(list(available_families.items())[:50])
    fam_latent: dict[str, dict] = {}
    for fam, pids in eval_fams_latent.items():
        within = []
        for a, b in combinations(pids, 2):
            if a in latent_states and b in latent_states:
                T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                within.append(d)
        cross_for_fam = []
        other_fams = [f for f in eval_fams_latent if f != fam]
        for other_fam in random.sample(other_fams, min(10, len(other_fams))):
            for a in pids[:2]:
                for b in eval_fams_latent[other_fam][:2]:
                    if a in latent_states and b in latent_states:
                        T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                        d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                        cross_for_fam.append(d)

        avg_w = sum(within) / max(len(within), 1)
        avg_c = sum(cross_for_fam) / max(len(cross_for_fam), 1)
        fam_latent[fam] = {"within": avg_w, "cross": avg_c,
                           "separation": avg_c / max(avg_w, 1e-8)}
    all_w = [v["within"] for v in fam_latent.values()]
    all_c = [v["cross"] for v in fam_latent.values()]
    print(f"  {len(fam_latent)} groups | within={sum(all_w)/len(all_w):.4f} "
          f"cross={sum(all_c)/len(all_c):.4f} "
          f"sep={sum(all_c)/len(all_c) / max(sum(all_w)/len(all_w), 1e-8):.2f}")
    results["per_family_latent"] = fam_latent

    # --- 5h. Hard-negative resistance (subsampled for speed) ---
    print("\n--- Hard-Negative Resistance ---")
    positive_pairs = []
    hardneg_pairs = []
    hn = None
    hn_fams = list(available_families.keys())[:40]
    with torch.no_grad():
        for fam in hn_fams:
            pids = available_families[fam]
            for a, b in combinations(pids, 2):
                if a in latent_states and b in latent_states:
                    positive_pairs.append((latent_states[a], latent_states[b]))
        for fi, fam_i in enumerate(hn_fams):
            for fam_j in hn_fams[fi + 1:]:
                for pid_i in available_families[fam_i][:3]:
                    vi = get_variant(pid_i)
                    for pid_j in available_families[fam_j][:3]:
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
            tt_id = transform_vocab.get(get_transform_type(pid_i, pid_j), 0)
            tr_ij = model.transport(h_i, h_j_a, transform_type_id=tt_id, return_operators=True)
            tr_ji = model.transport(h_j_a, h_i, transform_type_id=tt_id, return_operators=True)
            T_min = min(tr_ij.G_t.shape[0], tr_ji.G_t.shape[0])
            metrics = cycle_consistency_metrics(
                tr_ij.G_t[:T_min].cpu(), tr_ji.G_t[:T_min].cpu(),
            )
            cycle_errors.append(metrics)
            print(f"  {pid_i}<->{pid_j}: {metrics}")
    results["cycle_consistency"] = cycle_errors

    # --- 5j. Triple composition ---
    print("\n--- Triple Composition ---")
    comp_eval_fams = TRAIN_FAMILIES if len(TRAIN_FAMILIES) <= 20 else random.sample(TRAIN_FAMILIES, 20)
    comp_results = []
    with torch.no_grad():
        for fam_name in comp_eval_fams:
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
                tt_id = transform_vocab.get(get_transform_type(a, b), 0)
                tr = model.transport(h_a, h_b_a, transform_type_id=tt_id)
                factors[(a, b)] = (tr.U.cpu(), tr.V.cpu())
                tr_rev = model.transport(h_b_a, h_a, transform_type_id=tt_id)
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

    avg_comp = 0.0
    if comp_results:
        avg_comp = sum(r["error"] for r in comp_results) / len(comp_results)
        print(f"  Avg composition error: {avg_comp:.6f} ({len(comp_results)} triples)")
        for r in comp_results[:6]:
            print(f"    {r['triple']}: {r['error']:.6f}")
    results["composition"] = comp_results

    # --- 5k. Transport sharing ---
    print("\n--- Transport Sharing ---")
    sharing = {}
    with torch.no_grad():
        # Find train/holdout pairs sharing surface templates
        sharing_pairs = []
        for ft in TRAIN_FAMILIES[:20]:
            ft_variants = {get_variant(p) for p in available_families.get(ft, [])}
            for fh in HOLDOUT_FAMILIES[:20]:
                fh_variants = {get_variant(p) for p in available_families.get(fh, [])}
                if len(ft_variants & fh_variants) >= 2:
                    sharing_pairs.append((ft, fh))
                    if len(sharing_pairs) >= 6:
                        break
            if len(sharing_pairs) >= 6:
                break

        for fam_train, fam_holdout in sharing_pairs:
            train_v = [(pid, get_variant(pid)) for pid in available_families[fam_train]]
            holdout_v = [(pid, get_variant(pid)) for pid in available_families[fam_holdout]]
            for (pt1, v1), (pt2, v2) in combinations(train_v, 2):
                ph1 = next((p for p, v in holdout_v if v == v1), None)
                ph2 = next((p for p, v in holdout_v if v == v2), None)
                if ph1 is None or ph2 is None:
                    continue
                tt_id = transform_vocab.get(get_transform_type(pt1, pt2), 0)
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
        "identity": avg_identity, "train": avg_train_res,
        "holdout": avg_holdout_res, "improvement": improvement,
    }

    # --- 5m. Shuffle control ---
    print("\n--- Shuffle Control ---")
    all_pids_available = sorted(latent_states.keys())
    rng = random.Random(42)
    shuffle_within_dists, shuffle_cross_dists = [], []
    for _ in range(200):
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
    shuffle_sep = sc / max(sw, 1e-8)
    print(f"  Shuffle within ({len(shuffle_within_dists)} pairs):  {sw:.4f}")
    print(f"  Shuffle cross  ({len(shuffle_cross_dists)} pairs):   {sc:.4f}")
    print(f"  True within:   {avg_within:.4f}")
    print(f"  True cross:    {avg_cross:.4f}")
    print(f"  Shuffle separation: {shuffle_sep:.2f}")
    print(f"  True separation:    {separation:.2f}")
    results["shuffle_control"] = {
        "shuffle_within": sw, "shuffle_cross": sc,
        "shuffle_separation": shuffle_sep,
        "true_within": avg_within, "true_cross": avg_cross,
        "true_separation": separation,
    }

    # --- 5n. Holdout family debug (v4 diagnostic) ---
    print("\n--- Holdout Family Debug ---")
    debug_fam = HOLDOUT_FAMILIES[0] if HOLDOUT_FAMILIES else None
    if debug_fam and debug_fam in available_families:
        holdout_diag = family_debug(
            trajectories=trajectories,
            latent_states=latent_states,
            reconstruction_errors=rec_errors,
            transport_residuals=transport_residuals_map,
            family_pids=available_families[debug_fam],
            family_name=debug_fam,
        )
        print(f"  Family: {debug_fam}")
        print(f"  Avg recon MSE:     {holdout_diag.avg_reconstruction_mse:.6f}")
        print(f"  Avg transport res: {holdout_diag.avg_transport_residual:.6f}")
        print(f"  Traj length range: {holdout_diag.traj_length_range}")
        print(f"  Vel profile var:   {holdout_diag.velocity_profile_variance:.6f}")
        for pid, info in holdout_diag.per_pid.items():
            print(f"    {pid}: T={info['traj_length']} rec={info['reconstruction_mse']:.6f} "
                  f"|vel|={info['mean_velocity_norm']:.4f}")
        results["holdout_debug"] = asdict(holdout_diag)

    # --- 5o. Semantic Scorecard (v4 diagnostic) ---
    print("\n--- Semantic Scorecard ---")
    scorecard = compute_scorecard(
        latent_states=latent_states,
        families=available_families,
        retrieval_result=ret,
        hard_negative_result=hn,
        motion_result=ms,
        shuffle_separation=shuffle_sep,
    )
    print(scorecard.summary())
    results["scorecard"] = scorecard.to_dict()

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SDQ v3.7 SUMMARY")
    print("=" * 60)
    print(f"Epochs:                {args.epochs}")
    layer_desc = f"avg {args.layer_range[0]}–{args.layer_range[1]}" if args.layer_range else str(args.layer)
    print(f"Layer:                 {layer_desc}")
    print(f"Rank:                  {args.rank}")
    print(f"Normalization:         {norm_mode}")
    print(f"Triplet margin:        {triplet_margin}")
    print(f"Transport weight:      {args.transport_weight}")
    print(f"Composition weight:    {args.composition_weight}")
    print(f"Composition freq:      every {args.composition_freq} epochs")
    print(f"Contrastive freq:      every {CONTRASTIVE_FREQ} pairs")
    print(f"VICReg margins:        centroid={args.vicreg_centroid_margin}, hn={args.vicreg_hn_margin}")
    print(f"Final total loss:      {history[-1]['total']:.6f}")
    print(f"Final triplet:         {history[-1]['triplet']:.6f}")
    print(f"Final velocity:        {history[-1]['velocity']:.6f}")
    print(f"Final composition:     {history[-1]['composition']:.6f}")
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
    if hn:
        print(f"Hard-neg passes:       {hn.passes}")
        print(f"Hard-neg sep ratio:    {hn.separation_ratio:.2f}")
    print(f"Scorecard overall:     {scorecard.overall_score():.1f}/100")
    print("=" * 60)

    results["summary"] = {
        "epochs": args.epochs, "rank": args.rank,
        "norm_mode": norm_mode, "triplet_margin": triplet_margin,
        "transport_weight": args.transport_weight,
        "composition_weight": args.composition_weight,
        "composition_freq": args.composition_freq,
        "contrastive_freq": CONTRASTIVE_FREQ,
        "vicreg_centroid_margin": args.vicreg_centroid_margin,
        "vicreg_hn_margin": args.vicreg_hn_margin,
        "final_loss": history[-1],
        "latent_separation": separation,
        "retrieval_accuracy": ret.accuracy,
        "retrieval_mrr": ret.mean_reciprocal_rank,
        "motion_cosine_separation": ms.velocity_cosine_separation,
        "motion_l2_separation": ms.velocity_l2_separation,
        "composition_error": avg_comp,
        "train_transport_residual": avg_train_res,
        "holdout_transport_residual": avg_holdout_res,
        "identity_improvement": improvement,
        "scorecard": scorecard.overall_score(),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="SDQ v3.7 training")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--layer-range", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                        help="Average hidden states across layers LO..HI (overrides --layer). "
                             "e.g. --layer-range 10 16 targets Gemma-2-2B reasoning zone.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--gauge-dim", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--norm-mode", type=str, default="center_scale",
                        choices=["unit_norm", "center_scale", "standardize"])
    parser.add_argument("--triplet-margin", type=float, default=2.0)
    # v3.7 transport + structural args
    parser.add_argument("--transport-weight", type=float, default=3.5,
                        help="Transport consistency loss weight (default 3.5; v3.5 used 2.0)")
    parser.add_argument("--composition-weight", type=float, default=0.30,
                        help="Composition loss weight (default 0.30; v3.5 used 0.15)")
    parser.add_argument("--composition-freq", type=int, default=3,
                        help="Run composition loss every N epochs (default 3; v3.5 used 5)")
    parser.add_argument("--contrastive-freq", type=int, default=2,
                        help="VICReg fires every N pair steps (default 2, matching v3.5)")
    parser.add_argument("--vicreg-centroid-margin", type=float, default=8.0,
                        help="Centroid repulsion hinge margin in VICReg (default 8.0)")
    parser.add_argument("--vicreg-hn-margin", type=float, default=4.0,
                        help="Hard-negative repulsion hinge margin in VICReg (default 4.0)")
    parser.add_argument("--gate-init", type=float, default=0.1,
                        help="Transport correction gate initial value (default 0.1; v3.6 used 0.0)")
    parser.add_argument("--patience", type=int, default=300,
                        help="Early stopping: halt if total loss doesn't improve for N epochs (default 300 = disabled)")
    parser.add_argument("--output", type=str, default="artifacts/sdq_v37_results.json")
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
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
