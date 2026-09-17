#!/usr/bin/env python3
"""SDQ v4a training: v3.7 pipeline + attractor analysis + latent probe.

Built on the v3.7 training pipeline (MultiScaleConvEncoder, VICReg contrastive,
mini-batched encoding, transport gate_init=0.1, early stopping) and adds two
post-training evaluation stages:

  1. Attractor analysis — find fixed points of the learned LatentODE,
     classify stability via Jacobian eigenvalues, map basins of attraction,
     and correlate attractors with task answers.

  2. Latent probe — train linear classifiers on frozen z_t embeddings for
     task family classification and answer prediction, testing whether the
     latent space carries causally meaningful information.

Usage:
    python run_sdq_v4a.py [--epochs 300] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
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
from sdq.losses.triplet import semi_hard_triplet_loss
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

# New: attractor analysis and latent probe
from sdq.eval.attractor_analysis import (
    AttractorAnalysis, format_attractor_report,
)
from sdq.eval.latent_probe import (
    LatentProbe, train_probe, format_probe_result,
)

# New: manifold geometry
from sdq.manifold import ManifoldExtractor, spectral_profile, helix_score, subspace_overlap


# ---------------------------------------------------------------------------
# Semantic equivalence groups — populated dynamically from benchmark metadata
# ---------------------------------------------------------------------------
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
# Training step: one (i, j) pair — v3.7 style with detached reconstruction
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

    # Prototype regularization — soft penalty on ΔG
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
      - Invariance: MSE within-family -> pulls same-family members together
      - Variance: per-dim std >= 1.0 across batch -> prevents collapse
      - Covariance: decorrelates dimensions -> uses full latent capacity
      - Centroid repulsion: L2 hinge -> pushes family centroids apart
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
                    # Same variant, different family -> hard negative
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
# NEW: Attractor analysis + Latent probe evaluation
# ---------------------------------------------------------------------------
def run_attractor_analysis(
    model: SDQModel,
    latent_states: dict[str, torch.Tensor],
    PID_TO_FAMILY: dict[str, str],
    device: torch.device,
) -> dict:
    """Run attractor analysis on the trained dynamics."""
    print("\n" + "=" * 60)
    print("ATTRACTOR ANALYSIS")
    print("=" * 60)

    analyzer = AttractorAnalysis(
        dynamics=model.dynamics,
        convergence_threshold=1e-4,
        max_steps=2000,
        attractor_merge_radius=0.5,
    )

    # Build trajectory dict and answer_ids for the analyzer.
    # answer_ids should be actual answers (e.g., "14"), not semantic family names.
    z_trajectories: dict[str, torch.Tensor] = {}
    answer_ids: dict[str, str] = {}
    for pid, z in latent_states.items():
        z_trajectories[pid] = z.to(device)
        meta = _PID_META.get(pid, {})
        aid = meta.get("answer_id")
        if aid is not None:
            answer_ids[pid] = str(aid)
        elif pid in PID_TO_FAMILY:
            # Fallback: use semantic family as a proxy
            answer_ids[pid] = PID_TO_FAMILY[pid]

    report = analyzer.full_analysis(z_trajectories, answer_ids)
    print(format_attractor_report(report))

    # Serialize for JSON
    attractor_results = {
        "num_attractors": report.num_attractors,
        "num_stable": report.num_stable,
        "num_unstable": report.num_unstable,
        "answer_attractor_purity": report.answer_attractor_purity,
        "mean_convergence_steps": report.mean_convergence_steps,
        "basin_separation_score": report.basin_separation_score,
        "raw": report.raw,
    }

    if report.trajectory_stabilities:
        stable_count = sum(1 for s in report.trajectory_stabilities if s.is_stable)
        mean_lyap = sum(s.max_lyapunov for s in report.trajectory_stabilities) / len(report.trajectory_stabilities)
        attractor_results["stable_trajectories"] = stable_count
        attractor_results["total_trajectories_analyzed"] = len(report.trajectory_stabilities)
        attractor_results["mean_max_lyapunov"] = mean_lyap

    if report.fixed_points:
        attractor_results["fixed_points"] = [
            {
                "velocity_norm": fp.velocity_norm,
                "max_eigenvalue": fp.max_eigenvalue,
                "is_stable": fp.is_stable,
                "basin_size": fp.basin_size,
                "associated_answers": fp.associated_answers,
            }
            for fp in report.fixed_points
        ]

    return attractor_results


def run_manifold_geometry_eval(
    extractor: "ManifoldExtractor",
    latent_states: dict[str, torch.Tensor],
    available_families: dict[str, list[str]],
) -> dict:
    """Report manifold geometry statistics from a fitted ManifoldExtractor.

    Args:
        extractor:        Fitted ManifoldExtractor.
        latent_states:    {pid: [T, k]} latent trajectories.
        available_families: {family: [pid, ...]}

    Returns:
        dict suitable for JSON serialization.
    """
    print("\n" + "=" * 60)
    print("MANIFOLD GEOMETRY EVALUATION")
    print("=" * 60)

    results: dict = {}

    # --- Spectral profile ---
    sv = extractor.singular_values_
    if sv is not None:
        prof = spectral_profile(sv)
        results["spectral_profile"] = {
            "variance_explained": prof["variance_explained"].tolist(),
            "cumulative_variance": prof["cumulative_variance"].tolist(),
            "effective_rank": prof["effective_rank"],
            "spectral_entropy": prof["spectral_entropy"],
            "k_90": prof["k_90"],
            "k_95": prof["k_95"],
            "k_99": prof["k_99"],
        }
        print(f"  Effective rank:   {prof['effective_rank']:.2f}")
        print(f"  Spectral entropy: {prof['spectral_entropy']:.4f}")
        print(f"  Dims (90/95/99%): {prof['k_90']} / {prof['k_95']} / {prof['k_99']}")

    # --- Global helix score (all latent states concatenated) ---
    all_z = []
    for z in latent_states.values():
        all_z.append(z.reshape(-1, z.shape[-1]))
    if all_z:
        Z_all = torch.cat(all_z, dim=0)
        global_helix = helix_score(Z_all)
        results["global_helix_score"] = global_helix
        print(f"  Global helix score: {global_helix:.4f}")

    # --- Per-family helix scores ---
    family_helix: dict[str, float] = {}
    for fam, pids in available_families.items():
        fam_z = []
        for pid in pids:
            if pid in latent_states:
                fam_z.append(latent_states[pid].reshape(-1, latent_states[pid].shape[-1]))
        if fam_z:
            Z_fam = torch.cat(fam_z, dim=0)
            family_helix[fam] = helix_score(Z_fam)
    results["family_helix_scores"] = family_helix
    if family_helix:
        top5 = sorted(family_helix.items(), key=lambda x: -x[1])[:5]
        print("  Top-5 family helix scores:")
        for fam, s in top5:
            print(f"    {fam}: {s:.4f}")

    # --- Pairwise subspace overlap (top 10 families) ---
    top_fams = list(available_families.keys())[:10]
    V_k = extractor.V_k_
    if V_k is not None and len(top_fams) >= 2:
        # Build per-family subspaces via per-family PCA on latent states
        fam_subspaces: dict[str, torch.Tensor] = {}
        for fam in top_fams:
            fam_z = []
            for pid in available_families[fam]:
                if pid in latent_states:
                    fam_z.append(latent_states[pid].reshape(-1, latent_states[pid].shape[-1]))
            if len(fam_z) < 2:
                continue
            Z_f = torch.cat(fam_z, dim=0).float()
            Z_fc = Z_f - Z_f.mean(0)
            try:
                _, _, Vh = torch.linalg.svd(Z_fc, full_matrices=False)
                k_sub = min(4, Vh.shape[0])
                fam_subspaces[fam] = Vh[:k_sub].T  # [k, k_sub]
            except Exception:
                pass

        overlap_matrix: dict[str, dict[str, float]] = {}
        fam_list = list(fam_subspaces.keys())
        for i, fa in enumerate(fam_list):
            overlap_matrix[fa] = {}
            for j, fb in enumerate(fam_list):
                if fa == fb:
                    overlap_matrix[fa][fb] = 1.0
                else:
                    overlap_matrix[fa][fb] = subspace_overlap(
                        fam_subspaces[fa], fam_subspaces[fb]
                    )
        results["subspace_overlap_matrix"] = overlap_matrix
        if len(fam_list) >= 2:
            print(f"  Subspace overlap ({len(fam_list)}x{len(fam_list)} matrix computed)")

    return results


def run_latent_probes(
    model: SDQModel,
    latent_states: dict[str, torch.Tensor],
    PID_TO_FAMILY: dict[str, str],
    available_families: dict[str, list[str]],
    device: torch.device,
) -> dict:
    """Train and evaluate latent probes for task family + answer prediction."""
    print("\n" + "=" * 60)
    print("LATENT PROBE EVALUATION")
    print("=" * 60)

    # Build trajectory dict on CPU for probing
    z_trajectories: dict[str, torch.Tensor] = {}
    for pid, z in latent_states.items():
        z_trajectories[pid] = z.cpu()

    latent_dim = next(iter(z_trajectories.values())).shape[-1]
    probe_results: dict = {}

    # --- Probe 1: Task family classification ---
    print("\n--- Probe: Task Family ---")
    # Group by task_family (e.g. "arithmetic", "syllogism", "variable_chain")
    # rather than semantic_task_id, for a coarser and more meaningful probe
    task_fam_labels: dict[str, int] = {}
    task_fam_names: dict[int, str] = {}
    tf_to_idx: dict[str, int] = {}
    for pid in z_trajectories:
        meta = _PID_META.get(pid, {})
        tf = meta.get("task_family", PID_TO_FAMILY.get(pid, "unknown"))
        if tf not in tf_to_idx:
            idx = len(tf_to_idx)
            tf_to_idx[tf] = idx
            task_fam_names[idx] = tf
        task_fam_labels[pid] = tf_to_idx[tf]

    if task_fam_labels and len(tf_to_idx) > 1:
        num_tf = len(tf_to_idx)
        probe = LatentProbe(latent_dim, num_tf, aggregation="mean")
        train_res, test_res = train_probe(
            probe, z_trajectories, task_fam_labels, task_fam_names,
            epochs=200, lr=1e-3, device="cpu",
        )
        print(format_probe_result("Task Family", train_res, test_res))
        probe_results["task_family"] = {
            "train_accuracy": train_res.accuracy,
            "test_accuracy": test_res.accuracy,
            "num_classes": num_tf,
            "train_samples": train_res.num_samples,
            "test_samples": test_res.num_samples,
            "per_class_accuracy": test_res.per_class_accuracy,
        }

    # --- Probe 2: Semantic group classification ---
    print("\n--- Probe: Semantic Group ---")
    sem_labels: dict[str, int] = {}
    sem_names: dict[int, str] = {}
    sem_to_idx: dict[str, int] = {}
    for pid in z_trajectories:
        fam = PID_TO_FAMILY.get(pid)
        if fam is None:
            continue
        if fam not in sem_to_idx:
            idx = len(sem_to_idx)
            sem_to_idx[fam] = idx
            sem_names[idx] = fam
        sem_labels[pid] = sem_to_idx[fam]

    if sem_labels and len(sem_to_idx) > 1:
        num_sem = len(sem_to_idx)
        probe = LatentProbe(latent_dim, num_sem, aggregation="mean")
        train_res, test_res = train_probe(
            probe, z_trajectories, sem_labels, sem_names,
            epochs=200, lr=1e-3, device="cpu",
        )
        print(format_probe_result("Semantic Group", train_res, test_res))
        probe_results["semantic_group"] = {
            "train_accuracy": train_res.accuracy,
            "test_accuracy": test_res.accuracy,
            "num_classes": num_sem,
            "train_samples": train_res.num_samples,
            "test_samples": test_res.num_samples,
            "per_class_accuracy": test_res.per_class_accuracy,
        }

    # --- Probe 3: Surface variant (should NOT be predictable from z_t) ---
    print("\n--- Probe: Surface Variant (null test) ---")
    variant_labels: dict[str, int] = {}
    variant_names: dict[int, str] = {}
    var_to_idx: dict[str, int] = {}
    for pid in z_trajectories:
        v = get_variant(pid)
        if v not in var_to_idx:
            idx = len(var_to_idx)
            var_to_idx[v] = idx
            variant_names[idx] = v
        variant_labels[pid] = var_to_idx[v]

    if variant_labels and len(var_to_idx) > 1:
        num_variants = len(var_to_idx)
        probe = LatentProbe(latent_dim, num_variants, aggregation="mean")
        train_res, test_res = train_probe(
            probe, z_trajectories, variant_labels, variant_names,
            epochs=200, lr=1e-3, device="cpu",
        )
        print(format_probe_result("Surface Variant (null)", train_res, test_res))
        probe_results["surface_variant"] = {
            "train_accuracy": train_res.accuracy,
            "test_accuracy": test_res.accuracy,
            "num_classes": num_variants,
            "train_samples": train_res.num_samples,
            "test_samples": test_res.num_samples,
            "per_class_accuracy": test_res.per_class_accuracy,
        }

    # --- Probe 4: Semantic group from last timestep only ---
    print("\n--- Probe: Semantic Group (last-step) ---")
    if sem_labels and len(sem_to_idx) > 1:
        probe = LatentProbe(latent_dim, num_sem, aggregation="last")
        train_res, test_res = train_probe(
            probe, z_trajectories, sem_labels, sem_names,
            epochs=200, lr=1e-3, device="cpu",
        )
        print(format_probe_result("Semantic Group (last)", train_res, test_res))
        probe_results["semantic_group_last"] = {
            "train_accuracy": train_res.accuracy,
            "test_accuracy": test_res.accuracy,
        }

    # --- Probe 5: Answer prediction ---
    print("\n--- Probe: Answer Prediction ---")
    answer_labels: dict[str, int] = {}
    answer_names: dict[int, str] = {}
    ans_to_idx: dict[str, int] = {}
    for pid in z_trajectories:
        meta = _PID_META.get(pid, {})
        aid = meta.get("answer_id")
        if aid is None:
            continue
        if aid not in ans_to_idx:
            idx = len(ans_to_idx)
            ans_to_idx[aid] = idx
            answer_names[idx] = str(aid)
        answer_labels[pid] = ans_to_idx[aid]

    if answer_labels and len(ans_to_idx) > 1:
        num_answers = len(ans_to_idx)
        probe = LatentProbe(latent_dim, num_answers, aggregation="mean")
        train_res, test_res = train_probe(
            probe, z_trajectories, answer_labels, answer_names,
            epochs=200, lr=1e-3, device="cpu",
        )
        print(format_probe_result("Answer Prediction", train_res, test_res))
        probe_results["answer_prediction"] = {
            "train_accuracy": train_res.accuracy,
            "test_accuracy": test_res.accuracy,
            "num_classes": num_answers,
            "train_samples": train_res.num_samples,
            "test_samples": test_res.num_samples,
        }

    return probe_results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_training(args: argparse.Namespace) -> dict:
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load and normalize — dynamic benchmark loading (from v3.7)
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
    # 2. Build model — MultiScaleConvEncoder + prototype transport
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

    # Optionally replace encoder with analytic PCA manifold extractor
    manifold_extractor = None
    if args.encoder_mode == "manifold":
        all_h = [trajectories[pid].states for pid in trajectories]
        manifold_extractor = ManifoldExtractor(k=args.latent_dim).fit(all_h)
        manifold_extractor = manifold_extractor.to(device)
        model.encoder = manifold_extractor  # duck-typed replacement, no grad
        model._modules[0] = manifold_extractor

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}, rank={args.rank}")

    # v3.7 loss weights (semantic=3.0, velocity=3.0, triplet=5.0 for VICReg)
    loss_fn = SDQLoss(weights=LossWeights(
        reconstruction=1.0,
        semantic=3.0,
        transport=args.transport_weight,
        cycle=1.0,
        gauge=0.1,
        event=0.1,
        triplet=5.0,
        velocity=3.0,
        composition=args.composition_weight,
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
    print(f"Transport weight: {args.transport_weight}")
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
    # 5. Evaluation — v3.7 full suite
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    model.eval()
    results: dict = {"training_history": history, "family_diagnostics": family_diag}

    # --- Encode all latents ---
    latent_states: dict[str, torch.Tensor] = {}
    if args.encoder_mode == "learned":
        # Mini-batched encoding for the learned conv encoder
        EVAL_BATCH = 64
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
    else:
        # Manifold mode: encode each trajectory individually (no padding needed)
        with torch.no_grad():
            latent_states = {
                pid: manifold_extractor.encode(trajectories[pid].states.to(device)).cpu()
                for pid in trajectories
            }

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

    # --- 5g. Hard-negative resistance ---
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

    # --- 5h. Cycle consistency ---
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

    # --- 5i. Identity baseline ---
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
    avg_train_res_val = sum(train_avg_list) / max(len(train_avg_list), 1)
    avg_holdout_res_val = sum(holdout_avg_list) / max(len(holdout_avg_list), 1)
    improvement = 1 - (avg_train_res_val / max(avg_identity, 1e-8))
    print(f"  Identity baseline:     {avg_identity:.6f}")
    print(f"  Learned (train):       {avg_train_res_val:.6f}")
    print(f"  Learned (holdout):     {avg_holdout_res_val:.6f}")
    print(f"  Improvement over I:    {improvement:.1%}")
    results["identity_baseline"] = {
        "identity": avg_identity, "train": avg_train_res_val,
        "holdout": avg_holdout_res_val, "improvement": improvement,
    }

    # --- 5j. Shuffle control ---
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

    # --- 5k. Holdout family debug ---
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
        results["holdout_debug"] = asdict(holdout_diag)

    # --- 5l. Semantic Scorecard ---
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
    # 6. NEW: Attractor analysis
    # ------------------------------------------------------------------
    attractor_results = run_attractor_analysis(model, latent_states, PID_TO_FAMILY, device)
    results["attractor_analysis"] = attractor_results

    # ------------------------------------------------------------------
    # 7. NEW: Latent probe evaluation
    # ------------------------------------------------------------------
    probe_results = run_latent_probes(model, latent_states, PID_TO_FAMILY, available_families, device)
    results["latent_probes"] = probe_results

    # ------------------------------------------------------------------
    # 8. NEW: Manifold geometry evaluation (manifold mode only)
    # ------------------------------------------------------------------
    if args.encoder_mode == "manifold" and manifold_extractor is not None:
        results["manifold_geometry"] = run_manifold_geometry_eval(
            manifold_extractor, latent_states, available_families
        )

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SDQ v4a SUMMARY")
    print("=" * 60)
    print(f"Epochs:                {args.epochs}")
    layer_desc = f"avg {args.layer_range[0]}-{args.layer_range[1]}" if args.layer_range else str(args.layer)
    print(f"Layer:                 {layer_desc}")
    print(f"Rank:                  {args.rank}")
    print(f"Normalization:         {norm_mode}")
    print(f"Triplet margin:        {triplet_margin}")
    print(f"Transport weight:      {args.transport_weight}")
    print(f"Composition weight:    {args.composition_weight}")
    print(f"VICReg margins:        centroid={args.vicreg_centroid_margin}, hn={args.vicreg_hn_margin}")
    print(f"Final total loss:      {history[-1]['total']:.6f}")
    print(f"Latent separation:     {separation:.2f}")
    print(f"Retrieval accuracy:    {ret.accuracy:.2%}")
    print(f"Motion cos sep:        {ms.velocity_cosine_separation:.2f}")
    print(f"Transport vs identity: {improvement:.1%}")
    print(f"Scorecard overall:     {scorecard.overall_score():.1f}/100")
    print(f"--- Attractor Analysis ---")
    print(f"  Attractors found:    {attractor_results['num_attractors']}")
    print(f"  Stable:              {attractor_results['num_stable']}")
    print(f"  Answer-attractor purity: {attractor_results['answer_attractor_purity']:.1%}")
    print(f"  Basin separation:    {attractor_results['basin_separation_score']:.2f}")
    if attractor_results.get("mean_max_lyapunov") is not None:
        print(f"  Mean max Lyapunov:   {attractor_results['mean_max_lyapunov']:.4f}")
    print(f"--- Latent Probes ---")
    if "task_family" in probe_results:
        print(f"  Task family (test):  {probe_results['task_family']['test_accuracy']:.1%}")
    if "semantic_group" in probe_results:
        print(f"  Semantic grp (test): {probe_results['semantic_group']['test_accuracy']:.1%}")
    if "surface_variant" in probe_results:
        print(f"  Surface var (null):  {probe_results['surface_variant']['test_accuracy']:.1%}")
    if "answer_prediction" in probe_results:
        print(f"  Answer pred (test):  {probe_results['answer_prediction']['test_accuracy']:.1%}")
    print("=" * 60)

    results["summary"] = {
        "epochs": args.epochs, "rank": args.rank,
        "norm_mode": norm_mode, "triplet_margin": triplet_margin,
        "transport_weight": args.transport_weight,
        "composition_weight": args.composition_weight,
        "vicreg_centroid_margin": args.vicreg_centroid_margin,
        "vicreg_hn_margin": args.vicreg_hn_margin,
        "final_loss": history[-1],
        "latent_separation": separation,
        "retrieval_accuracy": ret.accuracy,
        "retrieval_mrr": ret.mean_reciprocal_rank,
        "motion_cosine_separation": ms.velocity_cosine_separation,
        "motion_l2_separation": ms.velocity_l2_separation,
        "identity_improvement": improvement,
        "scorecard": scorecard.overall_score(),
        "num_attractors": attractor_results["num_attractors"],
        "answer_attractor_purity": attractor_results["answer_attractor_purity"],
        "basin_separation_score": attractor_results["basin_separation_score"],
        "task_family_probe_acc": probe_results.get("task_family", {}).get("test_accuracy", 0),
        "semantic_group_probe_acc": probe_results.get("semantic_group", {}).get("test_accuracy", 0),
        "surface_variant_probe_acc": probe_results.get("surface_variant", {}).get("test_accuracy", 0),
        "answer_prediction_probe_acc": probe_results.get("answer_prediction", {}).get("test_accuracy", 0),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="SDQ v4a training + attractor/probe eval")
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
                        help="Transport consistency loss weight (default 3.5)")
    parser.add_argument("--composition-weight", type=float, default=0.30,
                        help="Composition loss weight (default 0.30)")
    parser.add_argument("--composition-freq", type=int, default=3,
                        help="Run composition loss every N epochs (default 3)")
    parser.add_argument("--contrastive-freq", type=int, default=2,
                        help="VICReg fires every N pair steps (default 2)")
    parser.add_argument("--vicreg-centroid-margin", type=float, default=8.0,
                        help="Centroid repulsion hinge margin in VICReg (default 8.0)")
    parser.add_argument("--vicreg-hn-margin", type=float, default=4.0,
                        help="Hard-negative repulsion hinge margin in VICReg (default 4.0)")
    parser.add_argument("--gate-init", type=float, default=0.1,
                        help="Transport correction gate initial value (default 0.1)")
    parser.add_argument("--patience", type=int, default=300,
                        help="Early stopping: halt if total loss doesn't improve for N epochs (default 300 = disabled)")
    parser.add_argument("--encoder-mode", default="learned",
                        choices=["learned", "manifold"],
                        help="'learned' = MultiScaleConvEncoder (default); "
                             "'manifold' = analytic PCA ManifoldExtractor")
    parser.add_argument("--output", type=str, default="artifacts/sdq_v4a_results.json")
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
