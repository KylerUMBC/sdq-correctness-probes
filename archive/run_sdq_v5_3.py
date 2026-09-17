#!/usr/bin/env python3
"""SDQ v5.3 training: contrastive geometry + supervised regime dynamics.

Phase 1: Same as v5.1 (SupCon + velocity cosine + ts_pull + DANN surface).
Phase 2: Per-regime velocity MLPs, Gumbel-softmax gate (tau annealed), task-family
  supervised gate loss (class-balanced), terminal velocity penalty, dynamics-focused
  training (contraction delayed to exhaustion, effectively pure dynamics).
Phase 3: Contraction fine-tuning with fresh AdamW optimizer and CosineAnnealingLR.
  Dynamics losses scaled by dyn_weight_phase3 (default 0.5), L_ms_contract weight
  boosted to ms_contract_weight_phase3 (default 15.0), contraction rollout extended
  to num_substeps_phase3 (default 25, vs K=10 in Phase 2) for stronger gradient
  signal matching the 50-step evaluation horizon (v5.3 iter8).

Usage:
    python run_sdq_v5_3.py [--device cuda] [--epochs-phase1 200] [--epochs-phase2 400]
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
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from sdq.instrumentation import collect_runs, set_seed
from sdq.trajectories import (
    extract_trajectory, Trajectory, normalize_trajectory, velocity_vectors,
)
from sdq.alignment import MonotoneAligner
from sdq.latent import (
    MultiScaleConvEncoder,
    ResidualDecoder,
    RegimeSwitchingDynamics,
)
from sdq.losses import reconstruction_loss
from sdq.losses.velocity import latent_velocity_cosine_loss
from sdq.eval.retrieval import retrieval_accuracy
from sdq.eval.motion_similarity import motion_similarity_eval, dynamics_motion_similarity_eval
from sdq.eval.hard_negatives import hard_negative_test
from sdq.eval.semantic_scorecard import compute_scorecard
from sdq.eval.latent_probe import LatentProbe, train_probe, format_probe_result
from sdq.eval.tube_analysis import full_tube_analysis, format_tube_report


# ---------------------------------------------------------------------------
# Global metadata lookup (populated during data loading)
# ---------------------------------------------------------------------------
_PID_META: dict[str, dict] = {}


def get_variant(pid: str) -> str:
    meta = _PID_META.get(pid)
    if meta:
        return meta["surface_template_id"]
    parts = pid.split("_", 1)
    return parts[1] if len(parts) > 1 else pid


def _levenshtein_int_seq(a: list[int], b: list[int]) -> int:
    """Edit distance between two integer sequences."""
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(prev + cost, dp[j] + 1, dp[j - 1] + 1)
            prev = cur
    return dp[lb]


def _balanced_tf_indices(
    tf_class_indices: dict[int, torch.Tensor],
    n_total: int,
    num_task_families: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample row indices into all_starts with ~equal counts per task-family class."""
    if n_total <= 0 or num_task_families <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    per_class = max(1, n_total // num_task_families)
    parts: list[torch.Tensor] = []
    for i in range(num_task_families):
        mask = tf_class_indices[i]
        if mask.numel() == 0:
            continue
        if mask.numel() >= per_class:
            perm = torch.randperm(mask.numel(), device=device)
            parts.append(mask[perm[:per_class]])
        else:
            j = torch.randint(0, mask.numel(), (per_class,), device=device)
            parts.append(mask[j])
    if not parts:
        return torch.zeros(0, dtype=torch.long, device=device)
    out = torch.cat(parts)
    if out.shape[0] > n_total:
        out = out[:n_total]
    elif out.shape[0] < n_total:
        all_masks = [
            tf_class_indices[i]
            for i in range(num_task_families)
            if tf_class_indices[i].numel() > 0
        ]
        if all_masks:
            all_idx = torch.cat(all_masks)
            need = n_total - out.shape[0]
            j = torch.randint(0, all_idx.numel(), (need,), device=device)
            out = torch.cat([out, all_idx[j]])
    return out


def compute_regime_metrics(
    dynamics: RegimeSwitchingDynamics,
    latent_states_raw: dict[str, torch.Tensor],
    available_families: dict[str, list[str]],
    device: torch.device,
    num_substeps: int,
    max_pairs_within: int = 40,
    max_pairs_cross: int = 80,
) -> dict:
    """Regime consistency, per-regime flow MSE, edit-distance alignment, gate entropy."""
    dynamics.eval()
    eps = 1e-8
    within_kls: list[float] = []
    cross_kls: list[float] = []

    # Within-family symmetric KL of gate sequences
    for fam, pids in available_families.items():
        pids_ok = [p for p in pids if p in latent_states_raw]
        if len(pids_ok) < 2:
            continue
        pairs_done = 0
        for i, a in enumerate(pids_ok):
            for b in pids_ok[i + 1 :]:
                if pairs_done >= max_pairs_within:
                    break
                za = latent_states_raw[a].to(device)
                zb = latent_states_raw[b].to(device)
                tmin = min(za.shape[0], zb.shape[0])
                if tmin < 1:
                    continue
                with torch.no_grad():
                    pa = dynamics.gate_probs(za[:tmin])
                    pb = dynamics.gate_probs(zb[:tmin])
                    kl_ab = (pa * (pa.clamp_min(eps).log() - pb.clamp_min(eps).log())).sum(dim=-1)
                    kl_ba = (pb * (pb.clamp_min(eps).log() - pa.clamp_min(eps).log())).sum(dim=-1)
                    within_kls.append(0.5 * (kl_ab + kl_ba).mean().item())
                pairs_done += 1
            if pairs_done >= max_pairs_within:
                break

    fam_list = list(available_families.keys())
    rng = random.Random(42)
    for _ in range(max_pairs_cross):
        if len(fam_list) < 2:
            break
        fa, fb = rng.sample(fam_list, 2)
        pa_list = [p for p in available_families[fa] if p in latent_states_raw]
        pb_list = [p for p in available_families[fb] if p in latent_states_raw]
        if not pa_list or not pb_list:
            continue
        a, b = rng.choice(pa_list), rng.choice(pb_list)
        za = latent_states_raw[a].to(device)
        zb = latent_states_raw[b].to(device)
        tmin = min(za.shape[0], zb.shape[0])
        if tmin < 1:
            continue
        with torch.no_grad():
            pa = dynamics.gate_probs(za[:tmin])
            pb = dynamics.gate_probs(zb[:tmin])
            kl_ab = (pa * (pa.clamp_min(eps).log() - pb.clamp_min(eps).log())).sum(dim=-1)
            kl_ba = (pb * (pb.clamp_min(eps).log() - pa.clamp_min(eps).log())).sum(dim=-1)
            cross_kls.append(0.5 * (kl_ab + kl_ba).mean().item())

    mean_w = sum(within_kls) / max(len(within_kls), 1)
    mean_c = sum(cross_kls) / max(len(cross_kls), 1)
    if len(within_kls) == 0:
        regime_consistency_ratio = 0.0
    else:
        regime_consistency_ratio = mean_c / max(mean_w, 1e-8)

    # Argmax regime sequences + edit distances
    within_edits: list[float] = []
    cross_edits: list[float] = []
    for fam, pids in available_families.items():
        pids_ok = [p for p in pids if p in latent_states_raw]
        if len(pids_ok) < 2:
            continue
        seqs: list[list[int]] = []
        for pid in pids_ok:
            z = latent_states_raw[pid].to(device)
            with torch.no_grad():
                seqs.append(dynamics.gate_probs(z).argmax(dim=-1).cpu().tolist())
        for i, sa in enumerate(seqs):
            for sb in seqs[i + 1 :]:
                within_edits.append(float(_levenshtein_int_seq(sa, sb)))

    for _ in range(max_pairs_cross):
        if len(fam_list) < 2:
            break
        fa, fb = rng.sample(fam_list, 2)
        pa_list = [p for p in available_families[fa] if p in latent_states_raw]
        pb_list = [p for p in available_families[fb] if p in latent_states_raw]
        if not pa_list or not pb_list:
            continue
        a, b = rng.choice(pa_list), rng.choice(pb_list)
        za = latent_states_raw[a].to(device)
        zb = latent_states_raw[b].to(device)
        with torch.no_grad():
            sa = dynamics.gate_probs(za).argmax(dim=-1).cpu().tolist()
            sb = dynamics.gate_probs(zb).argmax(dim=-1).cpu().tolist()
        cross_edits.append(float(_levenshtein_int_seq(sa, sb)))

    mean_we = sum(within_edits) / max(len(within_edits), 1)
    mean_ce = sum(cross_edits) / max(len(cross_edits), 1)
    if len(within_edits) == 0:
        edit_distance_ratio = 0.0
    else:
        edit_distance_ratio = mean_ce / max(mean_we, 1e-8)

    # Per-regime MSE + gate entropy / utilization
    k = dynamics.num_regimes
    bucket_err: dict[int, list[float]] = {r: [] for r in range(k)}
    all_ent: list[float] = []
    util = torch.zeros(k, device=device)
    n_points = 0

    for pid, z in list(latent_states_raw.items())[:300]:
        z = z.to(device)
        if z.shape[0] < 2:
            continue
        with torch.no_grad():
            gp = dynamics.gate_probs(z)
            ent = -(gp * gp.clamp_min(1e-8).log()).sum(dim=-1)
            all_ent.append(ent.mean().item())
            util = util + gp.sum(dim=0)
            n_points += z.shape[0]
            for t in range(z.shape[0] - 1):
                zt = z[t : t + 1]
                dom = dynamics.gate_probs(zt).argmax(dim=-1).item()
                pred = dynamics.multi_step_predict(zt, num_substeps)
                err = (pred[0] - z[t + 1]).pow(2).sum().item()
                bucket_err[dom].append(err)

    # Per-timestep regime histogram (T_max ~ 20)
    t_max = max((latent_states_raw[p].shape[0] for p in latent_states_raw), default=0)
    hist = torch.zeros(t_max, k, device=device)
    n_hist = 0
    for pid, z in list(latent_states_raw.items())[:200]:
        z = z.to(device)
        with torch.no_grad():
            argm = dynamics.gate_probs(z).argmax(dim=-1)
        for t in range(z.shape[0]):
            if t < t_max:
                hist[t, argm[t]] += 1.0
                n_hist += 1

    per_regime_mse = {
        str(r): (sum(v) / len(v) if v else 0.0) for r, v in bucket_err.items()
    }
    util_norm = (util / max(n_points, 1)).cpu().tolist()

    return {
        "regime_kl_within_mean": mean_w,
        "regime_kl_cross_mean": mean_c,
        "regime_consistency_ratio": regime_consistency_ratio,
        "edit_distance_within_mean": mean_we,
        "edit_distance_cross_mean": mean_ce,
        "edit_distance_ratio": edit_distance_ratio,
        "gate_entropy_mean": sum(all_ent) / max(len(all_ent), 1),
        "regime_utilization": util_norm,
        "per_regime_dynamics_mse": per_regime_mse,
        "per_timestep_regime_mass": hist.cpu().tolist() if t_max > 0 else [],
    }


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (SupCon)
# ---------------------------------------------------------------------------
def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    hard_neg_mask: torch.Tensor | None = None,
    hard_neg_weight: float = 3.0,
) -> torch.Tensor:
    """Supervised contrastive loss with optional hard-negative upweighting.

    Args:
        embeddings: [N, D] L2-normalized embeddings.
        labels: [N] integer class labels.
        temperature: scaling temperature (lower = sharper).
        hard_neg_mask: [N, N] bool mask where True = hard-negative pair
                       (same surface template, different family).
        hard_neg_weight: multiplier for hard-negative pairs in denominator.

    Returns:
        Scalar loss.
    """
    N = embeddings.shape[0]
    device = embeddings.device

    # Pairwise cosine similarity (embeddings already L2-normed)
    sim = embeddings @ embeddings.T  # [N, N]
    sim = sim / temperature

    # Mask: positive pairs share the same label (excluding self)
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
    self_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = label_eq & self_mask  # [N, N]

    # Count positives per anchor; skip anchors with no positives
    num_pos = pos_mask.sum(dim=1)  # [N]
    valid = num_pos > 0

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Upweight hard negatives in the denominator
    # This makes the model work harder to push apart same-template, different-family pairs
    denom_weight = self_mask.float()
    if hard_neg_mask is not None:
        denom_weight = denom_weight + (hard_neg_mask.float() * (hard_neg_weight - 1.0))

    # Log-softmax over all non-self entries
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    log_exp = sim - sim_max  # [N, N]
    exp_sum = (torch.exp(log_exp) * denom_weight).sum(dim=1, keepdim=True)  # [N, 1]
    log_prob = log_exp - torch.log(exp_sum + 1e-12)  # [N, N]

    # Average log-prob over positive pairs
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)  # [N]
    pos_log_prob = pos_log_prob[valid] / num_pos[valid]

    return -pos_log_prob.mean()


# ---------------------------------------------------------------------------
# Gradient reversal for surface template invariance (DANN-style)
# ---------------------------------------------------------------------------
class GradientReversal(torch.autograd.Function):
    """Reverses gradient during backward pass."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class SurfaceClassifier(nn.Module):
    """Adversarial classifier: predicts surface template from encoder output.

    Gradient reversal makes the encoder MAXIMIZE classification loss,
    i.e., learn to NOT encode surface template information.
    """
    def __init__(self, input_dim: int, num_variants: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_variants),
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        x = GradientReversal.apply(x, alpha)
        return self.net(x)


# ---------------------------------------------------------------------------
# Projection head for contrastive learning (discarded after Phase 1)
# ---------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    """2-layer MLP projection head (SimCLR / SupCon style).

    Maps encoder output to a smaller space where SupCon operates.
    Discarded after training — eval uses raw encoder output.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Model bundle (simplified — no gauge encoder, no transport; v5.2 uses RegimeSwitchingDynamics)
# ---------------------------------------------------------------------------
class SDQv5Model(nn.Module):
    def __init__(self, encoder, decoder, dynamics, aligner, proj_head,
                 surface_classifier=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dynamics = dynamics
        self.aligner = aligner
        self.proj_head = proj_head
        self.surface_classifier = surface_classifier


# ---------------------------------------------------------------------------
# Batch encoding utility
# ---------------------------------------------------------------------------
def batch_encode(
    encoder: nn.Module,
    trajectories: dict[str, Trajectory],
    pids: list[str],
    device: torch.device,
    chunk_size: int = 64,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Encode all trajectories, return raw latents and time-averaged embeddings.

    Returns:
        z_full: {pid: [T, D_z]} full latent trajectories (raw encoder output)
        z_mean: {pid: [D_z]} time-averaged embeddings (raw, not normalized)
    """
    z_full: dict[str, torch.Tensor] = {}
    z_mean: dict[str, torch.Tensor] = {}

    for start in range(0, len(pids), chunk_size):
        chunk = pids[start:start + chunk_size]
        h_list = []
        lengths = []
        for pid in chunk:
            h = trajectories[pid].states.to(device)
            h_list.append(h)
            lengths.append(h.shape[0])

        max_T = max(lengths)
        padded = torch.zeros(len(chunk), max_T, h_list[0].shape[1], device=device)
        for i, h in enumerate(h_list):
            padded[i, :lengths[i]] = h

        z_batch = encoder(padded)  # [B, max_T, D_z]

        for i, pid in enumerate(chunk):
            z_i = z_batch[i, :lengths[i]]  # [T_i, D_z]
            z_full[pid] = z_i
            z_mean[pid] = z_i.mean(dim=0)

    return z_full, z_mean


# ---------------------------------------------------------------------------
# Evaluation helpers (adapted from v4a)
# ---------------------------------------------------------------------------
def run_tube_analysis(
    model: SDQv5Model,
    latent_states: dict[str, torch.Tensor],
    latent_states_raw: dict[str, torch.Tensor],
    available_families: dict[str, list[str]],
    device: torch.device,
    run_interventions: bool = True,
) -> dict:
    """Run tube/manifold analysis on latent trajectories.

    Uses blended states for geometry experiments (1-4) and raw states
    for dynamics-based intervention experiments (5).
    """
    z_blended = {pid: z.to(device) for pid, z in latent_states.items()}
    z_raw = {pid: z.to(device) for pid, z in latent_states_raw.items()}

    has_dynamics = run_interventions and hasattr(model, "dynamics")
    report = full_tube_analysis(
        z_blended, available_families,
        dynamics=model.dynamics if has_dynamics else None,
        run_interventions=has_dynamics,
        latent_states_raw=z_raw,
    )
    print(format_tube_report(report))

    from dataclasses import asdict
    tube_results: dict = {
        "coherence": {
            "mean_within_spread": report.coherence.mean_within_spread,
            "mean_cross_spread": report.coherence.mean_cross_spread,
            "tube_tightening_ratio": report.coherence.tube_tightening_ratio,
            "per_timestep_within_spread": report.coherence.per_timestep_within_spread,
            "per_timestep_cross_spread": report.coherence.per_timestep_cross_spread,
            "per_timestep_classification_acc": report.coherence.per_timestep_classification_acc,
            "separation_over_time": report.coherence.separation_over_time,
        },
        "transverse": {
            "transverse_contraction_ratio": report.transverse.transverse_contraction_ratio,
            "along_fraction": report.transverse.along_fraction,
            "transverse_fraction": report.transverse.transverse_fraction,
            "per_timestep_along_spread": report.transverse.per_timestep_along_spread,
            "per_timestep_transverse_spread": report.transverse.per_timestep_transverse_spread,
        },
        "reasoning_stage": {
            "timestep_probe_accuracy": report.reasoning_stage.timestep_probe_accuracy,
            "along_tube_monotonicity": report.reasoning_stage.along_tube_monotonicity,
            "mean_along_velocity": report.reasoning_stage.mean_along_velocity,
        },
        "separation": {
            "mean_classification_margin": report.separation.mean_classification_margin,
            "tube_overlap_fraction": report.separation.tube_overlap_fraction,
            "confused_pairs": [
                {"fam_a": fa, "fam_b": fb, "overlap": ov}
                for fa, fb, ov in report.separation.confused_pairs
            ],
            "per_timestep_classification_margin": report.separation.per_timestep_classification_margin,
        },
    }
    if report.intervention is not None:
        tube_results["intervention"] = {
            "transverse_recovery_rate": report.intervention.transverse_recovery_rate,
            "cross_tube_persistence_rate": report.intervention.cross_tube_persistence_rate,
            "mean_transverse_recovery_steps": report.intervention.mean_transverse_recovery_steps,
            "mean_recovery_distance": report.intervention.mean_recovery_distance,
        }

    if report.dynamics_tube is not None:
        dt = report.dynamics_tube
        tube_results["dynamics_tube"] = {
            "dynamics_tightening_ratio": dt.dynamics_tightening_ratio,
            "per_step_within_spread": dt.per_step_within_spread,
            "per_step_cross_spread": dt.per_step_cross_spread,
            "per_step_vel_cosine": dt.per_step_vel_cosine,
            "per_step_classification_acc": dt.per_step_classification_acc,
        }

    return tube_results


def run_latent_probes(
    model: SDQv5Model,
    latent_states: dict[str, torch.Tensor],
    PID_TO_FAMILY: dict[str, str],
    available_families: dict[str, list[str]],
    device: torch.device,
) -> dict:
    print("\n" + "=" * 60)
    print("LATENT PROBE EVALUATION")
    print("=" * 60)

    z_trajectories = {pid: z.cpu() for pid, z in latent_states.items()}
    latent_dim = next(iter(z_trajectories.values())).shape[-1]
    probe_results: dict = {}

    # Probe 1: Task family classification
    print("\n--- Probe: Task Family ---")
    tf_to_idx: dict[str, int] = {}
    task_fam_names: dict[int, str] = {}
    task_fam_labels: dict[str, int] = {}
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
        }

    # Probe 2: Semantic group classification
    print("\n--- Probe: Semantic Group ---")
    sem_to_idx: dict[str, int] = {}
    sem_names: dict[int, str] = {}
    sem_labels: dict[str, int] = {}
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
        }

    # Probe 3: Surface variant (null test — should NOT be predictable)
    print("\n--- Probe: Surface Variant (null test) ---")
    var_to_idx: dict[str, int] = {}
    variant_names: dict[int, str] = {}
    variant_labels: dict[str, int] = {}
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
        }

    # Probe 4: Answer prediction
    print("\n--- Probe: Answer Prediction ---")
    ans_to_idx: dict[str, int] = {}
    ans_names: dict[int, str] = {}
    ans_labels: dict[str, int] = {}
    for pid in z_trajectories:
        meta = _PID_META.get(pid, {})
        aid = meta.get("answer_id")
        if aid is None:
            continue
        aid_str = str(aid)
        if aid_str not in ans_to_idx:
            idx = len(ans_to_idx)
            ans_to_idx[aid_str] = idx
            ans_names[idx] = aid_str
        ans_labels[pid] = ans_to_idx[aid_str]

    if ans_labels and len(ans_to_idx) > 1:
        num_answers = len(ans_to_idx)
        probe = LatentProbe(latent_dim, num_answers, aggregation="mean")
        train_res, test_res = train_probe(
            probe, z_trajectories, ans_labels, ans_names,
            epochs=200, lr=1e-3, device="cpu",
        )
        print(format_probe_result("Answer Prediction", train_res, test_res))
        probe_results["answer_prediction"] = {
            "train_accuracy": train_res.accuracy,
            "test_accuracy": test_res.accuracy,
            "num_classes": num_answers,
        }

    return probe_results


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def run_training(args: argparse.Namespace) -> dict:
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n=== Loading benchmark data ===")
    benchmark_path = Path("data/prompts/benchmark_v1.json")
    split_path = Path("data/prompts/splits/split_semantic_family.json")
    with open(benchmark_path, encoding="utf-8") as f:
        benchmark_data = json.load(f)
    with open(split_path, encoding="utf-8") as f:
        split_data = json.load(f)

    global _PID_META
    examples_list = benchmark_data["examples"]
    _PID_META = {e["example_id"]: e for e in examples_list}

    _sg: dict[str, list[str]] = defaultdict(list)
    for ex in examples_list:
        _sg[ex["semantic_task_id"]].append(ex["example_id"])
    semantic_groups = dict(_sg)

    train_id_set = set(split_data["train_ids"])
    holdout_id_set = set(split_data["test_ids"])
    print(f"Benchmark: {len(examples_list)} examples, "
          f"{len(semantic_groups)} semantic groups")

    runs = collect_runs("data/runs", prefix="bench_", device="cpu")
    print(f"Loaded {len(runs)} benchmark runs")

    available_families: dict[str, list[str]] = {}
    PID_TO_FAMILY: dict[str, str] = {}
    for stid, pids in semantic_groups.items():
        found = [pid for pid in pids if pid in runs]
        if found:
            available_families[stid] = found
            for pid in found:
                PID_TO_FAMILY[pid] = stid

    TRAIN_FAMILIES: list[str] = []
    HOLDOUT_FAMILIES: list[str] = []
    for stid, pids in available_families.items():
        if any(pid in train_id_set for pid in pids):
            TRAIN_FAMILIES.append(stid)
        elif any(pid in holdout_id_set for pid in pids):
            HOLDOUT_FAMILIES.append(stid)

    tf_counts = Counter(_PID_META[pid]["task_family"]
                        for pids in available_families.values()
                        for pid in pids)
    print(f"Available: {len(available_families)} groups "
          f"({len(TRAIN_FAMILIES)} train, {len(HOLDOUT_FAMILIES)} holdout)")
    for tf, count in sorted(tf_counts.items()):
        print(f"  {tf}: {count} runs")

    # Normalize trajectories
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
    print(f"Hidden dim: {hidden_dim}, latent dim: {args.latent_dim}")

    # Build family-to-label mapping for SupCon
    train_families_avail = {f: available_families[f]
                            for f in TRAIN_FAMILIES if f in available_families}
    train_pids = [pid for fam in TRAIN_FAMILIES if fam in available_families
                  for pid in available_families[fam]]

    fam_to_label: dict[str, int] = {}
    for i, fam in enumerate(sorted(train_families_avail.keys())):
        fam_to_label[fam] = i
    pid_labels = {pid: fam_to_label[PID_TO_FAMILY[pid]] for pid in train_pids}

    # Task-family labels for v5.3 gate supervision (must match num_regimes in dynamics)
    task_family_names = sorted({_PID_META[pid]["task_family"] for pid in train_pids})
    tf_to_idx = {name: i for i, name in enumerate(task_family_names)}
    num_task_families = len(tf_to_idx)
    pid_to_tf_idx = {
        pid: tf_to_idx[_PID_META[pid]["task_family"]] for pid in train_pids
    }

    # Build surface-template IDs for hard-negative mining
    pid_variants = {pid: get_variant(pid) for pid in train_pids}
    # Hard-negative mask: same surface template, different semantic family
    N_train = len(train_pids)
    hn_mask = torch.zeros(N_train, N_train, dtype=torch.bool, device=device)
    for i in range(N_train):
        for j in range(i + 1, N_train):
            if (pid_variants[train_pids[i]] == pid_variants[train_pids[j]]
                    and pid_labels[train_pids[i]] != pid_labels[train_pids[j]]):
                hn_mask[i, j] = True
                hn_mask[j, i] = True
    hn_count = hn_mask.sum().item() // 2

    # Build surface variant labels for adversarial training
    unique_variants = sorted(set(pid_variants.values()))
    variant_to_label = {v: i for i, v in enumerate(unique_variants)}
    pid_variant_labels = {pid: variant_to_label[pid_variants[pid]] for pid in train_pids}
    num_variants = len(unique_variants)
    variant_label_tensor = torch.tensor(
        [pid_variant_labels[pid] for pid in train_pids],
        device=device, dtype=torch.long,
    )

    print(f"Training PIDs: {len(train_pids)}, families: {len(fam_to_label)}, "
          f"hard-negative pairs: {hn_count}, surface variants: {num_variants}")
    print(f"  Task families (gate supervision): {num_task_families} -> {task_family_names}")

    if args.num_regimes != num_task_families:
        print(
            f"  NOTE: --num-regimes {args.num_regimes} overridden by "
            f"{num_task_families} task families from benchmark",
        )

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = SDQv5Model(
        encoder=MultiScaleConvEncoder(hidden_dim, args.latent_dim),
        decoder=ResidualDecoder(args.latent_dim, args.gauge_dim, hidden_dim),
        dynamics=RegimeSwitchingDynamics(
            args.latent_dim, num_regimes=num_task_families,
        ),
        aligner=MonotoneAligner(hidden_dim, align_dim=min(128, hidden_dim)),
        proj_head=ProjectionHead(args.latent_dim, hidden_dim=256, output_dim=128),
        surface_classifier=SurfaceClassifier(args.latent_dim, num_variants),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # ------------------------------------------------------------------
    # 2b. Optional checkpoint loading (skip Phase 1)
    # ------------------------------------------------------------------
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        ckpt_state = load_file(args.checkpoint)
        # Only load encoder/decoder/proj_head weights — train dynamics from scratch
        # Old checkpoints may have dynamics weights from incompatible architectures
        # (e.g. time-gated model with learned decay_rate)
        encoder_state = {
            k: v for k, v in ckpt_state.items()
            if not k.startswith("dynamics.")
        }
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        dyn_skipped = [k for k in ckpt_state if k.startswith("dynamics.")]
        if dyn_skipped:
            print(f"  Skipped {len(dyn_skipped)} dynamics keys (training from scratch)")
        if missing:
            print(f"  Missing keys (initialized fresh): {[k for k in missing if not k.startswith('dynamics.')]}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
        print("  Checkpoint loaded — skipping Phase 1")

    # ------------------------------------------------------------------
    # 3. Phase 1: Contrastive Geometry (skipped if checkpoint provided)
    # ------------------------------------------------------------------
    _skip_phase1 = bool(args.checkpoint)
    if _skip_phase1:
        print("\n  [Phase 1 skipped — encoder loaded from checkpoint]")

    if not _skip_phase1:
        print("\n" + "=" * 60)
        print("PHASE 1: CONTRASTIVE GEOMETRY")
        print("=" * 60)

    phase1_params = (list(model.encoder.parameters())
                     + list(model.decoder.parameters())
                     + list(model.proj_head.parameters())
                     + list(model.surface_classifier.parameters()))
    opt1 = torch.optim.AdamW(phase1_params, lr=args.lr_phase1, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=args.epochs_phase1, eta_min=args.lr_phase1 * 0.01,
    )

    REC_SAMPLES = 32  # trajectories sampled for reconstruction per epoch
    history: list[dict] = []

    temperature = args.temperature

    for epoch in range(1, (0 if _skip_phase1 else args.epochs_phase1) + 1):
        model.train()

        # 1. Batch encode all training trajectories (raw, unnormalized)
        z_full_raw, z_mean_raw = batch_encode(
            model.encoder, trajectories, train_pids, device,
            chunk_size=args.encode_batch,
        )

        # Also compute first-half and second-half means for temporal contrastive
        z_half1 = {}
        z_half2 = {}
        for pid in train_pids:
            z = z_full_raw[pid]  # [T, D_z]
            T = z.shape[0]
            mid = T // 2
            if mid > 0:
                z_half1[pid] = z[:mid].mean(dim=0)
                z_half2[pid] = z[mid:].mean(dim=0)
            else:
                z_half1[pid] = z.mean(dim=0)
                z_half2[pid] = z.mean(dim=0)

        # 2a. Projection-head SupCon (main contrastive through proj head)
        emb_list = [z_mean_raw[pid] for pid in train_pids]
        emb_stack = torch.stack(emb_list)  # [N, D_z]
        proj = model.proj_head(emb_stack)  # [N, proj_dim]
        proj_normed = F.normalize(proj, dim=1)
        label_tensor = torch.tensor(
            [pid_labels[pid] for pid in train_pids],
            device=device, dtype=torch.long,
        )

        L_supcon_proj = supervised_contrastive_loss(
            proj_normed, label_tensor, temperature=temperature,
            hard_neg_mask=hn_mask, hard_neg_weight=10.0,
        )

        # 2b. Direct encoder SupCon (forces encoder itself to separate)
        enc_normed = F.normalize(emb_stack, dim=1)
        L_supcon_enc = supervised_contrastive_loss(
            enc_normed, label_tensor, temperature=temperature,
            hard_neg_mask=hn_mask, hard_neg_weight=10.0,
        )

        # 2c. Temporal segment SupCon (separation at sub-trajectory level)
        h1_stack = torch.stack([z_half1[pid] for pid in train_pids])
        h2_stack = torch.stack([z_half2[pid] for pid in train_pids])
        seg_stack = torch.cat([h1_stack, h2_stack], dim=0)  # [2N, D_z]
        seg_normed = F.normalize(seg_stack, dim=1)
        seg_labels = torch.cat([label_tensor, label_tensor], dim=0)
        L_supcon_seg = supervised_contrastive_loss(
            seg_normed, seg_labels, temperature=temperature,
        )

        # Store raw normed means for diagnostics (no projection head)
        z_mean_dict = {pid: F.normalize(z_mean_raw[pid], dim=0)
                       for pid in train_pids}

        # 3. Variance + Covariance regularization on encoder output
        emb_centered = emb_stack - emb_stack.mean(dim=0)
        emb_std_per_dim = emb_stack.std(dim=0)
        L_var = F.relu(1.0 - emb_std_per_dim).mean()
        N_batch = emb_stack.shape[0]
        cov = (emb_centered.T @ emb_centered) / max(N_batch - 1, 1)
        cov_diag = cov.diag()
        off_diag = cov - torch.diag(cov_diag)
        L_cov = (off_diag ** 2).sum() / emb_stack.shape[1]

        # 4. Per-timestep same-family pull loss (RAW squared L2)
        #    Matches hard_neg metric: .pow(2).sum(dim=-1).mean()
        #    Directly reduces positive_distance in the hard_neg test.
        #    Values ~190 initially, so weight must be very small (0.0005).
        N_PULL_PAIRS = 30
        L_ts_pull = torch.tensor(0.0, device=device)
        pull_count = 0
        sample_fam_keys = random.sample(
            list(train_families_avail.keys()),
            min(N_PULL_PAIRS, len(train_families_avail)),
        )
        for fam in sample_fam_keys:
            fam_pids_avail = [p for p in train_families_avail[fam]
                              if p in z_full_raw]
            if len(fam_pids_avail) >= 2:
                p1, p2 = random.sample(fam_pids_avail, 2)
                z1 = z_full_raw[p1]  # [T1, D]
                z2 = z_full_raw[p2]  # [T2, D]
                T_min = min(z1.shape[0], z2.shape[0])
                # Raw squared L2 per timestep (same metric as hard_neg_test)
                L_ts_pull = L_ts_pull + (z1[:T_min] - z2[:T_min]).pow(2).sum(dim=-1).mean()
                pull_count += 1
        if pull_count > 0:
            L_ts_pull = L_ts_pull / pull_count

        # 4b. Per-timestep velocity cosine loss (same pairs as ts_pull)
        #     Trains encoder to produce velocity-aligned trajectories within family.
        #     latent_velocity_cosine_loss returns mean(1 - cos(dz_i, dz_j)), in [0, 2].
        #     Expected initial value ~0.9 (near-random velocity alignment).
        L_vel_cos = torch.tensor(0.0, device=device)
        vel_count = 0
        for fam in sample_fam_keys:
            fam_pids_avail = [p for p in train_families_avail[fam]
                              if p in z_full_raw]
            if len(fam_pids_avail) >= 2:
                p1, p2 = random.sample(fam_pids_avail, 2)
                z1 = z_full_raw[p1]
                z2 = z_full_raw[p2]
                T_min = min(z1.shape[0], z2.shape[0])
                if T_min >= 2:  # need at least 2 timesteps for velocity
                    L_vel_cos = L_vel_cos + latent_velocity_cosine_loss(
                        z1[:T_min], z2[:T_min],
                    )
                    vel_count += 1
        if vel_count > 0:
            L_vel_cos = L_vel_cos / vel_count

        # 5. Gradient reversal: adversarial surface template classifier
        #    Encoder gradient is REVERSED — encoder learns to strip surface info
        adv_alpha = min(epoch / 50.0, 1.0)  # ramp up gradually
        surf_logits = model.surface_classifier(emb_stack, alpha=adv_alpha)
        L_surf = F.cross_entropy(surf_logits, variant_label_tensor)

        # 6. Reconstruction loss (sampled)
        rec_pids = random.sample(train_pids, min(REC_SAMPLES, len(train_pids)))
        L_rec = torch.tensor(0.0, device=device)
        for pid in rec_pids:
            z_raw = z_full_raw[pid]
            h = trajectories[pid].states.to(device)
            h_hat = model.decoder(z_raw)
            L_rec = L_rec + reconstruction_loss(h, h_hat)
        L_rec = L_rec / len(rec_pids)

        # 7. Total loss
        #    ts_pull raw values ~190, weight 0.003 → ~0.57 gradient (38% of SupCon)
        #    vel_cos values ~0.9, weight 0.3 → ~0.27 gradient
        loss = (L_supcon_proj
                + 0.5 * L_supcon_enc
                + 0.3 * L_supcon_seg
                + 0.5 * L_var
                + 0.5 * L_cov
                + 0.003 * L_ts_pull
                + 0.3 * L_vel_cos
                + 0.3 * L_surf
                + args.rec_weight * L_rec)

        opt1.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(phase1_params, max_norm=1.0)
        opt1.step()
        sched1.step()

        # Diagnostics
        with torch.no_grad():
            emb_std = emb_stack.std(dim=0).mean().item()
            # Sample within/cross cosine similarities on ENCODER output
            within_cos, cross_cos = [], []
            sample_fams = random.sample(
                list(train_families_avail.keys()),
                min(20, len(train_families_avail)),
            )
            for fam in sample_fams:
                fam_pids = [p for p in train_families_avail[fam] if p in z_mean_dict]
                if len(fam_pids) >= 2:
                    for a, b in combinations(fam_pids[:4], 2):
                        cos = F.cosine_similarity(
                            z_mean_dict[a].unsqueeze(0),
                            z_mean_dict[b].unsqueeze(0),
                        ).item()
                        within_cos.append(cos)
            # Cross-family: sample random pairs from different families
            for _ in range(50):
                f1, f2 = random.sample(sample_fams, 2)
                p1 = random.choice(train_families_avail[f1])
                p2 = random.choice(train_families_avail[f2])
                if p1 in z_mean_dict and p2 in z_mean_dict:
                    cos = F.cosine_similarity(
                        z_mean_dict[p1].unsqueeze(0),
                        z_mean_dict[p2].unsqueeze(0),
                    ).item()
                    cross_cos.append(cos)

        avg_within = sum(within_cos) / max(len(within_cos), 1)
        avg_cross = sum(cross_cos) / max(len(cross_cos), 1)

        # Surface classifier accuracy (diagnostic — should stay low if adversarial works)
        with torch.no_grad():
            surf_acc = (surf_logits.argmax(dim=1) == variant_label_tensor).float().mean().item()

        entry = {
            "phase": 1, "epoch": epoch,
            "supcon_proj": L_supcon_proj.item(),
            "supcon_enc": L_supcon_enc.item(),
            "supcon_seg": L_supcon_seg.item(),
            "var": L_var.item(), "cov": L_cov.item(),
            "ts_pull": L_ts_pull.item(),
            "vel_cos": L_vel_cos.item(),
            "surf_adv": L_surf.item(), "surf_acc": surf_acc,
            "rec": L_rec.item(), "total": loss.item(),
            "embed_std": emb_std,
            "within_cos": avg_within, "cross_cos": avg_cross,
            "lr": sched1.get_last_lr()[0],
        }
        history.append(entry)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  P1 Epoch {epoch:3d} | "
                f"proj={L_supcon_proj.item():.3f} "
                f"enc={L_supcon_enc.item():.3f} "
                f"pull={L_ts_pull.item():.1f} "
                f"vcos={L_vel_cos.item():.3f} "
                f"surf={L_surf.item():.2f}({surf_acc:.0%}) "
                f"std={emb_std:.4f} "
                f"w_cos={avg_within:.3f} "
                f"x_cos={avg_cross:.3f}"
            )

        # Early convergence on ENCODER output (not projection head)
        # Raised from 100→150 to give velocity cosine loss more training time
        if epoch >= 150 and avg_within > 0.95 and avg_cross < 0.1:
            print(f"  Phase 1 converged at epoch {epoch} "
                  f"(within_cos={avg_within:.3f}, cross_cos={avg_cross:.3f})")
            break

    # ------------------------------------------------------------------
    # 4. Phase 2: v5.3 Regime dynamics (encoder FROZEN)
    #    Per-regime MLPs, Gumbel gate (tau annealed), task-family CE, terminal vel.
    # ------------------------------------------------------------------
    if args.epochs_phase2 > 0:
        print("\n" + "=" * 60)
        print(
            f"PHASE 2: v5.3 REGIME DYNAMICS (encoder frozen, K={args.num_substeps}, "
            f"{num_task_families} regimes = task families)",
        )
        print("=" * 60)

        # Freeze encoder to preserve Phase 1 geometry
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.encoder.eval()

        # Pre-compute all latents once (encoder is frozen, won't change)
        print("  Pre-computing frozen latent embeddings...")
        with torch.no_grad():
            frozen_z_full, frozen_z_mean = batch_encode(
                model.encoder, trajectories, train_pids, device,
                chunk_size=args.encode_batch,
            )
        # Detach all to avoid any accidental gradient flow to encoder
        frozen_z_full = {pid: z.detach() for pid, z in frozen_z_full.items()}
        frozen_z_mean = {pid: z.detach() for pid, z in frozen_z_mean.items()}

        # Only dynamics parameters
        phase2_params = list(model.dynamics.parameters())
        opt2 = torch.optim.AdamW(phase2_params, lr=args.lr_phase2, weight_decay=1e-4)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=args.epochs_phase2, eta_min=args.lr_phase2 * 0.01,
        )

        # Linear warmup for first 10 epochs
        WARMUP_EPOCHS = 10
        K = args.num_substeps

        # Pre-concatenate all trajectory start/target pairs (frozen — computed once)
        all_starts = []
        all_targets = []
        for pid in train_pids:
            z = frozen_z_full[pid]  # [T, D_z]
            if z.shape[0] < 2:
                continue
            all_starts.append(z[:-1])
            all_targets.append(z[1:])
        all_starts = torch.cat(all_starts, dim=0)   # [N_total, D_z]
        all_targets = torch.cat(all_targets, dim=0)  # [N_total, D_z]
        dz_true = all_targets - all_starts            # [N_total, D_z]
        print(f"  Batched dynamics: {all_starts.shape[0]} start-target pairs")

        # Task-family label per start position (same for all timesteps of a prompt)
        all_tf_labels: list[int] = []
        for pid in train_pids:
            z = frozen_z_full[pid]
            if z.shape[0] < 2:
                continue
            tf_i = pid_to_tf_idx[pid]
            all_tf_labels.extend([tf_i] * (z.shape[0] - 1))
        all_tf_tensor = torch.tensor(all_tf_labels, device=device, dtype=torch.long)
        assert all_tf_tensor.shape[0] == all_starts.shape[0]

        tf_class_indices = {
            i: (all_tf_tensor == i).nonzero(as_tuple=True)[0]
            for i in range(num_task_families)
        }

        # Terminal states for terminal-velocity loss
        terminal_list = [
            frozen_z_full[pid][-1]
            for pid in train_pids
            if pid in frozen_z_full and frozen_z_full[pid].shape[0] >= 1
        ]
        terminals_stack = (
            torch.stack(terminal_list, dim=0) if terminal_list else None
        )

        # Build points + family labels for velocity contrastive loss (InfoNCE)
        vcoh_points = []
        vcoh_labels = []
        fam_to_idx = {}
        for fam, pids_in_fam in available_families.items():
            fam_pids = [p for p in pids_in_fam if p in frozen_z_full and frozen_z_full[p].shape[0] >= 3]
            if len(fam_pids) < 2:
                continue
            if fam not in fam_to_idx:
                fam_to_idx[fam] = len(fam_to_idx)
            fi = fam_to_idx[fam]
            fam_sample = fam_pids[:3]
            T_min = min(frozen_z_full[p].shape[0] for p in fam_sample)
            for pid in fam_sample:
                z_mid = frozen_z_full[pid][1:T_min - 1]  # [T_mid, D]
                vcoh_points.append(z_mid)
                vcoh_labels.extend([fi] * z_mid.shape[0])
        if vcoh_points:
            vcoh_z = torch.cat(vcoh_points, dim=0)  # [N, D_z]
            vcoh_fam = torch.tensor(vcoh_labels, device=device)
            print(f"  Velocity contrast points: {vcoh_z.shape[0]} across {len(fam_to_idx)} families")
        else:
            vcoh_z = vcoh_fam = None
            print("  WARNING: No velocity contrast points found")

        for epoch in range(1, args.epochs_phase2 + 1):
            model.dynamics.train()

            # LR warmup
            if epoch <= WARMUP_EPOCHS:
                warmup_factor = epoch / WARMUP_EPOCHS
                for pg in opt2.param_groups:
                    pg["lr"] = args.lr_phase2 * warmup_factor

            # Gumbel temperature: tau_high -> tau_low over Phase 2
            denom = max(args.epochs_phase2 - 1, 1)
            tau = (
                args.gate_tau_low
                + (args.gate_tau_high - args.gate_tau_low)
                * (1.0 - (epoch - 1) / denom)
            )

            # Contraction warmup: delay then ramp from 0 to 1
            contract_warmup = max(0.0, min(1.0,
                (epoch - args.contract_delay_epochs)
                / max(args.contract_warmup_epochs, 1),
            ))

            # --- Loss 1 & 2: Batched multi-step dynamics + direction ---
            z_curr = all_starts
            for _ in range(K):
                z_curr = z_curr + model.dynamics.velocity_field(z_curr, tau=tau)

            L_dyn = (z_curr - all_targets).pow(2).sum(dim=-1).mean()

            dz_pred = z_curr - all_starts
            cos = F.cosine_similarity(dz_pred, dz_true, dim=-1)
            L_vel_dir = (1.0 - cos).mean()

            # --- Velocity magnitude matching ---
            mag_pred = dz_pred.norm(dim=-1)
            mag_true = dz_true.norm(dim=-1)
            L_mag = ((mag_pred - mag_true).pow(2) / (mag_true.pow(2) + 1e-6)).mean()

            # --- Velocity contrastive (InfoNCE on velocity directions) ---
            L_vcoh = torch.tensor(0.0, device=device)
            if vcoh_z is not None and args.vcoh_weight > 0:
                batch_size = min(256, vcoh_z.shape[0])
                idx = torch.randperm(vcoh_z.shape[0], device=device)[:batch_size]
                L_vcoh = model.dynamics.velocity_contrastive_loss(
                    vcoh_z[idx], vcoh_fam[idx], temperature=0.1, tau=tau,
                )

            # --- Task-family supervised gate (cross-entropy, class-balanced) ---
            n_gate = min(args.gate_sup_batch, all_starts.shape[0])
            gidx = _balanced_tf_indices(
                tf_class_indices, n_gate, num_task_families, device,
            )
            L_gate_sup = model.dynamics.gate_supervised_loss(
                all_starts[gidx], all_tf_tensor[gidx],
            )

            # --- Regime consistency (same-family gate sequences) ---
            L_reg = torch.tensor(0.0, device=device)
            reg_count = 0
            N_REGIME_PAIRS = 32
            fam_keys = list(train_families_avail.keys())
            for _ in range(N_REGIME_PAIRS):
                fam = random.choice(fam_keys)
                fam_pids = [
                    p for p in train_families_avail[fam]
                    if p in frozen_z_full and frozen_z_full[p].shape[0] >= 2
                ]
                if len(fam_pids) < 2:
                    continue
                p1, p2 = random.sample(fam_pids, 2)
                z1 = frozen_z_full[p1]
                z2 = frozen_z_full[p2]
                t_min = min(z1.shape[0], z2.shape[0])
                if t_min >= 1:
                    L_reg = L_reg + model.dynamics.regime_consistency_loss(
                        z1[:t_min], z2[:t_min],
                    )
                    reg_count += 1
            if reg_count > 0:
                L_reg = L_reg / reg_count

            # --- Transverse-only contraction + gate entropy (class-balanced) ---
            n_s = min(512, all_starts.shape[0])
            idx_s = _balanced_tf_indices(
                tf_class_indices, n_s, num_task_families, device,
            )
            z_s = all_starts[idx_s]
            tangent = dz_true[idx_s]
            tangent_n = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            perturb_scale = 0.1 + random.random() * 0.4
            delta_c = torch.randn_like(z_s)
            delta_c = delta_c - (delta_c * tangent_n).sum(
                dim=-1, keepdim=True,
            ) * tangent_n
            bad_c = delta_c.norm(dim=-1) < 1e-6
            if bad_c.any():
                alt = torch.randn_like(z_s)
                alt = alt - (alt * tangent_n).sum(dim=-1, keepdim=True) * tangent_n
                delta_c = torch.where(bad_c.unsqueeze(-1), alt, delta_c)
            delta_c = (
                delta_c
                / delta_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                * perturb_scale
            )
            with torch.no_grad():
                z_next_c = z_s + model.dynamics.velocity_field(z_s, tau=tau)
            z_pert_next = (z_s + delta_c) + model.dynamics.velocity_field(
                z_s + delta_c, tau=tau,
            )
            ratio_c = (z_pert_next - z_next_c.detach()).norm(
                dim=-1,
            ) / delta_c.norm(dim=-1).clamp(min=1e-8)
            L_contract = F.relu(ratio_c - args.contract_target).mean()
            L_gate_ent = model.dynamics.gate_entropy_loss(z_s)

            # --- Multi-step transverse contraction (attractor basin) ---
            L_ms_contract = torch.tensor(0.0, device=device)
            if args.ms_contract_weight > 0:
                n_ms = min(args.ms_contract_batch, all_starts.shape[0])
                ms_idx = _balanced_tf_indices(
                    tf_class_indices, n_ms, num_task_families, device,
                )
                z_ms = all_starts[ms_idx]
                ms_tangent = dz_true[ms_idx]
                ms_tangent_n = ms_tangent / ms_tangent.norm(
                    dim=-1, keepdim=True,
                ).clamp(min=1e-8)
                ms_delta = torch.randn_like(z_ms)
                ms_delta = ms_delta - (ms_delta * ms_tangent_n).sum(
                    dim=-1, keepdim=True,
                ) * ms_tangent_n
                bad_ms = ms_delta.norm(dim=-1) < 1e-6
                if bad_ms.any():
                    alt_ms = torch.randn_like(z_ms)
                    alt_ms = alt_ms - (alt_ms * ms_tangent_n).sum(
                        dim=-1, keepdim=True,
                    ) * ms_tangent_n
                    ms_delta = torch.where(bad_ms.unsqueeze(-1), alt_ms, ms_delta)
                ms_scale = 0.5 + random.random() * 1.0
                ms_delta = (
                    ms_delta
                    / ms_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    * ms_scale
                )
                z_orig = z_ms
                z_pert_ms = z_ms + ms_delta
                for _ in range(K):
                    z_orig = z_orig + model.dynamics.velocity_field(
                        z_orig, tau=tau,
                    )
                    z_pert_ms = z_pert_ms + model.dynamics.velocity_field(
                        z_pert_ms, tau=tau,
                    )
                d_final = (z_pert_ms - z_orig).norm(dim=-1)
                d_init = ms_delta.norm(dim=-1)
                L_ms_contract = F.relu(
                    d_final / (d_init + 1e-6) - args.ms_contract_target,
                ).mean()

            # --- Tube tightening: same semantic-family pairs, penalize distance growth ---
            L_tighten = torch.tensor(0.0, device=device)
            if args.tighten_weight > 0:
                tighten_z1: list[torch.Tensor] = []
                tighten_z2: list[torch.Tensor] = []
                fam_keys_tighten = list(train_families_avail.keys())
                for _ in range(args.tighten_pairs):
                    fam = random.choice(fam_keys_tighten)
                    fam_pids = [
                        p for p in train_families_avail[fam]
                        if p in frozen_z_full and frozen_z_full[p].shape[0] >= 2
                    ]
                    if len(fam_pids) < 2:
                        continue
                    p1, p2 = random.sample(fam_pids, 2)
                    z1 = frozen_z_full[p1]
                    z2 = frozen_z_full[p2]
                    t_max = min(z1.shape[0], z2.shape[0]) - 2
                    if t_max < 0:
                        continue
                    t_common = random.randint(0, t_max)
                    tighten_z1.append(z1[t_common])
                    tighten_z2.append(z2[t_common])
                if tighten_z1:
                    z1_b = torch.stack(tighten_z1, dim=0)
                    z2_b = torch.stack(tighten_z2, dim=0)
                    z1_end = model.dynamics.multi_step_predict(
                        z1_b, num_substeps=K, tau=tau,
                    )
                    z2_end = model.dynamics.multi_step_predict(
                        z2_b, num_substeps=K, tau=tau,
                    )
                    d_start = (z1_b - z2_b).norm(dim=-1)
                    d_end = (z1_end - z2_end).norm(dim=-1)
                    valid = d_start > 0.5
                    if valid.any():
                        ratio = (d_end[valid] / d_start[valid]).clamp(max=5.0)
                        L_tighten = F.relu(
                            ratio - args.tighten_target,
                        ).mean()

            # --- Terminal velocity penalty ---
            L_terminal = torch.tensor(0.0, device=device)
            if terminals_stack is not None and args.terminal_weight > 0:
                n_t = min(512, terminals_stack.shape[0])
                tidx = torch.randperm(terminals_stack.shape[0], device=device)[:n_t]
                z_t = terminals_stack[tidx]
                L_terminal = model.dynamics.terminal_velocity_loss(
                    z_t, margin=args.terminal_margin, tau=tau,
                )

            # --- Total loss (contraction losses warmed up) ---
            loss = (
                L_dyn
                + args.vel_dir_weight * L_vel_dir
                + args.mag_weight * L_mag
                + args.vcoh_weight * L_vcoh
                + args.gate_sup_weight * L_gate_sup
                + args.regime_weight * L_reg
                + args.contract_weight * contract_warmup * L_contract
                + args.entropy_weight * L_gate_ent
                + args.terminal_weight * L_terminal
                + args.tighten_weight * contract_warmup * L_tighten
                + args.ms_contract_weight * contract_warmup * L_ms_contract
            )

            opt2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(phase2_params, max_norm=1.0)
            opt2.step()
            if epoch > WARMUP_EPOCHS:
                sched2.step()

            # Diagnostics: mean velocity magnitude
            with torch.no_grad():
                sample_z = torch.stack([
                    frozen_z_full[pid][frozen_z_full[pid].shape[0] // 2]
                    for pid in train_pids[:100]
                ])
                mean_vel = model.dynamics.velocity_field(
                    sample_z, tau=tau,
                ).norm(dim=-1).mean().item()

            entry = {
                "phase": 2, "epoch": epoch,
                "dyn": L_dyn.item(),
                "vel_dir": L_vel_dir.item(),
                "mag": L_mag.item(),
                "vcoh": L_vcoh.item(),
                "gate_sup": L_gate_sup.item(),
                "regime": L_reg.item(),
                "contract": L_contract.item(),
                "gate_ent": L_gate_ent.item(),
                "terminal": L_terminal.item(),
                "tighten": L_tighten.item(),
                "ms_contract": L_ms_contract.item(),
                "contract_warmup": contract_warmup,
                "tau": tau,
                "total": loss.item(),
                "mean_vel": mean_vel,
                "lr": opt2.param_groups[0]["lr"],
            }
            history.append(entry)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  P2 Epoch {epoch:3d} | "
                    f"dyn={L_dyn.item():.4f} "
                    f"gsup={L_gate_sup.item():.3f} "
                    f"term={L_terminal.item():.4f} "
                    f"tighten={L_tighten.item():.4f} "
                    f"ms_ctr={L_ms_contract.item():.4f} "
                    f"cwarm={contract_warmup:.2f} "
                    f"vcoh={L_vcoh.item():.3f} "
                    f"tau={tau:.3f} "
                    f"|f|={mean_vel:.4f} "
                    f"lr={opt2.param_groups[0]['lr']:.6f}"
                )

        # --------------------------------------------------------------
        # 4b. Phase 3: Contraction fine-tuning (fresh optimizer)
        #     Same loss structure, contract_warmup=1.0, tau=gate_tau_low.
        # --------------------------------------------------------------
        if args.epochs_phase3 > 0:
            print("\n" + "=" * 60)
            print(
                f"PHASE 3: CONTRACTION FINE-TUNING "
                f"({args.epochs_phase3} epochs, lr={args.lr_phase3}, "
                f"dyn_w={args.dyn_weight_phase3}, ms_ctr_w={args.ms_contract_weight_phase3}, "
                f"K_p3={args.num_substeps_phase3})",
            )
            print("=" * 60)

            opt3 = torch.optim.AdamW(
                phase2_params, lr=args.lr_phase3, weight_decay=1e-4,
            )
            sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt3, T_max=args.epochs_phase3,
                eta_min=args.lr_phase3 * 0.01,
            )
            P3_WARMUP = 5
            tau_p3 = args.gate_tau_low
            K_p3 = args.num_substeps_phase3

            for epoch in range(1, args.epochs_phase3 + 1):
                model.dynamics.train()

                if epoch <= P3_WARMUP:
                    for pg in opt3.param_groups:
                        pg["lr"] = args.lr_phase3 * (epoch / P3_WARMUP)

                # --- L_dyn (dynamics anchor) ---
                z_curr = all_starts
                for _ in range(K):
                    z_curr = z_curr + model.dynamics.velocity_field(
                        z_curr, tau=tau_p3,
                    )
                L_dyn = (z_curr - all_targets).pow(2).sum(dim=-1).mean()

                dz_pred = z_curr - all_starts
                cos = F.cosine_similarity(dz_pred, dz_true, dim=-1)
                L_vel_dir = (1.0 - cos).mean()

                mag_pred = dz_pred.norm(dim=-1)
                mag_true = dz_true.norm(dim=-1)
                L_mag = (
                    (mag_pred - mag_true).pow(2)
                    / (mag_true.pow(2) + 1e-6)
                ).mean()

                # --- Velocity contrastive (InfoNCE) ---
                L_vcoh = torch.tensor(0.0, device=device)
                if vcoh_z is not None and args.vcoh_weight > 0:
                    v_z = model.dynamics.velocity_field(vcoh_z, tau=tau_p3)
                    v_n = F.normalize(v_z, dim=-1)
                    sim_v = v_n @ v_n.T / 0.1
                    mask_pos = vcoh_fam.unsqueeze(0) == vcoh_fam.unsqueeze(1)
                    mask_pos.fill_diagonal_(False)
                    if mask_pos.any():
                        exp_all = torch.exp(
                            sim_v - sim_v.max(dim=1, keepdim=True).values,
                        )
                        exp_pos = exp_all * mask_pos.float()
                        denom_v = exp_all.sum(dim=1) - torch.diag(exp_all)
                        L_vcoh = -(
                            torch.log(
                                exp_pos.sum(dim=1)
                                / denom_v.clamp(min=1e-8)
                                + 1e-8,
                            )
                        ).mean()

                # --- Gate supervision ---
                n_sup = min(args.gate_sup_batch, all_starts.shape[0])
                sup_idx = _balanced_tf_indices(
                    tf_class_indices, n_sup, num_task_families, device,
                )
                z_sup = all_starts[sup_idx]
                tf_sup = all_tf_tensor[sup_idx]
                L_gate_sup = model.dynamics.gate_supervised_loss(
                    z_sup, tf_sup,
                )

                # --- Regime consistency ---
                L_reg = torch.tensor(0.0, device=device)
                if args.regime_weight > 0:
                    reg_fam_keys = list(train_families_avail.keys())
                    random.shuffle(reg_fam_keys)
                    kl_list: list[torch.Tensor] = []
                    for fam in reg_fam_keys[:10]:
                        fam_pids = [
                            p for p in train_families_avail[fam]
                            if p in frozen_z_full
                        ]
                        if len(fam_pids) < 2:
                            continue
                        pair = random.sample(fam_pids, 2)
                        g0 = model.dynamics.gate_probs(
                            frozen_z_mean[pair[0]],
                        )
                        g1 = model.dynamics.gate_probs(
                            frozen_z_mean[pair[1]],
                        )
                        kl = (
                            F.kl_div(
                                g0.log(), g1, reduction="batchmean",
                            )
                            + F.kl_div(
                                g1.log(), g0, reduction="batchmean",
                            )
                        ) / 2.0
                        kl_list.append(kl)
                    if kl_list:
                        L_reg = torch.stack(kl_list).mean()

                L_gate_ent = model.dynamics.gate_entropy_loss(
                    all_starts[sup_idx],
                )

                # --- Multi-step transverse contraction (primary Phase 3 objective) ---
                L_ms_contract = torch.tensor(0.0, device=device)
                if args.ms_contract_weight_phase3 > 0:
                    n_ms = min(args.ms_contract_batch, all_starts.shape[0])
                    ms_idx = _balanced_tf_indices(
                        tf_class_indices, n_ms, num_task_families, device,
                    )
                    z_ms = all_starts[ms_idx]
                    ms_tangent = dz_true[ms_idx]
                    ms_tangent_n = ms_tangent / ms_tangent.norm(
                        dim=-1, keepdim=True,
                    ).clamp(min=1e-8)
                    ms_delta = torch.randn_like(z_ms)
                    ms_delta = ms_delta - (ms_delta * ms_tangent_n).sum(
                        dim=-1, keepdim=True,
                    ) * ms_tangent_n
                    bad_ms = ms_delta.norm(dim=-1) < 1e-6
                    if bad_ms.any():
                        alt_ms = torch.randn_like(z_ms)
                        alt_ms = alt_ms - (alt_ms * ms_tangent_n).sum(
                            dim=-1, keepdim=True,
                        ) * ms_tangent_n
                        ms_delta = torch.where(
                            bad_ms.unsqueeze(-1), alt_ms, ms_delta,
                        )
                    ms_scale = 0.5 + random.random() * 1.0
                    ms_delta = (
                        ms_delta
                        / ms_delta.norm(dim=-1, keepdim=True).clamp(
                            min=1e-8,
                        )
                        * ms_scale
                    )
                    z_orig = z_ms
                    z_pert_ms = z_ms + ms_delta
                    for _ in range(K_p3):
                        z_orig = z_orig + model.dynamics.velocity_field(
                            z_orig, tau=tau_p3,
                        )
                        z_pert_ms = (
                            z_pert_ms
                            + model.dynamics.velocity_field(
                                z_pert_ms, tau=tau_p3,
                            )
                        )
                    d_final = (z_pert_ms - z_orig).norm(dim=-1)
                    d_init = ms_delta.norm(dim=-1)
                    L_ms_contract = F.relu(
                        d_final / (d_init + 1e-6) - args.ms_contract_target,
                    ).mean()

                # --- Terminal velocity penalty ---
                L_terminal = torch.tensor(0.0, device=device)
                if terminals_stack is not None and args.terminal_weight > 0:
                    n_t = min(512, terminals_stack.shape[0])
                    tidx = torch.randperm(
                        terminals_stack.shape[0], device=device,
                    )[:n_t]
                    z_t = terminals_stack[tidx]
                    L_terminal = model.dynamics.terminal_velocity_loss(
                        z_t, margin=args.terminal_margin, tau=tau_p3,
                    )

                # --- Total loss (contraction-dominant, dynamics scaled down) ---
                dw = args.dyn_weight_phase3
                loss_p3 = (
                    dw * L_dyn
                    + dw * args.vel_dir_weight * L_vel_dir
                    + dw * args.mag_weight * L_mag
                    + dw * args.vcoh_weight * L_vcoh
                    + args.gate_sup_weight * L_gate_sup
                    + args.regime_weight * L_reg
                    + args.entropy_weight * L_gate_ent
                    + args.terminal_weight * L_terminal
                    + args.ms_contract_weight_phase3 * L_ms_contract
                )

                opt3.zero_grad()
                loss_p3.backward()
                nn.utils.clip_grad_norm_(phase2_params, max_norm=1.0)
                opt3.step()
                if epoch > P3_WARMUP:
                    sched3.step()

                with torch.no_grad():
                    sample_z = torch.stack([
                        frozen_z_full[pid][
                            frozen_z_full[pid].shape[0] // 2
                        ]
                        for pid in train_pids[:100]
                    ])
                    mean_vel = model.dynamics.velocity_field(
                        sample_z, tau=tau_p3,
                    ).norm(dim=-1).mean().item()

                entry = {
                    "phase": 3, "epoch": epoch,
                    "dyn": L_dyn.item(),
                    "vel_dir": L_vel_dir.item(),
                    "mag": L_mag.item(),
                    "vcoh": L_vcoh.item(),
                    "gate_sup": L_gate_sup.item(),
                    "regime": L_reg.item(),
                    "gate_ent": L_gate_ent.item(),
                    "terminal": L_terminal.item(),
                    "ms_contract": L_ms_contract.item(),
                    "total": loss_p3.item(),
                    "mean_vel": mean_vel,
                    "lr": opt3.param_groups[0]["lr"],
                }
                history.append(entry)

                if epoch % 10 == 0 or epoch == 1:
                    print(
                        f"  P3 Epoch {epoch:3d} | "
                        f"dyn={L_dyn.item():.4f} "
                        f"ms_ctr={L_ms_contract.item():.4f} "
                        f"gsup={L_gate_sup.item():.3f} "
                        f"term={L_terminal.item():.4f} "
                        f"vcoh={L_vcoh.item():.3f} "
                        f"|f|={mean_vel:.4f} "
                        f"lr={opt3.param_groups[0]['lr']:.6f}"
                    )

        # Unfreeze encoder for eval (no training, just for state_dict consistency)
        for p in model.encoder.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    model.eval()
    results: dict = {"training_history": history}

    # Encode all latents, then create two representations:
    #   1) Mean-blended + L2-normalized: for distance metrics (scorecard, hard_neg, motion)
    #      Blending injects the mean (where family structure is strong) into each timestep,
    #      so per-timestep distances reflect semantic structure instead of surface template.
    #   2) Raw: for probes (preserve full information for linear classification)
    blend_alpha = args.blend_alpha
    with torch.no_grad():
        all_pids = list(trajectories.keys())
        z_full_eval, z_mean_eval = batch_encode(
            model.encoder, trajectories, all_pids, device,
            chunk_size=64,
        )
        latent_states = {}
        for pid, z in z_full_eval.items():
            mu = z.mean(dim=0, keepdim=True)  # [1, D]
            z_blend = blend_alpha * mu + (1 - blend_alpha) * z  # [T, D]
            latent_states[pid] = F.normalize(z_blend, dim=-1).cpu()
        # Raw latents preserved separately for probes
        latent_states_raw = {pid: z.cpu() for pid, z in z_full_eval.items()}
    print(f"  Eval blend_alpha={blend_alpha}")

    # --- Latent semantic consistency ---
    print("\n--- Latent Semantic Consistency ---")
    within_dists, cross_dists = [], []
    for fam, pids in available_families.items():
        for a, b in combinations(pids, 2):
            if a in latent_states and b in latent_states:
                T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                within_dists.append(d)

    fam_names = list(available_families.keys())
    for i in range(min(50, len(fam_names))):
        for j in range(i + 1, min(i + 11, len(fam_names))):
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

    # --- Retrieval accuracy ---
    print("\n--- Retrieval Accuracy ---")
    ret = retrieval_accuracy(latent_states, PID_TO_FAMILY)
    print(f"  Accuracy:  {ret.accuracy:.2%} ({ret.num_queries} queries)")
    print(f"  MRR:       {ret.mean_reciprocal_rank:.4f}")
    results["retrieval"] = {
        "accuracy": ret.accuracy, "mrr": ret.mean_reciprocal_rank,
        "num_queries": ret.num_queries,
    }

    # --- Motion similarity ---
    print("\n--- Latent Motion Similarity ---")
    ms = motion_similarity_eval(latent_states, available_families)
    print(f"  Within velocity cosine:  {ms.within_velocity_cosine:.4f}")
    print(f"  Cross velocity cosine:   {ms.cross_velocity_cosine:.4f}")
    print(f"  Cosine separation:       {ms.velocity_cosine_separation:.2f}")
    results["motion_similarity"] = asdict(ms)

    # --- Hard-negative resistance ---
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

    # --- Per-family reconstruction ---
    print("\n--- Per-Family Reconstruction ---")
    rec_errors: dict[str, float] = {}
    with torch.no_grad():
        for pid in trajectories:
            h = trajectories[pid].states.to(device)
            z = model.encoder(h)
            h_hat = model.decoder(z)
            rec_errors[pid] = reconstruction_loss(h, h_hat).item()

    avg_rec = sum(rec_errors.values()) / max(len(rec_errors), 1)
    print(f"  {len(rec_errors)} runs, overall avg MSE = {avg_rec:.6f}")
    results["avg_reconstruction_mse"] = avg_rec

    # --- Shuffle control ---
    print("\n--- Shuffle Control ---")
    all_pids_available = sorted(latent_states.keys())
    rng = random.Random(42)
    shuffle_within, shuffle_cross = [], []
    for _ in range(200):
        a, b = rng.sample(all_pids_available, 2)
        T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
        d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
        fam_a = PID_TO_FAMILY.get(a, "?")
        fam_b = PID_TO_FAMILY.get(b, "?")
        if fam_a == fam_b:
            shuffle_within.append(d)
        else:
            shuffle_cross.append(d)

    sw = sum(shuffle_within) / max(len(shuffle_within), 1)
    sc = sum(shuffle_cross) / max(len(shuffle_cross), 1)
    shuffle_sep = sc / max(sw, 1e-8)
    print(f"  Shuffle separation: {shuffle_sep:.2f}")
    print(f"  True separation:    {separation:.2f}")
    results["shuffle_control"] = {
        "shuffle_within": sw, "shuffle_cross": sc,
        "shuffle_separation": shuffle_sep,
    }

    # --- Semantic Scorecard ---
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

    # --- Cosine similarity matrix (top 10 families, diagnostic) ---
    print("\n--- Cosine Similarity Matrix (sample) ---")
    sample_pids = []
    sample_fams_list = list(available_families.keys())[:10]
    for fam in sample_fams_list:
        sample_pids.append(available_families[fam][0])  # 1 per family
    cos_matrix: dict[str, dict[str, float]] = {}
    for a in sample_pids:
        cos_matrix[a] = {}
        za = latent_states[a].mean(dim=0)
        for b in sample_pids:
            zb = latent_states[b].mean(dim=0)
            cos_matrix[a][b] = F.cosine_similarity(
                za.unsqueeze(0), zb.unsqueeze(0)
            ).item()
    results["cosine_similarity_matrix"] = cos_matrix

    # Print a compact version
    print("     " + "  ".join(f"{p[:8]:>8}" for p in sample_pids))
    for a in sample_pids:
        row = [f"{cos_matrix[a][b]:.3f}" for b in sample_pids]
        print(f"{a[:8]:>8} " + "  ".join(f"{v:>8}" for v in row))

    # --- Dynamics motion similarity (uses learned velocity field) ---
    if args.epochs_phase2 > 0:
        print("\n--- Dynamics Motion Similarity (learned f(z)) ---")
        dms = dynamics_motion_similarity_eval(
            latent_states_raw, available_families, model.dynamics,
        )
        print(f"  Within velocity cosine (dyn):  {dms.within_velocity_cosine:.4f}")
        print(f"  Cross velocity cosine (dyn):   {dms.cross_velocity_cosine:.4f}")
        print(f"  Cosine separation (dyn):       {dms.velocity_cosine_separation:.2f}")
        results["dynamics_motion_similarity"] = asdict(dms)

    # --- Dynamics diagnostics ---
    if args.epochs_phase2 > 0:
        print("\n--- Dynamics Diagnostics ---")
        with torch.no_grad():
            all_vel = []
            all_vel_terminal = []
            all_dyn_errors = []
            for pid in list(latent_states_raw.keys())[:200]:
                z = latent_states_raw[pid].to(device)
                vel = model.dynamics.velocity_field(z)
                all_vel.append(vel.norm(dim=-1).mean().item())
                # Terminal velocity (last position)
                all_vel_terminal.append(vel[-1].norm().item())
                # Multi-step dynamics error
                if z.shape[0] >= 2:
                    err = model.dynamics.dynamics_consistency_loss(z, args.num_substeps).item()
                    all_dyn_errors.append(err)
        mean_vel = sum(all_vel) / max(len(all_vel), 1)
        mean_vel_term = sum(all_vel_terminal) / max(len(all_vel_terminal), 1)
        mean_err = sum(all_dyn_errors) / max(len(all_dyn_errors), 1)
        print(f"  Mean ||f(z)||:          {mean_vel:.4f}")
        print(f"  Mean ||f(z)|| terminal: {mean_vel_term:.4f}")
        print(f"  Mean dynamics error:    {mean_err:.4f}")
        results["dynamics_diagnostics"] = {
            "mean_velocity_norm": mean_vel,
            "mean_velocity_norm_terminal": mean_vel_term,
            "mean_dynamics_error": mean_err,
        }

    # --- Regime-specific metrics (v5.3) ---
    if args.epochs_phase2 > 0:
        print("\n--- Regime Metrics (v5.3) ---")
        regime_metrics = compute_regime_metrics(
            model.dynamics,
            latent_states_raw,
            available_families,
            device,
            args.num_substeps,
        )
        print(
            f"  Regime KL within/cross: {regime_metrics['regime_kl_within_mean']:.4f} / "
            f"{regime_metrics['regime_kl_cross_mean']:.4f} "
            f"(ratio={regime_metrics['regime_consistency_ratio']:.2f})",
        )
        print(
            f"  Edit dist within/cross: {regime_metrics['edit_distance_within_mean']:.2f} / "
            f"{regime_metrics['edit_distance_cross_mean']:.2f} "
            f"(ratio={regime_metrics['edit_distance_ratio']:.2f})",
        )
        print(f"  Mean gate entropy:      {regime_metrics['gate_entropy_mean']:.4f}")
        print(f"  Regime utilization:    {regime_metrics['regime_utilization']}")
        results["regime_metrics"] = regime_metrics

        with torch.no_grad():
            g_correct = 0
            g_tokens = 0
            for pid, z in latent_states_raw.items():
                meta = _PID_META.get(pid)
                if meta is None:
                    continue
                tfn = meta["task_family"]
                if tfn not in tf_to_idx:
                    continue
                gt = tf_to_idx[tfn]
                z = z.to(device)
                pred = model.dynamics.gate_logits(z).argmax(dim=-1)
                g_correct += (pred == gt).sum().item()
                g_tokens += z.shape[0]
        gate_tf_acc = g_correct / max(g_tokens, 1)
        print(f"  Gate argmax vs task-family: {gate_tf_acc:.2%} ({g_tokens} tokens)")
        results["gate_task_family_accuracy"] = gate_tf_acc

    # --- Tube / manifold analysis ---
    tube_results = run_tube_analysis(
        model, latent_states, latent_states_raw, available_families, device,
        run_interventions=(args.epochs_phase2 > 0),
    )
    results["tube_analysis"] = tube_results

    # --- Latent probes ---
    probe_results = run_latent_probes(model, latent_states_raw, PID_TO_FAMILY, available_families, device)
    results["latent_probes"] = probe_results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SDQ v5.3 SUMMARY")
    print("=" * 60)
    print(f"Phase 1 epochs:       {args.epochs_phase1}")
    print(f"Phase 2 epochs:       {args.epochs_phase2}")
    print(f"Phase 3 epochs:       {args.epochs_phase3}")
    print(f"Normalization:        {norm_mode}")
    print(f"Temperature:          {args.temperature}")
    print(f"Latent separation:    {separation:.2f}")
    print(f"Retrieval accuracy:   {ret.accuracy:.2%}")
    print(f"Motion cos sep:       {ms.velocity_cosine_separation:.2f}")
    print(f"Scorecard overall:    {scorecard.overall_score():.1f}/100")
    print(f"--- Dynamics ---")
    if "dynamics_diagnostics" in results:
        dd = results["dynamics_diagnostics"]
        print(f"  Mean ||f(z)||:      {dd['mean_velocity_norm']:.4f}")
        print(f"  Mean ||f(z)|| term: {dd['mean_velocity_norm_terminal']:.4f}")
        print(f"  Dynamics error:     {dd['mean_dynamics_error']:.4f}")
    if "regime_metrics" in results:
        rm = results["regime_metrics"]
        print(f"  Regime KL ratio:    {rm['regime_consistency_ratio']:.2f}")
        print(f"  Edit dist ratio:    {rm['edit_distance_ratio']:.2f}")
        print(f"  Gate entropy mean:  {rm['gate_entropy_mean']:.4f}")
    if "gate_task_family_accuracy" in results:
        print(f"  Gate vs task-fam:   {results['gate_task_family_accuracy']:.2%}")
    if "dynamics_motion_similarity" in results:
        print(f"  Dyn vel cos sep:    {results['dynamics_motion_similarity']['velocity_cosine_separation']:.2f}")
    print(f"--- Tube Analysis ---")
    tc = tube_results.get("coherence", {})
    tt = tube_results.get("transverse", {})
    ts = tube_results.get("reasoning_stage", {})
    tsep = tube_results.get("separation", {})
    print(f"  Tube tightening:    {tc.get('tube_tightening_ratio', 0):.3f}")
    print(f"  Within spread:      {tc.get('mean_within_spread', 0):.4f}")
    print(f"  Cross spread:       {tc.get('mean_cross_spread', 0):.4f}")
    print(f"  Transverse contract:{tt.get('transverse_contraction_ratio', 0):.3f}")
    print(f"  Along-tube mono:    {ts.get('along_tube_monotonicity', 0):.3f}")
    print(f"  Stage probe acc:    {ts.get('timestep_probe_accuracy', 0):.1%}")
    print(f"  Tube margin:        {tsep.get('mean_classification_margin', 0):.4f}")
    print(f"  Tube overlap:       {tsep.get('tube_overlap_fraction', 0):.1%}")
    if "intervention" in tube_results:
        ti = tube_results["intervention"]
        print(f"  Trans recovery:     {ti['transverse_recovery_rate']:.1%}")
        print(f"  Cross persistence:  {ti['cross_tube_persistence_rate']:.1%}")
    if "dynamics_tube" in tube_results:
        dtr = tube_results["dynamics_tube"]
        print(f"  --- Dynamics Tube (Exp 6) ---")
        print(f"  Dyn tightening:     {dtr['dynamics_tightening_ratio']:.3f}")
        if dtr["per_step_vel_cosine"]:
            mean_vc = sum(dtr["per_step_vel_cosine"]) / len(dtr["per_step_vel_cosine"])
            print(f"  Dyn vel cos (mean): {mean_vc:.3f}")
        if dtr["per_step_classification_acc"]:
            mean_acc = sum(dtr["per_step_classification_acc"]) / len(dtr["per_step_classification_acc"])
            print(f"  Dyn 1-NN (mean):    {mean_acc:.1%}")
    print(f"--- Latent Probes ---")
    if "task_family" in probe_results:
        print(f"  Task family (test): {probe_results['task_family']['test_accuracy']:.1%}")
    if "semantic_group" in probe_results:
        print(f"  Semantic grp (test):{probe_results['semantic_group']['test_accuracy']:.1%}")
    if "surface_variant" in probe_results:
        print(f"  Surface var (null): {probe_results['surface_variant']['test_accuracy']:.1%}")
    if "answer_prediction" in probe_results:
        print(f"  Answer pred (test): {probe_results['answer_prediction']['test_accuracy']:.1%}")
    print("=" * 60)

    results["summary"] = {
        "epochs_phase1": args.epochs_phase1,
        "epochs_phase2": args.epochs_phase2,
        "epochs_phase3": args.epochs_phase3,
        "norm_mode": norm_mode,
        "temperature": args.temperature,
        "blend_alpha": args.blend_alpha,
        "latent_separation": separation,
        "retrieval_accuracy": ret.accuracy,
        "retrieval_mrr": ret.mean_reciprocal_rank,
        "motion_cosine_separation": ms.velocity_cosine_separation,
        "dynamics_vel_cos_sep": results.get("dynamics_motion_similarity", {}).get("velocity_cosine_separation", 0),
        "dynamics_mean_vel": results.get("dynamics_diagnostics", {}).get("mean_velocity_norm", 0),
        "dynamics_mean_vel_terminal": results.get("dynamics_diagnostics", {}).get("mean_velocity_norm_terminal", 0),
        "dynamics_mean_error": results.get("dynamics_diagnostics", {}).get("mean_dynamics_error", 0),
        "scorecard": scorecard.overall_score(),
        "tube_tightening_ratio": tc.get("tube_tightening_ratio", 0),
        "tube_within_spread": tc.get("mean_within_spread", 0),
        "tube_cross_spread": tc.get("mean_cross_spread", 0),
        "transverse_contraction_ratio": tt.get("transverse_contraction_ratio", 0),
        "along_tube_monotonicity": ts.get("along_tube_monotonicity", 0),
        "timestep_probe_accuracy": ts.get("timestep_probe_accuracy", 0),
        "tube_classification_margin": tsep.get("mean_classification_margin", 0),
        "tube_overlap_fraction": tsep.get("tube_overlap_fraction", 0),
        "transverse_recovery_rate": tube_results.get("intervention", {}).get("transverse_recovery_rate", 0),
        "cross_tube_persistence_rate": tube_results.get("intervention", {}).get("cross_tube_persistence_rate", 0),
        "dynamics_tightening_ratio": tube_results.get("dynamics_tube", {}).get("dynamics_tightening_ratio", 0),
        "dynamics_tube_vel_cosine_mean": (
            sum(tube_results.get("dynamics_tube", {}).get("per_step_vel_cosine", [0])) /
            max(len(tube_results.get("dynamics_tube", {}).get("per_step_vel_cosine", [0])), 1)
        ),
        "dynamics_tube_classification_mean": (
            sum(tube_results.get("dynamics_tube", {}).get("per_step_classification_acc", [0])) /
            max(len(tube_results.get("dynamics_tube", {}).get("per_step_classification_acc", [0])), 1)
        ),
        "task_family_probe_acc": probe_results.get("task_family", {}).get("test_accuracy", 0),
        "semantic_group_probe_acc": probe_results.get("semantic_group", {}).get("test_accuracy", 0),
        "surface_variant_probe_acc": probe_results.get("surface_variant", {}).get("test_accuracy", 0),
        "answer_prediction_probe_acc": probe_results.get("answer_prediction", {}).get("test_accuracy", 0),
        "regime_consistency_ratio": results.get("regime_metrics", {}).get("regime_consistency_ratio", 0),
        "regime_edit_distance_ratio": results.get("regime_metrics", {}).get("edit_distance_ratio", 0),
        "regime_gate_entropy_mean": results.get("regime_metrics", {}).get("gate_entropy_mean", 0),
        "num_regimes": num_task_families,
        "gate_task_family_accuracy": results.get("gate_task_family_accuracy", 0),
    }
    return results, model


def main():
    parser = argparse.ArgumentParser(
        description="SDQ v5.3: contrastive geometry + supervised regime dynamics",
    )
    # Phase 1 args
    parser.add_argument("--epochs-phase1", type=int, default=200)
    parser.add_argument("--lr-phase1", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="SupCon temperature (default 0.1)")
    parser.add_argument("--rec-weight", type=float, default=0.01,
                        help="Reconstruction loss weight in Phase 1 (default 0.01)")
    parser.add_argument("--blend-alpha", type=float, default=0.9,
                        help="Eval mean-blend ratio: z_eval(t) = α*mean + (1-α)*z(t). "
                             "Higher → more mean-centric (default 0.9)")
    # Phase 2 args (dynamics training)
    parser.add_argument("--epochs-phase2", type=int, default=400)
    parser.add_argument("--lr-phase2", type=float, default=5e-4)
    parser.add_argument("--vel-dir-weight", type=float, default=0.5,
                        help="Velocity direction loss weight in Phase 2 (default 0.5)")
    parser.add_argument("--mag-weight", type=float, default=0.5,
                        help="Velocity magnitude loss weight (default 0.5)")
    parser.add_argument("--vcoh-weight", type=float, default=5.0,
                        help="Velocity contrastive (InfoNCE) weight — v5.3 default 5.0")
    parser.add_argument("--num-regimes", type=int, default=6,
                        help="Ignored if Phase 2 runs: regimes = task families in train set")
    parser.add_argument("--gate-sup-weight", type=float, default=5.0,
                        help="Task-family supervised gate CE weight (default 5.0)")
    parser.add_argument("--gate-sup-batch", type=int, default=4096,
                        help="Max points per epoch for gate supervision (default 4096)")
    parser.add_argument("--gate-tau-high", type=float, default=1.0,
                        help="Gumbel temperature at start of Phase 2 (default 1.0)")
    parser.add_argument("--gate-tau-low", type=float, default=0.1,
                        help="Gumbel temperature at end of Phase 2 (default 0.1)")
    parser.add_argument("--terminal-weight", type=float, default=2.0,
                        help="Terminal velocity ReLU penalty weight (default 2.0)")
    parser.add_argument("--terminal-margin", type=float, default=0.5,
                        help="Margin for terminal ||f(z)|| (default 0.5)")
    parser.add_argument("--regime-weight", type=float, default=2.0,
                        help="Regime consistency (same-family gate KL) weight (default 2.0)")
    parser.add_argument("--contract-weight", type=float, default=0,
                        help="Single-step transverse contraction weight (iter5 default 0, disabled)")
    parser.add_argument("--entropy-weight", type=float, default=0.5,
                        help="Gate entropy regularizer weight (v5.3 default 0.5)")
    parser.add_argument("--contract-target", type=float, default=0.7,
                        help="Target ratio for single-step transverse contraction (default 0.7)")
    parser.add_argument("--tighten-weight", type=float, default=0,
                        help="Tube tightening loss weight (iter5 default 0, disabled)")
    parser.add_argument("--tighten-target", type=float, default=0.8,
                        help="Target ratio d_end/d_start for tube tightening (default 0.8)")
    parser.add_argument("--ms-contract-weight", type=float, default=3.0,
                        help="Multi-step transverse contraction weight (iter5 default 3.0)")
    parser.add_argument("--contract-delay-epochs", type=int, default=250,
                        help="Epochs of pure dynamics before contraction begins (iter5 default 250)")
    parser.add_argument("--contract-warmup-epochs", type=int, default=150,
                        help="Epochs to ramp contraction from 0 to full after delay (iter5 default 150)")
    parser.add_argument("--ms-contract-target", type=float, default=0.5,
                        help="Target d_final/d_init over K substeps (default 0.5)")
    parser.add_argument("--ms-contract-batch", type=int, default=128,
                        help="Batch size for multi-step contraction (default 128)")
    parser.add_argument("--tighten-pairs", type=int, default=32,
                        help="Same-family pairs per epoch for tube tightening (default 32)")
    parser.add_argument("--num-substeps", type=int, default=10,
                        help="Euler sub-steps per observed interval for multi-step dynamics (default 10)")
    # Phase 3 args (contraction fine-tuning)
    parser.add_argument("--epochs-phase3", type=int, default=200,
                        help="Phase 3 contraction fine-tuning epochs (default 200, 0 to skip)")
    parser.add_argument("--lr-phase3", type=float, default=1e-4,
                        help="Phase 3 peak learning rate (default 1e-4)")
    parser.add_argument("--ms-contract-weight-phase3", type=float, default=15.0,
                        help="L_ms_contract weight in Phase 3 (default 15.0)")
    parser.add_argument("--dyn-weight-phase3", type=float, default=0.5,
                        help="Scale factor for dynamics losses (L_dyn, vel_dir, mag, vcoh) in Phase 3")
    parser.add_argument("--num-substeps-phase3", type=int, default=25,
                        help="Contraction rollout substeps in Phase 3 (default 25, eval uses 50)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .safetensors checkpoint to skip Phase 1")
    # Model args
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--gauge-dim", type=int, default=32)
    parser.add_argument("--encode-batch", type=int, default=64,
                        help="Batch size for encoding (default 64)")
    # Data args
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--layer-range", type=int, nargs=2, metavar=("LO", "HI"), default=None)
    parser.add_argument("--norm-mode", type=str, default="standardize",
                        choices=["unit_norm", "center_scale", "standardize"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="artifacts/sdq_v5_3_results.json")
    args = parser.parse_args()

    t0 = time.time()
    results, model = run_training(args)
    elapsed = time.time() - t0

    results["elapsed_seconds"] = elapsed
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    # Save model checkpoint
    ckpt_path = out_path.with_stem(out_path.stem + "_model")
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, str(ckpt_path.with_suffix(".safetensors")))
    print(f"Model checkpoint saved to {ckpt_path.with_suffix('.safetensors')}")


if __name__ == "__main__":
    main()
