#!/usr/bin/env python3
"""SDQ v6: dual-dynamics recovery model.

Phase 1: Contrastive representation learning (same as v5.3).
Phase 2: Nominal dynamics training — per-regime velocity MLPs, Gumbel-softmax
  gate, task-family supervised gate loss, terminal velocity penalty.
  No contraction losses.  Goal: clean rollout field before asking it to repair.
Phase 3: Recovery dynamics training — explicit off-tube perturbation recovery
  with a dedicated corrector network.  Nominal dynamics frozen.
Phase 4: Conservative joint fine-tune — both nominal and recovery trainable,
  nominal fidelity kept as a hard anchor.

Usage:
    python run_sdq_v6.py [--device cuda] [--epochs-phase1 200] [--epochs-phase2 400]
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
    TubeAwareDynamics,
    TubeGeometry,
)
from sdq.losses import reconstruction_loss
from sdq.losses.velocity import latent_velocity_cosine_loss
from sdq.eval.retrieval import retrieval_accuracy
from sdq.eval.motion_similarity import motion_similarity_eval, dynamics_motion_similarity_eval
from sdq.eval.hard_negatives import hard_negative_test
from sdq.eval.semantic_scorecard import compute_scorecard
from sdq.eval.latent_probe import LatentProbe, train_probe, format_probe_result
from sdq.eval.tube_analysis import full_tube_analysis, format_tube_report
from sdq.eval.attractor_analysis import AttractorAnalysis, format_attractor_report


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
    dynamics: TubeAwareDynamics,
    latent_states_raw: dict[str, torch.Tensor],
    available_families: dict[str, list[str]],
    device: torch.device,
    num_substeps: int,
    max_pairs_within: int = 40,
    max_pairs_cross: int = 80,
) -> dict:
    dynamics.eval()
    eps = 1e-8
    within_kls: list[float] = []
    cross_kls: list[float] = []

    for fam, pids in available_families.items():
        pids_ok = [p for p in pids if p in latent_states_raw]
        if len(pids_ok) < 2:
            continue
        pairs_done = 0
        for i, a in enumerate(pids_ok):
            for b in pids_ok[i + 1:]:
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
    regime_consistency_ratio = mean_c / max(mean_w, 1e-8) if within_kls else 0.0

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
            for sb in seqs[i + 1:]:
                within_edits.append(float(_levenshtein_int_seq(sa, sb)))

    for _ in range(max_pairs_cross):
        if len(fam_list) < 2:
            break
        fa, fb = rng.sample(fam_list, 2)
        pa_l = [p for p in available_families[fa] if p in latent_states_raw]
        pb_l = [p for p in available_families[fb] if p in latent_states_raw]
        if not pa_l or not pb_l:
            continue
        a, b = rng.choice(pa_l), rng.choice(pb_l)
        za = latent_states_raw[a].to(device)
        zb = latent_states_raw[b].to(device)
        with torch.no_grad():
            sa = dynamics.gate_probs(za).argmax(dim=-1).cpu().tolist()
            sb = dynamics.gate_probs(zb).argmax(dim=-1).cpu().tolist()
        cross_edits.append(float(_levenshtein_int_seq(sa, sb)))

    mean_we = sum(within_edits) / max(len(within_edits), 1)
    mean_ce = sum(cross_edits) / max(len(cross_edits), 1)
    edit_distance_ratio = mean_ce / max(mean_we, 1e-8) if within_edits else 0.0

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
                zt = z[t:t + 1]
                dom = dynamics.gate_probs(zt).argmax(dim=-1).item()
                pred = dynamics.multi_step_predict(zt, num_substeps)
                err = (pred[0] - z[t + 1]).pow(2).sum().item()
                bucket_err[dom].append(err)

    t_max = max((latent_states_raw[p].shape[0] for p in latent_states_raw), default=0)
    hist = torch.zeros(t_max, k, device=device)
    for pid, z in list(latent_states_raw.items())[:200]:
        z = z.to(device)
        with torch.no_grad():
            argm = dynamics.gate_probs(z).argmax(dim=-1)
        for t in range(z.shape[0]):
            if t < t_max:
                hist[t, argm[t]] += 1.0

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
# SupCon, gradient reversal, projection head, surface classifier
# ---------------------------------------------------------------------------
def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    hard_neg_mask: torch.Tensor | None = None,
    hard_neg_weight: float = 3.0,
) -> torch.Tensor:
    N = embeddings.shape[0]
    device = embeddings.device
    sim = embeddings @ embeddings.T / temperature
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = label_eq & self_mask
    num_pos = pos_mask.sum(dim=1)
    valid = num_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    denom_weight = self_mask.float()
    if hard_neg_mask is not None:
        denom_weight = denom_weight + (hard_neg_mask.float() * (hard_neg_weight - 1.0))
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    log_exp = sim - sim_max
    exp_sum = (torch.exp(log_exp) * denom_weight).sum(dim=1, keepdim=True)
    log_prob = log_exp - torch.log(exp_sum + 1e-12)
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
    pos_log_prob = pos_log_prob[valid] / num_pos[valid]
    return -pos_log_prob.mean()


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class SurfaceClassifier(nn.Module):
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


class ProjectionHead(nn.Module):
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
# Model bundle
# ---------------------------------------------------------------------------
class SDQv6Model(nn.Module):
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
# Batch encoding
# ---------------------------------------------------------------------------
def batch_encode(
    encoder: nn.Module,
    trajectories: dict[str, Trajectory],
    pids: list[str],
    device: torch.device,
    chunk_size: int = 64,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
        z_batch = encoder(padded)
        for i, pid in enumerate(chunk):
            z_i = z_batch[i, :lengths[i]]
            z_full[pid] = z_i
            z_mean[pid] = z_i.mean(dim=0)
    return z_full, z_mean


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def run_tube_analysis(
    model: SDQv6Model,
    latent_states: dict[str, torch.Tensor],
    latent_states_raw: dict[str, torch.Tensor],
    available_families: dict[str, list[str]],
    device: torch.device,
    run_interventions: bool = True,
) -> dict:
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
    model: SDQv6Model,
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
# Recovery helpers / evaluation (v6-specific)
# ---------------------------------------------------------------------------
def scheduled_recovery_horizon(
    epoch: int,
    total_epochs: int,
    final_horizon: int,
) -> int:
    """Curriculum from short recovery rollouts to eval-matched horizons."""
    stages = [1, 5, 10, 25, 50]
    stages = [s for s in stages if s <= final_horizon]
    if final_horizon not in stages:
        stages.append(final_horizon)
    if not stages:
        return max(final_horizon, 1)
    progress = 0.0 if total_epochs <= 1 else (epoch - 1) / (total_epochs - 1)
    idx = min(int(progress * len(stages)), len(stages) - 1)
    return stages[idx]


def scheduled_recovery_scales(epoch: int, total_epochs: int) -> list[float]:
    """Curriculum from moderate perturbations to full eval scale range."""
    progress = 0.0 if total_epochs <= 1 else (epoch - 1) / (total_epochs - 1)
    if progress < 0.25:
        return [0.7, 1.0]
    if progress < 0.5:
        return [0.5, 0.7, 1.0]
    if progress < 0.75:
        return [0.3, 0.7, 1.0, 1.5]
    return [0.3, 0.5, 1.0, 1.5, 2.0]


def summarize_attractor_report(report) -> dict:
    """Convert attractor report dataclasses into JSON-safe primitives."""
    return {
        "num_attractors": report.num_attractors,
        "num_stable": report.num_stable,
        "num_unstable": report.num_unstable,
        "answer_attractor_purity": report.answer_attractor_purity,
        "mean_convergence_steps": report.mean_convergence_steps,
        "basin_separation_score": report.basin_separation_score,
        "fixed_points": [
            {
                "velocity_norm": fp.velocity_norm,
                "max_eigenvalue": fp.max_eigenvalue,
                "is_stable": fp.is_stable,
                "basin_size": fp.basin_size,
                "associated_answers": fp.associated_answers,
            }
            for fp in report.fixed_points
        ],
        "trajectory_stabilities": [
            {
                "max_lyapunov": stab.max_lyapunov,
                "is_stable": stab.is_stable,
                "convergence_rate": stab.convergence_rate,
                "nearest_attractor_idx": stab.nearest_attractor_idx,
                "attractor_distance": stab.attractor_distance,
            }
            for stab in report.trajectory_stabilities
        ],
        "raw": report.raw,
    }


def directional_progress_penalty(
    chosen_progress: torch.Tensor,
    expected_progress: torch.Tensor,
    backward_weight: float,
    overshoot_weight: float,
    overshoot_slack: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize backward jumps much more than mild forward overshoot."""
    backward = F.relu(expected_progress - chosen_progress)
    overshoot = F.relu(chosen_progress - (expected_progress + overshoot_slack))
    penalty = (
        backward_weight * backward + overshoot_weight * overshoot
    ).mean()
    return penalty, backward.mean(), overshoot.mean()


@torch.no_grad()
def run_recovery_analysis(
    dynamics: TubeAwareDynamics,
    latent_states_raw: dict[str, torch.Tensor],
    families: dict[str, list[str]],
    tube_geo: TubeGeometry,
    family_to_idx: dict[str, int],
    device: torch.device,
    perturbation_scales: list[float] | None = None,
    rollout_steps: int = 50,
    num_samples: int = 60,
    progress_window: int | None = None,
    backward_window: int = 0,
    forward_window: int | None = None,
    run_attractor_analysis: bool = False,
    attractor_num_samples: int = 40,
) -> dict:
    """Recovery analysis across perturbation scales.

    Returns a dict suitable for direct inclusion in the results JSON.
    """
    if perturbation_scales is None:
        perturbation_scales = [0.3, 0.5, 1.0, 1.5, 2.0]

    dynamics.eval()
    had_geo = dynamics.has_tube_geometry
    dynamics.set_tube_geometry(tube_geo)

    if forward_window is None:
        forward_window = progress_window if progress_window is not None else 0

    fam_names = sorted(families.keys())
    valid_fams = [
        f for f in fam_names
        if sum(1 for p in families[f] if p in latent_states_raw) >= 2
    ]

    null_corrections: list[float] = []

    pid_to_family_idx = {
        pid: family_to_idx[fam]
        for fam, pids in families.items()
        if fam in family_to_idx
        for pid in pids
    }

    def analyze_mode(force_family: bool) -> dict:
        per_scale: dict[str, dict] = {}
        same_family_rates: list[float] = []
        wrong_family_rates: list[float] = []

        for scale in perturbation_scales:
            recoveries = []
            distances = []
            identity_ok = []
            same_family_anchor = []
            wrong_family_anchor = []
            backward_progress = []
            overshoot_progress = []
            # Mean perturbed-vs-reference (teacher) divergence per rollout step
            div_curve_sum = None
            ratio_curve_sum = None
            div_curve_count = 0
            samples_done = 0

            for fam in valid_fams:
                if samples_done >= num_samples:
                    break
                if fam not in family_to_idx:
                    continue
                fam_idx = family_to_idx[fam]
                pids = [p for p in families[fam] if p in latent_states_raw]
                trajs = [latent_states_raw[p].to(device) for p in pids]
                T_min = min(z.shape[0] for z in trajs)
                mean_path = torch.stack([z[:T_min] for z in trajs]).mean(dim=0)
                dz = mean_path[1:] - mean_path[:-1]
                tangents = dz / dz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                t_mid = T_min // 2

                for pid in pids[:3]:
                    if samples_done >= num_samples:
                        break
                    z = latent_states_raw[pid][:T_min].to(device)
                    z_start = z[t_mid]
                    tang = tangents[min(t_mid, tangents.shape[0] - 1)]

                    delta = torch.randn_like(z_start)
                    delta = delta - (delta @ tang) * tang
                    delta = delta / delta.norm().clamp(min=1e-8) * scale

                    z_pert = z_start + delta

                    fam_tensor = torch.tensor(fam_idx, device=device)
                    progress = torch.tensor(t_mid, dtype=torch.long, device=device)
                    orig_traj = dynamics.rollout(
                        z_start,
                        rollout_steps,
                        family_ids=fam_tensor,
                        progress_hint=progress,
                        backward_window=backward_window,
                        forward_window=forward_window,
                    )

                    pert_states = [z_pert]
                    z_roll = z_pert.unsqueeze(0)
                    step_progress = progress.clone()
                    local_same = []
                    local_wrong = []
                    local_backward = []
                    local_overshoot = []
                    for _ in range(rollout_steps - 1):
                        z_nominal = dynamics.nominal_step(z_roll)
                        z_roll, ctx = dynamics.recover_step(
                            z_nominal,
                            family_ids=fam_tensor if force_family else None,
                            progress_hint=step_progress,
                            backward_window=backward_window,
                            forward_window=forward_window,
                            return_context=True,
                        )
                        global_ctx = tube_geo.contextualize(z_roll)
                        matched_family = (
                            global_ctx["family_ids"].reshape(-1)[0].item() == fam_idx
                        )
                        local_same.append(float(matched_family))
                        local_wrong.append(float(not matched_family))
                        chosen_progress = int(ctx["progress"].reshape(-1)[0].item())
                        prev_progress = int(step_progress.reshape(-1)[0].item())
                        local_backward.append(float(chosen_progress < prev_progress))
                        local_overshoot.append(float(chosen_progress > prev_progress + 2))
                        step_progress = torch.maximum(
                            ctx["progress"], step_progress,
                        ) + 1
                        pert_states.append(z_roll.squeeze(0))
                    pert_traj = torch.stack(pert_states)

                    divergence = (orig_traj - pert_traj).norm(dim=-1)
                    d_init = divergence[0].item()
                    d_final = divergence[-1].item()
                    # Align lengths (orig_traj may be [T], pert_traj [T])
                    Tdiv = min(orig_traj.shape[0], pert_traj.shape[0])
                    div_slice = divergence[:Tdiv].detach().cpu()
                    if div_curve_sum is None:
                        div_curve_sum = div_slice.clone()
                        ratio_curve_sum = (
                            div_slice / max(d_init, 1e-8)
                        ).clone()
                    else:
                        div_curve_sum = div_curve_sum + div_slice
                        ratio_curve_sum = ratio_curve_sum + (
                            div_slice / max(d_init, 1e-8)
                        )
                    div_curve_count += 1

                    recoveries.append(d_final < d_init * 0.5)
                    if d_init > 1e-8:
                        distances.append(d_final / d_init)

                    final_gate = dynamics.gate_probs(pert_traj[-1].unsqueeze(0))
                    identity_ok.append(
                        final_gate.argmax(dim=-1).item()
                        == dynamics.gate_probs(orig_traj[-1].unsqueeze(0)).argmax(dim=-1).item()
                    )
                    same_family_anchor.append(
                        sum(local_same) / max(len(local_same), 1)
                    )
                    wrong_family_anchor.append(
                        sum(local_wrong) / max(len(local_wrong), 1)
                    )
                    backward_progress.append(
                        sum(local_backward) / max(len(local_backward), 1)
                    )
                    overshoot_progress.append(
                        sum(local_overshoot) / max(len(local_overshoot), 1)
                    )
                    samples_done += 1

            mean_div_curve = (
                (div_curve_sum / max(div_curve_count, 1)).tolist()
                if div_curve_sum is not None else []
            )
            mean_ratio_curve = (
                (ratio_curve_sum / max(div_curve_count, 1)).tolist()
                if ratio_curve_sum is not None else []
            )
            per_scale[str(scale)] = {
                "recovery_rate": sum(recoveries) / max(len(recoveries), 1),
                "mean_distance_ratio": sum(distances) / max(len(distances), 1),
                "identity_preservation": sum(identity_ok) / max(len(identity_ok), 1),
                "same_family_anchor_rate": (
                    sum(same_family_anchor) / max(len(same_family_anchor), 1)
                ),
                "wrong_family_anchor_rate": (
                    sum(wrong_family_anchor) / max(len(wrong_family_anchor), 1)
                ),
                "backward_progress_rate": (
                    sum(backward_progress) / max(len(backward_progress), 1)
                ),
                "overshoot_progress_rate": (
                    sum(overshoot_progress) / max(len(overshoot_progress), 1)
                ),
                "num_samples": len(recoveries),
                "mean_reference_divergence_curve": mean_div_curve,
                "mean_divergence_ratio_curve": mean_ratio_curve,
            }
            same_family_rates.append(
                per_scale[str(scale)]["same_family_anchor_rate"]
            )
            wrong_family_rates.append(
                per_scale[str(scale)]["wrong_family_anchor_rate"]
            )

        overall_rates = [
            v["recovery_rate"] for v in per_scale.values()
        ]
        return {
            "per_scale": per_scale,
            "overall_recovery_rate": sum(overall_rates) / max(len(overall_rates), 1),
            "same_family_anchor_rate": sum(same_family_rates) / max(len(same_family_rates), 1),
            "wrong_family_anchor_rate": sum(wrong_family_rates) / max(len(wrong_family_rates), 1),
        }

    # Null correction norm (on-tube states should get near-zero corrections)
    null_bins = {
        "very_near": [],
        "near": [],
        "far": [],
    }
    all_pids = list(latent_states_raw.keys())[:100]
    for pid in all_pids:
        if pid not in pid_to_family_idx:
            continue
        z = latent_states_raw[pid].to(device)
        t_mid = z.shape[0] // 2
        z_pt = z[t_mid]
        fam_idx = torch.tensor([pid_to_family_idx[pid]], device=device)
        progress = torch.tensor([t_mid], dtype=torch.long, device=device)
        ctx = tube_geo.contextualize(
            z_pt.unsqueeze(0),
            family_ids=fam_idx,
            progress_hint=progress,
            backward_window=backward_window,
            forward_window=forward_window,
        )
        anchor, tangent = ctx["anchor"], ctx["tangent"]
        corr = dynamics.recovery_field(z_pt.unsqueeze(0), anchor, tangent)
        corr_norm = corr.norm().item()
        null_corrections.append(corr_norm)
        tube_dist = ctx["transverse_norm"].item()
        if tube_dist < 0.1:
            null_bins["very_near"].append(corr_norm)
        elif tube_dist < 0.5:
            null_bins["near"].append(corr_norm)
        else:
            null_bins["far"].append(corr_norm)

    inferred_results = analyze_mode(force_family=False)
    oracle_results = analyze_mode(force_family=True)

    attractor_results = None
    if run_attractor_analysis:
        class RecoveryRolloutField(torch.nn.Module):
            def __init__(self, wrapped: TubeAwareDynamics):
                super().__init__()
                self.wrapped = wrapped

            def velocity_field(self, z: torch.Tensor) -> torch.Tensor:
                return self.wrapped.step(z) - z

        sample_trajs: dict[str, torch.Tensor] = {}
        for i, pid in enumerate(all_pids[:attractor_num_samples]):
            z = latent_states_raw[pid].to(device)
            mid = z.shape[0] // 2
            sample_trajs[pid] = dynamics.rollout(z[mid], 20).detach().cpu()
        if sample_trajs:
            analyzer = AttractorAnalysis(
                RecoveryRolloutField(dynamics),
                convergence_threshold=0.1,
                max_steps=200,
                attractor_merge_radius=3.0,
            )
            report = analyzer.full_analysis(sample_trajs)
            print(format_attractor_report(report))
            attractor_results = summarize_attractor_report(report)

    if not had_geo:
        dynamics.set_tube_geometry(None)

    def _mean_final_divergence(per_scale: dict) -> float:
        finals = [
            v["mean_reference_divergence_curve"][-1]
            for v in per_scale.values()
            if v.get("mean_reference_divergence_curve")
        ]
        return sum(finals) / max(len(finals), 1)

    return {
        "per_scale": inferred_results["per_scale"],
        "overall_recovery_rate": inferred_results["overall_recovery_rate"],
        "same_family_anchor_rate": inferred_results["same_family_anchor_rate"],
        "wrong_family_anchor_rate": inferred_results["wrong_family_anchor_rate"],
        "oracle_family": oracle_results,
        "inferred_family": inferred_results,
        "oracle_overall_recovery_rate": oracle_results["overall_recovery_rate"],
        "oracle_same_family_anchor_rate": oracle_results["same_family_anchor_rate"],
        "mean_final_reference_divergence_inferred": _mean_final_divergence(
            inferred_results["per_scale"],
        ),
        "mean_final_reference_divergence_oracle": _mean_final_divergence(
            oracle_results["per_scale"],
        ),
        "mean_null_correction_norm": sum(null_corrections) / max(len(null_corrections), 1),
        "null_correction_by_distance_bin": {
            key: (sum(vals) / len(vals) if vals else 0.0)
            for key, vals in null_bins.items()
        },
        "attractor_diagnostics": attractor_results,
    }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def run_training(args: argparse.Namespace) -> dict:
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data (identical to v5.3)
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

    train_families_avail = {f: available_families[f]
                            for f in TRAIN_FAMILIES if f in available_families}
    train_pids = [pid for fam in TRAIN_FAMILIES if fam in available_families
                  for pid in available_families[fam]]

    fam_to_label: dict[str, int] = {}
    for i, fam in enumerate(sorted(train_families_avail.keys())):
        fam_to_label[fam] = i
    pid_labels = {pid: fam_to_label[PID_TO_FAMILY[pid]] for pid in train_pids}

    task_family_names = sorted({_PID_META[pid]["task_family"] for pid in train_pids})
    tf_to_idx = {name: i for i, name in enumerate(task_family_names)}
    num_task_families = len(tf_to_idx)
    pid_to_tf_idx = {
        pid: tf_to_idx[_PID_META[pid]["task_family"]] for pid in train_pids
    }

    pid_variants = {pid: get_variant(pid) for pid in train_pids}
    N_train = len(train_pids)
    hn_mask = torch.zeros(N_train, N_train, dtype=torch.bool, device=device)
    for i in range(N_train):
        for j in range(i + 1, N_train):
            if (pid_variants[train_pids[i]] == pid_variants[train_pids[j]]
                    and pid_labels[train_pids[i]] != pid_labels[train_pids[j]]):
                hn_mask[i, j] = True
                hn_mask[j, i] = True

    unique_variants = sorted(set(pid_variants.values()))
    variant_to_label = {v: i for i, v in enumerate(unique_variants)}
    pid_variant_labels = {pid: variant_to_label[pid_variants[pid]] for pid in train_pids}
    num_variants = len(unique_variants)
    variant_label_tensor = torch.tensor(
        [pid_variant_labels[pid] for pid in train_pids],
        device=device, dtype=torch.long,
    )

    print(f"Training PIDs: {len(train_pids)}, families: {len(fam_to_label)}, "
          f"surface variants: {num_variants}")
    print(f"  Task families (gate supervision): {num_task_families} -> {task_family_names}")

    # ------------------------------------------------------------------
    # 2. Build model (v6: TubeAwareDynamics)
    # ------------------------------------------------------------------
    model = SDQv6Model(
        encoder=MultiScaleConvEncoder(hidden_dim, args.latent_dim),
        decoder=ResidualDecoder(args.latent_dim, args.gauge_dim, hidden_dim),
        dynamics=TubeAwareDynamics(
            args.latent_dim,
            num_regimes=num_task_families,
            num_tube_families=len(fam_to_label),
            progress_coord_cap=args.progress_coord_cap,
        ),
        aligner=MonotoneAligner(hidden_dim, align_dim=min(128, hidden_dim)),
        proj_head=ProjectionHead(args.latent_dim, hidden_dim=256, output_dim=128),
        surface_classifier=SurfaceClassifier(args.latent_dim, num_variants),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # ------------------------------------------------------------------
    # 2b. Optional checkpoint loading
    # ------------------------------------------------------------------
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        ckpt_state = load_file(args.checkpoint)
        encoder_state = {
            k: v for k, v in ckpt_state.items()
            if not k.startswith("dynamics.")
        }
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        dyn_skipped = [k for k in ckpt_state if k.startswith("dynamics.")]
        if dyn_skipped:
            print(f"  Skipped {len(dyn_skipped)} dynamics keys (training from scratch)")
        if missing:
            print(f"  Missing keys (initialized fresh): "
                  f"{[k for k in missing if not k.startswith('dynamics.')]}")
        print("  Checkpoint loaded -- skipping Phase 1")

    # ------------------------------------------------------------------
    # 3. Phase 1: Contrastive Geometry (same as v5.3)
    # ------------------------------------------------------------------
    _skip_phase1 = bool(args.checkpoint)
    if _skip_phase1:
        print("\n  [Phase 1 skipped -- encoder loaded from checkpoint]")

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

    REC_SAMPLES = 32
    history: list[dict] = []
    temperature = args.temperature

    for epoch in range(1, (0 if _skip_phase1 else args.epochs_phase1) + 1):
        model.train()
        z_full_raw, z_mean_raw = batch_encode(
            model.encoder, trajectories, train_pids, device,
            chunk_size=args.encode_batch,
        )
        z_half1 = {}
        z_half2 = {}
        for pid in train_pids:
            z = z_full_raw[pid]
            T = z.shape[0]
            mid = T // 2
            if mid > 0:
                z_half1[pid] = z[:mid].mean(dim=0)
                z_half2[pid] = z[mid:].mean(dim=0)
            else:
                z_half1[pid] = z.mean(dim=0)
                z_half2[pid] = z.mean(dim=0)

        emb_list = [z_mean_raw[pid] for pid in train_pids]
        emb_stack = torch.stack(emb_list)
        proj = model.proj_head(emb_stack)
        proj_normed = F.normalize(proj, dim=1)
        label_tensor = torch.tensor(
            [pid_labels[pid] for pid in train_pids],
            device=device, dtype=torch.long,
        )

        L_supcon_proj = supervised_contrastive_loss(
            proj_normed, label_tensor, temperature=temperature,
            hard_neg_mask=hn_mask, hard_neg_weight=10.0,
        )
        enc_normed = F.normalize(emb_stack, dim=1)
        L_supcon_enc = supervised_contrastive_loss(
            enc_normed, label_tensor, temperature=temperature,
            hard_neg_mask=hn_mask, hard_neg_weight=10.0,
        )
        h1_stack = torch.stack([z_half1[pid] for pid in train_pids])
        h2_stack = torch.stack([z_half2[pid] for pid in train_pids])
        seg_stack = torch.cat([h1_stack, h2_stack], dim=0)
        seg_normed = F.normalize(seg_stack, dim=1)
        seg_labels = torch.cat([label_tensor, label_tensor], dim=0)
        L_supcon_seg = supervised_contrastive_loss(
            seg_normed, seg_labels, temperature=temperature,
        )

        z_mean_dict = {pid: F.normalize(z_mean_raw[pid], dim=0) for pid in train_pids}

        emb_centered = emb_stack - emb_stack.mean(dim=0)
        emb_std_per_dim = emb_stack.std(dim=0)
        L_var = F.relu(1.0 - emb_std_per_dim).mean()
        N_batch = emb_stack.shape[0]
        cov = (emb_centered.T @ emb_centered) / max(N_batch - 1, 1)
        cov_diag = cov.diag()
        off_diag = cov - torch.diag(cov_diag)
        L_cov = (off_diag ** 2).sum() / emb_stack.shape[1]

        N_PULL_PAIRS = 30
        L_ts_pull = torch.tensor(0.0, device=device)
        pull_count = 0
        sample_fam_keys = random.sample(
            list(train_families_avail.keys()),
            min(N_PULL_PAIRS, len(train_families_avail)),
        )
        for fam in sample_fam_keys:
            fam_pids_avail = [p for p in train_families_avail[fam] if p in z_full_raw]
            if len(fam_pids_avail) >= 2:
                p1, p2 = random.sample(fam_pids_avail, 2)
                z1 = z_full_raw[p1]
                z2 = z_full_raw[p2]
                T_min = min(z1.shape[0], z2.shape[0])
                L_ts_pull = L_ts_pull + (z1[:T_min] - z2[:T_min]).pow(2).sum(dim=-1).mean()
                pull_count += 1
        if pull_count > 0:
            L_ts_pull = L_ts_pull / pull_count

        L_vel_cos = torch.tensor(0.0, device=device)
        vel_count = 0
        for fam in sample_fam_keys:
            fam_pids_avail = [p for p in train_families_avail[fam] if p in z_full_raw]
            if len(fam_pids_avail) >= 2:
                p1, p2 = random.sample(fam_pids_avail, 2)
                z1 = z_full_raw[p1]
                z2 = z_full_raw[p2]
                T_min = min(z1.shape[0], z2.shape[0])
                if T_min >= 2:
                    L_vel_cos = L_vel_cos + latent_velocity_cosine_loss(
                        z1[:T_min], z2[:T_min],
                    )
                    vel_count += 1
        if vel_count > 0:
            L_vel_cos = L_vel_cos / vel_count

        adv_alpha = min(epoch / 50.0, 1.0)
        surf_logits = model.surface_classifier(emb_stack, alpha=adv_alpha)
        L_surf = F.cross_entropy(surf_logits, variant_label_tensor)

        rec_pids = random.sample(train_pids, min(REC_SAMPLES, len(train_pids)))
        L_rec = torch.tensor(0.0, device=device)
        for pid in rec_pids:
            z_raw = z_full_raw[pid]
            h = trajectories[pid].states.to(device)
            h_hat = model.decoder(z_raw)
            L_rec = L_rec + reconstruction_loss(h, h_hat)
        L_rec = L_rec / len(rec_pids)

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

        with torch.no_grad():
            emb_std = emb_stack.std(dim=0).mean().item()
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
                            z_mean_dict[a].unsqueeze(0), z_mean_dict[b].unsqueeze(0),
                        ).item()
                        within_cos.append(cos)
            for _ in range(50):
                f1, f2 = random.sample(sample_fams, 2)
                p1 = random.choice(train_families_avail[f1])
                p2 = random.choice(train_families_avail[f2])
                if p1 in z_mean_dict and p2 in z_mean_dict:
                    cos = F.cosine_similarity(
                        z_mean_dict[p1].unsqueeze(0), z_mean_dict[p2].unsqueeze(0),
                    ).item()
                    cross_cos.append(cos)

        avg_within = sum(within_cos) / max(len(within_cos), 1)
        avg_cross = sum(cross_cos) / max(len(cross_cos), 1)

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

        if epoch >= 150 and avg_within > 0.95 and avg_cross < 0.1:
            print(f"  Phase 1 converged at epoch {epoch} "
                  f"(within_cos={avg_within:.3f}, cross_cos={avg_cross:.3f})")
            break

    # ------------------------------------------------------------------
    # 4. Phase 2: Nominal Dynamics (encoder FROZEN, no contraction)
    # ------------------------------------------------------------------
    if args.epochs_phase2 > 0:
        print("\n" + "=" * 60)
        print(
            f"PHASE 2: NOMINAL DYNAMICS "
            f"(encoder frozen, K={args.num_substeps}, "
            f"{num_task_families} regimes)",
        )
        print("=" * 60)

        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.encoder.eval()

        print("  Pre-computing frozen latent embeddings...")
        with torch.no_grad():
            frozen_z_full, frozen_z_mean = batch_encode(
                model.encoder, trajectories, train_pids, device,
                chunk_size=args.encode_batch,
            )
        frozen_z_full = {pid: z.detach() for pid, z in frozen_z_full.items()}
        frozen_z_mean = {pid: z.detach() for pid, z in frozen_z_mean.items()}

        # Only nominal dynamics parameters (gate + velocity_nets)
        nominal_params = (
            list(model.dynamics.gate.parameters())
            + list(model.dynamics.velocity_nets.parameters())
        )
        opt2 = torch.optim.AdamW(nominal_params, lr=args.lr_phase2, weight_decay=1e-4)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=args.epochs_phase2, eta_min=args.lr_phase2 * 0.01,
        )

        WARMUP_EPOCHS = 10
        K = args.num_substeps

        all_starts = []
        all_targets = []
        for pid in train_pids:
            z = frozen_z_full[pid]
            if z.shape[0] < 2:
                continue
            all_starts.append(z[:-1])
            all_targets.append(z[1:])
        all_starts = torch.cat(all_starts, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        dz_true = all_targets - all_starts
        print(f"  Batched dynamics: {all_starts.shape[0]} start-target pairs")

        all_tf_labels: list[int] = []
        all_sem_labels: list[int] = []
        all_step_ids: list[int] = []
        for pid in train_pids:
            z = frozen_z_full[pid]
            if z.shape[0] < 2:
                continue
            tf_i = pid_to_tf_idx[pid]
            all_tf_labels.extend([tf_i] * (z.shape[0] - 1))
            all_sem_labels.extend([pid_labels[pid]] * (z.shape[0] - 1))
            all_step_ids.extend(list(range(z.shape[0] - 1)))
        all_tf_tensor = torch.tensor(all_tf_labels, device=device, dtype=torch.long)
        all_sem_tensor = torch.tensor(all_sem_labels, device=device, dtype=torch.long)
        all_step_tensor = torch.tensor(all_step_ids, device=device, dtype=torch.long)

        tf_class_indices = {
            i: (all_tf_tensor == i).nonzero(as_tuple=True)[0]
            for i in range(num_task_families)
        }

        terminal_list = [
            frozen_z_full[pid][-1]
            for pid in train_pids
            if pid in frozen_z_full and frozen_z_full[pid].shape[0] >= 1
        ]
        terminals_stack = torch.stack(terminal_list, dim=0) if terminal_list else None

        vcoh_points = []
        vcoh_labels = []
        fam_to_idx_vcoh = {}
        for fam, pids_in_fam in available_families.items():
            fam_pids = [p for p in pids_in_fam
                        if p in frozen_z_full and frozen_z_full[p].shape[0] >= 3]
            if len(fam_pids) < 2:
                continue
            if fam not in fam_to_idx_vcoh:
                fam_to_idx_vcoh[fam] = len(fam_to_idx_vcoh)
            fi = fam_to_idx_vcoh[fam]
            fam_sample = fam_pids[:3]
            T_min = min(frozen_z_full[p].shape[0] for p in fam_sample)
            for pid in fam_sample:
                z_mid = frozen_z_full[pid][1:T_min - 1]
                vcoh_points.append(z_mid)
                vcoh_labels.extend([fi] * z_mid.shape[0])
        if vcoh_points:
            vcoh_z = torch.cat(vcoh_points, dim=0)
            vcoh_fam = torch.tensor(vcoh_labels, device=device)
            print(f"  Velocity contrast points: {vcoh_z.shape[0]} "
                  f"across {len(fam_to_idx_vcoh)} families")
        else:
            vcoh_z = vcoh_fam = None

        for epoch in range(1, args.epochs_phase2 + 1):
            model.dynamics.train()

            if epoch <= WARMUP_EPOCHS:
                warmup_factor = epoch / WARMUP_EPOCHS
                for pg in opt2.param_groups:
                    pg["lr"] = args.lr_phase2 * warmup_factor

            denom = max(args.epochs_phase2 - 1, 1)
            tau = (
                args.gate_tau_low
                + (args.gate_tau_high - args.gate_tau_low)
                * (1.0 - (epoch - 1) / denom)
            )

            # --- Dynamics MSE ---
            z_curr = all_starts
            for _ in range(K):
                z_curr = z_curr + model.dynamics.velocity_field(z_curr, tau=tau)
            L_dyn = (z_curr - all_targets).pow(2).sum(dim=-1).mean()

            dz_pred = z_curr - all_starts
            cos = F.cosine_similarity(dz_pred, dz_true, dim=-1)
            L_vel_dir = (1.0 - cos).mean()

            mag_pred = dz_pred.norm(dim=-1)
            mag_true = dz_true.norm(dim=-1)
            L_mag = ((mag_pred - mag_true).pow(2) / (mag_true.pow(2) + 1e-6)).mean()

            # --- Velocity contrastive ---
            L_vcoh = torch.tensor(0.0, device=device)
            if vcoh_z is not None and args.vcoh_weight > 0:
                batch_size = min(256, vcoh_z.shape[0])
                idx = torch.randperm(vcoh_z.shape[0], device=device)[:batch_size]
                L_vcoh = model.dynamics.velocity_contrastive_loss(
                    vcoh_z[idx], vcoh_fam[idx], temperature=0.1, tau=tau,
                )

            # --- Gate supervision ---
            n_gate = min(args.gate_sup_batch, all_starts.shape[0])
            gidx = _balanced_tf_indices(
                tf_class_indices, n_gate, num_task_families, device,
            )
            L_gate_sup = model.dynamics.gate_supervised_loss(
                all_starts[gidx], all_tf_tensor[gidx],
            )

            # --- Regime consistency ---
            L_reg = torch.tensor(0.0, device=device)
            reg_count = 0
            fam_keys = list(train_families_avail.keys())
            for _ in range(32):
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

            # --- Gate entropy ---
            n_s = min(512, all_starts.shape[0])
            idx_s = _balanced_tf_indices(
                tf_class_indices, n_s, num_task_families, device,
            )
            L_gate_ent = model.dynamics.gate_entropy_loss(all_starts[idx_s])

            # --- Terminal velocity ---
            L_terminal = torch.tensor(0.0, device=device)
            if terminals_stack is not None and args.terminal_weight > 0:
                n_t = min(512, terminals_stack.shape[0])
                tidx = torch.randperm(terminals_stack.shape[0], device=device)[:n_t]
                L_terminal = model.dynamics.terminal_velocity_loss(
                    terminals_stack[tidx], margin=args.terminal_margin, tau=tau,
                )

            # --- Total (no contraction) ---
            loss = (
                L_dyn
                + args.vel_dir_weight * L_vel_dir
                + args.mag_weight * L_mag
                + args.vcoh_weight * L_vcoh
                + args.gate_sup_weight * L_gate_sup
                + args.regime_weight * L_reg
                + args.entropy_weight * L_gate_ent
                + args.terminal_weight * L_terminal
            )

            opt2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(nominal_params, max_norm=1.0)
            opt2.step()
            if epoch > WARMUP_EPOCHS:
                sched2.step()

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
                "gate_ent": L_gate_ent.item(),
                "terminal": L_terminal.item(),
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
                    f"vcoh={L_vcoh.item():.3f} "
                    f"tau={tau:.3f} "
                    f"|f|={mean_vel:.4f} "
                    f"lr={opt2.param_groups[0]['lr']:.6f}"
                )

        # ==============================================================
        # 5. Phase 3: Recovery Dynamics (nominal FROZEN)
        # ==============================================================
        if args.epochs_phase3 > 0:
            print("\n" + "=" * 60)
            print(
                f"PHASE 3: RECOVERY DYNAMICS "
                f"({args.epochs_phase3} epochs, lr={args.lr_phase3}, "
                f"K_final={args.recovery_substeps_final})",
            )
            print("=" * 60)

            # Freeze nominal dynamics, train only recovery_net
            for p in model.dynamics.gate.parameters():
                p.requires_grad_(False)
            for p in model.dynamics.velocity_nets.parameters():
                p.requires_grad_(False)

            recovery_params = (
                list(model.dynamics.recovery_net.parameters())
                + list(model.dynamics.recovery_gain.parameters())
                + list(model.dynamics.recovery_along.parameters())
                + list(model.dynamics.progress_head.parameters())
            )
            if model.dynamics.family_router is not None:
                recovery_params += list(model.dynamics.family_router.parameters())
            opt3 = torch.optim.AdamW(
                recovery_params, lr=args.lr_phase3, weight_decay=1e-4,
            )
            sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt3, T_max=args.epochs_phase3,
                eta_min=args.lr_phase3 * 0.01,
            )
            P3_WARMUP = 5
            tau_p3 = args.gate_tau_low

            # Build tube geometry from frozen latents
            print("  Building tube geometry...")
            tube_geo = TubeGeometry.build(
                frozen_z_full, train_families_avail, fam_to_label, device,
            )
            print(f"  Tube anchors: {tube_geo.anchors.shape[0]}")
            model.dynamics.set_tube_geometry(tube_geo)

            for epoch in range(1, args.epochs_phase3 + 1):
                model.dynamics.recovery_net.train()
                model.dynamics.recovery_gain.train()
                model.dynamics.recovery_along.train()
                model.dynamics.progress_head.train()
                if model.dynamics.family_router is not None:
                    model.dynamics.family_router.train()

                if epoch <= P3_WARMUP:
                    for pg in opt3.param_groups:
                        pg["lr"] = args.lr_phase3 * (epoch / P3_WARMUP)

                K_rec = scheduled_recovery_horizon(
                    epoch, args.epochs_phase3, args.recovery_substeps_final,
                )
                recovery_scales = scheduled_recovery_scales(
                    epoch, args.epochs_phase3,
                )

                # --- Recovery rollout losses ---
                n_rec = min(args.recovery_batch, all_starts.shape[0])
                rec_idx = _balanced_tf_indices(
                    tf_class_indices, n_rec, num_task_families, device,
                )
                z_batch = all_starts[rec_idx]
                task_tf_batch = all_tf_tensor[rec_idx]
                sem_batch = all_sem_tensor[rec_idx]
                step_batch = all_step_tensor[rec_idx]
                tangent_batch = dz_true[rec_idx]
                tangent_n = tangent_batch / tangent_batch.norm(
                    dim=-1, keepdim=True,
                ).clamp(min=1e-8)

                L_recover_transverse = torch.tensor(0.0, device=device)
                L_recover_progress = torch.tensor(0.0, device=device)
                L_recover_smooth = torch.tensor(0.0, device=device)
                L_basin_margin = torch.tensor(0.0, device=device)
                basin_margin_ok = torch.tensor(0.0, device=device)
                L_ref_num = torch.tensor(0.0, device=device)
                L_ref_den = 0
                z_recovered_last = None

                step_cap = all_step_tensor.max().clamp(min=1).float()
                tgt_prog = (step_batch.float() / step_cap).clamp(0, 1)
                L_prog_head = F.mse_loss(
                    model.dynamics.progress_scalar(z_batch).squeeze(-1),
                    tgt_prog,
                )
                L_router = torch.tensor(0.0, device=device)
                n_router = 0
                if model.dynamics.family_router is not None:
                    L_router = L_router + F.cross_entropy(
                        model.dynamics.family_router_logits(z_batch),
                        sem_batch.long(),
                    )
                    n_router += 1

                for scale in recovery_scales:
                    delta = torch.randn_like(z_batch)
                    delta = delta - (delta * tangent_n).sum(
                        dim=-1, keepdim=True,
                    ) * tangent_n
                    bad = delta.norm(dim=-1) < 1e-6
                    if bad.any():
                        alt = torch.randn_like(z_batch)
                        alt = alt - (alt * tangent_n).sum(
                            dim=-1, keepdim=True,
                        ) * tangent_n
                        delta = torch.where(bad.unsqueeze(-1), alt, delta)
                    delta = delta / delta.norm(
                        dim=-1, keepdim=True,
                    ).clamp(min=1e-8) * scale

                    z_pert = z_batch + delta
                    z_roll = z_pert
                    z_orig = z_batch
                    progress_roll = step_batch.clone()
                    prev_correction = None

                    if model.dynamics.family_router is not None:
                        L_router = L_router + F.cross_entropy(
                            model.dynamics.family_router_logits(z_pert),
                            sem_batch.long(),
                        )
                        n_router += 1

                    for step_i in range(K_rec):
                        z_orig = z_orig + model.dynamics.velocity_field(
                            z_orig, tau=tau_p3,
                        )
                        z_nominal = z_roll + model.dynamics.velocity_field(
                            z_roll, tau=tau_p3,
                        )
                        z_roll, ctx = model.dynamics.recover_step(
                            z_nominal,
                            family_ids=sem_batch,
                            progress_hint=progress_roll,
                            backward_window=args.recovery_backward_window,
                            forward_window=args.recovery_forward_window,
                            return_context=True,
                        )
                        L_recover_transverse = L_recover_transverse + (
                            ctx["transverse_norm"] / scale
                        ).mean()
                        expected_progress = progress_roll.float() + 1.0
                        progress_penalty, _, _ = directional_progress_penalty(
                            ctx["progress"].float(),
                            expected_progress,
                            backward_weight=args.backward_progress_weight,
                            overshoot_weight=args.forward_overshoot_weight,
                            overshoot_slack=args.forward_overshoot_slack,
                        )
                        L_recover_progress = L_recover_progress + progress_penalty
                        L_basin_margin = L_basin_margin + F.relu(
                            ctx["transverse_norm"]
                            + args.basin_margin
                            - ctx["wrong_family_distance"]
                        ).mean()
                        basin_margin_ok = basin_margin_ok + (
                            ctx["transverse_norm"] + args.basin_margin
                            < ctx["wrong_family_distance"]
                        ).float().mean()
                        if prev_correction is not None:
                            L_recover_smooth = L_recover_smooth + (
                                ctx["correction"] - prev_correction
                            ).norm(dim=-1).mean()
                        prev_correction = ctx["correction"]
                        progress_roll = torch.maximum(
                            ctx["progress"], progress_roll,
                        ) + 1
                        w_ref = (
                            args.reference_rollout_early_weight
                            if K_rec <= 1 or step_i < (K_rec // 2)
                            else args.reference_rollout_late_weight
                        )
                        term_ref = (
                            (z_roll - z_orig.detach()).pow(2).sum(dim=-1).mean()
                        )
                        L_ref_num = L_ref_num + w_ref * term_ref
                        L_ref_den += 1

                    d_final = (z_roll - z_orig.detach()).norm(dim=-1)
                    d_init = delta.norm(dim=-1)
                    L_recover_transverse = L_recover_transverse + (
                        d_final / d_init.clamp(min=1e-8)
                    ).mean()
                    z_recovered_last = z_roll

                denom_rec = max(len(recovery_scales) * (K_rec + 1), 1)
                L_recover_transverse = L_recover_transverse / denom_rec
                L_recover_progress = L_recover_progress / max(
                    len(recovery_scales) * K_rec, 1,
                )
                L_recover_smooth = L_recover_smooth / max(
                    len(recovery_scales) * max(K_rec - 1, 1), 1,
                )
                L_basin_margin = L_basin_margin / max(
                    len(recovery_scales) * K_rec, 1,
                )
                basin_margin_ok = basin_margin_ok / max(
                    len(recovery_scales) * K_rec, 1,
                )
                L_ref = L_ref_num / max(L_ref_den, 1)
                if n_router > 0:
                    L_router = L_router / n_router

                # --- L_null: correction should be near-zero for on-tube states ---
                null_n = min(256, all_starts.shape[0])
                null_idx = torch.randperm(
                    all_starts.shape[0], device=device,
                )[:null_n]
                z_on_tube = all_starts[null_idx]
                sem_null = all_sem_tensor[null_idx]
                step_null = all_step_tensor[null_idx]
                z_on_corr, ctx_null = model.dynamics.recover_step(
                    z_on_tube,
                    family_ids=sem_null,
                    progress_hint=step_null,
                    backward_window=args.recovery_backward_window,
                    forward_window=args.recovery_forward_window,
                    return_context=True,
                )
                corr_null_norm = ctx_null["correction"].norm(dim=-1)
                L_null = (
                    corr_null_norm
                    / (ctx_null["transverse_norm"].detach() + 1e-3)
                ).mean()

                # --- L_identity: preserve family after recovery ---
                L_identity = torch.tensor(0.0, device=device)
                if z_recovered_last is not None and args.identity_weight > 0:
                    rec_logits = model.dynamics.gate_logits(z_recovered_last)
                    L_identity = F.cross_entropy(
                        rec_logits, task_tf_batch.long(),
                    )

                # --- Total ---
                loss_p3 = (
                    args.recover_weight * L_recover_transverse
                    + args.progress_weight * L_recover_progress
                    + args.smoothness_weight * L_recover_smooth
                    + args.basin_margin_weight * L_basin_margin
                    + args.reference_rollout_weight * L_ref
                    + args.progress_head_weight * L_prog_head
                    + args.family_router_weight * L_router
                    + args.null_weight * L_null
                    + args.identity_weight * L_identity
                )

                opt3.zero_grad()
                loss_p3.backward()
                nn.utils.clip_grad_norm_(recovery_params, max_norm=1.0)
                opt3.step()
                if epoch > P3_WARMUP:
                    sched3.step()

                entry = {
                    "phase": 3, "epoch": epoch,
                    "recover": L_recover_transverse.item(),
                    "recover_progress": L_recover_progress.item(),
                    "recover_smooth": L_recover_smooth.item(),
                    "basin_margin": L_basin_margin.item(),
                    "basin_margin_ok": basin_margin_ok.item(),
                    "reference_rollout": L_ref.item(),
                    "progress_head": L_prog_head.item(),
                    "family_router": L_router.item(),
                    "null": L_null.item(),
                    "identity": L_identity.item(),
                    "recovery_horizon": K_rec,
                    "recovery_scales": recovery_scales,
                    "total": loss_p3.item(),
                    "lr": opt3.param_groups[0]["lr"],
                }
                history.append(entry)

                if epoch % 10 == 0 or epoch == 1:
                    print(
                        f"  P3 Epoch {epoch:3d} | "
                        f"recover={L_recover_transverse.item():.4f} "
                        f"prog={L_recover_progress.item():.4f} "
                        f"smooth={L_recover_smooth.item():.4f} "
                        f"margin={L_basin_margin.item():.4f} "
                        f"route={basin_margin_ok.item():.2%} "
                        f"ref={L_ref.item():.4f} "
                        f"ph={L_prog_head.item():.4f} "
                        f"rtr={L_router.item():.4f} "
                        f"null={L_null.item():.4f} "
                        f"identity={L_identity.item():.3f} "
                        f"K={K_rec} "
                        f"total={loss_p3.item():.4f} "
                        f"lr={opt3.param_groups[0]['lr']:.6f}"
                    )

            # Unfreeze nominal for Phase 4
            for p in model.dynamics.gate.parameters():
                p.requires_grad_(True)
            for p in model.dynamics.velocity_nets.parameters():
                p.requires_grad_(True)

        # ==============================================================
        # 6. Phase 4: Joint Fine-tune (conservative)
        # ==============================================================
        if args.epochs_phase4 > 0 and args.epochs_phase3 > 0:
            print("\n" + "=" * 60)
            print(
                f"PHASE 4: JOINT FINE-TUNE "
                f"({args.epochs_phase4} epochs, lr={args.lr_phase4})",
            )
            print("=" * 60)

            all_dyn_params = list(model.dynamics.parameters())
            opt4 = torch.optim.AdamW(
                all_dyn_params, lr=args.lr_phase4, weight_decay=1e-4,
            )
            sched4 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt4, T_max=args.epochs_phase4,
                eta_min=args.lr_phase4 * 0.01,
            )
            P4_WARMUP = 3
            tau_p4 = args.gate_tau_low

            if not model.dynamics.has_tube_geometry:
                tube_geo = TubeGeometry.build(
                    frozen_z_full, train_families_avail, fam_to_label, device,
                )
                model.dynamics.set_tube_geometry(tube_geo)

            for epoch in range(1, args.epochs_phase4 + 1):
                model.dynamics.train()

                if epoch <= P4_WARMUP:
                    for pg in opt4.param_groups:
                        pg["lr"] = args.lr_phase4 * (epoch / P4_WARMUP)

                K_joint = scheduled_recovery_horizon(
                    epoch, args.epochs_phase4, args.recovery_substeps_final,
                )
                phase4_scales = scheduled_recovery_scales(
                    epoch, args.epochs_phase4,
                )

                # --- Nominal fidelity (hard anchor) ---
                z_curr = all_starts
                for _ in range(K):
                    z_curr = z_curr + model.dynamics.velocity_field(
                        z_curr, tau=tau_p4,
                    )
                L_dyn = (z_curr - all_targets).pow(2).sum(dim=-1).mean()

                dz_pred = z_curr - all_starts
                cos = F.cosine_similarity(dz_pred, dz_true, dim=-1)
                L_vel_dir = (1.0 - cos).mean()

                # --- Recovery (same as Phase 3 but smaller batch) ---
                n_rec = min(args.recovery_batch // 2, all_starts.shape[0])
                rec_idx = _balanced_tf_indices(
                    tf_class_indices, n_rec, num_task_families, device,
                )
                z_batch_j = all_starts[rec_idx]
                sem_batch_j = all_sem_tensor[rec_idx]
                step_batch_j = all_step_tensor[rec_idx]
                tangent_j = dz_true[rec_idx]
                tangent_jn = tangent_j / tangent_j.norm(
                    dim=-1, keepdim=True,
                ).clamp(min=1e-8)

                L_recover_j = torch.tensor(0.0, device=device)
                L_progress_j = torch.tensor(0.0, device=device)
                L_smooth_j = torch.tensor(0.0, device=device)
                L_basin_margin_j = torch.tensor(0.0, device=device)
                basin_margin_ok_j = torch.tensor(0.0, device=device)
                L_ref_num_j = torch.tensor(0.0, device=device)
                L_ref_den_j = 0
                step_cap_j = all_step_tensor.max().clamp(min=1).float()
                tgt_prog_j = (step_batch_j.float() / step_cap_j).clamp(0, 1)
                L_prog_head_j = F.mse_loss(
                    model.dynamics.progress_scalar(z_batch_j).squeeze(-1),
                    tgt_prog_j,
                )
                L_router_j = torch.tensor(0.0, device=device)
                n_router_j = 0
                if model.dynamics.family_router is not None:
                    L_router_j = L_router_j + F.cross_entropy(
                        model.dynamics.family_router_logits(z_batch_j),
                        sem_batch_j.long(),
                    )
                    n_router_j += 1

                for scale_j in phase4_scales:
                    delta_j = torch.randn_like(z_batch_j)
                    delta_j = delta_j - (delta_j * tangent_jn).sum(
                        dim=-1, keepdim=True,
                    ) * tangent_jn
                    delta_j = delta_j / delta_j.norm(
                        dim=-1, keepdim=True,
                    ).clamp(min=1e-8) * scale_j

                    z_pert_j = z_batch_j + delta_j
                    z_roll_j = z_pert_j
                    z_orig_j = z_batch_j
                    progress_j = step_batch_j.clone()
                    prev_corr_j = None
                    if model.dynamics.family_router is not None:
                        L_router_j = L_router_j + F.cross_entropy(
                            model.dynamics.family_router_logits(z_pert_j),
                            sem_batch_j.long(),
                        )
                        n_router_j += 1
                    for step_ij in range(K_joint):
                        z_orig_j = z_orig_j + model.dynamics.velocity_field(
                            z_orig_j, tau=tau_p4,
                        )
                        z_nom_j = z_roll_j + model.dynamics.velocity_field(
                            z_roll_j, tau=tau_p4,
                        )
                        z_roll_j, ctx_j = model.dynamics.recover_step(
                            z_nom_j,
                            family_ids=sem_batch_j,
                            progress_hint=progress_j,
                            backward_window=args.recovery_backward_window,
                            forward_window=args.recovery_forward_window,
                            return_context=True,
                        )
                        L_recover_j = L_recover_j + (
                            ctx_j["transverse_norm"] / scale_j
                        ).mean()
                        progress_penalty_j, _, _ = directional_progress_penalty(
                            ctx_j["progress"].float(),
                            progress_j.float() + 1.0,
                            backward_weight=args.backward_progress_weight,
                            overshoot_weight=args.forward_overshoot_weight,
                            overshoot_slack=args.forward_overshoot_slack,
                        )
                        L_progress_j = L_progress_j + progress_penalty_j
                        L_basin_margin_j = L_basin_margin_j + F.relu(
                            ctx_j["transverse_norm"]
                            + args.basin_margin
                            - ctx_j["wrong_family_distance"]
                        ).mean()
                        basin_margin_ok_j = basin_margin_ok_j + (
                            ctx_j["transverse_norm"] + args.basin_margin
                            < ctx_j["wrong_family_distance"]
                        ).float().mean()
                        if prev_corr_j is not None:
                            L_smooth_j = L_smooth_j + (
                                ctx_j["correction"] - prev_corr_j
                            ).norm(dim=-1).mean()
                        prev_corr_j = ctx_j["correction"]
                        progress_j = torch.maximum(
                            ctx_j["progress"], progress_j,
                        ) + 1
                        w_rj = (
                            args.reference_rollout_early_weight
                            if K_joint <= 1 or step_ij < (K_joint // 2)
                            else args.reference_rollout_late_weight
                        )
                        trj = (
                            (z_roll_j - z_orig_j.detach()).pow(2).sum(dim=-1).mean()
                        )
                        L_ref_num_j = L_ref_num_j + w_rj * trj
                        L_ref_den_j += 1

                    d_final_j = (z_roll_j - z_orig_j.detach()).norm(dim=-1)
                    d_init_j = delta_j.norm(dim=-1)
                    L_recover_j = L_recover_j + (
                        d_final_j / d_init_j.clamp(min=1e-8)
                    ).mean()

                denom_joint = max(len(phase4_scales) * (K_joint + 1), 1)
                L_recover_j = L_recover_j / denom_joint
                L_progress_j = L_progress_j / max(
                    len(phase4_scales) * K_joint, 1,
                )
                L_smooth_j = L_smooth_j / max(
                    len(phase4_scales) * max(K_joint - 1, 1), 1,
                )
                L_basin_margin_j = L_basin_margin_j / max(
                    len(phase4_scales) * K_joint, 1,
                )
                basin_margin_ok_j = basin_margin_ok_j / max(
                    len(phase4_scales) * K_joint, 1,
                )
                L_ref_j = L_ref_num_j / max(L_ref_den_j, 1)
                if n_router_j > 0:
                    L_router_j = L_router_j / n_router_j

                # Null
                null_n_j = min(128, all_starts.shape[0])
                null_idx_j = torch.randperm(
                    all_starts.shape[0], device=device,
                )[:null_n_j]
                sem_null_j = all_sem_tensor[null_idx_j]
                step_null_j = all_step_tensor[null_idx_j]
                _, ctx_null_j = model.dynamics.recover_step(
                    all_starts[null_idx_j],
                    family_ids=sem_null_j,
                    progress_hint=step_null_j,
                    backward_window=args.recovery_backward_window,
                    forward_window=args.recovery_forward_window,
                    return_context=True,
                )
                L_null_j = (
                    ctx_null_j["correction"].norm(dim=-1)
                    / (ctx_null_j["transverse_norm"].detach() + 1e-3)
                ).mean()

                L_stab = torch.tensor(0.0, device=device)
                if args.stability_regularizer_weight > 0:
                    L_stab = model.dynamics.regime_consistency_loss(
                        z_roll_j, z_orig_j.detach(),
                    )

                # --- Joint total ---
                dw = args.dyn_weight_phase4
                loss_p4 = (
                    dw * L_dyn
                    + dw * args.vel_dir_weight * L_vel_dir
                    + args.recover_weight * L_recover_j
                    + args.progress_weight * L_progress_j
                    + args.smoothness_weight * L_smooth_j
                    + args.basin_margin_weight * L_basin_margin_j
                    + args.reference_rollout_weight * L_ref_j
                    + args.progress_head_weight * L_prog_head_j
                    + args.family_router_weight * L_router_j
                    + args.null_weight * L_null_j
                    + args.stability_regularizer_weight * L_stab
                )

                opt4.zero_grad()
                loss_p4.backward()
                nn.utils.clip_grad_norm_(all_dyn_params, max_norm=1.0)
                opt4.step()
                if epoch > P4_WARMUP:
                    sched4.step()

                entry = {
                    "phase": 4, "epoch": epoch,
                    "dyn": L_dyn.item(),
                    "vel_dir": L_vel_dir.item(),
                    "recover": L_recover_j.item(),
                    "recover_progress": L_progress_j.item(),
                    "recover_smooth": L_smooth_j.item(),
                    "basin_margin": L_basin_margin_j.item(),
                    "basin_margin_ok": basin_margin_ok_j.item(),
                    "reference_rollout": L_ref_j.item(),
                    "progress_head": L_prog_head_j.item(),
                    "family_router": L_router_j.item(),
                    "stability_reg": L_stab.item(),
                    "null": L_null_j.item(),
                    "recovery_horizon": K_joint,
                    "recovery_scales": phase4_scales,
                    "total": loss_p4.item(),
                    "lr": opt4.param_groups[0]["lr"],
                }
                history.append(entry)

                if epoch % 10 == 0 or epoch == 1:
                    print(
                        f"  P4 Epoch {epoch:3d} | "
                        f"dyn={L_dyn.item():.4f} "
                        f"vdir={L_vel_dir.item():.4f} "
                        f"rec={L_recover_j.item():.4f} "
                        f"prog={L_progress_j.item():.4f} "
                        f"smooth={L_smooth_j.item():.4f} "
                        f"margin={L_basin_margin_j.item():.4f} "
                        f"route={basin_margin_ok_j.item():.2%} "
                        f"ref={L_ref_j.item():.4f} "
                        f"stab={L_stab.item():.4f} "
                        f"null={L_null_j.item():.4f} "
                        f"K={K_joint} "
                        f"lr={opt4.param_groups[0]['lr']:.6f}"
                    )

        # Unfreeze encoder for eval
        for p in model.encoder.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    model.eval()
    results: dict = {"training_history": history}

    blend_alpha = args.blend_alpha
    with torch.no_grad():
        all_pids = list(trajectories.keys())
        z_full_eval, z_mean_eval = batch_encode(
            model.encoder, trajectories, all_pids, device, chunk_size=64,
        )
        latent_states = {}
        for pid, z in z_full_eval.items():
            mu = z.mean(dim=0, keepdim=True)
            z_blend = blend_alpha * mu + (1 - blend_alpha) * z
            latent_states[pid] = F.normalize(z_blend, dim=-1).cpu()
        latent_states_raw = {pid: z.cpu() for pid, z in z_full_eval.items()}
    print(f"  Eval blend_alpha={blend_alpha}")

    # Build tube geometry for eval (uses all available semantic families)
    if args.epochs_phase2 > 0:
        eval_fam_to_idx = {
            fam: i for i, fam in enumerate(sorted(available_families.keys()))
        }
        eval_tube_geo = TubeGeometry.build(
            {pid: z.to(device) for pid, z in latent_states_raw.items()},
            available_families,
            eval_fam_to_idx,
            device,
        )
        model.dynamics.set_tube_geometry(eval_tube_geo)
        print(f"  Tube geometry set for eval ({eval_tube_geo.anchors.shape[0]} anchors)")

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

    # --- Retrieval ---
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
    print(f"  Cosine separation: {ms.velocity_cosine_separation:.2f}")
    results["motion_similarity"] = asdict(ms)

    # --- Hard-negative ---
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
        print(f"  Separation ratio: {hn.separation_ratio:.2f}")
        results["hard_negative"] = {
            "positive_distance": hn.positive_distance,
            "hard_negative_distance": hn.hard_negative_distance,
            "separation_ratio": hn.separation_ratio,
            "passes": hn.passes,
        }

    # --- Reconstruction ---
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
    results["shuffle_control"] = {
        "shuffle_within": sw, "shuffle_cross": sc, "shuffle_separation": shuffle_sep,
    }

    # --- Scorecard ---
    print("\n--- Semantic Scorecard ---")
    scorecard = compute_scorecard(
        latent_states=latent_states, families=available_families,
        retrieval_result=ret, hard_negative_result=hn,
        motion_result=ms, shuffle_separation=shuffle_sep,
    )
    print(scorecard.summary())
    results["scorecard"] = scorecard.to_dict()

    # --- Dynamics motion similarity ---
    if args.epochs_phase2 > 0:
        print("\n--- Dynamics Motion Similarity ---")
        dms = dynamics_motion_similarity_eval(
            latent_states_raw, available_families, model.dynamics,
        )
        print(f"  Cosine separation (dyn): {dms.velocity_cosine_separation:.2f}")
        results["dynamics_motion_similarity"] = asdict(dms)

    # --- Dynamics diagnostics ---
    if args.epochs_phase2 > 0:
        print("\n--- Dynamics Diagnostics ---")
        with torch.no_grad():
            all_vel, all_vel_terminal, all_dyn_errors = [], [], []
            for pid in list(latent_states_raw.keys())[:200]:
                z = latent_states_raw[pid].to(device)
                vel = model.dynamics.velocity_field(z)
                all_vel.append(vel.norm(dim=-1).mean().item())
                all_vel_terminal.append(vel[-1].norm().item())
                if z.shape[0] >= 2:
                    err = model.dynamics.dynamics_consistency_loss(
                        z, args.num_substeps,
                    ).item()
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

    # --- Regime metrics ---
    if args.epochs_phase2 > 0:
        print("\n--- Regime Metrics ---")
        regime_metrics = compute_regime_metrics(
            model.dynamics, latent_states_raw, available_families,
            device, args.num_substeps,
        )
        print(f"  Regime KL ratio:    {regime_metrics['regime_consistency_ratio']:.2f}")
        print(f"  Gate entropy mean:  {regime_metrics['gate_entropy_mean']:.4f}")
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
        print(f"  Gate vs task-family: {gate_tf_acc:.2%}")
        results["gate_task_family_accuracy"] = gate_tf_acc

    # --- Tube analysis (uses recovery-augmented rollout) ---
    tube_results = run_tube_analysis(
        model, latent_states, latent_states_raw, available_families, device,
        run_interventions=(args.epochs_phase2 > 0),
    )
    results["tube_analysis"] = tube_results

    # --- Recovery-specific analysis ---
    if args.epochs_phase3 > 0 and model.dynamics.has_tube_geometry:
        print("\n--- Recovery Analysis (v6) ---")
        recovery_results = run_recovery_analysis(
            model.dynamics,
            {pid: z.to(device) for pid, z in latent_states_raw.items()},
            available_families,
            eval_tube_geo,
            eval_fam_to_idx,
            device,
            progress_window=args.recovery_progress_window,
            backward_window=args.recovery_backward_window,
            forward_window=args.recovery_forward_window,
            run_attractor_analysis=args.run_attractor_analysis,
            attractor_num_samples=args.attractor_num_samples,
        )
        print(f"  Inferred recovery rate: {recovery_results['overall_recovery_rate']:.1%}")
        print(f"  Oracle recovery rate:   {recovery_results['oracle_overall_recovery_rate']:.1%}")
        print(f"  Null correction norm:  {recovery_results['mean_null_correction_norm']:.4f}")
        print(f"  Inferred same-family:   {recovery_results['same_family_anchor_rate']:.1%}")
        print(f"  Oracle same-family:     {recovery_results['oracle_same_family_anchor_rate']:.1%}")
        print(
            "  Mean final ||orig-pert|| (inferred): "
            f"{recovery_results['mean_final_reference_divergence_inferred']:.4f}"
        )
        print(
            "  Mean final ||orig-pert|| (oracle):   "
            f"{recovery_results['mean_final_reference_divergence_oracle']:.4f}"
        )
        for sc_str, sc_data in recovery_results["inferred_family"]["per_scale"].items():
            oracle_scale = recovery_results["oracle_family"]["per_scale"][sc_str]
            print(f"  Scale {sc_str}: "
                  f"infer_rec={sc_data['recovery_rate']:.1%} "
                  f"oracle_rec={oracle_scale['recovery_rate']:.1%} "
                  f"same={sc_data['same_family_anchor_rate']:.1%} "
                  f"back={sc_data['backward_progress_rate']:.1%}")
        results["recovery_analysis"] = recovery_results

    # --- Latent probes ---
    probe_results = run_latent_probes(
        model, latent_states_raw, PID_TO_FAMILY, available_families, device,
    )
    results["latent_probes"] = probe_results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SDQ v6 SUMMARY")
    print("=" * 60)
    print(f"Phase 1 epochs:       {args.epochs_phase1}")
    print(f"Phase 2 epochs:       {args.epochs_phase2}")
    print(f"Phase 3 epochs:       {args.epochs_phase3}")
    print(f"Phase 4 epochs:       {args.epochs_phase4}")
    print(f"Latent separation:    {separation:.2f}")
    print(f"Retrieval accuracy:   {ret.accuracy:.2%}")
    print(f"Scorecard overall:    {scorecard.overall_score():.1f}/100")
    if "dynamics_diagnostics" in results:
        dd = results["dynamics_diagnostics"]
        print(f"  Dynamics error:     {dd['mean_dynamics_error']:.4f}")
    if "dynamics_motion_similarity" in results:
        print(f"  Dyn vel cos sep:    "
              f"{results['dynamics_motion_similarity']['velocity_cosine_separation']:.2f}")
    tc = tube_results.get("coherence", {})
    tt = tube_results.get("transverse", {})
    print(f"  Tube tightening:    {tc.get('tube_tightening_ratio', 0):.3f}")
    if "intervention" in tube_results:
        ti = tube_results["intervention"]
        print(f"  Trans recovery:     {ti['transverse_recovery_rate']:.1%}")
        print(f"  Cross persistence:  {ti['cross_tube_persistence_rate']:.1%}")
        print(f"  Mean rec distance:  {ti['mean_recovery_distance']:.3f}")
    if "recovery_analysis" in results:
        ra = results["recovery_analysis"]
        print(f"  V6 inferred rec:    {ra['overall_recovery_rate']:.1%}")
        print(f"  V6 oracle rec:      {ra['oracle_overall_recovery_rate']:.1%}")
        print(f"  Null corr norm:     {ra['mean_null_correction_norm']:.4f}")
        print(f"  Same-family anchor: {ra['same_family_anchor_rate']:.1%}")
    if "dynamics_tube" in tube_results:
        dtr = tube_results["dynamics_tube"]
        print(f"  Dyn tightening:     {dtr['dynamics_tightening_ratio']:.3f}")
    print(f"--- Latent Probes ---")
    if "task_family" in probe_results:
        print(f"  Task family (test): {probe_results['task_family']['test_accuracy']:.1%}")
    if "answer_prediction" in probe_results:
        print(f"  Answer pred (test): {probe_results['answer_prediction']['test_accuracy']:.1%}")
    print("=" * 60)

    results["summary"] = {
        "epochs_phase1": args.epochs_phase1,
        "epochs_phase2": args.epochs_phase2,
        "epochs_phase3": args.epochs_phase3,
        "epochs_phase4": args.epochs_phase4,
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
        "regime_gate_entropy_mean": results.get("regime_metrics", {}).get("gate_entropy_mean", 0),
        "num_regimes": num_task_families,
        "gate_task_family_accuracy": results.get("gate_task_family_accuracy", 0),
        "v6_recovery_rate": results.get("recovery_analysis", {}).get("overall_recovery_rate", 0),
        "v6_oracle_recovery_rate": results.get("recovery_analysis", {}).get("oracle_overall_recovery_rate", 0),
        "v6_mean_final_reference_divergence_inferred": results.get(
            "recovery_analysis", {},
        ).get("mean_final_reference_divergence_inferred", 0),
        "v6_mean_final_reference_divergence_oracle": results.get(
            "recovery_analysis", {},
        ).get("mean_final_reference_divergence_oracle", 0),
        "v6_null_correction_norm": results.get("recovery_analysis", {}).get("mean_null_correction_norm", 0),
        "v6_same_family_anchor_rate": results.get("recovery_analysis", {}).get("same_family_anchor_rate", 0),
        "v6_wrong_family_anchor_rate": results.get("recovery_analysis", {}).get("wrong_family_anchor_rate", 0),
        "v6_oracle_same_family_anchor_rate": results.get("recovery_analysis", {}).get("oracle_same_family_anchor_rate", 0),
        "v6_backward_progress_rate": (
            sum(
                scale.get("backward_progress_rate", 0.0)
                for scale in results.get("recovery_analysis", {}).get("inferred_family", {}).get("per_scale", {}).values()
            ) / max(
                len(results.get("recovery_analysis", {}).get("inferred_family", {}).get("per_scale", {})),
                1,
            )
        ),
    }
    return results, model


def main():
    parser = argparse.ArgumentParser(
        description="SDQ v6: dual-dynamics recovery model",
    )
    # Phase 1
    parser.add_argument("--epochs-phase1", type=int, default=200)
    parser.add_argument("--lr-phase1", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--rec-weight", type=float, default=0.01)
    parser.add_argument("--blend-alpha", type=float, default=0.9)
    # Phase 2 (nominal dynamics)
    parser.add_argument("--epochs-phase2", type=int, default=400)
    parser.add_argument("--lr-phase2", type=float, default=5e-4)
    parser.add_argument("--vel-dir-weight", type=float, default=0.5)
    parser.add_argument("--mag-weight", type=float, default=0.5)
    parser.add_argument("--vcoh-weight", type=float, default=5.0)
    parser.add_argument("--num-regimes", type=int, default=6)
    parser.add_argument("--gate-sup-weight", type=float, default=5.0)
    parser.add_argument("--gate-sup-batch", type=int, default=4096)
    parser.add_argument("--gate-tau-high", type=float, default=1.0)
    parser.add_argument("--gate-tau-low", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=2.0)
    parser.add_argument("--terminal-margin", type=float, default=0.5)
    parser.add_argument("--regime-weight", type=float, default=2.0)
    parser.add_argument("--entropy-weight", type=float, default=0.5)
    parser.add_argument("--num-substeps", type=int, default=10)
    # Phase 3 (recovery)
    parser.add_argument("--epochs-phase3", type=int, default=200)
    parser.add_argument("--lr-phase3", type=float, default=1e-3)
    parser.add_argument("--recovery-substeps-final", type=int, default=50,
                        help="Final recovery rollout horizon; curriculum ramps up to this")
    parser.add_argument("--recovery-batch", type=int, default=256)
    parser.add_argument("--recover-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.5)
    parser.add_argument("--smoothness-weight", type=float, default=0.25)
    parser.add_argument("--basin-margin-weight", type=float, default=0.5)
    parser.add_argument("--basin-margin", type=float, default=0.5)
    parser.add_argument("--null-weight", type=float, default=0.5)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--recovery-progress-window", type=int, default=2,
                        help="Legacy symmetric lookup window used as fallback when directional windows are unset")
    parser.add_argument("--recovery-backward-window", type=int, default=0,
                        help="How far recovery lookup may search backward along the target tube")
    parser.add_argument("--recovery-forward-window", type=int, default=3,
                        help="How far recovery lookup may search forward along the target tube")
    parser.add_argument("--backward-progress-weight", type=float, default=1.0)
    parser.add_argument("--forward-overshoot-weight", type=float, default=0.25)
    parser.add_argument("--forward-overshoot-slack", type=int, default=2)
    parser.add_argument("--progress-coord-cap", type=int, default=256,
                        help="Clamp for discrete progress hints from progress_head")
    parser.add_argument("--reference-rollout-weight", type=float, default=1.0,
                        help="Pull perturbed recovery states toward nominal teacher rollout")
    parser.add_argument("--reference-rollout-early-weight", type=float, default=1.0,
                        help="Weight for teacher-matching loss in first half of recovery horizon")
    parser.add_argument("--reference-rollout-late-weight", type=float, default=0.5,
                        help="Weight for teacher-matching loss in second half of recovery horizon")
    parser.add_argument("--progress-head-weight", type=float, default=0.25,
                        help="MSE supervising progress_head vs normalized step index")
    parser.add_argument("--family-router-weight", type=float, default=0.5,
                        help="Cross-entropy for explicit semantic-family router")
    parser.add_argument("--stability-regularizer-weight", type=float, default=0.0,
                        help="Phase-4 regime consistency recovered-vs-teacher (optional)")
    # Phase 4 (joint)
    parser.add_argument("--epochs-phase4", type=int, default=50)
    parser.add_argument("--lr-phase4", type=float, default=1e-5)
    parser.add_argument("--dyn-weight-phase4", type=float, default=5.0)
    parser.add_argument("--run-attractor-analysis", action="store_true",
                        help="Run optional attractor/basin diagnostics after training")
    parser.add_argument("--attractor-num-samples", type=int, default=40,
                        help="Number of rollout seeds for optional attractor diagnostics")
    # Checkpoint / model
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--gauge-dim", type=int, default=32)
    parser.add_argument("--encode-batch", type=int, default=64)
    # Data
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--layer-range", type=int, nargs=2, metavar=("LO", "HI"), default=None)
    parser.add_argument("--norm-mode", type=str, default="standardize",
                        choices=["unit_norm", "center_scale", "standardize"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="artifacts/sdq_v6_results.json")
    args = parser.parse_args()

    t0 = time.time()
    results, model = run_training(args)
    elapsed = time.time() - t0

    results["elapsed_seconds"] = elapsed
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    ckpt_path = out_path.with_stem(out_path.stem + "_model")
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, str(ckpt_path.with_suffix(".safetensors")))
    print(f"Model checkpoint saved to {ckpt_path.with_suffix('.safetensors')}")


if __name__ == "__main__":
    main()
