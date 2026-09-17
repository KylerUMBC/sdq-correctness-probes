#!/usr/bin/env python3
"""SDQ v5 training: two-phase contrastive + dynamics pipeline.

Fixes v4a's representation collapse by using Supervised Contrastive Learning
(SupCon) as the primary objective in Phase 1, then adding dynamics/velocity
matching in Phase 2 with a fully frozen encoder.

Key differences from v4a:
  - SupCon (batch-level) replaces VICReg (interleaved per-pair)
  - Two-phase training: geometry first, then structure
  - Encoder gets reconstruction gradient (no z.detach()) in Phase 1
  - Encoder FROZEN in Phase 2 (prevents geometry destruction)
  - L2 normalization applied only to mean embeddings for SupCon + eval metrics
  - Fixed temperature=0.1 for sharp contrastive gradients throughout
  - 2-3 losses instead of 9+
  - Batch encoding instead of per-pair forward passes

Usage:
    python run_sdq_v5.py [--device cuda] [--epochs-phase1 200] [--epochs-phase2 50]
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

from sdq.instrumentation import collect_runs, set_seed
from sdq.trajectories import (
    extract_trajectory, Trajectory, normalize_trajectory, velocity_vectors,
)
from sdq.alignment import MonotoneAligner
from sdq.latent import (
    MultiScaleConvEncoder,
    ResidualDecoder,
    LatentODE,
)
from sdq.losses import reconstruction_loss
from sdq.losses.velocity import latent_velocity_cosine_loss
from sdq.eval.retrieval import retrieval_accuracy
from sdq.eval.motion_similarity import motion_similarity_eval
from sdq.eval.hard_negatives import hard_negative_test
from sdq.eval.semantic_scorecard import compute_scorecard
from sdq.eval.attractor_analysis import AttractorAnalysis, format_attractor_report
from sdq.eval.latent_probe import LatentProbe, train_probe, format_probe_result


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
# Model bundle (simplified — no gauge encoder, no transport)
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
def run_attractor_analysis(
    model: SDQv5Model,
    latent_states: dict[str, torch.Tensor],
    PID_TO_FAMILY: dict[str, str],
    device: torch.device,
) -> dict:
    print("\n" + "=" * 60)
    print("ATTRACTOR ANALYSIS")
    print("=" * 60)

    analyzer = AttractorAnalysis(
        dynamics=model.dynamics,
        convergence_threshold=1e-4,
        max_steps=2000,
        attractor_merge_radius=0.5,
    )

    z_trajectories: dict[str, torch.Tensor] = {}
    answer_ids: dict[str, str] = {}
    for pid, z in latent_states.items():
        z_trajectories[pid] = z.to(device)
        meta = _PID_META.get(pid, {})
        aid = meta.get("answer_id")
        if aid is not None:
            answer_ids[pid] = str(aid)
        elif pid in PID_TO_FAMILY:
            answer_ids[pid] = PID_TO_FAMILY[pid]

    report = analyzer.full_analysis(z_trajectories, answer_ids)
    print(format_attractor_report(report))

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

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = SDQv5Model(
        encoder=MultiScaleConvEncoder(hidden_dim, args.latent_dim),
        decoder=ResidualDecoder(args.latent_dim, args.gauge_dim, hidden_dim),
        dynamics=LatentODE(args.latent_dim),
        aligner=MonotoneAligner(hidden_dim, align_dim=min(128, hidden_dim)),
        proj_head=ProjectionHead(args.latent_dim, hidden_dim=256, output_dim=128),
        surface_classifier=SurfaceClassifier(args.latent_dim, num_variants),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # ------------------------------------------------------------------
    # 3. Phase 1: Contrastive Geometry
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1: CONTRASTIVE GEOMETRY")
    print("=" * 60)

    # Phase 1 trains encoder + decoder + projection head + surface classifier
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

    for epoch in range(1, args.epochs_phase1 + 1):
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
    # 4. Phase 2: SDQ Structure (dynamics + velocity)
    #    Encoder is FROZEN — only dynamics and aligner train.
    #    NOTE: Disabled by default (epochs_phase2=0) because:
    #      - Velocity loss has zero gradient (frozen detached inputs, no learnable transform)
    #      - Dynamics module trains but is never used in evaluation
    #      - All velocity alignment now handled by vel_cos loss in Phase 1
    # ------------------------------------------------------------------
    if args.epochs_phase2 > 0:
        print("\n" + "=" * 60)
        print("PHASE 2: SDQ STRUCTURE (encoder frozen)")
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
        # Detach all to avoid any accidental gradient flow
        frozen_z_full = {pid: z.detach() for pid, z in frozen_z_full.items()}
        frozen_z_mean = {pid: z.detach() for pid, z in frozen_z_mean.items()}

        phase2_params = list(model.dynamics.parameters()) + list(model.aligner.parameters())
        opt2 = torch.optim.AdamW(phase2_params, lr=args.lr_phase2, weight_decay=1e-4)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=args.epochs_phase2, eta_min=args.lr_phase2 * 0.01,
        )

        # Build same-family pairs for velocity matching
        all_train_pairs: list[tuple[str, str, str]] = []
        for fam_name, pids in train_families_avail.items():
            for p_i, p_j in combinations(pids, 2):
                all_train_pairs.append((fam_name, p_i, p_j))

        PAIRS_PER_EPOCH = min(50, len(all_train_pairs))

        for epoch in range(1, args.epochs_phase2 + 1):
            model.dynamics.train()
            model.aligner.train()

            # Pair-level: velocity cosine + dynamics consistency on frozen latents
            pairs = random.sample(all_train_pairs, PAIRS_PER_EPOCH)
            L_vel_total = torch.tensor(0.0, device=device)
            L_dyn_total = torch.tensor(0.0, device=device)
            for fam_name, pid_i, pid_j in pairs:
                z_i = frozen_z_full[pid_i]
                z_j = frozen_z_full[pid_j]

                # Velocity cosine on the frozen latent trajectories
                T_min = min(z_i.shape[0], z_j.shape[0])
                L_vel_total = L_vel_total + latent_velocity_cosine_loss(
                    z_i[:T_min], z_j[:T_min],
                )
                L_dyn_total = L_dyn_total + model.dynamics.dynamics_consistency_loss(z_i)

            L_vel = L_vel_total / PAIRS_PER_EPOCH
            L_dyn = L_dyn_total / PAIRS_PER_EPOCH

            loss = args.vel_weight * L_vel + args.dyn_weight * L_dyn

            opt2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(phase2_params, max_norm=1.0)
            opt2.step()
            sched2.step()

            entry = {
                "phase": 2, "epoch": epoch,
                "velocity": L_vel.item(),
                "dynamics": L_dyn.item(),
                "total": loss.item(),
                "lr": sched2.get_last_lr()[0],
            }
            history.append(entry)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  P2 Epoch {epoch:3d} | "
                    f"vel={L_vel.item():.4f} "
                    f"dyn={L_dyn.item():.4f} "
                    f"total={loss.item():.4f} "
                    f"lr={sched2.get_last_lr()[0]:.6f}"
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

    # --- Attractor analysis ---
    attractor_results = run_attractor_analysis(model, latent_states_raw, PID_TO_FAMILY, device)
    results["attractor_analysis"] = attractor_results

    # --- Latent probes ---
    probe_results = run_latent_probes(model, latent_states_raw, PID_TO_FAMILY, available_families, device)
    results["latent_probes"] = probe_results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SDQ v5 SUMMARY")
    print("=" * 60)
    print(f"Phase 1 epochs:       {args.epochs_phase1}")
    print(f"Phase 2 epochs:       {args.epochs_phase2}")
    print(f"Normalization:        {norm_mode}")
    print(f"Temperature:          {args.temperature}")
    print(f"Latent separation:    {separation:.2f}")
    print(f"Retrieval accuracy:   {ret.accuracy:.2%}")
    print(f"Motion cos sep:       {ms.velocity_cosine_separation:.2f}")
    print(f"Scorecard overall:    {scorecard.overall_score():.1f}/100")
    print(f"--- Attractor Analysis ---")
    print(f"  Attractors found:   {attractor_results['num_attractors']}")
    print(f"  Purity:             {attractor_results['answer_attractor_purity']:.1%}")
    print(f"  Basin separation:   {attractor_results['basin_separation_score']:.2f}")
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
        "norm_mode": norm_mode,
        "temperature": args.temperature,
        "blend_alpha": args.blend_alpha,
        "latent_separation": separation,
        "retrieval_accuracy": ret.accuracy,
        "retrieval_mrr": ret.mean_reciprocal_rank,
        "motion_cosine_separation": ms.velocity_cosine_separation,
        "scorecard": scorecard.overall_score(),
        "num_attractors": attractor_results["num_attractors"],
        "answer_attractor_purity": attractor_results["answer_attractor_purity"],
        "basin_separation_score": attractor_results["basin_separation_score"],
        "task_family_probe_acc": probe_results.get("task_family", {}).get("test_accuracy", 0),
        "semantic_group_probe_acc": probe_results.get("semantic_group", {}).get("test_accuracy", 0),
        "surface_variant_probe_acc": probe_results.get("surface_variant", {}).get("test_accuracy", 0),
        "answer_prediction_probe_acc": probe_results.get("answer_prediction", {}).get("test_accuracy", 0),
    }
    return results, model


def main():
    parser = argparse.ArgumentParser(description="SDQ v5 two-phase training")
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
    # Phase 2 args
    parser.add_argument("--epochs-phase2", type=int, default=0)
    parser.add_argument("--lr-phase2", type=float, default=3e-4)
    parser.add_argument("--vel-weight", type=float, default=1.0,
                        help="Velocity cosine weight in Phase 2 (default 1.0)")
    parser.add_argument("--dyn-weight", type=float, default=0.5,
                        help="Dynamics consistency weight in Phase 2 (default 0.5)")
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
    parser.add_argument("--output", type=str, default="artifacts/sdq_v5_results.json")
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
    ckpt_path = out_path.with_suffix(".pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
