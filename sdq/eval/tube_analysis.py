"""Tube / manifold analysis for latent dynamics.

Replaces point-attractor analysis with the tube paradigm: reasoning is
a controlled evolution through a constrained region of state space, not
convergence to a fixed point.  Different prompts with the same task may
begin far apart, then get pulled into the same dynamical regime while
retaining degrees of freedom along the way.

Five experiments:
    1. Tube Coherence      — do same-family trajectories stay close? tighten?
    2. Transverse Contract — is spread transverse to the mean path contracting?
    3. Reasoning Stage     — does along-tube position predict reasoning stage?
    4. Tube Separation     — do different answers live in different tubes?
    5. Tube Intervention   — can transverse perturbations recover?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TubeCoherenceResult:
    """Experiment 1: do same-family trajectories stay close?"""
    per_timestep_within_spread: list[float]   # [T_min] avg within-family L2
    per_timestep_cross_spread: list[float]     # [T_min] avg cross-family L2
    tube_tightening_ratio: float               # spread_last / spread_first (< 1 = tightening)
    per_timestep_classification_acc: list[float]  # [T_min] 1-NN family accuracy
    mean_within_spread: float
    mean_cross_spread: float
    separation_over_time: list[float]          # cross / within at each t


@dataclass
class TransverseContractionResult:
    """Experiment 2: decompose spread into along-tube and transverse."""
    per_timestep_along_spread: list[float]     # [T_min] along the mean path
    per_timestep_transverse_spread: list[float]  # [T_min] perpendicular
    transverse_contraction_ratio: float        # transverse_last / transverse_first
    along_fraction: float                      # fraction of total spread that is along-tube
    transverse_fraction: float                 # fraction that is transverse


@dataclass
class ReasoningStageResult:
    """Experiment 3: does along-tube position predict reasoning stage?"""
    timestep_probe_accuracy: float      # linear probe: along-tube → timestep
    along_tube_monotonicity: float      # Spearman rank corr of along-tube coord vs t
    mean_along_velocity: float          # avg speed along the tube direction


@dataclass
class TubeSeparationResult:
    """Experiment 4: do different answers live in different tubes?"""
    per_timestep_classification_margin: list[float]  # [T_min] margin between closest pair
    mean_classification_margin: float
    tube_overlap_fraction: float         # frac of timesteps where tubes overlap
    confused_pairs: list[tuple[str, str, float]]  # (fam_a, fam_b, overlap)


@dataclass
class TubeInterventionResult:
    """Experiment 5: tube-relative interventions."""
    transverse_recovery_rate: float      # frac of transverse perturbations that return
    cross_tube_persistence_rate: float   # frac of cross-tube perturbations that stay
    mean_transverse_recovery_steps: float
    mean_recovery_distance: float        # distance at end / distance at start


@dataclass
class DynamicsTubeResult:
    """Experiment 6: dynamics-predicted tube coherence."""
    per_step_within_spread: list[float]      # [rollout_steps] avg within-family L2
    per_step_cross_spread: list[float]       # [rollout_steps] avg cross-family L2
    dynamics_tightening_ratio: float         # within_last / within_first (< 1 = tightening)
    per_step_vel_cosine: list[float]         # [rollout_steps-1] within-family velocity cosine
    per_step_classification_acc: list[float] # [rollout_steps] 1-NN family accuracy


@dataclass
class TubeAnalysisReport:
    """Full tube analysis report."""
    coherence: TubeCoherenceResult
    transverse: TransverseContractionResult
    reasoning_stage: ReasoningStageResult
    separation: TubeSeparationResult
    intervention: TubeInterventionResult | None
    dynamics_tube: DynamicsTubeResult | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_trajectories(
    trajectories: list[Tensor],
) -> tuple[Tensor, int]:
    """Truncate trajectories to common length, stack into [N, T_min, D].

    Args:
        trajectories: list of [T_i, D] tensors.

    Returns:
        (stacked [N, T_min, D], T_min)
    """
    T_min = min(z.shape[0] for z in trajectories)
    stacked = torch.stack([z[:T_min] for z in trajectories])  # [N, T_min, D]
    return stacked, T_min


def _mean_path(stacked: Tensor) -> Tensor:
    """Mean trajectory across the family: [T_min, D]."""
    return stacked.mean(dim=0)


def _tangent_vectors(mean_path: Tensor) -> Tensor:
    """Unit tangent vectors along the mean path: [T_min-1, D].

    Returns normalized dz/dt of the mean path.
    """
    dz = mean_path[1:] - mean_path[:-1]  # [T-1, D]
    norms = dz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return dz / norms


# ---------------------------------------------------------------------------
# Experiment 1: Tube Coherence
# ---------------------------------------------------------------------------

def tube_coherence(
    latent_states: dict[str, Tensor],
    families: dict[str, list[str]],
) -> TubeCoherenceResult:
    """Do same-family trajectories stay close? Does the tube tighten?

    For each timestep, computes within-family and cross-family L2 spread,
    plus a 1-NN classifier accuracy.
    """
    fam_names = sorted(families.keys())

    # Build per-family stacked tensors (truncated to family T_min)
    family_stacks: dict[str, Tensor] = {}
    for fam in fam_names:
        pids = [p for p in families[fam] if p in latent_states]
        if len(pids) < 2:
            continue
        trajs = [latent_states[p] for p in pids]
        stacked, _ = _align_trajectories(trajs)
        family_stacks[fam] = stacked  # [N_fam, T_fam, D]

    if not family_stacks:
        return TubeCoherenceResult(
            per_timestep_within_spread=[], per_timestep_cross_spread=[],
            tube_tightening_ratio=1.0, per_timestep_classification_acc=[],
            mean_within_spread=0.0, mean_cross_spread=0.0,
            separation_over_time=[],
        )

    # Global T_min across families
    T_min = min(s.shape[1] for s in family_stacks.values())
    valid_fams = sorted(family_stacks.keys())

    # Per-timestep within-family spread
    within_spread = []
    for t in range(T_min):
        dists = []
        for fam in valid_fams:
            stack = family_stacks[fam][:, t, :]  # [N_fam, D]
            centroid = stack.mean(dim=0, keepdim=True)
            dists.append((stack - centroid).norm(dim=-1).mean().item())
        within_spread.append(sum(dists) / len(dists))

    # Per-timestep cross-family spread (sample pairs)
    cross_spread = []
    for t in range(T_min):
        dists = []
        for fi, fam_i in enumerate(valid_fams):
            centroid_i = family_stacks[fam_i][:, t, :].mean(dim=0)
            for fam_j in valid_fams[fi + 1:]:
                centroid_j = family_stacks[fam_j][:, t, :].mean(dim=0)
                dists.append((centroid_i - centroid_j).norm().item())
        cross_spread.append(sum(dists) / max(len(dists), 1))

    # Tube tightening
    if within_spread[0] > 1e-8:
        tightening = within_spread[-1] / within_spread[0]
    else:
        tightening = 1.0

    # Per-timestep 1-NN classification
    classification_acc = []
    for t in range(T_min):
        # Collect all points with labels
        all_points = []
        all_labels = []
        for li, fam in enumerate(valid_fams):
            pts = family_stacks[fam][:, t, :]  # [N_fam, D]
            all_points.append(pts)
            all_labels.extend([li] * pts.shape[0])
        all_points = torch.cat(all_points, dim=0)  # [N_total, D]
        all_labels = torch.tensor(all_labels, device=all_points.device)

        # Leave-one-out 1-NN
        dists = torch.cdist(all_points, all_points)  # [N, N]
        dists.fill_diagonal_(float('inf'))
        nn_idx = dists.argmin(dim=1)
        correct = (all_labels[nn_idx] == all_labels).float().mean().item()
        classification_acc.append(correct)

    separation = [
        c / max(w, 1e-8) for w, c in zip(within_spread, cross_spread)
    ]

    return TubeCoherenceResult(
        per_timestep_within_spread=within_spread,
        per_timestep_cross_spread=cross_spread,
        tube_tightening_ratio=tightening,
        per_timestep_classification_acc=classification_acc,
        mean_within_spread=sum(within_spread) / len(within_spread),
        mean_cross_spread=sum(cross_spread) / len(cross_spread),
        separation_over_time=separation,
    )


# ---------------------------------------------------------------------------
# Experiment 2: Transverse Contraction
# ---------------------------------------------------------------------------

def transverse_contraction(
    latent_states: dict[str, Tensor],
    families: dict[str, list[str]],
) -> TransverseContractionResult:
    """Decompose within-family spread into along-tube and transverse components.

    The mean path defines the tube axis.  For each family member at each
    timestep, project the deviation from centroid onto the local tangent
    (along-tube) and take the residual (transverse).
    """
    fam_names = sorted(families.keys())

    all_along = []
    all_transverse = []

    for fam in fam_names:
        pids = [p for p in families[fam] if p in latent_states]
        if len(pids) < 2:
            continue
        trajs = [latent_states[p] for p in pids]
        stacked, T_min = _align_trajectories(trajs)  # [N, T, D]
        if T_min < 2:
            continue

        mean = _mean_path(stacked)  # [T, D]
        tangents = _tangent_vectors(mean)  # [T-1, D]

        # For each timestep (except last, where we don't have tangent)
        along_per_t = []
        trans_per_t = []
        for t in range(T_min - 1):
            deviations = stacked[:, t, :] - mean[t].unsqueeze(0)  # [N, D]
            tang = tangents[t]  # [D]

            # Project onto tangent
            along_proj = (deviations * tang).sum(dim=-1, keepdim=True)  # [N, 1]
            along_component = along_proj * tang  # [N, D]
            transverse_component = deviations - along_component  # [N, D]

            along_per_t.append(along_component.norm(dim=-1).mean().item())
            trans_per_t.append(transverse_component.norm(dim=-1).mean().item())

        all_along.append(along_per_t)
        all_transverse.append(trans_per_t)

    if not all_along:
        return TransverseContractionResult(
            per_timestep_along_spread=[], per_timestep_transverse_spread=[],
            transverse_contraction_ratio=1.0,
            along_fraction=0.5, transverse_fraction=0.5,
        )

    # Average across families
    T = min(len(a) for a in all_along)
    avg_along = [
        sum(a[t] for a in all_along) / len(all_along) for t in range(T)
    ]
    avg_trans = [
        sum(a[t] for a in all_transverse) / len(all_transverse) for t in range(T)
    ]

    # Transverse contraction ratio
    if avg_trans[0] > 1e-8:
        trans_ratio = avg_trans[-1] / avg_trans[0]
    else:
        trans_ratio = 1.0

    # Along vs transverse fractions
    total_along = sum(avg_along)
    total_trans = sum(avg_trans)
    total = total_along + total_trans + 1e-8
    along_frac = total_along / total
    trans_frac = total_trans / total

    return TransverseContractionResult(
        per_timestep_along_spread=avg_along,
        per_timestep_transverse_spread=avg_trans,
        transverse_contraction_ratio=trans_ratio,
        along_fraction=along_frac,
        transverse_fraction=trans_frac,
    )


# ---------------------------------------------------------------------------
# Experiment 3: Reasoning Stage Predictiveness
# ---------------------------------------------------------------------------

def reasoning_stage_analysis(
    latent_states: dict[str, Tensor],
    families: dict[str, list[str]],
) -> ReasoningStageResult:
    """Does along-tube position predict reasoning stage (timestep)?

    Projects each trajectory point onto the family mean path and checks
    whether the projection coordinate increases monotonically with t.
    Also trains a tiny linear probe: along-tube coord → timestep bin.
    """
    fam_names = sorted(families.keys())

    monotonicity_scores = []
    along_velocities = []
    probe_X = []
    probe_y = []

    for fam in fam_names:
        pids = [p for p in families[fam] if p in latent_states]
        if len(pids) < 2:
            continue
        trajs = [latent_states[p] for p in pids]
        stacked, T_min = _align_trajectories(trajs)  # [N, T, D]
        if T_min < 3:
            continue

        mean = _mean_path(stacked)  # [T, D]
        tangents = _tangent_vectors(mean)  # [T-1, D]

        for n in range(stacked.shape[0]):
            # Cumulative along-tube coordinate
            along_coords = []
            cumulative = 0.0
            for t in range(T_min - 1):
                dev = stacked[n, t] - mean[t]
                proj = (dev * tangents[t]).sum().item()
                # Also add the step along the mean path
                step = (mean[t + 1] - mean[t]).norm().item()
                cumulative += step + proj
                along_coords.append(cumulative)

            along_coords_t = torch.tensor(along_coords)
            timesteps = torch.arange(len(along_coords), dtype=torch.float32)

            # Spearman rank correlation (monotonicity)
            rank_x = along_coords_t.argsort().argsort().float()
            rank_y = timesteps.argsort().argsort().float()
            rank_x = rank_x - rank_x.mean()
            rank_y = rank_y - rank_y.mean()
            denom = rank_x.norm() * rank_y.norm()
            if denom > 1e-8:
                spearman = (rank_x * rank_y).sum() / denom
                monotonicity_scores.append(spearman.item())

            # Along-tube velocity
            if len(along_coords) >= 2:
                diffs = along_coords_t[1:] - along_coords_t[:-1]
                along_velocities.append(diffs.mean().item())

            # For probe: normalized along-tube coord → timestep bin
            if len(along_coords) >= 2:
                ac = along_coords_t
                ac_norm = (ac - ac.min()) / (ac.max() - ac.min() + 1e-8)
                # Bin timesteps into 5 bins
                T_eff = len(along_coords)
                bins = torch.clamp((timesteps / T_eff * 5).long(), 0, 4)
                for i in range(len(ac_norm)):
                    probe_X.append(ac_norm[i].unsqueeze(0))
                    probe_y.append(bins[i])

    avg_mono = sum(monotonicity_scores) / max(len(monotonicity_scores), 1)
    avg_vel = sum(along_velocities) / max(len(along_velocities), 1)

    # Simple linear probe: along-tube scalar → 5-class timestep bin
    probe_acc = 0.0
    if len(probe_X) > 20:
        X = torch.stack(probe_X)  # [N, 1]
        y = torch.tensor(probe_y)  # [N]
        # Train/test split
        n = X.shape[0]
        perm = torch.randperm(n)
        split = int(0.7 * n)
        X_tr, X_te = X[perm[:split]], X[perm[split:]]
        y_tr, y_te = y[perm[:split]], y[perm[split:]]
        # Fit linear probe (closed-form via least squares isn't great for
        # classification, so use a small SGD loop)
        probe = torch.nn.Linear(1, 5)
        opt = torch.optim.Adam(probe.parameters(), lr=0.01)
        for _ in range(200):
            logits = probe(X_tr)
            loss = F.cross_entropy(logits, y_tr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = probe(X_te).argmax(dim=1)
            probe_acc = (pred == y_te).float().mean().item()

    return ReasoningStageResult(
        timestep_probe_accuracy=probe_acc,
        along_tube_monotonicity=avg_mono,
        mean_along_velocity=avg_vel,
    )


# ---------------------------------------------------------------------------
# Experiment 4: Tube Separation
# ---------------------------------------------------------------------------

def tube_separation(
    latent_states: dict[str, Tensor],
    families: dict[str, list[str]],
) -> TubeSeparationResult:
    """Do different answers live in different tubes?

    At each timestep, measures the margin between family centroids
    relative to within-family spread.  Identifies confused pairs.
    """
    fam_names = sorted(families.keys())

    # Build per-family stacks
    family_stacks: dict[str, Tensor] = {}
    for fam in fam_names:
        pids = [p for p in families[fam] if p in latent_states]
        if len(pids) < 2:
            continue
        trajs = [latent_states[p] for p in pids]
        stacked, _ = _align_trajectories(trajs)
        family_stacks[fam] = stacked

    valid_fams = sorted(family_stacks.keys())
    if len(valid_fams) < 2:
        return TubeSeparationResult(
            per_timestep_classification_margin=[], mean_classification_margin=0.0,
            tube_overlap_fraction=0.0, confused_pairs=[],
        )

    T_min = min(s.shape[1] for s in family_stacks.values())

    per_t_margin = []
    overlap_count = 0
    pair_overlaps: dict[tuple[str, str], float] = {}

    for t in range(T_min):
        # Compute centroids and spreads
        centroids = {}
        spreads = {}
        for fam in valid_fams:
            pts = family_stacks[fam][:, t, :]  # [N, D]
            centroids[fam] = pts.mean(dim=0)
            spreads[fam] = (pts - centroids[fam].unsqueeze(0)).norm(dim=-1).mean().item()

        # Pairwise margin: (centroid_dist - spread_a - spread_b)
        margins = []
        for fi, fa in enumerate(valid_fams):
            for fb in valid_fams[fi + 1:]:
                dist = (centroids[fa] - centroids[fb]).norm().item()
                margin = dist - spreads[fa] - spreads[fb]
                margins.append(margin)

                if margin < 0:
                    key = (fa, fb)
                    pair_overlaps[key] = pair_overlaps.get(key, 0) + 1.0

        avg_margin = sum(margins) / max(len(margins), 1)
        per_t_margin.append(avg_margin)
        if avg_margin < 0:
            overlap_count += 1

    tube_overlap_frac = overlap_count / max(T_min, 1)

    # Normalize overlap counts and find worst pairs
    confused = []
    for (fa, fb), count in sorted(pair_overlaps.items(), key=lambda x: -x[1]):
        confused.append((fa, fb, count / T_min))
    confused = confused[:10]  # top 10

    mean_margin = sum(per_t_margin) / max(len(per_t_margin), 1)

    return TubeSeparationResult(
        per_timestep_classification_margin=per_t_margin,
        mean_classification_margin=mean_margin,
        tube_overlap_fraction=tube_overlap_frac,
        confused_pairs=confused,
    )


# ---------------------------------------------------------------------------
# Experiment 5: Tube-Relative Interventions
# ---------------------------------------------------------------------------

@torch.no_grad()
def tube_intervention(
    latent_states: dict[str, Tensor],
    families: dict[str, list[str]],
    dynamics,
    perturbation_scale: float = 1.0,
    rollout_steps: int = 50,
    num_samples: int = 30,
) -> TubeInterventionResult:
    """Perturb trajectories transversely and cross-tube, measure recovery.

    Transverse perturbation: perpendicular to the family mean path direction.
    Cross-tube perturbation: toward a different family's centroid.

    Recovery = does the perturbed rollout return closer to the tube?
    """
    fam_names = sorted(families.keys())

    # Build per-family mean paths
    family_means: dict[str, Tensor] = {}
    family_tangents: dict[str, Tensor] = {}
    for fam in fam_names:
        pids = [p for p in families[fam] if p in latent_states]
        if len(pids) < 2:
            continue
        trajs = [latent_states[p] for p in pids]
        stacked, T_min = _align_trajectories(trajs)
        if T_min < 3:
            continue
        mean = _mean_path(stacked)  # [T, D]
        family_means[fam] = mean
        family_tangents[fam] = _tangent_vectors(mean)

    valid_fams = sorted(family_means.keys())
    if len(valid_fams) < 2:
        return TubeInterventionResult(
            transverse_recovery_rate=0.0, cross_tube_persistence_rate=0.0,
            mean_transverse_recovery_steps=0.0, mean_recovery_distance=1.0,
        )

    transverse_recoveries = []
    cross_tube_persist = []
    recovery_steps_list = []
    recovery_dists = []

    samples_done = 0
    for fam in valid_fams:
        if samples_done >= num_samples:
            break
        pids = [p for p in families[fam] if p in latent_states]
        mean = family_means[fam]
        tangents = family_tangents[fam]
        T_min = mean.shape[0]
        t_mid = T_min // 2

        for pid in pids[:3]:
            if samples_done >= num_samples:
                break
            z = latent_states[pid][:T_min].to(mean.device)
            z_start = z[t_mid]  # [D]

            # --- Transverse perturbation ---
            tang = tangents[min(t_mid, tangents.shape[0] - 1)]  # [D]
            random_dir = torch.randn_like(z_start)
            # Remove tangent component → purely transverse
            random_dir = random_dir - (random_dir @ tang) * tang
            random_dir = random_dir / random_dir.norm().clamp(min=1e-8)
            pert_transverse = random_dir * perturbation_scale

            z_pert = z_start + pert_transverse

            # Rollout both
            orig_traj = dynamics.rollout(z_start, rollout_steps)  # [T, D]
            pert_traj = dynamics.rollout(z_pert, rollout_steps)    # [T, D]

            divergence = (orig_traj - pert_traj).norm(dim=-1)  # [T]
            initial_div = divergence[0].item()
            final_div = divergence[-1].item()

            # Recovery: final divergence < 50% of initial
            recovered = final_div < initial_div * 0.5
            transverse_recoveries.append(recovered)

            if initial_div > 1e-8:
                recovery_dists.append(final_div / initial_div)

            # Find first step where divergence drops below 50%
            half_thresh = initial_div * 0.5
            rec_step = rollout_steps
            for s in range(rollout_steps):
                if divergence[s].item() < half_thresh:
                    rec_step = s
                    break
            recovery_steps_list.append(rec_step)

            # --- Cross-tube perturbation ---
            other_fams = [f for f in valid_fams if f != fam]
            if other_fams:
                other_fam = other_fams[0]
                other_mean = family_means[other_fam]
                other_t = min(t_mid, other_mean.shape[0] - 1)
                cross_dir = other_mean[other_t] - z_start
                cross_dir = cross_dir / cross_dir.norm().clamp(min=1e-8)
                pert_cross = cross_dir * perturbation_scale

                z_pert_cross = z_start + pert_cross
                cross_traj = dynamics.rollout(z_pert_cross, rollout_steps)
                cross_div = (orig_traj - cross_traj).norm(dim=-1)
                # Persistence: divergence stays or grows
                persists = cross_div[-1].item() >= cross_div[0].item() * 0.5
                cross_tube_persist.append(persists)

            samples_done += 1

    trans_recovery_rate = sum(transverse_recoveries) / max(len(transverse_recoveries), 1)
    cross_persist_rate = sum(cross_tube_persist) / max(len(cross_tube_persist), 1)
    mean_rec_steps = sum(recovery_steps_list) / max(len(recovery_steps_list), 1)
    mean_rec_dist = sum(recovery_dists) / max(len(recovery_dists), 1)

    return TubeInterventionResult(
        transverse_recovery_rate=trans_recovery_rate,
        cross_tube_persistence_rate=cross_persist_rate,
        mean_transverse_recovery_steps=mean_rec_steps,
        mean_recovery_distance=mean_rec_dist,
    )


# ---------------------------------------------------------------------------
# Experiment 6: Dynamics Tube Coherence
# ---------------------------------------------------------------------------

@torch.no_grad()
def dynamics_tube_coherence(
    latent_states_raw: dict[str, Tensor],
    families: dict[str, list[str]],
    dynamics,
    rollout_steps: int = 20,
) -> DynamicsTubeResult:
    """Rollout raw states through dynamics and measure family clustering.

    For each family member, takes z at t=T//2 and rolls out through
    dynamics.rollout(). Then measures whether the dynamics-predicted
    trajectories cluster by family — independent of encoder blending.

    Args:
        latent_states_raw: {prompt_id: [T, D]} raw latent trajectories.
        families: {family_name: [prompt_id, ...]}.
        dynamics: trained LatentODE with .rollout() and .velocity_field().
        rollout_steps: number of dynamics steps to roll out.

    Returns:
        DynamicsTubeResult with per-step metrics.
    """
    fam_names = sorted(families.keys())

    # Collect starting points and family labels for each member
    start_points: list[Tensor] = []     # [D] each
    fam_labels: list[int] = []
    valid_fams: list[str] = []
    fam_idx_map: dict[str, int] = {}

    for fam in fam_names:
        pids = [p for p in families[fam] if p in latent_states_raw]
        if len(pids) < 2:
            continue
        if fam not in fam_idx_map:
            fam_idx_map[fam] = len(valid_fams)
            valid_fams.append(fam)
        fi = fam_idx_map[fam]
        for pid in pids:
            z = latent_states_raw[pid]
            t_mid = z.shape[0] // 2
            start_points.append(z[t_mid])
            fam_labels.append(fi)

    if len(start_points) < 4 or len(valid_fams) < 2:
        return DynamicsTubeResult(
            per_step_within_spread=[], per_step_cross_spread=[],
            dynamics_tightening_ratio=1.0,
            per_step_vel_cosine=[], per_step_classification_acc=[],
        )

    # Stack starting points and rollout all at once
    starts = torch.stack(start_points)  # [N, D]
    labels = torch.tensor(fam_labels, device=starts.device)  # [N]

    # Rollout each point: [rollout_steps, N, D]
    # dynamics.rollout expects [N, D] and returns [T, N, D]
    trajectories = dynamics.rollout(starts, rollout_steps)  # [T, N, D]

    # Per-step metrics
    per_step_within = []
    per_step_cross = []
    per_step_acc = []
    per_step_vel_cos = []

    for step in range(rollout_steps):
        pts = trajectories[step]  # [N, D]

        # Within-family spread: avg distance to family centroid
        within_dists = []
        centroids = {}
        for fi in range(len(valid_fams)):
            mask = labels == fi
            if mask.sum() < 2:
                continue
            fam_pts = pts[mask]  # [N_fam, D]
            centroid = fam_pts.mean(dim=0, keepdim=True)
            centroids[fi] = centroid.squeeze(0)
            within_dists.append((fam_pts - centroid).norm(dim=-1).mean().item())
        per_step_within.append(sum(within_dists) / max(len(within_dists), 1))

        # Cross-family spread: avg distance between centroids
        cross_dists = []
        cent_keys = sorted(centroids.keys())
        for i, fi in enumerate(cent_keys):
            for fj in cent_keys[i + 1:]:
                cross_dists.append((centroids[fi] - centroids[fj]).norm().item())
        per_step_cross.append(sum(cross_dists) / max(len(cross_dists), 1))

        # 1-NN classification
        dists_mat = torch.cdist(pts, pts)  # [N, N]
        dists_mat.fill_diagonal_(float('inf'))
        nn_idx = dists_mat.argmin(dim=1)
        correct = (labels[nn_idx] == labels).float().mean().item()
        per_step_acc.append(correct)

    # Per-step within-family velocity cosine
    for step in range(rollout_steps - 1):
        vel = trajectories[step + 1] - trajectories[step]  # [N, D]
        cos_sims = []
        for fi in range(len(valid_fams)):
            mask = labels == fi
            if mask.sum() < 2:
                continue
            fam_vel = vel[mask]  # [N_fam, D]
            # Pairwise cosine similarity within family
            n_fam = fam_vel.shape[0]
            for i in range(n_fam):
                for j in range(i + 1, n_fam):
                    cos = F.cosine_similarity(
                        fam_vel[i].unsqueeze(0), fam_vel[j].unsqueeze(0),
                    ).item()
                    cos_sims.append(cos)
        per_step_vel_cos.append(sum(cos_sims) / max(len(cos_sims), 1))

    # Tightening ratio
    if per_step_within and per_step_within[0] > 1e-8:
        tightening = per_step_within[-1] / per_step_within[0]
    else:
        tightening = 1.0

    return DynamicsTubeResult(
        per_step_within_spread=per_step_within,
        per_step_cross_spread=per_step_cross,
        dynamics_tightening_ratio=tightening,
        per_step_vel_cosine=per_step_vel_cos,
        per_step_classification_acc=per_step_acc,
    )


# ---------------------------------------------------------------------------
# Full analysis + formatting
# ---------------------------------------------------------------------------

def full_tube_analysis(
    latent_states: dict[str, Tensor],
    families: dict[str, list[str]],
    dynamics=None,
    run_interventions: bool = True,
    latent_states_raw: dict[str, Tensor] | None = None,
) -> TubeAnalysisReport:
    """Run all five tube experiments.

    Args:
        latent_states: {prompt_id: [T, D]} blended latent trajectories.
            Used for experiments 1-4 (geometry-based).
        families: {family_name: [prompt_id, ...]}.
        dynamics: trained LatentODE (needed for experiment 5).
        run_interventions: whether to run experiment 5 (requires dynamics).
        latent_states_raw: {prompt_id: [T, D]} raw (un-blended) trajectories.
            Used for experiment 5 (dynamics-based interventions).
            Falls back to latent_states if not provided.

    Returns:
        TubeAnalysisReport with all five results.
    """
    coherence = tube_coherence(latent_states, families)
    transverse = transverse_contraction(latent_states, families)
    stage = reasoning_stage_analysis(latent_states, families)
    separation = tube_separation(latent_states, families)

    intervention = None
    dyn_tube = None
    if run_interventions and dynamics is not None:
        # Use raw states for dynamics-based experiments
        intervention_states = latent_states_raw if latent_states_raw is not None else latent_states
        intervention = tube_intervention(intervention_states, families, dynamics)
        dyn_tube = dynamics_tube_coherence(intervention_states, families, dynamics)

    return TubeAnalysisReport(
        coherence=coherence,
        transverse=transverse,
        reasoning_stage=stage,
        separation=separation,
        intervention=intervention,
        dynamics_tube=dyn_tube,
    )


def format_tube_report(report: TubeAnalysisReport) -> str:
    """Human-readable summary of tube analysis."""
    c = report.coherence
    t = report.transverse
    s = report.reasoning_stage
    sep = report.separation
    lines = [
        "=" * 60,
        "TUBE / MANIFOLD ANALYSIS",
        "=" * 60,
        "",
        "--- Experiment 1: Tube Coherence ---",
        f"  Mean within-family spread:   {c.mean_within_spread:.4f}",
        f"  Mean cross-family spread:    {c.mean_cross_spread:.4f}",
        f"  Tube tightening ratio:       {c.tube_tightening_ratio:.3f}  "
        f"({'tightens' if c.tube_tightening_ratio < 0.9 else 'stable' if c.tube_tightening_ratio < 1.1 else 'expands'})",
    ]
    if c.per_timestep_classification_acc:
        acc_start = c.per_timestep_classification_acc[0]
        acc_end = c.per_timestep_classification_acc[-1]
        acc_mean = sum(c.per_timestep_classification_acc) / len(c.per_timestep_classification_acc)
        lines.append(f"  1-NN classification (start):  {acc_start:.1%}")
        lines.append(f"  1-NN classification (end):    {acc_end:.1%}")
        lines.append(f"  1-NN classification (mean):   {acc_mean:.1%}")
    if c.separation_over_time:
        sep_start = c.separation_over_time[0]
        sep_end = c.separation_over_time[-1]
        lines.append(f"  Separation ratio (start):    {sep_start:.2f}")
        lines.append(f"  Separation ratio (end):      {sep_end:.2f}")

    lines.extend([
        "",
        "--- Experiment 2: Transverse Contraction ---",
        f"  Transverse contraction ratio: {t.transverse_contraction_ratio:.3f}  "
        f"({'contracting' if t.transverse_contraction_ratio < 0.9 else 'neutral' if t.transverse_contraction_ratio < 1.1 else 'expanding'})",
        f"  Along-tube fraction:         {t.along_fraction:.1%}",
        f"  Transverse fraction:         {t.transverse_fraction:.1%}",
    ])

    lines.extend([
        "",
        "--- Experiment 3: Reasoning Stage ---",
        f"  Along-tube monotonicity:     {s.along_tube_monotonicity:.3f}  "
        f"(1.0 = perfectly ordered)",
        f"  Timestep probe accuracy:     {s.timestep_probe_accuracy:.1%}  "
        f"(chance = 20%)",
        f"  Mean along-tube velocity:    {s.mean_along_velocity:.4f}",
    ])

    lines.extend([
        "",
        "--- Experiment 4: Tube Separation ---",
        f"  Mean classification margin:  {sep.mean_classification_margin:.4f}  "
        f"(>0 = tubes don't overlap)",
        f"  Tube overlap fraction:       {sep.tube_overlap_fraction:.1%}",
    ])
    if sep.confused_pairs:
        lines.append("  Confused pairs (top 5):")
        for fa, fb, ov in sep.confused_pairs[:5]:
            lines.append(f"    {fa} <-> {fb}: {ov:.1%} overlap")

    if report.intervention is not None:
        iv = report.intervention
        lines.extend([
            "",
            "--- Experiment 5: Tube Interventions ---",
            f"  Transverse recovery rate:    {iv.transverse_recovery_rate:.1%}",
            f"  Cross-tube persistence rate: {iv.cross_tube_persistence_rate:.1%}",
            f"  Mean recovery steps:         {iv.mean_transverse_recovery_steps:.0f}",
            f"  Mean recovery distance:      {iv.mean_recovery_distance:.3f}  "
            f"(< 1 = recovering)",
        ])

    if report.dynamics_tube is not None:
        dt = report.dynamics_tube
        lines.extend([
            "",
            "--- Experiment 6: Dynamics Tube Coherence ---",
            f"  Dynamics tightening ratio:   {dt.dynamics_tightening_ratio:.3f}  "
            f"({'tightens' if dt.dynamics_tightening_ratio < 0.9 else 'stable' if dt.dynamics_tightening_ratio < 1.1 else 'expands'})",
        ])
        if dt.per_step_within_spread:
            lines.append(f"  Within spread (start):       {dt.per_step_within_spread[0]:.4f}")
            lines.append(f"  Within spread (end):         {dt.per_step_within_spread[-1]:.4f}")
        if dt.per_step_cross_spread:
            lines.append(f"  Cross spread (start):        {dt.per_step_cross_spread[0]:.4f}")
            lines.append(f"  Cross spread (end):          {dt.per_step_cross_spread[-1]:.4f}")
        if dt.per_step_vel_cosine:
            vel_start = dt.per_step_vel_cosine[0]
            vel_end = dt.per_step_vel_cosine[-1]
            vel_mean = sum(dt.per_step_vel_cosine) / len(dt.per_step_vel_cosine)
            lines.append(f"  Vel cosine (start):          {vel_start:.3f}")
            lines.append(f"  Vel cosine (end):            {vel_end:.3f}")
            lines.append(f"  Vel cosine (mean):           {vel_mean:.3f}")
        if dt.per_step_classification_acc:
            acc_start = dt.per_step_classification_acc[0]
            acc_end = dt.per_step_classification_acc[-1]
            acc_mean = sum(dt.per_step_classification_acc) / len(dt.per_step_classification_acc)
            lines.append(f"  1-NN classification (start): {acc_start:.1%}")
            lines.append(f"  1-NN classification (end):   {acc_end:.1%}")
            lines.append(f"  1-NN classification (mean):  {acc_mean:.1%}")

    return "\n".join(lines)
