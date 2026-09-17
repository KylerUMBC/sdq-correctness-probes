"""Latent motion similarity evaluation.

Measures whether semantically equivalent prompts share similar
latent motion (velocity direction and magnitude) after alignment.

This is the sharpest test of whether the latent captures semantic
*dynamics* rather than just static clustering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MotionSimilarityResult:
    """Result of latent motion similarity evaluation."""

    within_velocity_cosine: float  # avg cosine sim of velocities, same family
    cross_velocity_cosine: float   # avg cosine sim of velocities, diff family
    within_velocity_l2: float      # avg L2 of velocity diff, same family
    cross_velocity_l2: float       # avg L2 of velocity diff, diff family
    velocity_cosine_separation: float  # within / cross (higher = better)
    velocity_l2_separation: float      # cross / within (higher = better)


def _velocity_cosine_sim(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """Mean cosine similarity of velocities between two trajectories."""
    T = min(z_a.shape[0], z_b.shape[0])
    if T < 2:
        return 0.0
    dz_a = z_a[1:T] - z_a[:T - 1]
    dz_b = z_b[1:T] - z_b[:T - 1]
    dot = (dz_a * dz_b).sum(dim=-1)
    norm_a = dz_a.norm(dim=-1).clamp(min=1e-8)
    norm_b = dz_b.norm(dim=-1).clamp(min=1e-8)
    cos = dot / (norm_a * norm_b)
    return cos.mean().item()


def _velocity_l2(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """Mean L2 distance of velocities between two trajectories."""
    T = min(z_a.shape[0], z_b.shape[0])
    if T < 2:
        return 0.0
    dz_a = z_a[1:T] - z_a[:T - 1]
    dz_b = z_b[1:T] - z_b[:T - 1]
    return (dz_a - dz_b).pow(2).sum(dim=-1).mean().item()


def motion_similarity_eval(
    latent_states: dict[str, torch.Tensor],
    families: dict[str, list[str]],
) -> MotionSimilarityResult:
    """Evaluate latent motion similarity within vs across families.

    Args:
        latent_states: {prompt_id: [T, D_z]} latent trajectories.
        families: {family_name: [prompt_id, ...]} grouping.

    Returns:
        MotionSimilarityResult with within/cross comparisons.
    """
    from itertools import combinations

    within_cos, within_l2 = [], []
    cross_cos, cross_l2 = [], []

    fam_names = sorted(families.keys())

    # Within-group pairs
    for fam in fam_names:
        pids = [p for p in families[fam] if p in latent_states]
        for a, b in combinations(pids, 2):
            within_cos.append(_velocity_cosine_sim(latent_states[a], latent_states[b]))
            within_l2.append(_velocity_l2(latent_states[a], latent_states[b]))

    # Cross-group pairs
    for i in range(len(fam_names)):
        for j in range(i + 1, len(fam_names)):
            pids_i = [p for p in families[fam_names[i]] if p in latent_states]
            pids_j = [p for p in families[fam_names[j]] if p in latent_states]
            for a in pids_i[:2]:  # limit to avoid explosion
                for b in pids_j[:2]:
                    cross_cos.append(_velocity_cosine_sim(latent_states[a], latent_states[b]))
                    cross_l2.append(_velocity_l2(latent_states[a], latent_states[b]))

    avg_w_cos = sum(within_cos) / max(len(within_cos), 1)
    avg_c_cos = sum(cross_cos) / max(len(cross_cos), 1)
    avg_w_l2 = sum(within_l2) / max(len(within_l2), 1)
    avg_c_l2 = sum(cross_l2) / max(len(cross_l2), 1)

    return MotionSimilarityResult(
        within_velocity_cosine=avg_w_cos,
        cross_velocity_cosine=avg_c_cos,
        within_velocity_l2=avg_w_l2,
        cross_velocity_l2=avg_c_l2,
        velocity_cosine_separation=avg_w_cos / max(avg_c_cos, 1e-8),
        velocity_l2_separation=avg_c_l2 / max(avg_w_l2, 1e-8),
    )


def _dynamics_velocity_cosine_sim(
    z_a: torch.Tensor, z_b: torch.Tensor, dynamics,
) -> float:
    """Mean cosine similarity of *learned* velocity fields between trajectories.

    Uses dynamics.velocity_field(z_t) instead of finite differences z[t+1]-z[t].
    """
    T = min(z_a.shape[0], z_b.shape[0])
    if T < 1:
        return 0.0
    with torch.no_grad():
        vel_a = dynamics.velocity_field(z_a[:T].to(next(dynamics.parameters()).device))       # [T, D]
        vel_b = dynamics.velocity_field(z_b[:T].to(next(dynamics.parameters()).device))       # [T, D]
    dot = (vel_a * vel_b).sum(dim=-1)
    norm_a = vel_a.norm(dim=-1).clamp(min=1e-8)
    norm_b = vel_b.norm(dim=-1).clamp(min=1e-8)
    cos = dot / (norm_a * norm_b)
    return cos.mean().item()


def dynamics_motion_similarity_eval(
    latent_states: dict[str, torch.Tensor],
    families: dict[str, list[str]],
    dynamics,
) -> MotionSimilarityResult:
    """Motion similarity using the learned dynamics velocity field.

    Same structure as motion_similarity_eval but uses dynamics.velocity_field(z_t)
    instead of finite differences. Measures whether the learned flow field is
    family-coherent.
    """
    from itertools import combinations

    within_cos, cross_cos = [], []
    fam_names = sorted(families.keys())

    for fam in fam_names:
        pids = [p for p in families[fam] if p in latent_states]
        for a, b in combinations(pids, 2):
            within_cos.append(
                _dynamics_velocity_cosine_sim(latent_states[a], latent_states[b], dynamics)
            )

    for i in range(len(fam_names)):
        for j in range(i + 1, len(fam_names)):
            pids_i = [p for p in families[fam_names[i]] if p in latent_states]
            pids_j = [p for p in families[fam_names[j]] if p in latent_states]
            for a in pids_i[:2]:
                for b in pids_j[:2]:
                    cross_cos.append(
                        _dynamics_velocity_cosine_sim(latent_states[a], latent_states[b], dynamics)
                    )

    avg_w = sum(within_cos) / max(len(within_cos), 1)
    avg_c = sum(cross_cos) / max(len(cross_cos), 1)

    return MotionSimilarityResult(
        within_velocity_cosine=avg_w,
        cross_velocity_cosine=avg_c,
        within_velocity_l2=0.0,  # not computed for dynamics version
        cross_velocity_l2=0.0,
        velocity_cosine_separation=avg_w / max(avg_c, 1e-8),
        velocity_l2_separation=0.0,
    )
