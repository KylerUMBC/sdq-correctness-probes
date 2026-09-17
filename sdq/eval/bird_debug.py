"""Bird-focused diagnostic evaluation.

Provides detailed diagnostics for the bird family (or any specified
hard family) to identify alignment, transport, and reconstruction
failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sdq.trajectories import Trajectory


@dataclass
class FamilyDebugResult:
    """Detailed diagnostics for a single family."""

    family: str
    per_pid: dict[str, dict[str, float]] = field(default_factory=dict)
    avg_reconstruction_mse: float = 0.0
    avg_transport_residual: float = 0.0
    avg_alignment_displacement: float = 0.0
    avg_latent_velocity_norm: float = 0.0
    avg_state_norm: float = 0.0
    velocity_profile_variance: float = 0.0
    traj_length_range: tuple[int, int] = (0, 0)
    per_step_velocity_mismatch: list[float] = field(default_factory=list)


def family_debug(
    trajectories: dict[str, "Trajectory"],
    latent_states: dict[str, torch.Tensor],
    reconstruction_errors: dict[str, float],
    transport_residuals: dict[tuple[str, str], float],
    family_pids: list[str],
    family_name: str,
    encoder=None,
    decoder=None,
    gauge_encoder=None,
    aligner=None,
    transport_model=None,
    device: str = "cpu",
) -> FamilyDebugResult:
    """Run detailed diagnostics for a specific family.

    Args:
        trajectories: {pid: Trajectory} normalized trajectories.
        latent_states: {pid: [T, D_z]} encoded latents.
        reconstruction_errors: {pid: mse} per-prompt reconstruction.
        transport_residuals: {(pid_a, pid_b): residual} transport residuals.
        family_pids: list of prompt IDs in the family.
        family_name: name for the result.
        encoder, decoder, gauge_encoder, aligner, transport_model: optional models.
        device: device string.

    Returns:
        FamilyDebugResult with per-prompt and family-level diagnostics.
    """
    result = FamilyDebugResult(family=family_name)
    available = [pid for pid in family_pids if pid in trajectories]

    if not available:
        return result

    lengths = []
    vel_norms_all = []
    state_norms_all = []
    per_step_vel_profiles = []

    for pid in available:
        states = trajectories[pid].states
        T = states.shape[0]
        lengths.append(T)

        vel = states[1:] - states[:-1]
        vel_norm = vel.norm(dim=-1)  # [T-1]
        state_norm = states.norm(dim=-1).mean().item()

        per_pid_info = {
            "traj_length": T,
            "state_norm": state_norm,
            "mean_velocity_norm": vel_norm.mean().item(),
            "max_velocity_norm": vel_norm.max().item(),
            "min_velocity_norm": vel_norm.min().item(),
            "reconstruction_mse": reconstruction_errors.get(pid, 0.0),
        }

        # Latent diagnostics
        if pid in latent_states:
            z = latent_states[pid]
            z_vel = z[1:] - z[:-1]
            per_pid_info["latent_velocity_norm"] = z_vel.norm(dim=-1).mean().item()
            per_pid_info["latent_state_norm"] = z.norm(dim=-1).mean().item()
            per_pid_info["latent_state_var"] = z.var(dim=0).mean().item()
            vel_norms_all.append(z_vel.norm(dim=-1).mean().item())
        else:
            vel_norms_all.append(0.0)

        state_norms_all.append(state_norm)
        per_step_vel_profiles.append(vel_norm.tolist())

        result.per_pid[pid] = per_pid_info

    # Family-level aggregates
    result.traj_length_range = (min(lengths), max(lengths))
    result.avg_state_norm = sum(state_norms_all) / len(state_norms_all)
    result.avg_latent_velocity_norm = sum(vel_norms_all) / len(vel_norms_all)

    rec_vals = [reconstruction_errors.get(pid, 0.0) for pid in available]
    result.avg_reconstruction_mse = sum(rec_vals) / max(len(rec_vals), 1)

    # Transport residuals for this family
    res_vals = [v for (a, b), v in transport_residuals.items()
                if a in available and b in available]
    result.avg_transport_residual = sum(res_vals) / max(len(res_vals), 1)

    # Velocity profile variance across prompts
    if len(per_step_vel_profiles) > 1:
        min_len = min(len(p) for p in per_step_vel_profiles)
        profiles = torch.tensor([p[:min_len] for p in per_step_vel_profiles])
        result.velocity_profile_variance = profiles.var(dim=0).mean().item()

        # Per-step mismatch between all pairs
        from itertools import combinations
        mismatches = []
        for i, j in combinations(range(len(per_step_vel_profiles)), 2):
            ml = min(len(per_step_vel_profiles[i]), len(per_step_vel_profiles[j]))
            a_p = torch.tensor(per_step_vel_profiles[i][:ml])
            b_p = torch.tensor(per_step_vel_profiles[j][:ml])
            mismatches.extend((a_p - b_p).abs().tolist())
        result.per_step_velocity_mismatch = mismatches[:50]  # cap for display

    return result
