"""Causal intervention experiments on the latent dynamics.

Tests whether z_t is causally meaningful by perturbing z_t at specific
timesteps and measuring how the perturbation propagates through the dynamics.

Scientific question: if z_t captures semantics, then nudging z_t from one
attractor basin toward another should predictably shift the downstream
trajectory.  If the attractor structure is strong, the dynamics will pull
the perturbed trajectory back (effect_decays=True).  If the perturbation
crosses a basin boundary, the trajectory will diverge to a different attractor
and the divergence will persist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from sdq.latent.dynamics import LatentODE


@dataclass
class InterventionResult:
    """Results from a single intervention experiment."""

    original_trajectory: Tensor       # [T, D_z] rollout from intervention point
    perturbed_trajectory: Tensor      # [T, D_z] rollout after perturbation
    intervention_timestep: int        # index t where perturbation was applied
    perturbation_magnitude: float     # ||perturbation||
    trajectory_divergence: Tensor     # [T] L2 distance at each step
    mean_post_divergence: float       # mean divergence over all post-intervention steps
    effect_decays: bool               # True if divergence decreases after initial spike


@dataclass
class InterventionReport:
    """Aggregated results across many intervention experiments."""

    results: list[InterventionResult]
    mean_effect_magnitude: float       # mean of mean_post_divergence across all results
    fraction_with_decay: float         # fraction where attractor pulls trajectory back
    causal_score: float                # 0-1: mean_post_divergence / perturbation_magnitude
                                       # clamped to [0, 1]


class InterventionExperiment:
    """Run causal intervention experiments on a trained LatentODE.

    For each intervention, we:
      1. Take a known trajectory up to timestep t.
      2. Replace z_t with z_t + perturbation.
      3. Roll out both original and perturbed dynamics for num_steps.
      4. Measure how the trajectories diverge.
    """

    def __init__(self, dynamics: LatentODE, num_steps: int = 50):
        """
        Args:
            dynamics: a trained LatentODE used for rollouts.
            num_steps: how many steps to roll out after the intervention point.
        """
        self.dynamics = dynamics
        self.num_steps = num_steps

    @torch.no_grad()
    def intervene(
        self,
        z_trajectory: Tensor,
        timestep: int,
        perturbation: Tensor,
    ) -> InterventionResult:
        """Apply a perturbation at a given timestep and compare rollouts.

        Args:
            z_trajectory: [T, D_z] observed latent trajectory.
            timestep: index t at which to apply the perturbation (0 <= t < T).
            perturbation: [D_z] vector to add to z_trajectory[timestep].

        Returns:
            InterventionResult comparing original vs. perturbed rollouts.
        """
        T, D_z = z_trajectory.shape
        if timestep < 0 or timestep >= T:
            raise ValueError(
                f"timestep {timestep} out of range for trajectory of length {T}"
            )

        # Starting states for the two rollouts
        z_orig = z_trajectory[timestep]          # [D_z]
        z_pert = z_trajectory[timestep] + perturbation  # [D_z]

        # Roll out num_steps from each starting state (including the start itself)
        orig_traj = self.dynamics.rollout(z_orig, self.num_steps)  # [num_steps, D_z]
        pert_traj = self.dynamics.rollout(z_pert, self.num_steps)  # [num_steps, D_z]

        # Per-step L2 divergence
        divergence = (orig_traj - pert_traj).norm(dim=-1)  # [num_steps]

        mean_post_divergence = divergence.mean().item()
        perturbation_magnitude = perturbation.norm().item()

        # Detect decay: does divergence decrease after the initial spike?
        # We compare the first half vs. second half of the post-intervention divergence.
        effect_decays = False
        if self.num_steps >= 4:
            mid = self.num_steps // 2
            early_mean = divergence[:mid].mean().item()
            late_mean = divergence[mid:].mean().item()
            effect_decays = late_mean < early_mean

        return InterventionResult(
            original_trajectory=orig_traj,
            perturbed_trajectory=pert_traj,
            intervention_timestep=timestep,
            perturbation_magnitude=perturbation_magnitude,
            trajectory_divergence=divergence,
            mean_post_divergence=mean_post_divergence,
            effect_decays=effect_decays,
        )

    @torch.no_grad()
    def manifold_patch(
        self,
        z_original: Tensor,
        z_target: Tensor,
        extractor,
        timestep: int,
    ) -> InterventionResult:
        """Patch z_original toward z_target at a given timestep.

        Paper formula: h_patched = h_original - mu_orig + mu_target.
        In manifold coords: perturbation = z_target[timestep] - z_original[timestep].

        Args:
            z_original: [T, k] latent trajectory to perturb.
            z_target:   [T, k] reference trajectory supplying the patch direction.
            extractor:  ManifoldExtractor (unused in coord space but kept for API symmetry).
            timestep:   Index t at which to apply the perturbation.

        Returns:
            InterventionResult comparing original vs. patched rollouts.
        """
        perturbation = z_target[timestep] - z_original[timestep]
        return self.intervene(z_original, timestep, perturbation)

    @torch.no_grad()
    def run_experiment(
        self,
        z_trajectories: dict[str, Tensor],
        perturbation_magnitude: float = 1.0,
        num_interventions: int = 5,
    ) -> InterventionReport:
        """Run intervention experiments across a set of latent trajectories.

        For each trajectory, `num_interventions` random unit-vector perturbations
        (scaled by `perturbation_magnitude`) are applied at t = T // 2.

        Args:
            z_trajectories: {id: [T, D_z]} latent trajectories.
            perturbation_magnitude: scale of the perturbation vectors.
            num_interventions: number of random directions to try per trajectory.

        Returns:
            InterventionReport aggregating all results.
        """
        all_results: list[InterventionResult] = []

        for traj in z_trajectories.values():
            T, D_z = traj.shape
            timestep = T // 2
            device = traj.device

            for _ in range(num_interventions):
                # Sample a random unit vector and scale by perturbation_magnitude
                direction = torch.randn(D_z, device=device)
                direction = direction / direction.norm().clamp(min=1e-8)
                perturbation = direction * perturbation_magnitude

                result = self.intervene(traj, timestep, perturbation)
                all_results.append(result)

        if not all_results:
            return InterventionReport(
                results=[],
                mean_effect_magnitude=0.0,
                fraction_with_decay=0.0,
                causal_score=0.0,
            )

        mean_effect_magnitude = sum(r.mean_post_divergence for r in all_results) / len(all_results)
        fraction_with_decay = sum(1 for r in all_results if r.effect_decays) / len(all_results)

        # causal_score: how much does the perturbation actually move the trajectory?
        # Normalized by perturbation_magnitude so it's on a [0, 1] scale.
        if perturbation_magnitude > 0:
            raw_score = mean_effect_magnitude / perturbation_magnitude
        else:
            raw_score = 0.0
        causal_score = float(min(max(raw_score, 0.0), 1.0))

        return InterventionReport(
            results=all_results,
            mean_effect_magnitude=mean_effect_magnitude,
            fraction_with_decay=fraction_with_decay,
            causal_score=causal_score,
        )


def format_intervention_report(report: InterventionReport) -> str:
    """Concise human-readable summary of an InterventionReport."""
    n = len(report.results)
    lines = [
        "═══ INTERVENTION EXPERIMENT ═══",
        f"  Total interventions       : {n}",
        f"  Mean effect magnitude     : {report.mean_effect_magnitude:.4f}",
        f"  Fraction with decay       : {report.fraction_with_decay:.1%}",
        f"  Causal score (0-1)        : {report.causal_score:.4f}",
    ]

    if report.results:
        # Summarize effect decay interpretation
        if report.fraction_with_decay > 0.6:
            interpretation = "attractor strongly pulls trajectories back after perturbation"
        elif report.fraction_with_decay > 0.3:
            interpretation = "moderate attractor pull — some perturbations escape basins"
        else:
            interpretation = "perturbations persist — weak or absent attractor structure"
        lines.append(f"  Interpretation            : {interpretation}")

        # Causal score interpretation
        if report.causal_score > 0.5:
            causal_interp = "perturbations have large causal effect on downstream trajectory"
        elif report.causal_score > 0.1:
            causal_interp = "perturbations have moderate causal effect"
        else:
            causal_interp = "perturbations have little causal effect (z_t may not be causally central)"
        lines.append(f"  Causal interpretation     : {causal_interp}")

    return "\n".join(lines)
