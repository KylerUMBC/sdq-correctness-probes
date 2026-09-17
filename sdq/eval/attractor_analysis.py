"""Attractor analysis for the learned latent dynamics.

Given a trained LatentODE (z_{t+1} = z_t + f(z_t)), this module:
  1. Finds fixed points (attractors) by running dynamics forward until convergence
  2. Classifies stability via Jacobian eigenvalue analysis
  3. Maps basins of attraction — which initial states reach which attractors
  4. Correlates attractors with task answers (correct answer ↔ stable attractor)
  5. Computes local Lyapunov exponents along trajectories (detect instability)

The core claim: correct reasoning lives in stable attractor basins,
and errors correspond to trajectories escaping those basins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class FixedPoint:
    """A fixed point of the latent dynamics."""
    location: Tensor           # [D_z]
    velocity_norm: float       # ||f(z*)|| — should be near 0
    max_eigenvalue: float      # largest real part of Jacobian eigenvalues
    is_stable: bool            # max_eigenvalue < 0
    basin_size: int = 0        # number of trajectories that converge here
    associated_answers: dict[str, int] = field(default_factory=dict)  # answer_id → count


@dataclass
class TrajectoryStability:
    """Stability analysis for a single trajectory."""
    lyapunov_exponents: Tensor  # [D_z] local exponents
    max_lyapunov: float         # dominant exponent
    is_stable: bool             # max_lyapunov < 0
    convergence_rate: float     # how quickly trajectory approaches attractor
    nearest_attractor_idx: int  # which attractor this trajectory converges to
    attractor_distance: float   # final distance to that attractor


@dataclass
class AttractorReport:
    """Full analysis results."""
    fixed_points: list[FixedPoint]
    num_attractors: int
    num_stable: int
    num_unstable: int
    answer_attractor_purity: float      # do distinct answers map to distinct attractors?
    mean_convergence_steps: float       # average steps to reach attractor
    basin_separation_score: float       # how cleanly do basins separate?
    trajectory_stabilities: list[TrajectoryStability] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class AttractorAnalysis:
    """Analyze attractor structure in a trained LatentODE."""

    def __init__(
        self,
        dynamics: nn.Module,
        convergence_threshold: float = 0.1,
        max_steps: int = 2000,
        attractor_merge_radius: float = 3.0,
    ):
        self.dynamics = dynamics
        self.convergence_threshold = convergence_threshold
        self.max_steps = max_steps
        self.attractor_merge_radius = attractor_merge_radius

    @torch.no_grad()
    def run_to_convergence(self, z0: Tensor, return_trajectory: bool = False) -> dict:
        """Run dynamics from z0 until convergence or max_steps.

        Args:
            z0: initial states [N, D_z]
            return_trajectory: if True, return full trajectory [N, T, D_z]

        Returns:
            dict with 'final', 'converged', 'steps', and optionally 'trajectory'
        """
        z = z0.clone()
        N = z.shape[0]
        converged = torch.zeros(N, dtype=torch.bool, device=z.device)
        steps = torch.full((N,), self.max_steps, dtype=torch.long, device=z.device)
        trajectory = [z.clone()] if return_trajectory else None

        for t in range(self.max_steps):
            dz = self.dynamics.velocity_field(z)
            z_new = z + dz

            # Check convergence: ||dz|| < threshold
            vel_norm = dz.norm(dim=-1)
            newly_converged = (~converged) & (vel_norm < self.convergence_threshold)
            steps[newly_converged] = t
            converged |= newly_converged

            z = z_new
            if return_trajectory:
                trajectory.append(z.clone())

            if converged.all():
                break

        result = {
            "final": z,
            "converged": converged,
            "steps": steps,
            "velocity_at_end": dz.norm(dim=-1),
        }
        if return_trajectory:
            result["trajectory"] = torch.stack(trajectory, dim=1)
        return result

    def find_attractors(self, z_samples: Tensor) -> list[FixedPoint]:
        """Find distinct attractors by running many initial conditions forward.

        Args:
            z_samples: starting points [N, D_z], typically sampled from training data

        Returns:
            List of FixedPoint objects, deduplicated by merge_radius
        """
        # Run dynamics forward (no grad needed for this part)
        with torch.no_grad():
            result = self.run_to_convergence(z_samples)
        endpoints = result["final"]  # [N, D_z]
        converged = result["converged"]

        # Only consider converged points
        conv_points = endpoints[converged]
        if conv_points.shape[0] == 0:
            return []

        # Cluster nearby endpoints into distinct attractors
        attractors = []
        assigned = torch.zeros(conv_points.shape[0], dtype=torch.bool, device=conv_points.device)

        for i in range(conv_points.shape[0]):
            if assigned[i]:
                continue
            center = conv_points[i]
            dists = (conv_points - center).norm(dim=-1)
            cluster_mask = dists < self.attractor_merge_radius
            assigned |= cluster_mask

            # Average the cluster to get refined attractor location
            cluster_points = conv_points[cluster_mask]
            attractor_loc = cluster_points.mean(dim=0)

            # Compute velocity at attractor (should be ~0)
            with torch.no_grad():
                vel = self.dynamics.velocity_field(attractor_loc.unsqueeze(0)).squeeze(0)
            vel_norm = vel.norm().item()

            # Jacobian eigenvalue analysis for stability (needs gradients!)
            max_eig = self._max_jacobian_eigenvalue(attractor_loc)

            attractors.append(FixedPoint(
                location=attractor_loc,
                velocity_norm=vel_norm,
                max_eigenvalue=max_eig,
                is_stable=max_eig < 0,
                basin_size=cluster_mask.sum().item(),
            ))

        return attractors

    def _max_jacobian_eigenvalue(self, z: Tensor) -> float:
        """Compute discrete-time stability measure at fixed point z.

        For z_{t+1} = z_t + f(z_t), stability requires the spectral radius
        of the propagator (I + df/dz) to be < 1.  Returns max|1 + λ| - 1,
        where λ ranges over eigenvalues of df/dz.  Negative means stable.
        """
        D = z.shape[0]
        if D > 128:
            print(
                f"[AttractorAnalysis] Warning: latent_dim={D} > 128. "
                "Jacobian computation may be slow. Consider using a smaller "
                "latent_dim or sampling fewer trajectory points."
            )

        def _f(z_in: Tensor) -> Tensor:
            return self.dynamics.velocity_field(z_in.unsqueeze(0)).squeeze(0)

        # torch.autograd.functional.jacobian returns shape [D_out, D_in]
        jacobian = torch.autograd.functional.jacobian(_f, z.detach())
        # Ensure it is a plain 2-D tensor (the API can wrap results in a tuple
        # when the input/output are tuples; guard against that here).
        if isinstance(jacobian, tuple):
            jacobian = jacobian[0]
        jacobian = jacobian.reshape(D, D)

        # Eigenvalues of the discrete-time propagator I + J.
        # Stability requires spectral radius of (I + J) < 1, i.e. |1 + λ| < 1
        # for all eigenvalues λ of J.  We return max |1 + λ| - 1 so that
        # negative means stable (same sign convention as continuous-time λ_max).
        eigenvalues = torch.linalg.eigvals(jacobian)
        spectral_radii = (1 + eigenvalues).abs()
        max_real = (spectral_radii.max() - 1).item()
        return max_real

    @torch.no_grad()
    def map_basins(
        self,
        z_samples: Tensor,
        attractors: list[FixedPoint],
    ) -> Tensor:
        """For each sample point, determine which attractor it converges to.

        Args:
            z_samples: initial states [N, D_z]
            attractors: list of known attractors

        Returns:
            assignments: [N] indices into the attractor list (-1 if no convergence)
        """
        if not attractors:
            return torch.full((z_samples.shape[0],), -1, dtype=torch.long)

        result = self.run_to_convergence(z_samples)
        endpoints = result["final"]
        converged = result["converged"]

        attractor_locs = torch.stack([a.location for a in attractors])  # [K, D_z]

        # Assign each endpoint to nearest attractor
        dists = torch.cdist(endpoints, attractor_locs)  # [N, K]
        assignments = dists.argmin(dim=-1)  # [N]
        assignments[~converged] = -1

        return assignments

    def trajectory_lyapunov(self, z_trajectory: Tensor) -> TrajectoryStability:
        """Compute local Lyapunov exponents along a latent trajectory.

        Tracks how perturbations grow/shrink along the trajectory by
        accumulating the log of Jacobian singular values.

        NOTE: No @torch.no_grad() — we need gradients for Jacobian computation.

        Args:
            z_trajectory: [T, D_z] sequence of latent states
        """
        T, D = z_trajectory.shape
        if T < 2:
            return TrajectoryStability(
                lyapunov_exponents=torch.zeros(D),
                max_lyapunov=0.0,
                is_stable=True,
                convergence_rate=0.0,
                nearest_attractor_idx=-1,
                attractor_distance=float("inf"),
            )

        if D > 128:
            print(
                f"[AttractorAnalysis] Warning: latent_dim={D} > 128. "
                "Jacobian computation may be slow. Consider using a smaller "
                "latent_dim or sampling fewer trajectory points."
            )

        def _f(z_in: Tensor) -> Tensor:
            return self.dynamics.velocity_field(z_in.unsqueeze(0)).squeeze(0)

        log_sv_sum = torch.zeros(D, device=z_trajectory.device)
        count = 0

        for t in range(T - 1):
            z_t = z_trajectory[t].detach()

            # Compute full Jacobian in one vectorised call.
            # torch.autograd.functional.jacobian returns shape [D_out, D_in].
            jacobian = torch.autograd.functional.jacobian(_f, z_t)
            if isinstance(jacobian, tuple):
                jacobian = jacobian[0]
            jacobian = jacobian.reshape(D, D)

            # Local propagator is I + J (since z_{t+1} = z_t + f(z_t))
            propagator = torch.eye(D, device=z_trajectory.device) + jacobian
            sv = torch.linalg.svdvals(propagator)
            log_sv_sum += torch.log(sv.clamp(min=1e-10))
            count += 1

        lyapunov = log_sv_sum / max(count, 1)
        max_lyap = lyapunov.max().item()

        # Convergence rate: average velocity decrease along trajectory
        vel_norms = []
        with torch.no_grad():
            for t in range(T):
                v = self.dynamics.velocity_field(z_trajectory[t:t+1]).norm().item()
                vel_norms.append(v)
        if len(vel_norms) >= 2 and vel_norms[0] > 1e-8:
            convergence_rate = (vel_norms[0] - vel_norms[-1]) / vel_norms[0]
        else:
            convergence_rate = 0.0

        return TrajectoryStability(
            lyapunov_exponents=lyapunov.cpu(),
            max_lyapunov=max_lyap,
            is_stable=max_lyap < 0,
            convergence_rate=convergence_rate,
            nearest_attractor_idx=-1,  # filled in by caller
            attractor_distance=float("inf"),
        )

    def full_analysis(
        self,
        z_trajectories: dict[str, Tensor],
        answer_ids: dict[str, str] | None = None,
    ) -> AttractorReport:
        """Run complete attractor analysis on a set of encoded trajectories.

        Args:
            z_trajectories: {prompt_id: [T, D_z]} latent trajectories
            answer_ids: {prompt_id: answer_id} for answer-attractor correlation

        Returns:
            AttractorReport with full analysis
        """
        # Collect terminal latent states as seed points
        z_terminals = []
        pids = list(z_trajectories.keys())
        for pid in pids:
            z_terminals.append(z_trajectories[pid][-1])  # last timestep
        z_seeds = torch.stack(z_terminals)

        # Also sample from trajectory midpoints for better basin coverage
        z_midpoints = []
        for pid in pids:
            traj = z_trajectories[pid]
            mid = traj.shape[0] // 2
            z_midpoints.append(traj[mid])
        z_all_seeds = torch.cat([z_seeds, torch.stack(z_midpoints)], dim=0)

        # 1. Find attractors
        attractors = self.find_attractors(z_all_seeds)
        if not attractors:
            return AttractorReport(
                fixed_points=[],
                num_attractors=0,
                num_stable=0,
                num_unstable=0,
                answer_attractor_purity=0.0,
                mean_convergence_steps=float("inf"),
                basin_separation_score=0.0,
            )

        # 2. Map basins for the trajectory endpoints
        assignments = self.map_basins(z_seeds, attractors)
        conv_result = self.run_to_convergence(z_seeds)

        # 3. Answer-attractor correlation
        purity = 0.0
        if answer_ids:
            # For each attractor, check if all trajectories reaching it share an answer
            for k, attr in enumerate(attractors):
                mask = assignments == k
                if mask.sum() == 0:
                    continue
                reaching_pids = [pids[i] for i in range(len(pids)) if mask[i]]
                answers = [answer_ids.get(pid, "unknown") for pid in reaching_pids]
                attr.associated_answers = {}
                for a in answers:
                    attr.associated_answers[a] = attr.associated_answers.get(a, 0) + 1

            # Purity: fraction of trajectories where the attractor's plurality answer
            # matches this trajectory's answer
            correct = 0
            total = 0
            for k, attr in enumerate(attractors):
                if not attr.associated_answers:
                    continue
                plurality_answer = max(attr.associated_answers, key=attr.associated_answers.get)
                mask = assignments == k
                reaching_pids = [pids[i] for i in range(len(pids)) if mask[i]]
                for pid in reaching_pids:
                    total += 1
                    if answer_ids.get(pid, "") == plurality_answer:
                        correct += 1
            purity = correct / total if total > 0 else 0.0

        # 4. Per-trajectory stability analysis (sample up to 50 for efficiency)
        sample_pids = pids[:50] if len(pids) > 50 else pids
        stabilities = []
        for i, pid in enumerate(sample_pids):
            traj = z_trajectories[pid]
            stab = self.trajectory_lyapunov(traj)
            # Fill in attractor info
            if i < len(assignments):
                stab.nearest_attractor_idx = assignments[i].item()
                if stab.nearest_attractor_idx >= 0:
                    attr_loc = attractors[stab.nearest_attractor_idx].location
                    stab.attractor_distance = (traj[-1] - attr_loc).norm().item()
            stabilities.append(stab)

        # 5. Basin separation: average distance between attractor locations
        if len(attractors) >= 2:
            locs = torch.stack([a.location for a in attractors])
            pair_dists = torch.cdist(locs, locs)
            # Mean off-diagonal distance, normalized by mean attractor merge radius
            mask = ~torch.eye(len(attractors), dtype=torch.bool, device=locs.device)
            mean_sep = pair_dists[mask].mean().item()
            basin_separation = min(mean_sep / (self.attractor_merge_radius * 2), 5.0)
        else:
            basin_separation = 0.0

        mean_conv_steps = conv_result["steps"].float().mean().item()

        return AttractorReport(
            fixed_points=attractors,
            num_attractors=len(attractors),
            num_stable=sum(1 for a in attractors if a.is_stable),
            num_unstable=sum(1 for a in attractors if not a.is_stable),
            answer_attractor_purity=purity,
            mean_convergence_steps=mean_conv_steps,
            basin_separation_score=basin_separation,
            trajectory_stabilities=stabilities,
            raw={
                "num_seeds": z_all_seeds.shape[0],
                "num_converged": conv_result["converged"].sum().item(),
                "convergence_rate": conv_result["converged"].float().mean().item(),
                "mean_final_velocity": conv_result["velocity_at_end"].mean().item(),
                "attractor_locations": [a.location.tolist() for a in attractors],
                "attractor_eigenvalues": [a.max_eigenvalue for a in attractors],
            },
        )


def format_attractor_report(report: AttractorReport) -> str:
    """Human-readable summary of attractor analysis."""
    lines = [
        "═══ ATTRACTOR ANALYSIS ═══",
        f"  Attractors found     : {report.num_attractors}",
        f"    Stable             : {report.num_stable}",
        f"    Unstable           : {report.num_unstable}",
        f"  Answer-attractor purity : {report.answer_attractor_purity:.1%}",
        f"  Mean convergence steps  : {report.mean_convergence_steps:.0f}",
        f"  Basin separation score  : {report.basin_separation_score:.2f}",
    ]

    if report.fixed_points:
        lines.append("  ── Fixed points ──")
        for i, fp in enumerate(report.fixed_points):
            stability = "STABLE" if fp.is_stable else "UNSTABLE"
            lines.append(
                f"    [{i}] ||f(z*)||={fp.velocity_norm:.2e}  "
                f"λ_max={fp.max_eigenvalue:.4f}  {stability}  "
                f"basin={fp.basin_size}"
            )
            if fp.associated_answers:
                ans_str = ", ".join(
                    f"{a}:{c}" for a, c in sorted(
                        fp.associated_answers.items(), key=lambda x: -x[1]
                    )[:5]
                )
                lines.append(f"         answers: {ans_str}")

    if report.trajectory_stabilities:
        stable_count = sum(1 for s in report.trajectory_stabilities if s.is_stable)
        mean_lyap = sum(s.max_lyapunov for s in report.trajectory_stabilities) / len(report.trajectory_stabilities)
        mean_conv = sum(s.convergence_rate for s in report.trajectory_stabilities) / len(report.trajectory_stabilities)
        lines.append("  ── Trajectory stability ──")
        lines.append(f"    Stable trajectories  : {stable_count}/{len(report.trajectory_stabilities)}")
        lines.append(f"    Mean max Lyapunov    : {mean_lyap:.4f}")
        lines.append(f"    Mean convergence rate: {mean_conv:.2%}")

    return "\n".join(lines)
