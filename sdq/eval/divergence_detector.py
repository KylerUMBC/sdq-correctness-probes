"""Divergence detection for latent trajectories.

Monitors the velocity norm ||f(z_t)|| during generation and flags when a
trajectory leaves a known-good basin.  The core claim: correct reasoning
lives in stable attractor basins where the dynamics evolve smoothly (low
velocity or converging).  A velocity spike signals the model may be leaving
a valid reasoning manifold — a potential hallucination indicator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DivergenceEvent:
    """A single timestep where the velocity norm exceeded the spike threshold."""
    timestep: int
    velocity_norm: float
    baseline_velocity: float
    severity: float          # velocity_norm / baseline_velocity
    is_critical: bool        # severity >= critical_threshold


@dataclass
class DivergenceReport:
    """Full divergence analysis for one latent trajectory."""
    events: list[DivergenceEvent]
    max_severity: float
    mean_velocity: float
    baseline_velocity: float
    trajectory_length: int
    is_flagged: bool         # True if any critical event occurred
    flagged_at: int | None   # timestep of first critical event, or None


class DivergenceDetector:
    """Monitor latent trajectory velocity norms and flag basin escapes.

    Given a trained dynamics module with a ``velocity_field(z)`` method
    (e.g. LatentODE), this detector tracks ||f(z_t)|| at each timestep
    and compares it to a baseline velocity expected inside a stable basin.

    Spikes above ``spike_threshold * baseline`` produce DivergenceEvents;
    spikes above ``critical_threshold * baseline`` set ``is_critical=True``
    and cause the trajectory to be flagged.
    """

    def __init__(
        self,
        dynamics: nn.Module,
        baseline_velocity: float | None = None,
        spike_threshold: float = 2.0,
        critical_threshold: float = 5.0,
    ):
        self.dynamics = dynamics
        self.baseline_velocity = baseline_velocity
        self.spike_threshold = spike_threshold
        self.critical_threshold = critical_threshold

    @torch.no_grad()
    def estimate_baseline(self, z_trajectories: dict[str, Tensor]) -> float:
        """Estimate the typical in-basin velocity from a set of trajectories.

        Runs every latent state through the dynamics velocity field, collects
        all ||f(z_t)|| values, and stores (and returns) the median as the
        baseline.  Call this once on training / validation data before using
        ``monitor``.

        Args:
            z_trajectories: {prompt_id: [T, D_z]} latent trajectories.

        Returns:
            Estimated baseline velocity (median ||f(z_t)|| across all steps).
        """
        all_norms: list[Tensor] = []
        for traj in z_trajectories.values():
            # traj: [T, D_z]
            vel = self.dynamics.velocity_field(traj)   # [T, D_z]
            norms = vel.norm(dim=-1)                   # [T]
            all_norms.append(norms)

        if not all_norms:
            raise ValueError("z_trajectories is empty — cannot estimate baseline.")

        combined = torch.cat(all_norms)                # [total_steps]
        baseline = combined.median().item()
        self.baseline_velocity = baseline
        return baseline

    @torch.no_grad()
    def monitor(self, z_trajectory: Tensor) -> DivergenceReport:
        """Compute per-timestep velocity norms and build a DivergenceReport.

        Args:
            z_trajectory: [T, D_z] latent trajectory for a single sequence.

        Returns:
            DivergenceReport with events, severity scores, and flag status.
        """
        if self.baseline_velocity is None:
            raise RuntimeError(
                "baseline_velocity is not set.  Call estimate_baseline() first "
                "or pass baseline_velocity to __init__."
            )

        T = z_trajectory.shape[0]
        vel = self.dynamics.velocity_field(z_trajectory)   # [T, D_z]
        norms = vel.norm(dim=-1)                           # [T]

        baseline = self.baseline_velocity
        events: list[DivergenceEvent] = []
        flagged_at: int | None = None

        for t in range(T):
            vn = norms[t].item()
            severity = vn / baseline if baseline > 0 else float("inf")

            if severity >= self.spike_threshold:
                is_critical = severity >= self.critical_threshold
                events.append(DivergenceEvent(
                    timestep=t,
                    velocity_norm=vn,
                    baseline_velocity=baseline,
                    severity=severity,
                    is_critical=is_critical,
                ))
                if is_critical and flagged_at is None:
                    flagged_at = t

        max_severity = max((e.severity for e in events), default=0.0)
        mean_velocity = norms.mean().item()
        is_flagged = flagged_at is not None

        return DivergenceReport(
            events=events,
            max_severity=max_severity,
            mean_velocity=mean_velocity,
            baseline_velocity=baseline,
            trajectory_length=T,
            is_flagged=is_flagged,
            flagged_at=flagged_at,
        )

    def monitor_batch(
        self,
        z_trajectories: dict[str, Tensor],
    ) -> dict[str, DivergenceReport]:
        """Run ``monitor`` on every trajectory in the batch.

        Args:
            z_trajectories: {prompt_id: [T, D_z]} latent trajectories.

        Returns:
            {prompt_id: DivergenceReport}
        """
        return {pid: self.monitor(traj) for pid, traj in z_trajectories.items()}


def format_divergence_report(pid: str, report: DivergenceReport) -> str:
    """Return a human-readable multi-line summary of a DivergenceReport."""
    flag_str = "FLAGGED" if report.is_flagged else "ok"
    lines = [
        f"── Divergence [{pid}]  {flag_str} ──",
        f"  Trajectory length  : {report.trajectory_length}",
        f"  Mean velocity      : {report.mean_velocity:.4e}",
        f"  Baseline velocity  : {report.baseline_velocity:.4e}",
        f"  Max severity       : {report.max_severity:.2f}x",
        f"  Spike events       : {len(report.events)}",
    ]

    if report.is_flagged:
        lines.append(f"  First critical step: t={report.flagged_at}")

    if report.events:
        lines.append("  ── Events ──")
        for ev in report.events:
            crit_tag = " [CRITICAL]" if ev.is_critical else ""
            lines.append(
                f"    t={ev.timestep:4d}  ||f||={ev.velocity_norm:.4e}"
                f"  severity={ev.severity:.2f}x{crit_tag}"
            )

    return "\n".join(lines)
