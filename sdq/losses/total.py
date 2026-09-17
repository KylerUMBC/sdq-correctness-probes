"""Total SDQ objective combining all loss terms.

    L = L_rec
      + lambda_sem   * L_sem
      + lambda_trans  * L_trans
      + lambda_cycle  * L_cycle
      + lambda_gauge  * L_gauge
      + lambda_event  * L_event
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LossWeights:
    """Weights for the total SDQ objective."""

    reconstruction: float = 1.0
    semantic: float = 1.0
    transport: float = 1.0
    cycle: float = 0.5
    gauge: float = 0.1
    event: float = 0.1
    triplet: float = 0.0
    velocity: float = 0.0
    composition: float = 0.0
    retrieval: float = 0.0
    trajectory: float = 0.0


@dataclass
class LossBreakdown:
    """Itemized loss values for logging and diagnostics."""

    total: torch.Tensor
    reconstruction: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    semantic: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    transport: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    cycle: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    gauge: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    event: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    triplet: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    velocity: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    composition: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    retrieval: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    trajectory: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total.item(),
            "reconstruction": self.reconstruction.item(),
            "semantic": self.semantic.item(),
            "transport": self.transport.item(),
            "cycle": self.cycle.item(),
            "gauge": self.gauge.item(),
            "event": self.event.item(),
            "triplet": self.triplet.item(),
            "velocity": self.velocity.item(),
            "composition": self.composition.item(),
            "retrieval": self.retrieval.item(),
            "trajectory": self.trajectory.item(),
        }


class SDQLoss(nn.Module):
    """Combines all SDQ loss terms with configurable weights.

    Usage:
        loss_fn = SDQLoss(weights=LossWeights(semantic=2.0))
        breakdown = loss_fn(
            L_rec=reconstruction_loss(...),
            L_sem=semantic_consistency_loss(...),
            ...
        )
        breakdown.total.backward()
    """

    def __init__(self, weights: LossWeights | None = None):
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self,
        L_rec: torch.Tensor | None = None,
        L_sem: torch.Tensor | None = None,
        L_trans: torch.Tensor | None = None,
        L_cycle: torch.Tensor | None = None,
        L_gauge: torch.Tensor | None = None,
        L_event: torch.Tensor | None = None,
        L_triplet: torch.Tensor | None = None,
        L_velocity: torch.Tensor | None = None,
        L_composition: torch.Tensor | None = None,
        L_retrieval: torch.Tensor | None = None,
        L_trajectory: torch.Tensor | None = None,
    ) -> LossBreakdown:
        """Combine loss terms into total objective.

        Any term can be None (treated as 0). This allows incremental
        training where not all terms are available yet.

        Returns:
            LossBreakdown with per-term values and total.
        """
        device = None
        for term in (L_rec, L_sem, L_trans, L_cycle, L_gauge, L_event, L_triplet, L_velocity, L_composition, L_retrieval, L_trajectory):
            if term is not None:
                device = term.device
                break
        if device is None:
            device = torch.device("cpu")

        zero = torch.tensor(0.0, device=device)
        total = zero.clone()

        rec = L_rec if L_rec is not None else zero
        sem = L_sem if L_sem is not None else zero
        trans = L_trans if L_trans is not None else zero
        cyc = L_cycle if L_cycle is not None else zero
        gauge = L_gauge if L_gauge is not None else zero
        event = L_event if L_event is not None else zero
        trip = L_triplet if L_triplet is not None else zero
        vel = L_velocity if L_velocity is not None else zero
        comp = L_composition if L_composition is not None else zero
        retr = L_retrieval if L_retrieval is not None else zero
        traj = L_trajectory if L_trajectory is not None else zero

        total = (
            self.weights.reconstruction * rec
            + self.weights.semantic * sem
            + self.weights.transport * trans
            + self.weights.cycle * cyc
            + self.weights.gauge * gauge
            + self.weights.event * event
            + self.weights.triplet * trip
            + self.weights.velocity * vel
            + self.weights.composition * comp
            + self.weights.retrieval * retr
            + self.weights.trajectory * traj
        )

        return LossBreakdown(
            total=total,
            reconstruction=rec,
            semantic=sem,
            transport=trans,
            cycle=cyc,
            gauge=gauge,
            event=event,
            triplet=trip,
            velocity=vel,
            composition=comp,
            retrieval=retr,
            trajectory=traj,
        )
