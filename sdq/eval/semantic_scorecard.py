"""Semantic scorecard — unified multi-metric semantic evaluation.

Replaces reliance on a single within/cross separation number with
a comprehensive scorecard that reports all semantic quality dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import torch


@dataclass
class SemanticScorecard:
    """Complete semantic quality report."""

    # Core separation
    latent_separation: float = 0.0
    within_distance: float = 0.0
    cross_distance: float = 0.0

    # Retrieval
    retrieval_accuracy: float = 0.0
    retrieval_mrr: float = 0.0

    # Hard negatives
    hard_negative_passes: bool = False
    hard_negative_separation: float = 0.0

    # Motion
    velocity_cosine_separation: float = 0.0
    velocity_l2_separation: float = 0.0

    # Per-family
    per_family_separation: dict[str, float] = field(default_factory=dict)

    # Shuffle control
    shuffle_separation: float = 0.0
    shuffle_delta: float = 0.0  # true_sep - shuffle_sep (should be ~0)

    # Same-answer control
    same_answer_separation: float = 0.0

    def overall_score(self) -> float:
        """Weighted aggregate semantic quality score."""
        scores = []
        # Separation (capped at 10 for normalization)
        scores.append(min(self.latent_separation / 5.0, 2.0) * 20)
        # Retrieval accuracy
        scores.append(self.retrieval_accuracy * 25)
        # Hard-negative pass
        scores.append(25.0 if self.hard_negative_passes else 0.0)
        # Motion separation (capped at 10)
        scores.append(min(self.velocity_cosine_separation / 5.0, 2.0) * 15)
        # Consistency: shuffle delta should be near 0
        scores.append(max(0, 15.0 - abs(self.shuffle_delta) * 10))
        return sum(scores)

    def to_dict(self) -> dict[str, object]:
        return {
            "latent_separation": self.latent_separation,
            "within_distance": self.within_distance,
            "cross_distance": self.cross_distance,
            "retrieval_accuracy": self.retrieval_accuracy,
            "retrieval_mrr": self.retrieval_mrr,
            "hard_negative_passes": self.hard_negative_passes,
            "hard_negative_separation": self.hard_negative_separation,
            "velocity_cosine_separation": self.velocity_cosine_separation,
            "velocity_l2_separation": self.velocity_l2_separation,
            "per_family_separation": self.per_family_separation,
            "shuffle_separation": self.shuffle_separation,
            "shuffle_delta": self.shuffle_delta,
            "same_answer_separation": self.same_answer_separation,
            "overall_score": self.overall_score(),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== Semantic Scorecard ===",
            f"  Latent separation:     {self.latent_separation:.2f}",
            f"  Retrieval accuracy:    {self.retrieval_accuracy:.2%}",
            f"  Retrieval MRR:         {self.retrieval_mrr:.4f}",
            f"  Hard-neg passes:       {self.hard_negative_passes}",
            f"  Hard-neg separation:   {self.hard_negative_separation:.2f}",
            f"  Motion cos separation: {self.velocity_cosine_separation:.2f}",
            f"  Motion L2 separation:  {self.velocity_l2_separation:.2f}",
            f"  Shuffle separation:    {self.shuffle_separation:.2f}",
            f"  Shuffle delta:         {self.shuffle_delta:+.2f}",
        ]
        for fam, sep in self.per_family_separation.items():
            lines.append(f"  {fam} separation:       {sep:.2f}")
        lines.append(f"  Overall score:         {self.overall_score():.1f}/100")
        return "\n".join(lines)


def compute_scorecard(
    latent_states: dict[str, torch.Tensor],
    families: dict[str, list[str]],
    retrieval_result=None,
    hard_negative_result=None,
    motion_result=None,
    shuffle_separation: float = 0.0,
) -> SemanticScorecard:
    """Compute the full semantic scorecard.

    Args:
        latent_states: {pid: [T, D_z]} latent trajectories.
        families: {family_name: [prompt_ids]}.
        retrieval_result: RetrievalResult from retrieval_accuracy().
        hard_negative_result: HardNegativeResult.
        motion_result: MotionSimilarityResult.
        shuffle_separation: computed shuffle within/cross ratio.

    Returns:
        SemanticScorecard with all metrics filled.
    """
    sc = SemanticScorecard()

    # Core within/cross separation
    within_dists, cross_dists = [], []
    fam_names = list(families.keys())

    for fam, pids in families.items():
        fam_within = []
        for a, b in combinations(pids, 2):
            if a in latent_states and b in latent_states:
                T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                within_dists.append(d)
                fam_within.append(d)

        # Cross-family for this family
        fam_cross = []
        for other_fam in families:
            if other_fam == fam:
                continue
            for a in pids[:2]:
                for b in families[other_fam][:2]:
                    if a in latent_states and b in latent_states:
                        T_min = min(latent_states[a].shape[0], latent_states[b].shape[0])
                        d = (latent_states[a][:T_min] - latent_states[b][:T_min]).norm(dim=-1).mean().item()
                        cross_dists.append(d)
                        fam_cross.append(d)

        avg_w = sum(fam_within) / max(len(fam_within), 1)
        avg_c = sum(fam_cross) / max(len(fam_cross), 1)
        sc.per_family_separation[fam] = avg_c / max(avg_w, 1e-8)

    sc.within_distance = sum(within_dists) / max(len(within_dists), 1)
    sc.cross_distance = sum(cross_dists) / max(len(cross_dists), 1)
    sc.latent_separation = sc.cross_distance / max(sc.within_distance, 1e-8)

    # Retrieval
    if retrieval_result is not None:
        sc.retrieval_accuracy = retrieval_result.accuracy
        sc.retrieval_mrr = retrieval_result.mean_reciprocal_rank

    # Hard negatives
    if hard_negative_result is not None:
        sc.hard_negative_passes = hard_negative_result.passes
        sc.hard_negative_separation = hard_negative_result.separation_ratio

    # Motion
    if motion_result is not None:
        sc.velocity_cosine_separation = motion_result.velocity_cosine_separation
        sc.velocity_l2_separation = motion_result.velocity_l2_separation

    # Shuffle
    sc.shuffle_separation = shuffle_separation
    sc.shuffle_delta = sc.latent_separation - shuffle_separation

    return sc
