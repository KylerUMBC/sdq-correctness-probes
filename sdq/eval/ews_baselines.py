"""Baseline comparators for the early-warning system.

The EWS model must beat these to be useful:
  1. Logit-entropy baseline: mean output-logit entropy predicts failure
  2. Majority-class baseline: always predict the majority class per family
  3. Random baseline: AUROC = 0.5
  4. DivergenceDetector baseline: velocity-norm spikes from existing code
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from sdq.eval.early_warning_metrics import compute_auroc, compute_auprc


@dataclass
class BaselineResult:
    """Result of a single baseline evaluation."""

    name: str
    sequence_auroc: float
    sequence_auprc: float
    description: str


def logit_entropy_baseline(
    gen_logits_list: list[Tensor | None],
    labels: list[int],
) -> BaselineResult:
    """Baseline: mean logit entropy over generated tokens predicts failure.

    Higher entropy → less confident → more likely incorrect.

    Args:
        gen_logits_list: list of [T, vocab_size] or None per sequence.
        labels: list of 0 (correct) / 1 (incorrect).
    """
    scores = []
    valid_labels = []

    for gl, lab in zip(gen_logits_list, labels):
        if gl is None:
            continue
        probs = F.softmax(gl, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)  # [T]
        scores.append(entropy.mean().item())
        valid_labels.append(lab)

    if len(set(valid_labels)) < 2:
        return BaselineResult(
            name="logit_entropy",
            sequence_auroc=0.5,
            sequence_auprc=0.5,
            description="Insufficient data for logit-entropy baseline",
        )

    s = torch.tensor(scores)
    l = torch.tensor(valid_labels)
    return BaselineResult(
        name="logit_entropy",
        sequence_auroc=compute_auroc(s, l),
        sequence_auprc=compute_auprc(s, l),
        description="Mean logit entropy over generated tokens",
    )


def majority_class_baseline(
    labels: list[int],
    task_families: list[str],
) -> BaselineResult:
    """Baseline: predict the majority class (correct/incorrect) per family.

    This measures how much signal comes from just knowing which task family
    the example belongs to.

    Args:
        labels: list of 0/1 labels.
        task_families: list of family strings.
    """
    # Learn per-family majority
    family_rates: dict[str, float] = {}
    family_counts: dict[str, dict[str, int]] = {}

    for fam, lab in zip(task_families, labels):
        if fam not in family_counts:
            family_counts[fam] = {"pos": 0, "neg": 0}
        if lab == 1:
            family_counts[fam]["pos"] += 1
        else:
            family_counts[fam]["neg"] += 1

    for fam, counts in family_counts.items():
        total = counts["pos"] + counts["neg"]
        family_rates[fam] = counts["pos"] / total if total > 0 else 0.5

    # Score each example by its family's failure rate
    scores = [family_rates.get(fam, 0.5) for fam in task_families]

    if len(set(labels)) < 2:
        return BaselineResult(
            name="majority_class",
            sequence_auroc=0.5,
            sequence_auprc=0.5,
            description="Insufficient class variation",
        )

    s = torch.tensor(scores)
    l = torch.tensor(labels)
    return BaselineResult(
        name="majority_class",
        sequence_auroc=compute_auroc(s, l),
        sequence_auprc=compute_auprc(s, l),
        description="Per-family failure rate as risk score",
    )


def norm_velocity_baseline(
    gen_h_list: list[Tensor],
    labels: list[int],
) -> BaselineResult:
    """Baseline: mean hidden-state velocity norm predicts failure.

    This approximates the DivergenceDetector's logic without needing
    a trained dynamics model: just use raw ||h_t - h_{t-1}|| as a
    proxy for velocity.

    Args:
        gen_h_list: list of [T, D] hidden-state trajectories.
        labels: list of 0/1 labels.
    """
    scores = []

    for h in gen_h_list:
        if h.shape[0] < 2:
            scores.append(0.0)
            continue
        delta = h[1:] - h[:-1]
        vel_norms = delta.norm(dim=-1)  # [T-1]
        scores.append(vel_norms.mean().item())

    if len(set(labels)) < 2:
        return BaselineResult(
            name="norm_velocity",
            sequence_auroc=0.5,
            sequence_auprc=0.5,
            description="Insufficient class variation",
        )

    s = torch.tensor(scores)
    l = torch.tensor(labels)
    return BaselineResult(
        name="norm_velocity",
        sequence_auroc=compute_auroc(s, l),
        sequence_auprc=compute_auprc(s, l),
        description="Mean ||h_t - h_{t-1}|| over generation (DivergenceDetector proxy)",
    )


def cosine_instability_baseline(
    gen_h_list: list[Tensor],
    labels: list[int],
) -> BaselineResult:
    """Baseline: mean cosine distance between consecutive steps.

    Low cosine similarity between steps indicates erratic direction changes.

    Args:
        gen_h_list: list of [T, D] hidden-state trajectories.
        labels: list of 0/1 labels.
    """
    scores = []

    for h in gen_h_list:
        if h.shape[0] < 2:
            scores.append(0.0)
            continue
        cos = F.cosine_similarity(h[1:], h[:-1], dim=-1)  # [T-1]
        # Invert: lower cosine → higher instability → higher "risk" score
        scores.append((1.0 - cos.mean()).item())

    if len(set(labels)) < 2:
        return BaselineResult(
            name="cosine_instability",
            sequence_auroc=0.5,
            sequence_auprc=0.5,
            description="Insufficient class variation",
        )

    s = torch.tensor(scores)
    l = torch.tensor(labels)
    return BaselineResult(
        name="cosine_instability",
        sequence_auroc=compute_auroc(s, l),
        sequence_auprc=compute_auprc(s, l),
        description="Mean (1 - cos(h_t, h_{t-1})) over generation",
    )


def run_all_baselines(
    gen_h_list: list[Tensor],
    labels: list[int],
    task_families: list[str],
    gen_logits_list: list[Tensor | None] | None = None,
) -> list[BaselineResult]:
    """Run all baseline evaluations.

    Args:
        gen_h_list: list of [T, D] hidden-state trajectories.
        labels: list of 0/1 labels.
        task_families: list of family strings.
        gen_logits_list: Optional list of [T, vocab] logits.

    Returns:
        List of BaselineResult.
    """
    results = [
        majority_class_baseline(labels, task_families),
        norm_velocity_baseline(gen_h_list, labels),
        cosine_instability_baseline(gen_h_list, labels),
    ]

    if gen_logits_list is not None:
        results.append(logit_entropy_baseline(gen_logits_list, labels))

    return results


def format_baseline_comparison(
    ews_auroc: float,
    ews_auprc: float,
    baselines: list[BaselineResult],
) -> str:
    """Format a comparison table between EWS and baselines."""
    lines = [
        "=" * 60,
        "BASELINE COMPARISON",
        "=" * 60,
        "",
        f"{'Method':30s}  {'AUROC':>8s}  {'AUPRC':>8s}  {'vs EWS':>8s}",
        "-" * 60,
        f"{'EWS (ours)':30s}  {ews_auroc:8.4f}  {ews_auprc:8.4f}  {'---':>8s}",
    ]

    for b in baselines:
        delta = ews_auroc - b.sequence_auroc
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{b.name:30s}  {b.sequence_auroc:8.4f}  {b.sequence_auprc:8.4f}  "
            f"{sign}{delta:7.4f}"
        )

    return "\n".join(lines)
