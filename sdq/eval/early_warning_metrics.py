"""Evaluation metrics for the SDQ early-warning system.

Primary metrics:
  - AUROC / AUPRC for future-failure prediction
  - Average lead time before visible failure
  - Calibration of risk scores
  - False positive rate on self-correcting sequences

Secondary metrics:
  - Per-family breakdown
  - Per-timestep AUROC curve
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class EarlyWarningReport:
    """Full evaluation report for the early-warning system."""

    # Sequence-level
    sequence_auroc: float
    sequence_auprc: float

    # Per-timestep
    timestep_aurocs: dict[int, float]

    # Lead time
    first_above_chance_t: int | None     # first t where AUROC > 0.55
    first_strong_t: int | None           # first t where AUROC > 0.65
    average_lead_time: float | None      # mean tokens before failure where risk > threshold

    # Calibration
    calibration_bins: list[dict]         # [{predicted_prob, actual_freq, count}]
    expected_calibration_error: float

    # Self-correction
    false_positive_rate_on_recoverers: float | None

    # Per-family
    family_aurocs: dict[str, float] = field(default_factory=dict)


# ── Core metrics ─────────────────────────────────────────────────────────────

def compute_auroc(scores: Tensor, labels: Tensor) -> float:
    """Area under ROC curve (trapezoidal)."""
    if labels.unique().numel() < 2:
        return 0.5

    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    if n_pos == 0 or n_neg == 0:
        return 0.5

    sorted_indices = scores.argsort(descending=True)
    sorted_labels = labels[sorted_indices]

    tp = 0.0
    auc = 0.0
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            auc += tp

    return auc / (n_pos * n_neg)


def compute_auprc(scores: Tensor, labels: Tensor) -> float:
    """Area under precision-recall curve (average precision)."""
    if labels.unique().numel() < 2:
        return (labels == 1).float().mean().item()

    n_pos = (labels == 1).sum().item()
    if n_pos == 0:
        return 0.0

    sorted_indices = scores.argsort(descending=True)
    sorted_labels = labels[sorted_indices]

    tp = 0.0
    ap = 0.0
    for i, lab in enumerate(sorted_labels):
        if lab == 1:
            tp += 1
            precision = tp / (i + 1)
            ap += precision

    return ap / n_pos


def auroc_per_timestep(
    risk_curves: list[Tensor],
    labels: list[int],
    max_t: int | None = None,
) -> dict[int, float]:
    """Compute AUROC at each relative timestep.

    Args:
        risk_curves: list of [T_i] risk score tensors per sequence.
        labels: list of 0 (correct) / 1 (incorrect) per sequence.
        max_t: Maximum timestep to evaluate. Default: max sequence length.

    Returns:
        {timestep: AUROC}
    """
    if max_t is None:
        max_t = max(rc.shape[0] for rc in risk_curves) if risk_curves else 0

    results: dict[int, float] = {}
    for t in range(max_t):
        scores_t = []
        labs_t = []
        for rc, lab in zip(risk_curves, labels):
            if rc.shape[0] > t:
                scores_t.append(rc[t].item())
                labs_t.append(lab)

        if len(set(labs_t)) >= 2:
            results[t] = compute_auroc(torch.tensor(scores_t), torch.tensor(labs_t))
        else:
            results[t] = 0.5

    return results


def average_lead_time(
    risk_curves: list[Tensor],
    labels: list[int],
    threshold: float = 0.5,
) -> float | None:
    """Average number of tokens before end of sequence where risk first exceeds threshold.

    Only considers incorrect (label=1) sequences. Returns None if no
    incorrect sequence ever exceeds the threshold.

    Args:
        risk_curves: list of [T_i] risk score tensors.
        labels: list of 0/1 labels.
        threshold: Risk threshold for triggering a warning.

    Returns:
        Mean lead time in tokens, or None.
    """
    lead_times = []
    for rc, lab in zip(risk_curves, labels):
        if lab != 1:
            continue
        T = rc.shape[0]
        above = (rc > threshold).nonzero(as_tuple=True)[0]
        if above.numel() > 0:
            first_trigger = above[0].item()
            lead_times.append(T - first_trigger)

    if not lead_times:
        return None
    return sum(lead_times) / len(lead_times)


# ── Calibration ──────────────────────────────────────────────────────────────

def calibration_analysis(
    risk_scores: Tensor,
    labels: Tensor,
    n_bins: int = 10,
) -> tuple[list[dict], float]:
    """Compute calibration bins and expected calibration error.

    Args:
        risk_scores: [N] predicted probabilities.
        labels: [N] binary labels (0 or 1).
        n_bins: Number of calibration bins.

    Returns:
        (bins_list, ECE) where bins_list is [{predicted_prob, actual_freq, count}]
    """
    bins: list[dict] = []
    ece = 0.0
    total = labels.shape[0]

    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (risk_scores >= lo) & (risk_scores < hi)
        count = mask.sum().item()

        if count > 0:
            pred_prob = risk_scores[mask].mean().item()
            actual_freq = labels[mask].float().mean().item()
            ece += abs(pred_prob - actual_freq) * count / total
        else:
            pred_prob = (lo + hi) / 2
            actual_freq = 0.0

        bins.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "predicted_prob": pred_prob,
            "actual_freq": actual_freq,
            "count": int(count),
        })

    return bins, ece


# ── Self-correction false positives ──────────────────────────────────────────

def false_positive_rate_on_recoverers(
    risk_curves: list[Tensor],
    labels: list[int],
    recoverer_mask: list[bool],
    threshold: float = 0.5,
) -> float | None:
    """Fraction of self-correcting sequences flagged as risky.

    A "recoverer" is a sequence that wobbled (risk went up) but ultimately
    produced the correct answer (label=0). We want the false-positive rate
    on these to be low.

    Args:
        risk_curves: list of [T_i] risk tensors.
        labels: list of 0/1 labels.
        recoverer_mask: list of bools; True for sequences that are correct
            but showed transient elevated risk.
        threshold: Risk threshold.

    Returns:
        False positive rate, or None if no recoverers.
    """
    n_recoverers = sum(1 for m in recoverer_mask if m)
    if n_recoverers == 0:
        return None

    n_flagged = 0
    for rc, lab, is_recoverer in zip(risk_curves, labels, recoverer_mask):
        if not is_recoverer:
            continue
        if (rc > threshold).any():
            n_flagged += 1

    return n_flagged / n_recoverers


# ── Full evaluation ──────────────────────────────────────────────────────────

def evaluate_early_warning(
    risk_curves: list[Tensor],
    labels: list[int],
    task_families: list[str] | None = None,
    recoverer_mask: list[bool] | None = None,
    threshold: float = 0.5,
    max_t: int | None = None,
) -> EarlyWarningReport:
    """Run the full early-warning evaluation suite.

    Args:
        risk_curves: list of [T_i] per-timestep risk scores.
        labels: list of 0 (correct) / 1 (incorrect) per sequence.
        task_families: Optional list of family names for per-family breakdown.
        recoverer_mask: Optional list of bools for self-correction analysis.
        threshold: Risk threshold for lead-time and FP analysis.
        max_t: Max timestep for per-timestep AUROC.

    Returns:
        EarlyWarningReport with all metrics.
    """
    # Sequence-level scores (mean risk over sequence)
    seq_scores = torch.tensor([rc.mean().item() for rc in risk_curves])
    seq_labels = torch.tensor(labels)

    seq_auroc = compute_auroc(seq_scores, seq_labels)
    seq_auprc = compute_auprc(seq_scores, seq_labels)

    # Per-timestep
    ts_aurocs = auroc_per_timestep(risk_curves, labels, max_t)

    # Lead time
    first_above_chance = None
    first_strong = None
    for t in sorted(ts_aurocs.keys()):
        auc = ts_aurocs[t]
        if first_above_chance is None and auc > 0.55:
            first_above_chance = t
        if first_strong is None and auc > 0.65:
            first_strong = t

    avg_lead = average_lead_time(risk_curves, labels, threshold)

    # Calibration
    all_scores = torch.cat([rc for rc in risk_curves])
    all_labs = torch.cat([
        torch.full((rc.shape[0],), lab)
        for rc, lab in zip(risk_curves, labels)
    ])
    cal_bins, ece = calibration_analysis(all_scores, all_labs)

    # Self-correction FP
    fp_recover = None
    if recoverer_mask is not None:
        fp_recover = false_positive_rate_on_recoverers(
            risk_curves, labels, recoverer_mask, threshold,
        )

    # Per-family
    family_aurocs: dict[str, float] = {}
    if task_families is not None:
        fam_map: dict[str, list[int]] = {}
        for i, fam in enumerate(task_families):
            fam_map.setdefault(fam, []).append(i)

        for fam, indices in fam_map.items():
            fam_scores = torch.tensor([seq_scores[i].item() for i in indices])
            fam_labs = torch.tensor([labels[i] for i in indices])
            if fam_labs.unique().numel() >= 2:
                family_aurocs[fam] = compute_auroc(fam_scores, fam_labs)
            else:
                family_aurocs[fam] = 0.5

    return EarlyWarningReport(
        sequence_auroc=seq_auroc,
        sequence_auprc=seq_auprc,
        timestep_aurocs=ts_aurocs,
        first_above_chance_t=first_above_chance,
        first_strong_t=first_strong,
        average_lead_time=avg_lead,
        calibration_bins=cal_bins,
        expected_calibration_error=ece,
        false_positive_rate_on_recoverers=fp_recover,
        family_aurocs=family_aurocs,
    )


def format_early_warning_report(report: EarlyWarningReport) -> str:
    """Format an EarlyWarningReport as a human-readable string."""
    lines = [
        "=" * 60,
        "EARLY WARNING SYSTEM EVALUATION",
        "=" * 60,
        "",
        f"Sequence AUROC:   {report.sequence_auroc:.4f}",
        f"Sequence AUPRC:   {report.sequence_auprc:.4f}",
        f"ECE:              {report.expected_calibration_error:.4f}",
        "",
        "Lead time analysis:",
        f"  First t > 0.55 AUROC: {report.first_above_chance_t}",
        f"  First t > 0.65 AUROC: {report.first_strong_t}",
        f"  Average lead time:    {report.average_lead_time}",
    ]

    if report.false_positive_rate_on_recoverers is not None:
        lines.append(f"  FP rate on recoverers: {report.false_positive_rate_on_recoverers:.4f}")

    if report.family_aurocs:
        lines.append("")
        lines.append("Per-family AUROC:")
        for fam, auc in sorted(report.family_aurocs.items()):
            lines.append(f"  {fam:25s}  {auc:.4f}")

    lines.append("")
    lines.append("Per-timestep AUROC:")
    for t in sorted(report.timestep_aurocs.keys()):
        auc = report.timestep_aurocs[t]
        bar = "#" * int(auc * 40)
        lines.append(f"  t={t:3d}  {auc:.4f}  {bar}")

    return "\n".join(lines)
