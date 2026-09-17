"""Commitment geometry probes for SDQ Phase 3.

Scientific claim under test:

    Transformer prompt encodings contain a low-dimensional "commitment
    subspace" that predicts final answer correctness before any generation
    occurs. This subspace is partially family-conditional and partially
    general. It is causally load-bearing: perturbations along it change
    the output.

This module provides:
  - Linear and low-rank probes for correctness prediction from h_0 / z_0
  - Logit-confidence feature extraction (entropy, margin, top-k mass)
  - Stratified per-family AUROC evaluation
  - Leave-one-family-out cross-family transfer evaluation
  - PCA-based subspace rank estimation (AUROC vs. dimension curve)
  - Commitment direction extraction from trained probe weights

Stopping rule (Phase 3):
    within-family AUROC for the probe must exceed BOTH the family prior
    AND the logit-confidence baseline by >= 0.03 on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── AUROC (dependency-free) ─────────────────────────────────────────────────

def compute_auroc(scores: Tensor, labels: Tensor) -> float:
    """Mann-Whitney AUROC with proper tie handling. Labels: 1=positive, 0=negative.

    Tied scores receive 0.5 credit (average-rank method). This matters when
    scores are constant within groups — e.g. the family-prior baseline, where
    every member of a family shares one score and the AUROC would otherwise
    depend on arbitrary sort order.
    """
    if labels.unique().numel() < 2:
        return 0.5
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    s = scores.detach().float().flatten()
    y = labels.detach().flatten()
    order = s.argsort()
    sorted_s = s[order]

    # Average ranks (1-based) with ties sharing their mean rank
    n = s.numel()
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0  # mean of ranks i+1 .. j+1
        i = j + 1

    rank_sum_pos = ranks[y == 1].sum().item()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def per_family_auroc(
    scores: Tensor,
    labels: Tensor,
    families: list[str],
) -> dict[str, float]:
    """Compute AUROC stratified by task family."""
    fam_idx: dict[str, list[int]] = {}
    for i, f in enumerate(families):
        fam_idx.setdefault(f, []).append(i)
    out: dict[str, float] = {}
    for f, idx in fam_idx.items():
        s = scores[idx]
        y = labels[idx]
        out[f] = compute_auroc(s, y) if y.unique().numel() >= 2 else float("nan")
    return out


def mean_within_family_auroc(
    scores: Tensor,
    labels: Tensor,
    families: list[str],
) -> float:
    """Mean AUROC across families (excluding degenerate ones)."""
    fam_aurocs = per_family_auroc(scores, labels, families)
    valid = [v for v in fam_aurocs.values() if v == v]  # filter NaN
    return sum(valid) / len(valid) if valid else 0.5


# ── Standardization ─────────────────────────────────────────────────────────

@dataclass
class Standardizer:
    """Z-score standardizer fit on training data."""
    mu: Tensor
    sd: Tensor

    @staticmethod
    def fit(X: Tensor) -> "Standardizer":
        mu = X.mean(dim=0, keepdim=True)
        sd = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        return Standardizer(mu=mu, sd=sd)

    def transform(self, X: Tensor) -> Tensor:
        return (X - self.mu.to(X.device)) / self.sd.to(X.device)


# ── Probe models ─────────────────────────────────────────────────────────────

class LinearCommitmentProbe(nn.Module):
    """Single linear layer: h -> logit(P(incorrect))."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)

    @property
    def weight_vector(self) -> Tensor:
        """The D-dimensional commitment direction (detached)."""
        return self.linear.weight.data.squeeze(0).detach().clone()


class LowRankCommitmentProbe(nn.Module):
    """Bottleneck probe: h -> W1(D,k) -> ReLU -> W2(k,1) -> logit.

    The bottleneck dimension k constrains the probe to use at most a
    rank-k subspace of the input, allowing us to measure how much
    discriminative information lives in k dimensions.
    """

    def __init__(self, input_dim: int, rank: int):
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(input_dim, rank)
        self.up = nn.Linear(rank, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(F.relu(self.down(x))).squeeze(-1)

    @property
    def projection_matrix(self) -> Tensor:
        """The D x k projection (down layer weights, transposed)."""
        return self.down.weight.data.detach().clone().T  # [D, k]


# ── Logit confidence features ───────────────────────────────────────────────

def extract_logit_features(logits: Tensor) -> Tensor:
    """Compute interpretable confidence features from a logit vector.

    Returns a 5-dim feature vector:
        [0] Shannon entropy of softmax (nats)
        [1] max softmax probability
        [2] top1 - top2 logit gap (margin)
        [3] top-5 cumulative softmax probability
        [4] log of max softmax probability
    """
    p = F.softmax(logits, dim=-1)
    # Entropy
    entropy = -(p * p.clamp_min(1e-12).log()).sum().item()
    # Max prob
    max_prob = p.max().item()
    # Top-1 vs top-2 gap
    topk = logits.topk(min(5, logits.shape[-1]))
    margin = (topk.values[0] - topk.values[1]).item() if topk.values.shape[0] >= 2 else 0.0
    # Top-5 mass
    top5_mass = F.softmax(logits, dim=-1)[topk.indices].sum().item()
    # Log max prob
    log_max_prob = max_prob if max_prob > 0 else 1e-12
    log_max_prob = torch.tensor(log_max_prob).log().item()

    return torch.tensor([entropy, max_prob, margin, top5_mass, log_max_prob],
                        dtype=torch.float32)


# ── Probe training ──────────────────────────────────────────────────────────

def fit_probe(
    probe: nn.Module,
    X: Tensor,
    y: Tensor,
    epochs: int = 400,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    """Train a probe with full-batch AdamW + BCEWithLogits. Returns it in eval mode.

    `X` is used as-is — standardize/project before calling if needed.
    This is the single canonical training loop; every probe in the project
    should go through it so hyperparameters and behavior stay comparable.
    """
    probe = probe.to(device)
    Xd = X.to(device)
    yd = y.to(device).float()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(probe(Xd), yd)
        loss.backward()
        opt.step()
    probe.eval()
    return probe


@torch.no_grad()
def predict_probe_scores(probe: nn.Module, X: Tensor, device: str = "cpu") -> Tensor:
    """Sigmoid scores from a trained probe, returned on CPU."""
    return torch.sigmoid(probe(X.to(device))).cpu()


@dataclass
class ProbeResult:
    """Full evaluation of a trained commitment probe."""
    name: str
    pooled_auroc: float
    mean_within_family_auroc: float
    family_aurocs: dict[str, float]
    scores: Tensor          # [N_test] predicted risk scores
    labels: Tensor          # [N_test] ground truth
    families: list[str]     # [N_test] family names
    probe: nn.Module | None = None


def train_probe(
    probe: nn.Module,
    X_train: Tensor,
    y_train: Tensor,
    X_test: Tensor,
    y_test: Tensor,
    test_families: list[str],
    name: str = "probe",
    epochs: int = 400,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    device: str = "cpu",
) -> ProbeResult:
    """Train a commitment probe and evaluate.

    Args:
        probe: nn.Module with forward(X) -> logits [N].
        X_train, y_train: training features and labels (1=incorrect).
        X_test, y_test: test features and labels.
        test_families: family name per test example.
        name: human-readable probe name.
        epochs: SGD epochs.
        lr: learning rate.
        weight_decay: L2 penalty.
        device: torch device.

    Returns:
        ProbeResult with all metrics.
    """
    # Standardize
    scaler = Standardizer.fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    probe = fit_probe(probe, Xtr, y_train, epochs=epochs, lr=lr,
                      weight_decay=weight_decay, device=device)
    scores = predict_probe_scores(probe, Xte, device=device)
    labels_cpu = y_test.cpu()

    pooled = compute_auroc(scores, labels_cpu)
    fam_aurocs = per_family_auroc(scores, labels_cpu, test_families)
    mwf = mean_within_family_auroc(scores, labels_cpu, test_families)

    return ProbeResult(
        name=name,
        pooled_auroc=pooled,
        mean_within_family_auroc=mwf,
        family_aurocs=fam_aurocs,
        scores=scores,
        labels=labels_cpu,
        families=test_families,
        probe=probe,
    )


# ── Cross-family transfer (leave-one-out) ───────────────────────────────────

@dataclass
class TransferResult:
    """Leave-one-family-out transfer evaluation."""
    held_out_family: str
    auroc: float
    n_test: int
    n_pos: int
    n_neg: int


def cross_family_transfer(
    X: Tensor,
    y: Tensor,
    families: list[str],
    input_dim: int,
    epochs: int = 400,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    device: str = "cpu",
) -> list[TransferResult]:
    """Train on N-1 families, test on the held-out family.

    If the commitment subspace is genuinely semantic (not family
    memorization), it should transfer above chance (AUROC > 0.5).

    Returns one TransferResult per family.
    """
    unique_families = sorted(set(families))
    results: list[TransferResult] = []

    for held_out in unique_families:
        # Split
        train_mask = [f != held_out for f in families]
        test_mask = [f == held_out for f in families]

        train_idx = [i for i, m in enumerate(train_mask) if m]
        test_idx = [i for i, m in enumerate(test_mask) if m]

        if not train_idx or not test_idx:
            continue

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Skip if held-out family is degenerate (all same label)
        if y_test.unique().numel() < 2:
            results.append(TransferResult(
                held_out_family=held_out,
                auroc=float("nan"),
                n_test=len(test_idx),
                n_pos=int((y_test == 1).sum().item()),
                n_neg=int((y_test == 0).sum().item()),
            ))
            continue

        # Train fresh linear probe
        scaler = Standardizer.fit(X_train)
        probe = fit_probe(LinearCommitmentProbe(input_dim),
                          scaler.transform(X_train), y_train,
                          epochs=epochs, lr=lr, weight_decay=weight_decay,
                          device=device)
        scores = predict_probe_scores(probe, scaler.transform(X_test), device=device)

        auroc = compute_auroc(scores, y_test.cpu())
        results.append(TransferResult(
            held_out_family=held_out,
            auroc=auroc,
            n_test=len(test_idx),
            n_pos=int((y_test == 1).sum().item()),
            n_neg=int((y_test == 0).sum().item()),
        ))

    return results


# ── Subspace analysis ────────────────────────────────────────────────────────

@dataclass
class SubspaceResult:
    """Commitment subspace dimensionality analysis."""
    rank_aurocs: dict[int, float]             # {rank: pooled_auroc}
    rank_within_family_aurocs: dict[int, float]  # {rank: mean within-family auroc}
    full_auroc: float                         # full-rank probe AUROC
    rank_for_90pct: int | None                # smallest rank hitting 90% of full
    rank_for_95pct: int | None                # smallest rank hitting 95% of full
    pca_explained_variance: list[float]       # top eigenvalues


def subspace_rank_sweep(
    X_train: Tensor,
    y_train: Tensor,
    X_test: Tensor,
    y_test: Tensor,
    test_families: list[str],
    full_auroc: float,
    max_rank: int = 64,
    ranks: Sequence[int] | None = None,
    epochs: int = 400,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    device: str = "cpu",
) -> SubspaceResult:
    """Estimate commitment subspace dimensionality.

    Strategy: PCA-project h_0 to top-k components, train a linear probe
    on the projected features, sweep k from 1 to max_rank.

    The rank at which we hit 90% of the full-D probe's AUROC is the
    working estimate of the commitment subspace dimensionality.
    """
    D = X_train.shape[1]

    # PCA on training data
    scaler = Standardizer.fit(X_train)
    Xtr_std = scaler.transform(X_train)
    Xte_std = scaler.transform(X_test)

    # Covariance and eigen decomposition
    cov = (Xtr_std.T @ Xtr_std) / (Xtr_std.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # eigh returns ascending order; reverse
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    explained_variance = eigenvalues[:max_rank].tolist()

    if ranks is None:
        ranks = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
        ranks = [r for r in ranks if r <= min(max_rank, D)]

    rank_aurocs: dict[int, float] = {}
    rank_wf_aurocs: dict[int, float] = {}

    for k in ranks:
        # Project onto top-k PCs
        V_k = eigenvectors[:, :k].to(device)  # [D, k]
        Xtr_k = (Xtr_std.to(device) @ V_k)   # [N_train, k]
        Xte_k = (Xte_std.to(device) @ V_k)   # [N_test, k]

        # Train a linear probe on the projected features
        probe = fit_probe(LinearCommitmentProbe(k), Xtr_k, y_train,
                          epochs=epochs, lr=lr, weight_decay=weight_decay,
                          device=device)
        scores = predict_probe_scores(probe, Xte_k, device=device)

        auroc = compute_auroc(scores, y_test.cpu())
        wf_auroc = mean_within_family_auroc(scores, y_test.cpu(), test_families)
        rank_aurocs[k] = auroc
        rank_wf_aurocs[k] = wf_auroc

    # Find rank for 90% and 95% of full AUROC
    # "90% of full" means AUROC >= full_auroc * 0.90 (or more precisely,
    # we measure how much above 0.5 baseline — percentage of the "lift")
    lift = full_auroc - 0.5
    target_90 = 0.5 + 0.90 * lift
    target_95 = 0.5 + 0.95 * lift

    rank_90: int | None = None
    rank_95: int | None = None
    for k in sorted(rank_aurocs.keys()):
        if rank_90 is None and rank_aurocs[k] >= target_90:
            rank_90 = k
        if rank_95 is None and rank_aurocs[k] >= target_95:
            rank_95 = k

    return SubspaceResult(
        rank_aurocs=rank_aurocs,
        rank_within_family_aurocs=rank_wf_aurocs,
        full_auroc=full_auroc,
        rank_for_90pct=rank_90,
        rank_for_95pct=rank_95,
        pca_explained_variance=explained_variance,
    )


# ── Commitment direction extraction ─────────────────────────────────────────

@dataclass
class CommitmentDirection:
    """The extracted commitment direction from a trained probe."""
    direction: Tensor           # [D] unit vector
    weight_norm: float          # ||w|| before normalization
    bias: float                 # probe bias term
    top_components_idx: list[int]   # indices of top-contributing dimensions
    top_components_weight: list[float]  # corresponding weight magnitudes


def extract_commitment_direction(
    probe: LinearCommitmentProbe,
    top_k: int = 20,
) -> CommitmentDirection:
    """Extract the commitment direction from a trained linear probe.

    The weight vector w of the linear probe defines a 1D subspace.
    Projecting h_0 onto w (after normalization) gives the "commitment
    score" — higher values predict incorrectness.
    """
    w = probe.weight_vector  # [D]
    norm = w.norm().item()
    direction = w / w.norm().clamp_min(1e-8)
    bias = probe.linear.bias.data.item()

    # Top contributing dimensions
    abs_w = w.abs()
    topk = abs_w.topk(min(top_k, w.shape[0]))

    return CommitmentDirection(
        direction=direction,
        weight_norm=norm,
        bias=bias,
        top_components_idx=topk.indices.tolist(),
        top_components_weight=topk.values.tolist(),
    )


# ── Formatting ──────────────────────────────────────────────────────────────

def format_probe_result(result: ProbeResult) -> str:
    lines = [
        f"--- {result.name} ---",
        f"  Pooled AUROC:              {result.pooled_auroc:.4f}",
        f"  Mean within-family AUROC:  {result.mean_within_family_auroc:.4f}",
        f"  Per-family:",
    ]
    for fam, auc in sorted(result.family_aurocs.items()):
        s = f"{auc:.4f}" if auc == auc else "  N/A"
        lines.append(f"    {fam:25s}  {s}")
    return "\n".join(lines)


def format_transfer_results(results: list[TransferResult]) -> str:
    lines = ["--- Cross-family transfer (leave-one-out) ---"]
    valid = []
    for r in results:
        s = f"{r.auroc:.4f}" if r.auroc == r.auroc else "  N/A (degenerate)"
        lines.append(f"  held-out {r.held_out_family:25s}  AUROC={s}  "
                      f"(n={r.n_test}, pos={r.n_pos}, neg={r.n_neg})")
        if r.auroc == r.auroc:
            valid.append(r.auroc)
    if valid:
        mean_auc = sum(valid) / len(valid)
        lines.append(f"  Mean transfer AUROC: {mean_auc:.4f}")
        lines.append(f"  Transfers above chance (>0.55): "
                      f"{sum(1 for v in valid if v > 0.55)}/{len(valid)}")
    return "\n".join(lines)


def format_subspace_result(result: SubspaceResult) -> str:
    lines = [
        "--- Subspace rank sweep ---",
        f"  Full-rank AUROC: {result.full_auroc:.4f}",
        f"  90% lift rank:   {result.rank_for_90pct}",
        f"  95% lift rank:   {result.rank_for_95pct}",
        "",
        "  Rank    Pooled AUROC    Within-family AUROC",
    ]
    for k in sorted(result.rank_aurocs.keys()):
        pa = result.rank_aurocs[k]
        wfa = result.rank_within_family_aurocs.get(k, float("nan"))
        pct = (pa - 0.5) / max(result.full_auroc - 0.5, 1e-8) * 100
        lines.append(f"    {k:4d}    {pa:.4f}  ({pct:5.1f}%)    {wfa:.4f}")
    lines.append("")
    lines.append("  Top PCA explained variance ratios:")
    total_var = sum(result.pca_explained_variance) if result.pca_explained_variance else 1.0
    cumulative = 0.0
    for i, ev in enumerate(result.pca_explained_variance[:20]):
        cumulative += ev
        lines.append(f"    PC{i:3d}  {ev/total_var:8.4f}  cumulative={cumulative/total_var:.4f}")
    return "\n".join(lines)
