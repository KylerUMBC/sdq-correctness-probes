"""Linear probe: does z_t actually predict model behavior?

Without this, SDQ's latent space could be a well-regularized autoencoder
that happens to cluster nicely but carries no causal meaning. This module
trains a lightweight probe on top of frozen z_t embeddings to test whether
the latent representation contains task-relevant information.

Three probe targets:
  1. Task family classification   (arithmetic vs syllogistic vs ...)
  2. Answer prediction            (can z_t predict what the model will output?)
  3. Correctness prediction       (does z_t distinguish right from wrong reasoning?)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ProbeResult:
    """Evaluation results for a trained probe."""
    accuracy: float
    per_class_accuracy: dict[str, float]
    num_classes: int
    num_samples: int
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    loss: float = 0.0


class LatentProbe(nn.Module):
    """Linear probe over latent trajectory embeddings.

    Takes z_t trajectories, aggregates them (mean, last, or max),
    and classifies into discrete labels.
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.aggregation = aggregation
        self.linear = nn.Linear(latent_dim, num_classes)

    def aggregate(self, z: Tensor) -> Tensor:
        """Reduce a trajectory [T, D_z] to a single vector [D_z]."""
        if self.aggregation == "mean":
            return z.mean(dim=0)
        elif self.aggregation == "last":
            return z[-1]
        elif self.aggregation == "max":
            return z.max(dim=0).values
        raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def forward(self, z: Tensor) -> Tensor:
        """Classify a trajectory.

        Args:
            z: [T, D_z] single trajectory or [B, T, D_z] batch

        Returns:
            logits: [num_classes] or [B, num_classes]
        """
        if z.dim() == 2:
            return self.linear(self.aggregate(z))
        # Batched
        agg = torch.stack([self.aggregate(z[i]) for i in range(z.shape[0])])
        return self.linear(agg)


def train_probe(
    probe: LatentProbe,
    z_trajectories: dict[str, Tensor],
    labels: dict[str, int],
    label_names: dict[int, str],
    epochs: int = 200,
    lr: float = 1e-3,
    train_fraction: float = 0.8,
    device: str = "cpu",
) -> tuple[ProbeResult, ProbeResult]:
    """Train a linear probe and return train + test results.

    Args:
        probe: the probe module
        z_trajectories: {prompt_id: [T, D_z]}
        labels: {prompt_id: class_index}
        label_names: {class_index: human_readable_name}
        epochs: training epochs
        lr: learning rate
        train_fraction: fraction of data for training
        device: device string

    Returns:
        (train_result, test_result)
    """
    # Build dataset
    pids = [pid for pid in z_trajectories if pid in labels]
    if not pids:
        empty = ProbeResult(accuracy=0, per_class_accuracy={}, num_classes=0, num_samples=0)
        return empty, empty

    # Shuffle before splitting to avoid systematic bias
    # (e.g., all examples of one family grouped together)
    import random as _rng
    pids_shuffled = list(pids)
    _rng.Random(42).shuffle(pids_shuffled)
    pids = pids_shuffled

    # Split
    n_train = max(1, int(len(pids) * train_fraction))
    train_pids = pids[:n_train]
    test_pids = pids[n_train:]

    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Pre-aggregate all trajectories (probe trains on frozen z_t)
    train_z = torch.stack([probe.aggregate(z_trajectories[p].to(device)) for p in train_pids])
    train_y = torch.tensor([labels[p] for p in train_pids], dtype=torch.long, device=device)

    # Training loop
    probe.train()
    for _ in range(epochs):
        logits = probe.linear(train_z)
        loss = criterion(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    probe.eval()
    train_result = _evaluate_probe(probe, z_trajectories, train_pids, labels, label_names, device)
    train_result.loss = loss.item()

    if test_pids:
        test_result = _evaluate_probe(probe, z_trajectories, test_pids, labels, label_names, device)
    else:
        test_result = ProbeResult(accuracy=0, per_class_accuracy={}, num_classes=probe.num_classes, num_samples=0)

    return train_result, test_result


@torch.no_grad()
def _evaluate_probe(
    probe: LatentProbe,
    z_trajectories: dict[str, Tensor],
    pids: list[str],
    labels: dict[str, int],
    label_names: dict[int, str],
    device: str,
) -> ProbeResult:
    """Evaluate probe accuracy on a set of prompt IDs."""
    correct = 0
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}
    confusion: dict[str, dict[str, int]] = {}

    for pid in pids:
        z = z_trajectories[pid].to(device)
        logits = probe(z)
        pred = logits.argmax().item()
        true = labels[pid]

        true_name = label_names.get(true, str(true))
        pred_name = label_names.get(pred, str(pred))

        if true_name not in confusion:
            confusion[true_name] = {}
        confusion[true_name][pred_name] = confusion[true_name].get(pred_name, 0) + 1

        per_class_total[true_name] = per_class_total.get(true_name, 0) + 1
        if pred == true:
            correct += 1
            per_class_correct[true_name] = per_class_correct.get(true_name, 0) + 1

    per_class_acc = {
        name: per_class_correct.get(name, 0) / total
        for name, total in per_class_total.items()
    }

    return ProbeResult(
        accuracy=correct / len(pids) if pids else 0,
        per_class_accuracy=per_class_acc,
        num_classes=len(per_class_total),
        num_samples=len(pids),
        confusion=confusion,
    )


def format_probe_result(name: str, train: ProbeResult, test: ProbeResult) -> str:
    """Human-readable summary."""
    lines = [
        f"═══ PROBE: {name} ═══",
        f"  Train accuracy: {train.accuracy:.1%}  ({train.num_samples} samples)",
        f"  Test  accuracy: {test.accuracy:.1%}  ({test.num_samples} samples)",
    ]
    if test.per_class_accuracy:
        lines.append("  Per-class (test):")
        for cls, acc in sorted(test.per_class_accuracy.items()):
            lines.append(f"    {cls:30s} {acc:.1%}")
    return "\n".join(lines)
