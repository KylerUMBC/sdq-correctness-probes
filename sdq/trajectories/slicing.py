"""Layer and token-position slicing of hidden-state tensors.

Utility functions for selecting specific layers or token ranges
from raw activation tensors [num_layers, seq_len, hidden_dim].
"""

from __future__ import annotations

import torch


def slice_layer(activations: torch.Tensor, layer: int) -> torch.Tensor:
    """Extract a single layer. Returns [seq_len, hidden_dim]."""
    if layer == -1:
        layer = activations.shape[0] - 1
    return activations[layer]


def slice_layers(
    activations: torch.Tensor,
    start: int,
    end: int,
    mode: str = "mean",
) -> torch.Tensor:
    """Extract and reduce a range of layers. Returns [seq_len, hidden_dim].

    Args:
        activations: [num_layers, seq_len, hidden_dim]
        start: inclusive start layer index
        end: inclusive end layer index
        mode: 'mean' to average, 'concat' to concatenate along hidden dim
    """
    selected = activations[start : end + 1]  # [n_sel, seq_len, hidden_dim]
    if mode == "mean":
        return selected.mean(dim=0)
    elif mode == "concat":
        # [seq_len, n_sel * hidden_dim]
        return selected.permute(1, 0, 2).reshape(selected.shape[1], -1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def slice_token_range(
    states: torch.Tensor,
    start: int = 0,
    end: int | None = None,
) -> torch.Tensor:
    """Slice a token range from states [seq_len, hidden_dim] or [*, seq_len, hidden_dim]."""
    if end is None:
        return states[..., start:, :]
    return states[..., start:end, :]
