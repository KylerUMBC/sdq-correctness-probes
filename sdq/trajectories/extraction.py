"""Extract hidden-state trajectories from run artifacts.

A Trajectory wraps a tensor of shape [T, D] representing the
sequence of hidden states across token positions at a specific
layer (or averaged across a layer range).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sdq.instrumentation.run_storage import Run


@dataclass
class Trajectory:
    """A hidden-state trajectory: the core SDQ object.

    Attributes:
        states: [T, D] hidden states across token positions.
        prompt_id: which prompt produced this trajectory.
        layer: which layer (or layer description) this was extracted from.
        token_ids: corresponding token IDs.
        token_strings: corresponding token strings.
        metadata: any additional info.
    """

    states: torch.Tensor  # [T, D]
    prompt_id: str = ""
    layer: int | str = -1
    token_ids: list[int] = field(default_factory=list)
    token_strings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        """Sequence length."""
        return self.states.shape[0]

    @property
    def D(self) -> int:
        """Hidden dimension."""
        return self.states.shape[1]

    def to(self, device: str | torch.device) -> Trajectory:
        return Trajectory(
            states=self.states.to(device),
            prompt_id=self.prompt_id,
            layer=self.layer,
            token_ids=self.token_ids,
            token_strings=self.token_strings,
            metadata=self.metadata,
        )

    def float(self) -> Trajectory:
        return Trajectory(
            states=self.states.float(),
            prompt_id=self.prompt_id,
            layer=self.layer,
            token_ids=self.token_ids,
            token_strings=self.token_strings,
            metadata=self.metadata,
        )


def extract_trajectory(
    run: Run,
    layer: int = -1,
    layer_range: tuple[int, int] | None = None,
    token_slice: slice | None = None,
    to_float: bool = True,
) -> Trajectory:
    """Extract a single trajectory from a run.

    Args:
        run: A loaded Run with activations [num_layers, seq_len, hidden_dim].
        layer: Which layer to extract. -1 means last layer.
            Ignored if layer_range is set.
        layer_range: If set, average hidden states across this (inclusive) range.
        token_slice: Optional slice to select a subset of token positions.
        to_float: Convert bfloat16 to float32 for computation.

    Returns:
        A Trajectory with states [T, D].
    """
    act = run.activations  # [num_layers, seq_len, hidden_dim]

    if layer_range is not None:
        lo, hi = layer_range
        states = act[lo : hi + 1].mean(dim=0)  # [seq_len, hidden_dim]
        layer_label = f"avg_{lo}_{hi}"
    else:
        if layer == -1:
            layer = act.shape[0] - 1
        states = act[layer]  # [seq_len, hidden_dim]
        layer_label = layer

    if token_slice is not None:
        states = states[token_slice]
        token_ids = run.token_ids[token_slice]
        token_strings = run.token_strings[token_slice]
    else:
        token_ids = run.token_ids
        token_strings = run.token_strings

    if to_float:
        states = states.float()

    return Trajectory(
        states=states,
        prompt_id=run.prompt_id,
        layer=layer_label,
        token_ids=token_ids,
        token_strings=token_strings,
        metadata={"run_id": run.run_id},
    )


def extract_trajectories(
    runs: dict[str, Run],
    layer: int = -1,
    layer_range: tuple[int, int] | None = None,
    token_slice: slice | None = None,
    to_float: bool = True,
) -> dict[str, Trajectory]:
    """Extract trajectories from multiple runs.

    Returns a dict keyed by prompt_id.
    """
    return {
        pid: extract_trajectory(run, layer, layer_range, token_slice, to_float)
        for pid, run in runs.items()
    }
