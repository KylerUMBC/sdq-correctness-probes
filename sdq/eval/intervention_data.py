"""Shared data loading for SDQ intervention runners (Phases 4, 4b, 5).

This module exists to fix two problems found in the 2026-06 audit:

1. **Two different definitions of h_0 were in circulation.**
   `gen_activations.pt` stores per-generated-token states and *excludes* the
   prefill pass, so `gen_activations[0]` is the residual stream at the FIRST
   GENERATED token — after the model has already emitted it. The true
   prompt-final state ("before generation begins") is
   `activations.pt[layer, -1, :]`. Different scripts used different
   definitions, and one script could silently mix both via a fallback.
   Here the source is an explicit, mandatory choice and is never mixed.

2. **Every runner had its own fork of the loading/selection code**, with
   behaviors that drifted (e.g. only one script could resolve
   timestamp-suffixed run directories).

h_0 source semantics:
    "prompt_final" — activations.pt[layer, -1, :]. The state at the last
        prompt token, before any token is generated. This is the state the
        prefill intervention hook actually perturbs, and the honest basis
        for any "predicts before generation" claim.
    "first_gen"    — gen_activations.pt[0, layer, :]. The state at the first
        generated token (the historical definition used by Phases 2-5 before
        the audit). Kept for comparison: the AUROC gap between the two
        sources measures how much signal was first-token leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from sdq.labels.outcome_labeler import label_run

H0Source = Literal["prompt_final", "first_gen"]


def load_benchmark(path: str | Path) -> list[dict]:
    """Load benchmark examples from JSON (bare list or {"examples": [...]})."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("examples", [])


def resolve_run_dirs(runs_dir: Path, example_ids: list[str]) -> dict[str, Path]:
    """Map example_id -> run directory.

    Tries exact match first (`runs_dir/<eid>`), then a prefix scan for
    `<eid>_*` directories (matching save_run's `<prompt_id>_<timestamp>`
    layout). When multiple suffixed dirs match, the latest timestamp wins.
    """
    resolved: dict[str, Path] = {}
    eid_set = set(example_ids)

    for eid in example_ids:
        d = runs_dir / eid
        if (d / "metadata.json").exists():
            resolved[eid] = d

    missing = eid_set - resolved.keys()
    if not missing:
        return resolved

    by_prefix: dict[str, list[Path]] = {}
    for d in runs_dir.iterdir():
        if not d.is_dir() or not (d / "metadata.json").exists():
            continue
        for eid in missing:
            if d.name.startswith(eid + "_"):
                by_prefix.setdefault(eid, []).append(d)
                break
    for eid, dirs in by_prefix.items():
        dirs.sort()  # timestamp suffix sorts chronologically
        resolved[eid] = dirs[-1]

    return resolved


def load_h0(
    run_dir: Path,
    layer: int = -1,
    source: H0Source = "prompt_final",
) -> Tensor | None:
    """Load a single h_0 vector from a run directory, or None if unavailable.

    See module docstring for source semantics. `layer` indexes the captured
    transformer layers (negative indices allowed); both files share the same
    layer indexing (embedding layer excluded at capture time).
    """
    if source == "prompt_final":
        path = run_dir / "activations.pt"
        if not path.exists():
            return None
        act = torch.load(path, map_location="cpu", weights_only=True)
        return act[layer, -1, :].float()

    if source == "first_gen":
        path = run_dir / "gen_activations.pt"
        if not path.exists():
            return None
        gen = torch.load(path, map_location="cpu", weights_only=True)
        if gen.shape[0] == 0:
            return None
        return gen[0, layer, :].float()

    raise ValueError(f"Unknown h0 source: {source!r}")


def collect_h0_and_labels(
    examples: list[dict],
    runs_dir: Path,
    layer: int = -1,
    source: H0Source = "prompt_final",
) -> list[dict]:
    """Load h_0 and correctness for every example with a resolvable run.

    Returns example dicts augmented with `_h_0`, `_correct`, `_confidence`
    (answer-parser confidence), and `_run_dir`. Examples whose run lacks the
    requested activation file are skipped — sources are never mixed.
    """
    eids = [ex["example_id"] for ex in examples]
    dir_map = resolve_run_dirs(runs_dir, eids)

    out: list[dict] = []
    n_skipped_no_file = 0
    for ex in examples:
        run_dir = dir_map.get(ex["example_id"])
        if run_dir is None:
            continue
        with open(run_dir / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        new_tokens = meta.get("output", {}).get("new_tokens", "")
        outcome = label_run(new_tokens, ex["answer_id"], ex["task_family"])

        h_0 = load_h0(run_dir, layer=layer, source=source)
        if h_0 is None:
            n_skipped_no_file += 1
            continue
        out.append({
            **ex,
            "_h_0": h_0,
            "_correct": outcome.correct,
            "_confidence": outcome.confidence,
            "_run_dir": run_dir,
        })

    if n_skipped_no_file:
        print(f"  [collect_h0] skipped {n_skipped_no_file} runs without "
              f"the '{source}' activation file (sources are never mixed)")
    return out


def collect_h0_multi_layer(
    examples: list[dict],
    layers: list[int],
) -> dict[int, Tensor]:
    """Prompt-final states at several layers for already-collected examples.

    Reads each example's activations.pt once and slices all requested layers
    at the last prompt token. Used for per-layer magnitude calibration: an
    intervention at layer L must be calibrated against the natural variation
    of layer-L states, not last-layer states (residual norms grow with depth).

    Args:
        examples: dicts from collect_h0_and_labels (need `_run_dir`).
        layers: non-negative layer indices to extract.

    Returns:
        {layer: Tensor[N, D]} aligned with the order of `examples`.
        Raises if any example lacks activations.pt — calibration must not
        silently drop examples.
    """
    per_layer: dict[int, list[Tensor]] = {l: [] for l in layers}
    for ex in examples:
        path = Path(ex["_run_dir"]) / "activations.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"activations.pt missing for {ex['example_id']} — cannot "
                f"compute per-layer calibration")
        act = torch.load(path, map_location="cpu", weights_only=True)
        for l in layers:
            per_layer[l].append(act[l, -1, :].float())
    return {l: torch.stack(vs) for l, vs in per_layer.items()}


def compute_probe_scores(
    h0_matrix: Tensor,
    direction_std: Tensor,
    weight_norm: float,
    bias: float,
    scaler_mu: Tensor,
    scaler_sd: Tensor,
) -> Tensor:
    """P(incorrect) per example from the Phase 3 commitment probe."""
    mu = scaler_mu.squeeze()
    sd = scaler_sd.squeeze().clamp_min(1e-6)
    z = (h0_matrix.float() - mu) / sd
    logits = (z @ direction_std.float()) * weight_norm + bias
    return torch.sigmoid(logits)


def select_examples(
    all_examples: list[dict],
    probe_scores: Tensor,
    n_prompts: int,
    boundary_lo: float,
    boundary_hi: float,
) -> list[dict]:
    """Select correct examples for intervention, stratified by family.

    Within each family, near-boundary examples (probe score in
    [boundary_lo, boundary_hi]) are preferred, then others ordered by
    closeness to 0.5. Families are filled round-robin so no single family
    dominates the selection.
    """
    by_family: dict[str, list[dict]] = {}
    for i, ex in enumerate(all_examples):
        if not ex["_correct"]:
            continue
        e = {**ex, "_probe_score": float(probe_scores[i].item())}
        by_family.setdefault(e["task_family"], []).append(e)

    # Per-family priority order: boundary first, then by closeness to 0.5
    for fam, exs in by_family.items():
        boundary = [e for e in exs
                    if boundary_lo <= e["_probe_score"] <= boundary_hi]
        others = [e for e in exs
                  if not (boundary_lo <= e["_probe_score"] <= boundary_hi)]
        others.sort(key=lambda e: abs(e["_probe_score"] - 0.5))
        by_family[fam] = boundary + others

    # Round-robin fill across families
    selected: list[dict] = []
    queues = {fam: list(exs) for fam, exs in sorted(by_family.items())}
    while len(selected) < n_prompts and any(queues.values()):
        for fam in list(queues.keys()):
            if queues[fam]:
                selected.append(queues[fam].pop(0))
                if len(selected) >= n_prompts:
                    break
    return selected
