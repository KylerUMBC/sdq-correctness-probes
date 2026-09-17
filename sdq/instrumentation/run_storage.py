"""Persist and load instrumentation run artifacts.

On-disk format per run directory:
    activations.pt      — [num_layers, seq_len, hidden_dim] bfloat16
                          Prompt (prefill) states. The prompt-final state
                          "before generation begins" is [layer, -1, :].
    logits.pt           — [1, seq_len, vocab_size] bfloat16
    metadata.json       — prompt info, model config, generation output
    gen_activations.pt  — [num_gen_tokens - 1, num_layers, hidden_dim]
                          (EWS runs). Entry t is the state AT generated
                          token t (after it was emitted); the prefill pass
                          is excluded, so entry 0 is NOT the prompt-final
                          state. See GenerationCaptureResult docstring.
    gen_logits.pt       — [num_gen_tokens, vocab_size] (EWS runs, optional)
                          Entry t is the logits that *produced* token t
                          (entry 0 comes from prefill — note the asymmetry
                          with gen_activations).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sdq.instrumentation.hidden_capture import CaptureResult, GenerationCaptureResult


@dataclass
class Run:
    """A loaded instrumentation run."""

    run_id: str
    activations: torch.Tensor  # [num_layers, seq_len, hidden_dim]
    prompt_id: str
    prompt_text: str
    token_ids: list[int]
    token_strings: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    logits: torch.Tensor | None = None
    run_dir: Path | None = None

    @property
    def num_layers(self) -> int:
        return self.activations.shape[0]

    @property
    def seq_len(self) -> int:
        return self.activations.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self.activations.shape[2]


@dataclass
class GenerationRun(Run):
    """A loaded run that also includes per-generated-token hidden states."""

    gen_activations: torch.Tensor | None = None  # [num_gen_tokens, num_layers, hidden_dim]
    gen_logits: torch.Tensor | None = None        # [num_gen_tokens, vocab_size]
    gen_token_ids: list[int] = field(default_factory=list)
    gen_token_strings: list[str] = field(default_factory=list)

    @property
    def num_gen_tokens(self) -> int:
        if self.gen_activations is not None:
            return self.gen_activations.shape[0]
        return 0

    @property
    def has_generation_data(self) -> bool:
        return self.gen_activations is not None and self.gen_activations.shape[0] > 0


def load_run(
    run_dir: str | Path,
    load_logits: bool = False,
    load_generation: bool = False,
    device: str = "cpu",
) -> Run | GenerationRun:
    """Load a single run from disk.

    Args:
        run_dir: Path to the run directory.
        load_logits: Whether to load prompt logits.
        load_generation: Whether to load generation-time hidden states.
            If True and gen_activations.pt exists, returns a GenerationRun.
        device: Device for loaded tensors.
    """
    run_dir = Path(run_dir)

    with open(run_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    activations = torch.load(
        run_dir / "activations.pt",
        map_location=device,
        weights_only=True,
    )

    logits = None
    if load_logits and (run_dir / "logits.pt").exists():
        logits = torch.load(
            run_dir / "logits.pt",
            map_location=device,
            weights_only=True,
        )

    prompt_meta = meta.get("prompt", {})
    output_meta = meta.get("output", {})

    base_kwargs = dict(
        run_id=meta.get("metadata", {}).get("run_id", run_dir.name),
        activations=activations,
        prompt_id=prompt_meta.get("id", ""),
        prompt_text=prompt_meta.get("text", ""),
        token_ids=prompt_meta.get("token_ids", []),
        token_strings=prompt_meta.get("token_strings", []),
        metadata=meta,
        logits=logits,
        run_dir=run_dir,
    )

    gen_acts_path = run_dir / "gen_activations.pt"
    if load_generation and gen_acts_path.exists():
        gen_activations = torch.load(gen_acts_path, map_location=device, weights_only=True)
        gen_logits = None
        gen_logits_path = run_dir / "gen_logits.pt"
        if load_logits and gen_logits_path.exists():
            gen_logits = torch.load(gen_logits_path, map_location=device, weights_only=True)

        return GenerationRun(
            **base_kwargs,
            gen_activations=gen_activations,
            gen_logits=gen_logits,
            gen_token_ids=output_meta.get("gen_token_ids", []),
            gen_token_strings=output_meta.get("gen_token_strings", []),
        )

    return Run(**base_kwargs)


def collect_runs(
    runs_dir: str | Path,
    prompt_ids: list[str] | None = None,
    prefix: str = "",
    load_logits: bool = False,
    load_generation: bool = False,
    device: str = "cpu",
) -> dict[str, Run | GenerationRun]:
    """Load multiple runs, optionally filtering by prompt ID or prefix.

    Returns a dict keyed by prompt_id. If multiple runs match the same
    prompt_id, the latest (by directory name timestamp) wins.
    """
    runs_dir = Path(runs_dir)
    result: dict[str, Run | GenerationRun] = {}

    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or not (d / "metadata.json").exists():
            continue
        if prefix and not d.name.startswith(prefix):
            continue

        run = load_run(d, load_logits=load_logits, load_generation=load_generation, device=device)

        if prompt_ids is not None and run.prompt_id not in prompt_ids:
            continue

        result[run.prompt_id] = run

    return result


def save_run(
    capture: CaptureResult | GenerationCaptureResult,
    prompt_id: str,
    prompt_text: str,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Path:
    """Save a capture result to disk in the standard run format.

    If ``capture`` is a GenerationCaptureResult, also saves
    gen_activations.pt (and gen_logits.pt if available).
    """
    timestamp = datetime.now(timezone.utc)
    if run_id is None:
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        run_id = f"{prompt_id}_{ts_str}"

    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(capture.activations, run_dir / "activations.pt")
    torch.save(capture.logits, run_dir / "logits.pt")

    metadata: dict[str, Any] = {
        "metadata": {
            "run_id": run_id,
            "timestamp": timestamp.isoformat(),
            **capture.metadata,
        },
        "prompt": {
            "id": prompt_id,
            "text": prompt_text,
            "token_ids": capture.token_ids,
            "token_strings": capture.token_strings,
        },
        "output": {
            "generated_text": capture.generated_text,
            "new_tokens": capture.new_tokens,
        },
    }

    if isinstance(capture, GenerationCaptureResult):
        torch.save(capture.gen_activations, run_dir / "gen_activations.pt")
        if capture.gen_logits is not None:
            torch.save(capture.gen_logits, run_dir / "gen_logits.pt")
        metadata["output"]["gen_token_ids"] = capture.gen_token_ids
        metadata["output"]["gen_token_strings"] = capture.gen_token_strings

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return run_dir
