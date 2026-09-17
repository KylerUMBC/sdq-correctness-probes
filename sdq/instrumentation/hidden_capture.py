"""Hook-based hidden-state capture across layers and token positions.

Captures all-layer hidden states for the prompt tokens in a single
forward pass. Returns activations as [num_layers, seq_len, hidden_dim].

Extended to support generation-time capture: per-token hidden states
during autoregressive decoding for early-warning drift detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sdq.instrumentation.model_loader import ModelBundle


@dataclass
class CaptureResult:
    """Result of a single instrumented forward pass."""

    activations: torch.Tensor  # [num_layers, seq_len, hidden_dim]
    logits: torch.Tensor  # [1, seq_len, vocab_size]
    token_ids: list[int]
    token_strings: list[str]
    generated_text: str
    new_tokens: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationCaptureResult:
    """Result of a full generation capture including per-token hidden states.

    Extends CaptureResult with generation-time activations so every
    autoregressive step has its hidden-state snapshot available.

    IMPORTANT alignment semantics (audit 2026-06):
        gen_activations[t] is the residual stream AT generated token t —
        i.e. the forward pass that *reads* token t (already emitted) and
        predicts token t+1. The prefill pass is NOT included, so there are
        len(gen_token_ids) - 1 entries. gen_activations[0] is therefore NOT
        the prompt-final state; the prompt-final ("before generation") state
        is activations[layer, -1, :].
    """

    activations: torch.Tensor        # [num_layers, prompt_len, hidden_dim]
    gen_activations: torch.Tensor    # [num_gen_tokens - 1, num_layers, hidden_dim]
    logits: torch.Tensor             # [1, prompt_len, vocab_size]
    gen_logits: torch.Tensor | None  # [num_gen_tokens, vocab_size] or None
    token_ids: list[int]             # prompt token ids
    token_strings: list[str]         # prompt token strings
    gen_token_ids: list[int]         # generated token ids
    gen_token_strings: list[str]     # generated token strings
    generated_text: str
    new_tokens: str
    metadata: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def capture_hidden_states(
    bundle: ModelBundle,
    text: str,
    max_new_tokens: int | None = None,
) -> CaptureResult:
    """Run text through the model, capturing all-layer hidden states.

    Returns hidden states for the *prompt* tokens only (not generated).
    Shape: [num_layers, prompt_len, hidden_dim].
    """
    inf_cfg = bundle.config.get("inference", {})
    if max_new_tokens is None:
        max_new_tokens = inf_cfg.get("max_new_tokens", 32)

    inputs = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
    prompt_len = inputs["input_ids"].shape[1]

    token_ids = inputs["input_ids"][0].tolist()
    token_strings = [bundle.tokenizer.decode([tid]) for tid in token_ids]

    # Generate with hidden states for the prompt portion
    outputs = bundle.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=inf_cfg.get("do_sample", False),
        temperature=inf_cfg.get("temperature", 1.0),
        top_k=inf_cfg.get("top_k", 1),
        return_dict_in_generate=True,
        output_hidden_states=True,
    )

    # Extract prompt hidden states from the prefill step
    # outputs.hidden_states[0] is a tuple of (num_layers+1,) tensors from prefill
    # Each tensor is [batch, prompt_len, hidden_dim]
    # Index [1:] to skip the embedding layer, keeping only transformer layers
    prefill_hidden = outputs.hidden_states[0]
    layer_states = []
    for layer_idx in range(1, len(prefill_hidden)):
        layer_states.append(prefill_hidden[layer_idx][0].cpu())  # [prompt_len, hidden_dim]

    activations = torch.stack(layer_states)  # [num_layers, prompt_len, hidden_dim]

    # Get logits for prompt tokens via a separate forward pass
    with torch.no_grad():
        fwd = bundle.model(**inputs)
    logits = fwd.logits.cpu()  # [1, prompt_len, vocab_size]

    # Decode generated text
    full_ids = outputs.sequences[0].tolist()
    generated_text = bundle.tokenizer.decode(full_ids, skip_special_tokens=False)
    new_token_ids = full_ids[prompt_len:]
    new_tokens = bundle.tokenizer.decode(new_token_ids, skip_special_tokens=False)

    return CaptureResult(
        activations=activations,
        logits=logits,
        token_ids=token_ids,
        token_strings=token_strings,
        generated_text=generated_text,
        new_tokens=new_tokens,
        metadata={
            "model_name": bundle.model.config._name_or_path,
            "model_revision": bundle.config.get("model", {}).get("revision"),
            "device": str(bundle.device),
            "dtype": str(bundle.dtype),
            "prompt_len": prompt_len,
        },
    )


@torch.no_grad()
def capture_generation_hidden_states(
    bundle: ModelBundle,
    text: str,
    max_new_tokens: int | None = None,
    layers: list[int] | None = None,
) -> GenerationCaptureResult:
    """Run text through the model, capturing hidden states for every generated token.

    In addition to the prefill activations returned by ``capture_hidden_states``,
    this extracts one hidden-state vector per layer per generated token.

    Args:
        bundle: Loaded model bundle.
        text: Input prompt text.
        max_new_tokens: Maximum tokens to generate.
        layers: If provided, only capture these (0-indexed transformer) layers
                instead of all layers. Reduces memory for large models.

    Returns:
        GenerationCaptureResult with both prefill and generation activations.
    """
    inf_cfg = bundle.config.get("inference", {})
    if max_new_tokens is None:
        max_new_tokens = inf_cfg.get("max_new_tokens", 32)

    inputs = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)
    prompt_len = inputs["input_ids"].shape[1]

    token_ids = inputs["input_ids"][0].tolist()
    token_strings = [bundle.tokenizer.decode([tid]) for tid in token_ids]

    outputs = bundle.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=inf_cfg.get("do_sample", False),
        temperature=inf_cfg.get("temperature", 1.0),
        top_k=inf_cfg.get("top_k", 1),
        return_dict_in_generate=True,
        output_hidden_states=True,
    )

    # --- Prefill activations [num_layers, prompt_len, hidden_dim] ---
    prefill_hidden = outputs.hidden_states[0]
    num_transformer_layers = len(prefill_hidden) - 1  # skip embedding layer
    if layers is None:
        layer_indices = list(range(num_transformer_layers))
    else:
        layer_indices = [l for l in layers if 0 <= l < num_transformer_layers]

    prefill_states = []
    for li in layer_indices:
        prefill_states.append(prefill_hidden[li + 1][0].cpu())
    activations = torch.stack(prefill_states)  # [selected_layers, prompt_len, hidden_dim]

    # --- Generation-step activations ---
    # outputs.hidden_states[0] is the prefill pass (prompt positions); entries
    # 1..M-1 are decode steps. Decode step i reads generated token i-1, so
    # gen_activations[t] = state at generated token t (M-1 entries for M
    # generated tokens). The prompt-final state lives in `activations` above.
    gen_layer_states: list[torch.Tensor] = []
    for step_idx in range(1, len(outputs.hidden_states)):
        step_hidden = outputs.hidden_states[step_idx]
        # Each generation step: tuple of (num_layers+1,) tensors [batch, 1, hidden_dim]
        step_layers = torch.stack([
            step_hidden[li + 1][0, 0].cpu() for li in layer_indices
        ])  # [selected_layers, hidden_dim]
        gen_layer_states.append(step_layers)

    if gen_layer_states:
        gen_activations = torch.stack(gen_layer_states)  # [num_gen_tokens, selected_layers, hidden_dim]
    else:
        hidden_dim = activations.shape[-1]
        gen_activations = torch.empty(0, len(layer_indices), hidden_dim)

    # --- Logits for prompt tokens ---
    with torch.no_grad():
        fwd = bundle.model(**inputs)
    logits = fwd.logits.cpu()  # [1, prompt_len, vocab_size]

    # --- Per-step generation logits (from output_scores if available) ---
    gen_logits = None
    if hasattr(outputs, "scores") and outputs.scores:
        # outputs.scores is a tuple of [batch, vocab_size] per step
        gen_logits = torch.stack([s[0].cpu() for s in outputs.scores])  # [num_gen, vocab]

    # --- Decode generated text ---
    full_ids = outputs.sequences[0].tolist()
    generated_text = bundle.tokenizer.decode(full_ids, skip_special_tokens=False)
    new_token_ids = full_ids[prompt_len:]
    new_tokens_text = bundle.tokenizer.decode(new_token_ids, skip_special_tokens=False)
    gen_token_strings = [bundle.tokenizer.decode([tid]) for tid in new_token_ids]

    return GenerationCaptureResult(
        activations=activations,
        gen_activations=gen_activations,
        logits=logits,
        gen_logits=gen_logits,
        token_ids=token_ids,
        token_strings=token_strings,
        gen_token_ids=new_token_ids,
        gen_token_strings=gen_token_strings,
        generated_text=generated_text,
        new_tokens=new_tokens_text,
        metadata={
            "model_name": bundle.model.config._name_or_path,
            "model_revision": bundle.config.get("model", {}).get("revision"),
            "device": str(bundle.device),
            "dtype": str(bundle.dtype),
            "prompt_len": prompt_len,
            "num_gen_tokens": len(new_token_ids),
            "captured_layers": layer_indices,
        },
    )
