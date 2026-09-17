"""Causal intervention on the commitment subspace (Phase 4).

Scientific question:
    Is the commitment subspace causally load-bearing? If we perturb a
    correct prompt's hidden state along the commitment direction (toward
    "incorrect"), does the model's answer flip? And does the flip rate
    exceed a random-direction control at matched perturbation magnitude?

Pass condition (from plan):
    Directed perturbation causes >= 2x the answer-flip rate of
    matched-norm random perturbation.

Implementation strategy:
    We register a forward hook on the target transformer layer that fires
    once during the prefill pass. The hook adds a perturbation vector to
    the hidden state at the last prompt-token position. Generation then
    proceeds from the perturbed state, producing a new answer which we
    compare against the baseline (unperturbed) answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Standardization transform ────────────────────────────────────────────────

def direction_to_raw_space(
    direction_std: Tensor,
    scaler_sd: Tensor,
) -> Tensor:
    """Convert a probe direction from standardized space to raw hidden-state space.

    The probe was trained on z = (h - mu) / sd, so the learned weight w operates
    in z-space. The gradient of the probe score with respect to the raw hidden
    state h is:

        d(score)/d(h_i) = w_i / sd_i

    This function computes that gradient direction and normalizes to a unit vector.

    Args:
        direction_std: [hidden_dim] weight vector (or unit direction) in z-space.
        scaler_sd: [hidden_dim] or [1, hidden_dim] per-feature standard deviations.

    Returns:
        [hidden_dim] unit vector in raw h-space.
    """
    sd = scaler_sd.squeeze()  # [hidden_dim]
    w_raw = direction_std / sd.clamp_min(1e-8)
    return w_raw / w_raw.norm().clamp_min(1e-8)


# ── Hook for single-shot perturbation during prefill ─────────────────────────

class PrefillPerturbationHook:
    """Hook that perturbs a specific position's hidden state once during prefill.

    Designed to be registered on a transformer layer via register_forward_hook.
    Fires exactly once (the prefill pass), then becomes a no-op for all
    subsequent generation steps.

    The hook modifies the hidden state at `position` (typically the last
    prompt token) by adding `direction * magnitude`.
    """

    def __init__(
        self,
        direction: Tensor,
        magnitude: float,
        position: int = -1,
        all_prefill_positions: bool = False,
    ):
        self.direction = direction          # [hidden_dim], unit vector
        self.magnitude = magnitude
        self.position = position
        self.all_prefill_positions = all_prefill_positions
        self.fired = False

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Any,
    ) -> Any:
        if self.fired:
            return outputs

        self.fired = True

        # Transformer layer output format varies by model.
        # Gemma/Llama: (hidden_states, self_attn_weights, present_kv_cache)
        # We modify hidden_states in place.
        if isinstance(outputs, tuple):
            hidden = outputs[0]  # [batch, seq_len, hidden_dim]
        else:
            hidden = outputs     # Some models return tensor directly

        perturbation = (self.direction.to(hidden.device, hidden.dtype)
                        * self.magnitude)
        if self.all_prefill_positions:
            hidden += perturbation
        else:
            hidden[:, self.position, :] += perturbation

        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden

    def reset(self):
        """Allow the hook to fire again for another forward pass."""
        self.fired = False


# ── Hook for persistent perturbation (every forward pass) ────────────────────

class PersistentPerturbationHook:
    """Hook that perturbs the hidden state on every forward pass.

    Unlike PrefillPerturbationHook (fires once), this fires on every call,
    ensuring the perturbation is reapplied at each generation step. This
    defeats the KV cache washout problem where a single prefill perturbation
    is diluted by unperturbed cached states in subsequent steps.

    For generation steps (seq_len=1), always perturbs position -1 (the only
    token). For prefill (seq_len > 1), perturbs the specified position — or
    EVERY prompt position when `all_prefill_positions=True`, so the KV cache
    at downstream layers is built entirely from perturbed states. With the
    default (last position only), every other prompt position's KV entries
    are clean and generated tokens attend to them — the perturbation competes
    against an unperturbed cache of the whole prompt context. This surface
    difference is the "KV-cache bypass" hypothesis for the Phase 4/4b/5
    intervention nulls.
    """

    def __init__(
        self,
        direction: Tensor,
        magnitude: float,
        position: int = -1,
        max_firings: int = -1,
        all_prefill_positions: bool = False,
    ):
        self.direction = direction          # [hidden_dim], unit vector
        self.magnitude = magnitude
        self.position = position
        self.max_firings = max_firings      # -1 = unlimited
        self.all_prefill_positions = all_prefill_positions
        self.fire_count = 0

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Any,
    ) -> Any:
        if self.max_firings >= 0 and self.fire_count >= self.max_firings:
            return outputs

        self.fire_count += 1

        if isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        # For generation steps (seq_len=1), always perturb the only position
        seq_len = hidden.shape[1]

        perturbation = (self.direction.to(hidden.device, hidden.dtype)
                        * self.magnitude)
        if seq_len > 1 and self.all_prefill_positions:
            hidden += perturbation
        else:
            pos = -1 if seq_len == 1 else self.position
            hidden[:, pos, :] += perturbation

        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden

    def reset(self):
        """Reset the firing counter."""
        self.fire_count = 0


# ── Perturbation vector generators ───────────────────────────────────────────

def make_directed_perturbation(
    direction: Tensor,
    magnitude: float,
) -> Tensor:
    """Return direction * magnitude (the commitment direction scaled)."""
    d = direction / direction.norm().clamp_min(1e-8)
    return d * magnitude


def make_random_perturbation(
    dim: int,
    magnitude: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return a random unit vector scaled to the given magnitude."""
    v = torch.randn(dim, generator=generator)
    v = v / v.norm().clamp_min(1e-8)
    return v * magnitude


# ── Subspace perturbation tools ────────────────────────────────────────────

def compute_pca_basis_raw(
    h0_vectors: Tensor,
    scaler_mu: Tensor,
    scaler_sd: Tensor,
    k: int = 96,
) -> tuple[Tensor, Tensor]:
    """Compute top-k PCA basis of h_0 and convert to raw hidden-state space.

    Standardizes to z-space (matching Phase 3), runs PCA, then maps the
    eigenvectors back to raw h-space via v_raw[i] = v_z[i] * sd[i] and
    re-orthonormalizes with QR.

    Returns:
        (basis, eigenvalues) where basis is [D, k] orthonormal in raw
        h-space, eigenvalues is [k] in descending order.
    """
    mu = scaler_mu.squeeze()
    sd = scaler_sd.squeeze().clamp_min(1e-6)

    z = (h0_vectors.float() - mu) / sd
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(0)[:k]
    V_z = eigvecs.flip(1)[:, :k]  # [D, k] in z-space

    V_raw = V_z * sd.unsqueeze(1)  # [D, k] — each row scaled by sd

    Q, _ = torch.linalg.qr(V_raw)
    return Q, eigvals


def make_subspace_perturbation(
    basis: Tensor,
    magnitude: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Random perturbation within the subspace spanned by basis columns.

    Generates random coefficients and maps into the full D-space via basis,
    then normalizes and scales.
    """
    k = basis.shape[1]
    coeffs = torch.randn(k, generator=generator)
    v = basis @ coeffs
    return (v / v.norm().clamp_min(1e-8)) * magnitude


def make_complement_perturbation(
    basis: Tensor,
    dim: int,
    magnitude: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Random perturbation orthogonal to the subspace.

    Projects a full-space random vector onto the orthogonal complement
    of the subspace.
    """
    n = torch.randn(dim, generator=generator)
    proj = basis @ (basis.T @ n)
    v = n - proj
    return (v / v.norm().clamp_min(1e-8)) * magnitude


def make_random_subspace(
    dim: int,
    k: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate a random k-dimensional orthonormal subspace of R^dim.

    Used as a concentration control: if random-subspace perturbations cause
    the same flip rate as PCA-subspace perturbations, the effect is purely
    due to dimensional concentration, not the specific subspace.
    """
    M = torch.randn(dim, k, generator=generator)
    Q, _ = torch.linalg.qr(M)
    return Q  # [dim, k] orthonormal


# ── Generation with perturbation ─────────────────────────────────────────────

@torch.no_grad()
def generate_with_perturbation(
    model: nn.Module,
    tokenizer: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    layer_module: nn.Module,
    direction: Tensor,
    magnitude: float,
    position: int = -1,
    max_new_tokens: int = 64,
    do_sample: bool = False,
) -> tuple[str, list[int]]:
    """Run generation with a one-shot perturbation at the target layer.

    Args:
        model: The causal LM (e.g. GemmaForCausalLM).
        tokenizer: Corresponding tokenizer.
        input_ids: [1, seq_len] prompt token IDs.
        attention_mask: [1, seq_len] attention mask.
        layer_module: The transformer layer to hook (e.g. model.model.layers[-1]).
        direction: [hidden_dim] perturbation direction (unit vector).
        magnitude: Scalar magnitude of the perturbation.
        position: Token position to perturb (-1 = last prompt token).
        max_new_tokens: Max tokens to generate.
        do_sample: Whether to sample (False = greedy, matching capture).

    Returns:
        (generated_text, generated_token_ids) — the new tokens only.
    """
    hook = PrefillPerturbationHook(direction, magnitude, position)
    handle = layer_module.register_forward_hook(hook)

    try:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
        )
    finally:
        handle.remove()

    prompt_len = input_ids.shape[1]
    full_ids = outputs.sequences[0].tolist()
    new_ids = full_ids[prompt_len:]
    new_text = tokenizer.decode(new_ids, skip_special_tokens=False)

    return new_text, new_ids


@torch.no_grad()
def generate_baseline(
    model: nn.Module,
    tokenizer: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    max_new_tokens: int = 64,
    do_sample: bool = False,
) -> tuple[str, list[int]]:
    """Run unperturbed generation (baseline)."""
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_dict_in_generate=True,
    )
    prompt_len = input_ids.shape[1]
    full_ids = outputs.sequences[0].tolist()
    new_ids = full_ids[prompt_len:]
    new_text = tokenizer.decode(new_ids, skip_special_tokens=False)
    return new_text, new_ids


@torch.no_grad()
def generate_with_persistent_perturbation(
    model: nn.Module,
    tokenizer: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    layer_module: nn.Module,
    direction: Tensor,
    magnitude: float,
    position: int = -1,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    all_prefill_positions: bool = False,
) -> tuple[str, list[int], int]:
    """Run generation with a persistent perturbation at the target layer.

    Unlike generate_with_perturbation (fires once during prefill), this
    applies the perturbation on every forward pass, defeating KV cache
    washout of the *generated* tokens. The prompt's KV cache stays clean
    unless `all_prefill_positions=True` (see PersistentPerturbationHook).

    Returns:
        (generated_text, generated_token_ids, fire_count)
    """
    hook = PersistentPerturbationHook(direction, magnitude, position,
                                      all_prefill_positions=all_prefill_positions)
    handle = layer_module.register_forward_hook(hook)

    try:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
        )
    finally:
        handle.remove()

    prompt_len = input_ids.shape[1]
    full_ids = outputs.sequences[0].tolist()
    new_ids = full_ids[prompt_len:]
    new_text = tokenizer.decode(new_ids, skip_special_tokens=False)

    return new_text, new_ids, hook.fire_count


# ── Logit-shift diagnostic ──────────────────────────────────────────────────

@dataclass
class LogitShiftResult:
    """Result of measuring logit shift from a single perturbation."""
    kl_divergence: float          # KL(perturbed || baseline)
    js_divergence: float          # Jensen-Shannon
    top1_logit_change: float      # perturbed top1 logit - baseline top1 logit
    top1_prob_change: float       # perturbed top1 prob - baseline top1 prob
    top1_rank_change: int         # rank of baseline top1 in perturbed distribution


@torch.no_grad()
def measure_logit_shift(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    layer_module: nn.Module,
    direction: Tensor,
    magnitude: float,
    position: int = -1,
    all_prefill_positions: bool = False,
) -> LogitShiftResult:
    """Measure the effect of a perturbation on first-token logits.

    Runs two forward passes (baseline and perturbed) without generation,
    comparing the logit distributions at the last prompt position.
    """
    # Baseline forward pass
    baseline_out = model(input_ids=input_ids, attention_mask=attention_mask)
    baseline_logits = baseline_out.logits[0, -1, :]  # [vocab]

    # Perturbed forward pass
    hook = PrefillPerturbationHook(direction, magnitude, position,
                                   all_prefill_positions=all_prefill_positions)
    handle = layer_module.register_forward_hook(hook)
    try:
        perturbed_out = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    perturbed_logits = perturbed_out.logits[0, -1, :]  # [vocab]

    # Compute distributions
    baseline_probs = F.softmax(baseline_logits.float(), dim=0)
    perturbed_probs = F.softmax(perturbed_logits.float(), dim=0)

    # KL divergence: KL(perturbed || baseline)
    kl = F.kl_div(
        baseline_probs.log().unsqueeze(0),
        perturbed_probs.unsqueeze(0),
        reduction="batchmean",
    ).item()

    # Jensen-Shannon divergence
    m = 0.5 * (baseline_probs + perturbed_probs)
    jsd = 0.5 * (
        F.kl_div(m.log().unsqueeze(0), baseline_probs.unsqueeze(0),
                  reduction="batchmean").item()
        + F.kl_div(m.log().unsqueeze(0), perturbed_probs.unsqueeze(0),
                    reduction="batchmean").item()
    )

    # Top-1 changes
    baseline_top1 = baseline_logits.argmax()
    baseline_top1_logit = baseline_logits[baseline_top1].item()
    baseline_top1_prob = baseline_probs[baseline_top1].item()

    perturbed_top1_logit = perturbed_logits[baseline_top1].item()
    perturbed_top1_prob = perturbed_probs[baseline_top1].item()

    # Rank of baseline top-1 in perturbed distribution
    sorted_indices = perturbed_logits.argsort(descending=True)
    rank = (sorted_indices == baseline_top1).nonzero(as_tuple=True)[0].item()

    return LogitShiftResult(
        kl_divergence=kl,
        js_divergence=jsd,
        top1_logit_change=perturbed_top1_logit - baseline_top1_logit,
        top1_prob_change=perturbed_top1_prob - baseline_top1_prob,
        top1_rank_change=rank,
    )


# ── Trial result ─────────────────────────────────────────────────────────────

@dataclass
class InterventionTrial:
    """Result of a single perturbation trial on one prompt."""
    example_id: str
    task_family: str
    gold_answer: str
    magnitude: float
    perturbation_type: str              # "directed" or "random"
    baseline_text: str
    baseline_answer: str
    baseline_correct: bool
    perturbed_text: str
    perturbed_answer: str
    perturbed_correct: bool
    answer_flipped: bool                # baseline answer != perturbed answer
    flipped_to_incorrect: bool          # was correct, now incorrect


@dataclass
class MagnitudeResult:
    """Aggregated flip statistics at a single perturbation magnitude."""
    magnitude: float
    directed_flip_rate: float
    directed_flip_to_incorrect_rate: float
    random_flip_rate: float
    random_flip_to_incorrect_rate: float
    n_directed: int
    n_random: int
    flip_ratio: float                    # directed / max(random, epsilon)
    passes_threshold: bool               # flip_ratio >= 2.0


@dataclass
class InterventionReport:
    """Full Phase 4 intervention report."""
    magnitude_results: list[MagnitudeResult]
    per_trial: list[dict]                # serializable trial summaries
    n_prompts: int
    n_random_per_prompt: int
    magnitudes: list[float]
    direction_norm_stats: dict           # mean/std of h_0 projected onto direction
    phase4_pass: bool                    # any magnitude passes the threshold
    layer_idx: int = -1                  # which layer was hooked
    hook_mode: str = "prefill"           # "prefill" or "persistent"


# ── Statistics ───────────────────────────────────────────────────────────────

def compute_flip_statistics(
    directed_trials: list[InterventionTrial],
    random_trials: list[InterventionTrial],
    magnitude: float,
) -> MagnitudeResult:
    """Compute flip rates for directed vs random perturbations at one magnitude."""
    eps = 1e-6

    d_flips = sum(1 for t in directed_trials if t.answer_flipped)
    d_bad = sum(1 for t in directed_trials if t.flipped_to_incorrect)
    r_flips = sum(1 for t in random_trials if t.answer_flipped)
    r_bad = sum(1 for t in random_trials if t.flipped_to_incorrect)

    n_d = len(directed_trials) or 1
    n_r = len(random_trials) or 1

    d_rate = d_flips / n_d
    d_bad_rate = d_bad / n_d
    r_rate = r_flips / n_r
    r_bad_rate = r_bad / n_r

    ratio = d_rate / max(r_rate, eps)

    return MagnitudeResult(
        magnitude=magnitude,
        directed_flip_rate=d_rate,
        directed_flip_to_incorrect_rate=d_bad_rate,
        random_flip_rate=r_rate,
        random_flip_to_incorrect_rate=r_bad_rate,
        n_directed=len(directed_trials),
        n_random=len(random_trials),
        flip_ratio=ratio,
        passes_threshold=ratio >= 2.0 and d_rate > 0.05,  # require at least 5% directed flips
    )


def calibrate_magnitudes(
    h0_vectors: Tensor,
    direction: Tensor,
    n_magnitudes: int = 5,
) -> list[float]:
    """Choose perturbation magnitudes calibrated to the natural variation
    of h_0 projected onto the commitment direction.

    Returns magnitudes at [0.5, 1.0, 2.0, 3.0, 5.0] * std_along_direction.
    """
    d = direction / direction.norm().clamp_min(1e-8)
    projections = (h0_vectors.float() @ d.float())  # [N]
    std = projections.std().item()

    multipliers = [0.5, 1.0, 2.0, 3.0, 5.0][:n_magnitudes]
    return [std * m for m in multipliers]


# ── Formatting ───────────────────────────────────────────────────────────────

def format_intervention_report(report: InterventionReport) -> str:
    lines = [
        "=" * 70,
        "PHASE 4: CAUSAL INTERVENTION ON COMMITMENT SUBSPACE",
        "=" * 70,
        f"Layer: {report.layer_idx}  Hook mode: {report.hook_mode}",
        f"Prompts tested: {report.n_prompts}",
        f"Random controls per prompt per magnitude: {report.n_random_per_prompt}",
        f"Direction projection stats: mean={report.direction_norm_stats.get('mean', 0):.3f}, "
        f"std={report.direction_norm_stats.get('std', 0):.3f}",
        "",
        f"{'Magnitude':>10s}  {'Dir flip':>9s}  {'Dir->wrong':>10s}  "
        f"{'Rnd flip':>9s}  {'Rnd->wrong':>10s}  {'Ratio':>6s}  {'Pass':>5s}",
        "-" * 70,
    ]
    for mr in report.magnitude_results:
        lines.append(
            f"{mr.magnitude:10.2f}  {mr.directed_flip_rate:9.3f}  "
            f"{mr.directed_flip_to_incorrect_rate:10.3f}  "
            f"{mr.random_flip_rate:9.3f}  {mr.random_flip_to_incorrect_rate:10.3f}  "
            f"{mr.flip_ratio:6.2f}  {'YES' if mr.passes_threshold else 'no':>5s}"
        )
    lines.append("-" * 70)
    lines.append(f"PHASE 4 PASS: {report.phase4_pass}")
    if report.phase4_pass:
        passing = [mr for mr in report.magnitude_results if mr.passes_threshold]
        mags = [mr.magnitude for mr in passing]
        lines.append(f"  Passing magnitudes: {mags}")
        lines.append(f"  Best flip ratio: {max(mr.flip_ratio for mr in passing):.2f}")
    return "\n".join(lines)
