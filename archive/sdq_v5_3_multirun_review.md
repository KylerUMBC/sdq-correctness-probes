# SDQ v5.3 Multi-Run Review

## Scope

This document reviews `sdq_v5_3_results_colab3.json` through `sdq_v5_3_results_colab9.json` and tries to unify the behavior across runs instead of treating each colab result as an isolated anomaly.

The short version is:

- SDQ's semantic representation is already strong and is barely changing across these runs.
- The unstable part is the learned dynamics layer, especially under contraction-focused fine-tuning.
- The current system is learning "stay associated with the same task family" much more easily than "return to the original trajectory after a transverse perturbation."
- That is why we keep seeing `cross_tube_persistence_rate = 1.0` with `transverse_recovery_rate = 0.0`.

## Executive Diagnosis

Across `colab3` to `colab9`, the same pattern keeps repeating:

1. Static semantic metrics are nearly frozen.
2. Dynamics metrics move a lot from run to run.
3. Intervention recovery stays broken, except for a tiny blip in `colab3`.
4. Phase 3 can improve some contraction-adjacent numbers, but often by damaging rollout fidelity.

This suggests the current training setup is not yet learning a true attractor or basin structure. It is learning a vector field that often preserves semantic identity while failing to create strong transverse pull back to the tube.

## Cross-Run Table

Rounded summary of the main moving metrics:

| Run | Phase setup | Dyn error | Vel cosine sep | Dyn tighten | Recovery rate | Mean recovery dist | Gate TF acc | Gate entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `colab3` | 200 + 400 | 23.28 | 1.39 | 0.768 | 0.033 | 2.14 | 0.861 | 0.460 |
| `colab4` | 200 + 400 | 24.24 | 1.32 | 0.779 | 0.000 | 2.31 | 0.858 | 0.478 |
| `colab5` | 200 + 400 | 25.29 | 1.29 | 0.780 | 0.000 | 2.07 | 0.832 | 0.509 |
| `colab6` | 200 + 400 | 18.97 | 1.92 | 0.859 | 0.000 | 4.42 | 0.877 | 0.495 |
| `colab7` | 200 + 400 + 200 | 19.08 | 2.50 | 0.852 | 0.000 | 4.81 | 0.863 | 0.459 |
| `colab8` | 200 + 400 + 200 | 25.95 | 1.93 | 0.767 | 0.000 | 1.99 | 0.866 | 0.481 |
| `colab9` | 200 + 400 + 200 | 37.40 | 1.96 | 0.755 | 0.000 | 1.01 | 0.872 | 0.489 |

Important stable facts:

- `cross_tube_persistence_rate` is `1.0` in every run reviewed here.
- `transverse_recovery_rate` is non-zero only in `colab3`, and even there it is just `0.033`.
- The main representation metrics are effectively unchanged across runs.

## What Is Actually Stable

The strongest result in this entire set is that the semantic representation appears robust:

- retrieval stays very high
- latent separation stays high
- probe metrics stay strong
- task-family decoding from the learned state remains strong

That means SDQ is not failing at "semantic grouping." It is failing at the stronger claim: learning a geometry whose dynamics produce reliable recovery toward a reasoning tube after off-manifold perturbation.

This distinction matters because it explains why the results feel odd. The model is not random or broken. It is solving the easier problem consistently and the harder problem inconsistently.

## Unified Interpretation

### 1. SDQ currently has semantic manifolds, but not reliable basins

The current runs support:

- prompts with the same semantics are grouped well
- regimes encode meaningful task-family structure
- the learned field can preserve semantic identity over rollout

The current runs do not support:

- strong attractor behavior
- robust transverse correction
- a clean basin of attraction around the intended trajectory

This is the clearest unifying statement across the runs.

### 2. Persistence is not recovery

`cross_tube_persistence_rate = 1.0` sounds impressive, but paired with `transverse_recovery_rate = 0.0`, it means something narrower:

- perturbations usually do not jump into a different task tube
- but they also do not return to the original tube strongly enough within the evaluation horizon

So the vector field is semantically sticky, not geometrically restorative.

That is a very different capability from the one SDQ ultimately wants for reasoning correction or steering.

### 3. The system is learning tangent compatibility more easily than transverse attraction

This is the deepest pattern in the data.

The model can learn:

- along-trajectory continuation
- regime-consistent motion
- semantic persistence under rollout

But it does not reliably learn:

- contraction in the transverse directions that matter for recovery
- basin-shaping behavior over long horizons

In other words, the field knows how to move forward on the manifold better than it knows how to pull nearby off-manifold states back onto it.

### 4. Phase 3 exposes an objective conflict rather than fixing it

`colab7` is the cleanest Phase 3 result in terms of preserving dynamics quality:

- best `dynamics_vel_cos_sep`
- low `dynamics_mean_error`

But recovery is still zero.

`colab8` and especially `colab9` push recovery distance down, yet rollout error gets much worse:

- `colab9` reaches the best `mean_recovery_distance` at `1.01`
- but also the worst `dynamics_mean_error` at `37.40`

That is the signature of an optimization trade rather than a real solution. Phase 3 can make perturbed trajectories move closer in some average sense while simultaneously harming the fidelity of the learned rollout field.

### 5. Regime learning is useful, but partly collapsed

The regime switcher is not random, but it is not being used richly either. Across later runs, regime utilization is highly concentrated:

- regime `0`: about `0.37` to `0.38`
- regime `2`: about `0.42`
- regimes `1` and `3`: about `0.017`

So although six regimes exist, the effective system behaves much closer to a two-dominant-regime model with a few weak backup modes.

That matters because if transverse recovery depends on regime-specific corrective behavior, underused regimes will not learn stable correction fields.

### 6. The hardest errors are concentrated in specific regimes

In `colab9`, per-regime dynamics MSE is especially high for:

- regime `1`: `66.64`
- regime `5`: `52.64`

while the dominant regimes are lower, though still degraded.

This points to an important possibility: the model is not failing uniformly. It may be preserving common semantic flows while badly underfitting rare or boundary-case transition modes.

That would also help explain why global averages can look reasonable while interventions still fail.

## Why `colab9` Feels So Strange

`colab9` is not nonsense. It is a very informative failure mode.

It combines:

- best mean recovery distance in this set
- zero recovery rate
- perfect cross-tube persistence
- strong gate task-family accuracy
- worst dynamics mean error

The most coherent interpretation is:

- the Phase 3 objective is making perturbed states drift closer to the original tube on average
- but not enough to satisfy the recovery criterion
- and it is doing so by warping the field in ways that hurt rollout accuracy

So `colab9` is not evidence that contraction is useless. It is evidence that the current contraction formulation can partially shape local geometry without yielding a correct global attractor.

That is exactly the kind of "odd result" we should expect when a training signal is directionally relevant but not aligned with the full evaluation target.

## Likely Root Cause Stack

The behavior across runs is best explained by several layers at once:

### Primary cause: loss-eval mismatch

Training is rewarding a proxy for recovery, not the full thing the intervention metric measures.

This allows:

- better average closeness
- unchanged recovery threshold success
- preserved semantic identity
- degraded rollout fidelity

all at the same time.

### Primary cause: one-step or short-horizon learning is easier than basin formation

The field can learn local motion consistency without learning a real restoring geometry over 50 steps.

This is why rollout can look semantically reasonable while intervention recovery remains absent.

### Primary cause: contraction and fidelity are competing objectives

The Phase 3 runs strongly suggest that when contraction gets enough influence to matter, it often pulls against dynamics fidelity instead of complementing it.

### Secondary cause: effective regime collapse

Because only a subset of regimes are used heavily, the system may not have enough expressive or statistical support to learn distinct corrective behaviors across failure modes.

### Secondary cause: noisy contraction signal

The multi-step contraction estimate is stochastic and likely high-variance. That makes it easier to optimize a noisy surrogate than to steadily sculpt a reliable basin.

### Secondary cause: possible logging or run-configuration inconsistency

`colab8` shows a suspicious Phase 3 epoch-1 mismatch where the raw logged terms match `colab7` but the total differs in a way consistent with `ms_contract` not being included in the total. That does not explain the whole trend, but it is a warning against over-reading total-loss comparisons across runs.

## What The Runs Say About SDQ As A Research Direction

The results are better than they look if interpreted correctly.

They suggest:

- SDQ already captures semantic structure well
- regime-aware dynamics are learning something real
- the project is now bottlenecked by dynamical objectives, not by representation learning

That is a meaningful transition point. The project is no longer asking "can SDQ find semantic structure at all?" It is asking "can SDQ learn causal geometry with real attractors and recovery basins?"

That is a much more interesting problem.

## More Novel Interpretations And Directions

If the goal is to think more outside the box, the results point toward a shift in framing. The current setup treats recovery as something that should emerge from a learned rollout field. But these runs suggest that semantic grouping and corrective geometry may need to be modeled more explicitly.

Possible directions:

### 1. Learn tube coordinates directly

Instead of asking the field to implicitly discover "along-tube" and "off-tube" structure, explicitly parameterize:

- progress along the reasoning tube
- transverse displacement from the tube
- a restoring force in transverse coordinates

That would turn recovery into a first-class object rather than a side effect of rollout fitting.

### 2. Separate predictor dynamics from corrector dynamics

The same vector field may be doing two incompatible jobs:

- continue the nominal trajectory
- repair perturbed states

A more novel framing is to use:

- one field for nominal forward rollout
- another field or policy for recovery back toward the tube

This is closer to control than pure forecasting, and it may fit the end goal better.

### 3. Model basins as denoising rather than contraction

Instead of only penalizing rollout ratios, train the system to denoise perturbed latent states back toward a valid tube neighborhood.

That reframes the task from "contract distances under rollout" to "recover a valid reasoning state from corruption," which may align better with steering and hallucination correction.

### 4. Discover attractors explicitly

The project notes already point in this direction, and these results make it even more important. Rather than infer basin structure indirectly from intervention metrics, estimate:

- fixed points or terminal sets
- their local stability
- which trajectories belong to which basin

That would make it possible to ask whether the model has learned true semantic attractors or only smooth local continuation fields.

### 5. Treat regime transitions as events, not just soft mixtures

If reasoning truly moves through phases, then a continuously blended vector field may be too weak a model. A more event-based or hybrid-system view may be needed:

- continuous flow within a regime
- discrete transition rules between regimes

That could better capture why the current model preserves identity but struggles to restore perturbed states.

## Bottom Line

The multi-run story is coherent:

- SDQ's representation layer is working.
- The regime-aware dynamics are partially working.
- The current contraction machinery is not producing real transverse recovery.
- Phase 3 can move some proxy metrics in the right direction, but often by sacrificing rollout fidelity.

So the central unresolved issue is not "why are these results random?" It is:

How do we move from semantic persistence to true basin-shaped recovery dynamics?

That is the unifying question behind `colab3` through `colab9`, and it is probably the right axis for the next serious leap in SDQ.

## Concrete Takeaway

If we keep thinking in terms of small weight adjustments, we will probably keep seeing the same family of results.

If we instead treat this as a representation-vs-control mismatch, the path forward becomes clearer:

- preserve the strong semantic manifold learning
- redesign the dynamical objective around explicit recovery or basin structure
- evaluate attractors and correction directly rather than hoping they emerge from rollout fitting alone

That is the most defensible unified reading of the current evidence.
