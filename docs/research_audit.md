# SDQ research audit

**Status:** first forensic pass, 2026-09-10  
**Audited code:** Git commit `775a845` plus the explicitly listed untracked artifacts  
**Model:** `google/gemma-2-2b` (base model, not instruction-tuned), snapshot `c5ebcd40d208330abc697524c919956e692655cf`, greedy decoding  
**Purpose:** reconstruct experiments, evaluate evidence, and distinguish documentation limitations from methodological limitations

## Bottom line

The evidence supports the following limited conclusion:

> Prompt-final residual states predict whether Gemma 2 2B will answer a small synthetic reasoning problem correctly, even when paraphrases of the same problem are held out together. However, correctness-predictive directions did not show a reliable, specific causal advantage over norm- and subspace-matched controls in the tested interventions.

The positive result survived a semantic-group leakage check, benchmark exclusions, and a probe-blind human outcome audit. After excluding 185 cached prompts whose surface template changed the task or expected answer type without changing the stored label, and correcting the single-step arithmetic parser to use the first answer-bearing equation, a group-disjoint audit on 830 examples obtained mean within-family AUROC 0.821 at layer 0, 0.853 at layer 13, and 0.740 at layer 25. A hashed character-ngram prompt baseline reached 0.703, next-token confidence reached 0.680, and 13 hand-built surface features reached 0.654. These estimates are limited to the older six-family archive. The author was the human reviewer; this was not an independent external audit.

The selected exploratory component did not reproduce in the larger confirmation run, and the prompt-paired reanalysis finds no Holm-corrected effect. No reliable advantage was detected for the tested directions. This does not establish equivalence to random directions or prove that the information is non-causal or impossible to steer.

## Evidence labels used below

- **Keep:** defensible now, with narrow wording.
- **Provisional:** plausible and useful, but a named check is still load-bearing.
- **Historical/exploratory:** documents the research path but should not carry the abstract.
- **Do not claim:** not established by the current evidence.

## Reconstructed data and execution history

The Colab notebook resolves an important provenance ambiguity. Two different run populations were used.

| Population | What exists | Experiments that used it | Important limitation |
| --- | --- | --- | --- |
| Root `data/runs`, 1,015 examples, 190 semantic tasks, six families | Prompt activations and prompt logits; no generation activations in the local archive | Layer × position map; all-prefill intervention; new grouped audit on 830 conservatively retained rows | This is the older archive even when scripts point at `benchmark_v2.json`; 231 v2 examples are silently absent and 185 cached rows use a task/answer format inconsistent with their stored label |
| Nested `data/runs/gemma-2-2b/gemma-2-2b`, 1,246 examples, 244 semantic tasks, eight families | Existed in Colab; prompt and generation activations were captured there | Updated commitment probe, EWS probes, main mesoscale run, confirmation run, Phase 4 | The cached tensors are not in the current local archive, so these headline analyses cannot presently be rerun from the public checkout alone |

Evidence:

- The local `sdq_data.tar.gz` is 1.6 GB, SHA-256 `0DA07C78C3FCEE9515775EE9A009E9BB950A403A930FC1C84D067A88588403E8`, and contains 1,015 timestamped benchmark runs.
- The supplied Colab notebook is SHA-256 `340EF762BCE5E739212BA0097C975653834373615C4D763A72CE841425DC2C9A`. Its layer-map output explicitly reports 1,015 loaded and 231 missing examples. Its commitment and main mesoscale outputs explicitly report 1,246 examples from the nested directory.
- The current benchmark has 1,246 examples, 244 `semantic_task_id` values, and 3–6 paraphrases per semantic task.

The 1,015-example layer map and all-prefill experiment use a different population from the 1,246-example commitment/mesoscale runs. Historical script names below now resolve under `archive/legacy_experiments/`; original v1–v6 scripts remain directly under `archive/`.

## What each experiment actually did

### 1. Original SDQ trajectory models (v1–v6)

The original project learned encoders, decoders, local transport operators, and later ODE/regime-switching dynamics over hidden-state trajectories. Its objectives encouraged reconstruction, semantic-family grouping, surface-form invariance, transport consistency, and eventually contraction/recovery around learned “tubes.” The archived runs repeatedly achieved high task-family retrieval and decoding, while the learned dynamics were unstable across runs. The internal multirun review records cross-tube persistence of 1.0 alongside transverse recovery of 0.0 in almost every v5.3 run, with contraction improvements sometimes worsening rollout error.

**Assessment:** historical/exploratory. The representation-learning system learned task and template structure, but the results do not establish semantic attractors, stable reasoning manifolds, or corrective basins. Those hypotheses are preserved in the archived writeups, not asserted by the current README.

### 2. Early-warning system (EWS)

`capture_runs_ews.py` greedily generated up to 64 new tokens and cached residual states for each generated token. `run_sdq_ews.py` trained a convolutional encoder, drift features, and a GRU risk model. Every timestep in an ultimately incorrect generation received label 1; every timestep in a correct generation received label 0. Sequence risk was the mean timestep risk.

Saved headline: sequence AUROC 0.908 on 249 test examples.

**Assessment:** historical/exploratory, not a paper result. The main reasons are:

1. Examples were randomly split, so paraphrases of the same semantic problem crossed train/test.
2. Test AUROC was evaluated repeatedly during training and used to choose the “best” checkpoint; the same test set was then reported.
3. The per-batch loss is divided by a cumulative epoch-level `n_samples`, progressively downweighting later batches. The encoder loop has the same bug.
4. All timesteps inherit the final sequence label, so the experiment predicts final outcome but does not locate when a bad reasoning transition occurred.
5. “Average lead time” measures distance from threshold crossing to the end of generation, not time before an observed error event.
6. A “recoverer” is defined as a correct sequence whose score crossed 0.5, then the false-positive metric asks whether it crossed 0.5. The resulting false-positive rate is tautologically 1.
7. Generation was not called with `output_scores=True`; `gen_logits` is therefore absent and the reported 0.5 logit baseline is “missing data,” not a measured baseline.

The EWS results are preserved as historical motivation for the prompt-final probe, not used as a validated temporal error-localization result.

### 3. Prompt-final commitment probe

`run_sdq_commitment.py` loads the last prompt-token residual state, standardizes each dimension on the training set, and trains a 2,304-dimensional logistic probe to predict final incorrectness. It reports pooled and unweighted mean per-family AUROC, leave-one-family-out transfer, and an unsupervised PCA rank sweep. It then refits the direction on all examples for downstream interventions.

Saved 1,246-example result:

- prompt-final pooled AUROC: 0.923
- prompt-final mean within-family AUROC: 0.889
- first-generated-token pooled/within-family AUROC: 0.911/0.850
- family-prior pooled AUROC: 0.782
- cross-family transfer mean AUROC: 0.548, with two families inverted well below chance

**Critical split issue:** with the exact seed-42 split, 240 of 249 test examples (96.4%) had another paraphrase of the same `semantic_task_id` in training. Only two test semantic tasks were completely unseen. Within-family AUROC removes between-family base rates but does not remove this leakage.

**Saved-result issue:** the audited version defaulted `beats_logit` to `True` when generation logits were unavailable. The saved result therefore says the probe passed the predeclared “beat family prior and logit confidence” rule even though the logit baseline was never run. This audit patches future runs to use the equivalent prompt-final distribution in `logits.pt` and to report an inconclusive stopping rule if neither logit source exists; it does not retroactively validate the saved result.

**Updated assessment:** provisional positive result. The leakage-resistant 830-example rerun below makes it unlikely that paraphrase overlap, known invalid templates, or the two observed arithmetic parser errors explain the whole effect. The exact 1,246-example claim still lacks a group-disjoint rerun and should not be the paper's primary result.

### 4. New group-disjoint audit

`run_sdq_grouped_probe.py` expands the declarative exclusion rules in `data/prompts/benchmark_v2_exclusions.json`, holds out complete `semantic_task_id` groups in five folds, fits preprocessing on training folds only, assigns every example one out-of-fold score, and bootstraps whole semantic tasks for uncertainty. It compares three prompt-final layers with prompt-only and next-token-logit baselines.

| Signal | Pooled AUROC | Mean within-family AUROC | Group-bootstrap 95% CI |
| --- | ---: | ---: | ---: |
| Layer 0, final prompt token | 0.909 | 0.821 | 0.774–0.864 |
| Layer 13, final prompt token | 0.914 | 0.853 | 0.808–0.894 |
| Layer 25, final prompt token | 0.893 | 0.740 | 0.678–0.815 |
| 13 surface statistics | 0.834 | 0.654 | 0.583–0.730 |
| Hashed character 3–5 grams | 0.769 | 0.703 | 0.647–0.763 |
| Next-token confidence features | 0.778 | 0.680 | 0.617–0.732 |

Population: 830 conservatively retained examples from the 1,015-run archive, 190 held-out groups, six families, 456 correct and 374 incorrect. The exclusions remove 5 operand-reversing subtraction prompts, 110 incompatible/ambiguous set-inclusion prompts, 50 contradiction prompts whose templates reverse the target or reveal a proposition that is always true, and 20 syllogistic prompts that already state the conclusion instead of requesting an answer. “Layer 0” means the output of the first transformer block, not the raw embedding.

**Assessment:** keep as the paper's primary predictive result. The subsequent human audit found two arithmetic parser errors, repaired them, and triggered exclusion of the malformed syllogistic template; the reported rerun includes both changes. The result shows that the signal generalizes to unseen underlying problems and exceeds two input-only baselines. It does not show metacognitive awareness: the signal could encode prompt difficulty, familiarity, answer entropy, or other predictors of eventual success.

### 5. Layer × prompt-position map

`run_sdq_commitment_map.py` trained the same linear probe at 26 layers and nine prompt positions on the 1,015-example population. The same example-level, class-balanced split was reused for all 234 cells. Every test example had a semantic sibling in training. The map found weak BOS performance, increasing performance as more of the prompt was read, and broadly high performance across depth near the end of the prompt; its best selected cell was layer 13, four tokens from the end, at 0.907 mean within-family AUROC.

**Assessment:** provisional pattern, not defensible cell-level estimates. “Signal increases while reading the prompt and is not confined to late layers” is plausible. The numerical heatmap needs a grouped rerun, uncertainty, and no emphasis on the maximum of 234 searched cells.

### 6. Linear and subspace interventions

The intervention scripts added a direction to the residual stream either at the final prompt position or persistently during generation. Later versions calibrated magnitude using natural variation at each target layer and compared the learned direction with random directions in the learned span, its complement, random subspaces, and full-space random controls.

The original last-layer prefill intervention produced zero flips at all magnitudes. A later layer/mode sweep found a local 6.67× maximum, but it used last-layer magnitude calibration at other layers and selected the maximum without correction. Phase 4b found directed/random-subspace ratios near 1. Its raw result JSON is not present in the repository, so only derived tables and notes remain.

**Assessment:** the zero last-layer result is uninformative because the intervention surface had no demonstrated behavioral leverage. The later matched controls are scientifically useful, but missing raw artifacts prevent a complete audit.

### 7. Mesoscale basis, confirmation, and all-prefill intervention

The mesoscale script fit an eight-dimensional supervised bottleneck probe on prompt-final residual states, orthogonalized/rotated its down-projection, signed components using the full probe, and intervened along individual components, their coordinated sum, random directions within the span, and random full-space directions.

The main 1,246-example exploration used 80 correct prompts at layers 13 and 19. The best single-component cell was 1.95× its within-span control, just below a predeclared 2× threshold; none of 54 cells survived Holm correction. A post-hoc pooled component signal was followed up on 200 selected prompts and failed to replicate. The 1,015-example all-prefill run was also null after correction.

The original Fisher tests treat multiple generations from one prompt as independent. `run_sdq_reanalyze_interventions.py` now pairs each designated direction with random-subspace directions on the same prompt, uses the prompt as the unit of analysis, and applies a within-prompt exchangeability test plus cluster-bootstrap intervals.

Key reanalysis:

- Main exploration, layer 13, component 0, magnitude 3: paired difference +0.088; 95% CI +0.021 to +0.158; raw p=0.010; Holm p=0.263.
- Main exploration, layer 13, component 0, magnitude 2: +0.079; 95% CI +0.017 to +0.146; raw p=0.0068; Holm p=0.185.
- Confirmation, layer 13, component 0, magnitude 3: +0.020; 95% CI −0.007 to +0.051; Holm p=0.496.
- All-prefill, coordinated magnitude 2: +0.056; 95% CI −0.013 to +0.132; Holm p=0.631.

**Assessment:** the failed confirmation and non-significant matched-control comparisons qualify the original claim. The exploratory cells are not established discoveries. The historical `WAVE_SUPERPOSITION` verdict is unsupported: these tests do not distinguish distributed causality, redundancy, off-manifold damage, a bad direction estimate, an incorrect layer/surface, or insufficient power.

The new reanalysis applies Holm correction separately within each run/layer, not globally over the whole research history. Confidence intervals are pointwise. It retains legacy `flip_to_incorrect` labels rather than re-scoring every output with the repaired probe parser. Prompt clustering addresses repeated interventions on one prompt, but does not cluster paraphrases or shared sampled directions. The larger confirmation is not a semantic-group-disjoint holdout; its independence from every exploratory prompt is not established by the current summary.

## Load-bearing methodological issues

### Load-bearing checks and their status

1. **Completed — human-label adjudication.** The probe-blind audit covers 150 rows. After applying a frozen rule and excluding four sampled malformed prompts, the parser agrees on 144/146 evaluable rows. Two disagreements exposed the arithmetic parser's last-equation error. The raw annotations and adjudication log are preserved, and the stratified sample is not treated as a prevalence estimate.
2. **Completed — invalid benchmark rows.** The cached benchmark contains five operand-reversing subtraction prompts, 110 set-inclusion prompts whose surface form expects a different answer type or makes the intended entailment label ambiguous, 50 contradiction prompts whose surface form reverses polarity or makes the queried proposition trivially true, and 20 syllogistic prompts that already state the conclusion. The paper population excludes all 185 through a hashed, declarative manifest, and the generators are patched for future data. Legacy results did not exclude these rows.
3. **Completed — paper population decision.** The unrecoverable 1,246-example cache is not the paper population. The paper uses the 830 conservatively retained examples from the local archive and does not quote the 1,246 random-split AUROC as its primary result.
4. **Unresolved — held-out causal direction.** The legacy probe/basis was fit on all examples, including intervention prompts. This prevents interpreting the series as an independent group-held-out test of direction generalization.
5. **Unresolved — positive control for the intervention surface.** No task-relevant positive control demonstrated leverage at the same hook. A proposed foil-answer logit-gradient control has not been implemented or preregistered.
6. **Unresolved — answer changes versus degeneration.** At least 20% of “flip to incorrect” rows in the main mesoscale result and about 31% in the confirmation/all-prefill results exactly echo the input prompt. Other rows become unrelated or unanswerable. The saved error labels conflate these outcomes with parseable wrong answers.

### Important limitations, not fatal flaws

- One base model and a synthetic benchmark limit generality.
- Greedy decoding tests deterministic prompt sensitivity, not within-prompt stochastic error awareness.
- AUROC is macro-averaged equally over families. In the retained population, syllogistic has only 9 incorrect examples and set-inclusion has only 10 correct examples. Report per-family counts and cluster intervals.
- The 2,304-dimensional probe has more features than training semantic groups. Regularization sensitivity or a difference-of-means probe is a useful robustness check.
- A PCA “rank 96” result is not evidence for a unique 96-dimensional correctness mechanism. PCA selects high-variance directions, and the rank was read from the same evaluation curve.

The intervals bootstrap fitted out-of-fold scores, not the complete training
procedure. They omit model-selection and refitting uncertainty. The tested
layers were motivated by prior exploration. Baseline point-estimate differences
are not a paired significance test or a conditional-information analysis.

## Evidence retained in the current report

1. Correctness is linearly predictable before generation under semantic-group
   holdout on this benchmark.
2. The first-block state is predictive; the three evaluated layer estimates do
   not increase monotonically with depth. The dense leaky map does not establish
   a precise best layer or token position.
3. Residual-state probes have higher point estimates than the particular
   prompt-text and confidence baselines tested. This does not rule out stronger
   input-only predictors of difficulty.
4. The tested probe-derived interventions do not show a statistically reliable
   advantage over their matched controls under the saved analysis.
5. The selected exploratory component did not reproduce in the larger run.

None of these findings establishes awareness, absence of causal information,
wave-superposition encoding, or a general failure of activation steering.

## Documentation issues versus methodological issues

| Issue | Status and consequence |
| --- | --- |
| Stale writeups, scattered scripts, unreadable plots | Archived/indexed or repaired; does not alter experimental evidence |
| Artifact locations and populations unclear | Frozen JSONs and relocation/hash manifests now separate populations |
| Missing archive URL, completed annotation export, and original environment lock | Still limits outside reproduction; documentation alone cannot recover missing artifacts |
| Legacy semantic-task leakage | New grouped evaluation addresses it for the 830-example predictive population only |
| Outcome/benchmark errors | Human audit prompted parser fixes and 185 exclusions; one reviewer does not eliminate all labeling uncertainty |
| Repeated trials treated as independent | Prompt-paired reanalysis addresses within-prompt dependence, with remaining clustering limits noted above |
| Direction fit includes intervention prompts; no positive control | Still limits causal interpretation |

The present report can describe the existing predictive result and qualified
intervention audit without claiming the unresolved checks have been completed.
Personal planning and proposed follow-up designs are kept outside the public
snapshot. No additional experiment or outside review is claimed here.

## Verification performed during this audit

- Exact random-split semantic overlap was reproduced with the saved seed.
- A new group-disjoint probe runner was executed on all locally available cached examples.
- Existing raw intervention trials were reanalyzed at the prompt level.
- Before reorganization: **372 passed, 2 warnings**. After the 2026-09-17
  cleanup and new artifact checks: **382 passed, 3 warnings** (two PyTorch
  nested-tensor warnings and one Matplotlib/Pyparsing deprecation warning).
- Both figures were regenerated and visually inspected. Frozen probe,
  reanalysis, and intervention-input hashes match the pre-cleanup manifest.
- A reduced-iteration CPU smoke run successfully read all relocated
  intervention inputs and wrote a separate output. Its Monte Carlo estimates
  are not replacements for the frozen full-iteration analysis.

Passing tests show that the implemented units behave as expected. They do not, by themselves, validate the research design; the group split and parser audit are separate scientific checks.
