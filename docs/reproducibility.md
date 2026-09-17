# Reproducibility guide

This is the shortest path from a clean checkout to every number used in the
technical writeup. It also records what cannot yet be reproduced from a public
checkout.

## Frozen scope

- Model: `google/gemma-2-2b` (base, not instruction-tuned)
- Hugging Face snapshot: `c5ebcd40d208330abc697524c919956e692655cf`
- Decoding: greedy, 32 new tokens, seed 42, bfloat16 capture
- Paper population: 830 retained cached runs, 190 semantic-task groups, six families
- Outcome: whether the **first answer-bearing span** matches the reference
- Positive class in probe code: incorrect answer
- Primary probe metric: unweighted mean AUROC over task families
- Uncertainty: percentile bootstrap resampling complete semantic-task groups

The cached metadata records the model name and generation settings, but not a
revision. The snapshot above is the only snapshot in the local Hugging Face
cache and is therefore the best available reconstruction, not cryptographic
proof of what was loaded during the original March 2026 capture. Future
captures are pinned to it for both model and tokenizer.

## Environment

Python 3.10 or newer is required. On Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,paper]"
```

The frozen grouped result was produced with PyTorch `2.6.0+cu124`. Exact
versions of every transitive package were not recorded during the original
capture, so this is a source-level reproduction rather than a bit-for-bit
environment recreation. Figure generation uses Matplotlib 3.10.6. Installing a
CUDA build is unnecessary for the saved-result figure build and CPU analyses.

## Required inputs

The tracked benchmark and exclusions are sufficient to inspect prompt
construction:

- `data/prompts/benchmark_v2.json`
- `data/prompts/benchmark_v2_exclusions.json`

The grouped probe additionally needs the ignored `data/runs/` archive. Each run
directory contains `metadata.json`, `activations.pt`, and usually `logits.pt`.
The local `sdq_data.tar.gz` has SHA-256:

```text
0DA07C78C3FCEE9515775EE9A009E9BB950A403A930FC1C84D067A88588403E8
```

There is not yet a public download URL. An outside researcher can inspect
analysis code and saved out-of-fold scores, and reanalyze the intervention
records, but cannot refit the probe from the original activations in a clean
checkout. Recapturing with the model is a separate reproduction that needs
model access and compute; it is not a byte-identical replacement for the archive.

## Reproduce the grouped probe

With `data/runs/` populated:

```powershell
python run_sdq_grouped_probe.py `
  --benchmark data/prompts/benchmark_v2.json `
  --runs-dir data/runs `
  --exclude-file data/prompts/benchmark_v2_exclusions.json `
  --layers 0 13 25 `
  --folds 5 `
  --epochs 400 `
  --lr 0.01 `
  --weight-decay 0.001 `
  --bootstrap 1000 `
  --seed 42 `
  --device cpu `
  --text-hash-dim 2048 `
  --output outputs/reproduction/grouped_probe_results.json
```

Expected high-level counts are 830 examples, 190 groups, 456 correct, and 374
incorrect. Expected layer-13 mean within-family AUROC is approximately 0.853.
Small numeric variation may occur across PyTorch versions.

The script fits one standardized logistic probe per fold. Standardization is
fit on the training fold only. All paraphrases sharing `semantic_task_id` are
assigned to one fold, and every example receives one out-of-fold score.

## Reproduce the intervention statistics

This step uses only included JSON trial records and does not require a model or
GPU:

```powershell
python run_sdq_reanalyze_interventions.py `
  results/interventions/mesoscale_results.json `
  results/interventions/confirmation_results.json `
  results/interventions/prefill_all_results.json `
  --control within_span `
  --permutations 50000 `
  --bootstrap 10000 `
  --seed 42 `
  --output outputs/reproduction/intervention_paired_reanalysis.json
```

The unit of analysis is the prompt. A directed intervention is compared with
the mean of random within-span directions run on the same prompt. Confidence
intervals use a prompt-cluster bootstrap; p-values use a within-prompt
exchangeability test and Holm correction across component/magnitude cells
separately within each run/layer, not globally across runs. The bootstrap
intervals are pointwise. Semantic siblings and shared random directions are
not additional clustering dimensions in this legacy reanalysis.

The raw `flip_to_incorrect` labels and older populations are preserved; the
corrected probe label parser is not retrospectively applied to these trials.
Frozen source-path strings still refer to the former root locations. The
reorganization mapping is in `archive/relocation_manifest.json`; a rerun will
record the new paths, so its JSON need not have the same file hash even when
the numerical results agree. Neither runner overwrites frozen results by default.

## Rebuild paper tables and figures

```powershell
python paper/build_artifacts.py
```

This reads `results/grouped_probe_results.json` and
`results/intervention_paired_reanalysis.json`, and writes Markdown tables plus
SVG/PNG figures under `paper/generated/`. Matplotlib is an optional dependency;
`python paper/build_artifacts.py --tables-only` needs only the standard library.
To render rerun JSONs without altering the paper artifacts, pass
`--results-dir outputs/reproduction --output-dir outputs/reproduction/figures`.

## Run tests

```powershell
pytest
```

The suite checks analysis utilities, labels, grouping, and artifact consistency.
Tests do not establish the validity of scientific assumptions. Optional figure
layout tests are skipped if Matplotlib is not installed.

A clean checkout with the documented development/figure dependencies passes
378 tests and skips four integration tests that require local captured runs.
With that local data present, all 382 tests pass. A `.gitkeep` placeholder in
`data/runs/` is not treated as available run data.

## Human outcome audit

The raw workbook is intentionally preserved outside Git at
`outputs/01a01549-53a9-7883-9a47-836f66cb62f1/sdq_label_audit.xlsx`.
The frozen rubric, raw counts, and every listed adjudication are included in
`results/label_audit_adjudication.md`.

`results/label_audit_blind.csv` is the original blank packet, not the completed
annotations. A public row-level export of the completed workbook is not yet
included, so the full human annotation record is not independently inspectable
from a clean checkout.

The sample was stratified by family and parser verdict. Therefore 144/146
post-adjudication agreement is a check that exposed failure modes, not an
estimate of the parser's population accuracy. The audit led to two parser fixes
and exclusion of a malformed 20-prompt template.

## Provenance hashes

| Artifact | SHA-256 |
| --- | --- |
| Benchmark | `4BCD9844F5DD2EB396C909CAC45A4CF1F58AC347A5BB38B84255DB5F49E77867` |
| Exclusion manifest | `878F8589FC91C7E2FF561C63546419CFDB7F16B4CA13D288A5AB0988046C2A6D` |
| Grouped probe result | `6BE50835D9FBB3555BD598E9D0AFAB2A1D6BB6A04FF2F6960638F2A8EDAAA416` |
| Paired intervention reanalysis | `BCD3492FD75A0A5E2A4AFF4DFE54272D0C754616AC0F6848A223A6DC62F74792` |
| Local activation archive | `0DA07C78C3FCEE9515775EE9A009E9BB950A403A930FC1C84D067A88588403E8` |

The complete artifact inventory, including legacy populations, is in
`results/README.md`.

## Scope of this reproduction

The predictive result can be refit from the local activation archive. The
intervention statistics can be recomputed from saved trials. Neither operation
is a new held-out causal intervention. Missing original environment metadata,
the unavailable 1,246-example Colab tensor cache, and absent Phase 4b raw trials
remain provenance limitations. These gaps are disclosed rather than treated as
completed checks. This repository does not claim independent external review.
