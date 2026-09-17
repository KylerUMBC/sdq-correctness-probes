# Correctness probes and activation interventions in Gemma 2 2B

An independent research project by Kyler Gelissen. This repository contains
code, saved results, and a technical writeup; it is not a peer-reviewed paper.

This repository studies a narrow question: **if a linear probe can predict that
an answer will be wrong before generation begins, does the probe direction also
provide a reliable way to change the answer?**

On a controlled six-family reasoning benchmark, a probe on the final prompt-token
residual state predicts first-answer correctness on held-out semantic problems.
At layer 13, mean within-family AUROC is **0.853** (semantic-group bootstrap 95%
CI **0.808–0.894**), compared with **0.703** for a hashed prompt-text baseline,
**0.680** for next-token confidence, and **0.654** for hand-built surface
features. Probe-derived intervention directions, however, did not show a
reliable advantage over matched random directions after correcting for the
exploratory search, and the selected component failed a larger confirmation.

These results support a predictive claim, not a claim that the model “knows it
is wrong.” They also do not prove that correctness information is non-causal;
we did not detect a reliable advantage for the tested additive directions.

## Start here

- [`paper/preprint.md`](paper/preprint.md): detailed technical writeup
- [`docs/research_audit.md`](docs/research_audit.md): forensic reconstruction and methodological audit
- [`docs/reproducibility.md`](docs/reproducibility.md): exact inputs, commands, and known gaps
- [`docs/related_work.md`](docs/related_work.md): checked reading list with notes on relevance
- [`results/README.md`](results/README.md): result provenance and file hashes
- [`results/label_audit_adjudication.md`](results/label_audit_adjudication.md): human label-audit decisions
- [`archive/README.md`](archive/README.md): historical experiments and relocated files

## Frozen headline result

| Signal | Pooled AUROC | Mean within-family AUROC | 95% CI for within-family AUROC |
| --- | ---: | ---: | ---: |
| Residual stream, layer 13 | 0.914 | **0.853** | 0.808–0.894 |
| Residual stream, layer 0 | 0.909 | 0.821 | 0.774–0.864 |
| Residual stream, layer 25 | 0.893 | 0.740 | 0.678–0.815 |
| Hashed prompt text | 0.769 | 0.703 | 0.647–0.763 |
| Next-token confidence | 0.778 | 0.680 | 0.617–0.732 |
| Surface features | 0.834 | 0.654 | 0.583–0.730 |

Population: 830 cached generations, 190 semantic problems, six task
families, 456 correct and 374 incorrect. The positive label used by the code is
`incorrect`. All paraphrases of one semantic problem stay in the same fold.

## Quick verification

Create an environment and install the package:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,paper]"
```

Build the saved-result figures and run tests (no activation archive or GPU):

```powershell
python paper/build_artifacts.py
pytest
```

For statistical reanalysis, run `python run_sdq_reanalyze_interventions.py`.
For probe refitting, run `python run_sdq_grouped_probe.py`; this additionally
requires the local `data/runs/` activation archive, which has no public URL yet.
Both runners write to `outputs/reproduction/`, leaving frozen results untouched.
Full commands and provenance limits are in the reproducibility guide.

## What is current and what is historical

The current analysis entry points are:

- `run_sdq_grouped_probe.py`
- `run_sdq_reanalyze_interventions.py`
- `prepare_label_audit.py`
- `paper/build_artifacts.py`

```text
paper/        Technical writeup and generated figures/tables
results/      Frozen analysis outputs, intervention trials, and audit records
docs/         Methods audit, reproduction instructions, and related work
sdq/          Library code (including retained historical modules)
tests/        Automated checks
data/prompts/ Benchmark inputs and exclusion manifest
archive/      Superseded experiments, writeups, and figures
```

The original SDQ trajectory models, early-warning system, random-split probes,
and broad intervention sweeps are preserved under `archive/`. They are
exploratory or superseded. Their claims are assessed in the research audit;
they are not additional evidence for semantic manifolds or error awareness.

## Current status

The predictive
analysis uses semantic-group holdout; the intervention section is a reanalysis
of exploratory trials, not a new held-out causal experiment. The activation
archive still needs a public host for independent probe refitting. The human
annotation workbook is local; its adjudication and counts are documented.

This public repository is a fresh snapshot. The original private Git history,
personal planning notes, credentials, model cache, and large local checkpoints
are not included. Historical experiment code and results remain in `archive/`
with their limitations documented.

AI assistance is described in the technical writeup. No external peer review
is claimed. See [`docs/release_check.md`](docs/release_check.md) for the final
automated checks and the scope of this release.
