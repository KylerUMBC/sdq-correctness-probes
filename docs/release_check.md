# Public snapshot checks

This repository is a code-and-results release, not a paper submission. It does
not claim peer review or independent replication by another researcher.

## Release scope

- Current analysis code, benchmark inputs, saved results, figures, and writeup.
- Historical experiment code and results, labelled as exploratory or superseded.
- A fresh Git history. The original private repository is not made public.
- A GitHub no-reply commit address instead of the author's personal email.

Excluded: local credentials and `.env` files, personal planning notes, local
editor/assistant state, virtual environments, the model cache, the activation
archive, and large local checkpoints. Two small historical probe-direction
artifacts are retained for provenance; these are not language-model weights.

## Automated verification — 2026-09-17

- The clean snapshot passes 378 tests; four integration tests skip because
  captured activations are not distributed. With local run data, all 382 pass.
  Figure-layout and frozen-result consistency checks are included.
- Both current figures were rebuilt from the saved JSON results and visually
  inspected. Estimates and confidence intervals occupy a separate numeric column.
- Frozen analysis and intervention-input hashes match the pre-cleanup manifest.
- The active intervention analysis reads the relocated inputs and writes new
  results separately. The reduced-iteration smoke check does not replace the
  saved full-iteration analysis.
- Current documentation links were checked against the release files.
- Gitleaks 8.30.1 scanned the proposed release files and the original 71-commit
  history, with no detected secrets. The scanner binary's SHA-256 was checked
  against the official release digest. The actual credential value in the local
  `.env` was also checked against current files and 510 historical Git blobs;
  no match was found. Secret values and scan workspaces are not published.

Automated scans reduce risk but do not prove that every possible secret is
absent. Passing software tests likewise does not establish a scientific claim.

## Known limits

Probe refitting requires an activation archive that is not included or publicly
hosted. The complete human annotation workbook is local; the public files
contain the original sampling packet and adjudication summary. Saved-result
figures and intervention statistics can be reproduced without the model.

Scientific limitations are described in the writeup and research audit. They
include one small base model, synthetic tasks, noisy continuations, and the
absence of a group-held-out causal test with a task-relevant positive control.
No stronger conclusion is implied by making the repository public.
