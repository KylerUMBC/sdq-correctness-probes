# Result artifact manifest

This manifest records artifacts examined in the September 2026 audit. Current
analysis outputs live here; trial-level inputs live in `interventions/`.
Superseded results live in `archive/legacy_experiments/`. The 2026-09-17
relocation changed paths, not experimental JSON contents or their hashes.
Original path strings inside saved metadata are retained for provenance; the
mapping is in [`archive/relocation_manifest.json`](../archive/relocation_manifest.json).

The artifacts do **not** all describe the same dataset population.  Do not
compare their sample counts without checking the `Population` column.

| Artifact | Population | Status | SHA-256 |
| --- | --- | --- | --- |
| [`commitment_results_updated.json`](../archive/legacy_experiments/commitment_results_updated.json) | 1,246 examples; eight families; nested Colab cache | Legacy random-split probe; semantic-task leakage; missing logit baseline was incorrectly treated as passed | `78F1E14995B8C27CC26218384EE7DC4B9DCDD2C07DB2619B126F13B3F26F7531` |
| [`ews_results2.json`](../archive/legacy_experiments/ews_results2.json) | 1,246 examples; eight families; nested Colab cache | Historical only; test-set checkpoint selection and training-loss accounting problems | `5422D8CB18DCAA75BCBCB641FA6AA12E33B60758CCCB04C34479AC043EDC0112` |
| [`commitment_map_results.json`](../archive/legacy_experiments/commitment_map_results.json) | 1,015 examples; six families; root archive | Legacy random-split layer/position scan; every test example had a training paraphrase | `FCC5433AB394ADE0079C68F08581E7F2E29541A40A0EB1CCAB289AAB80431B9E` |
| [`mesoscale_results.json`](interventions/mesoscale_results.json) | 80 selected prompts from the 1,246-example cache | Exploratory intervention search; no Holm-corrected cell | `1E1AC6668144AB34882F4CA998802C94B92258F9A3CA5946F04346C93345CCF8` |
| [`confirmation_results.json`](interventions/confirmation_results.json) | 200 selected prompts from the 1,246-example cache; 196 baseline-correct | Larger intervention run; selected component did not replicate | `D58EEE00D8394F2874959C9ED5F457FC424E21EEA97376297078F1F99AF92F1F` |
| [`prefill_all_results.json`](interventions/prefill_all_results.json) | 1,015-example root archive; 78 usable selected prompts | All-prefill follow-up; null after correction | `B03DED4188E1293EC3B2BBB2BDC67BA343965F716299B512CE356590E5C24CF7` |
| [`grouped_probe_results.json`](grouped_probe_results.json) | 830 conservatively retained examples after 185 exclusions; 190 held-out semantic groups; six families | Leakage-resistant audit updated after the human label audit | `6BE50835D9FBB3555BD598E9D0AFAB2A1D6BB6A04FF2F6960638F2A8EDAAA416` |
| [`intervention_paired_reanalysis.json`](intervention_paired_reanalysis.json) | Existing main, confirmation, and all-prefill trial records | Prompt-paired reanalysis; no Holm-corrected result within run/layer | `BCD3492FD75A0A5E2A4AFF4DFE54272D0C754616AC0F6848A223A6DC62F74792` |

## Input and provenance hashes

| Input | SHA-256 | Note |
| --- | --- | --- |
| `data/prompts/benchmark_v2.json` | `4BCD9844F5DD2EB396C909CAC45A4CF1F58AC347A5BB38B84255DB5F49E77867` | Current 1,246-example benchmark |
| `data/prompts/benchmark_v2_exclusions.json` | `878F8589FC91C7E2FF561C63546419CFDB7F16B4CA13D288A5AB0988046C2A6D` | Declarative rules plus five explicit IDs exclude 185 cached prompts with task/label inconsistencies |
| Original audited `configs/model.yaml` | `820AC64D14678AA908CE84C112218F4E534ED15FCC040CCA21904A271B70F40F` | `google/gemma-2-2b`, greedy generation; the later reproducibility patch pins the recovered snapshot |
| Current pinned `configs/model.yaml` | `05E321035E19D5E7CD0A5A66013B398ADCF818C70AA53BF41AC871585B04C191` | Same generation settings with snapshot `c5ebcd40d208330abc697524c919956e692655cf` pinned for future captures |
| `sdq_data.tar.gz` | `0DA07C78C3FCEE9515775EE9A009E9BB950A403A930FC1C84D067A88588403E8` | Locally available 1,015-run archive; ignored by Git |
| `archive/legacy_experiments/commitment_direction.pt` | `AAB5016182CDC846F9EAC7C61B9BE229A22AFAB138D5808DA256CC25E7A66AF7` | Direction fit on the full 1,246-example population; not held out for intervention |
| Supplied `SDQ_ColabRunner.ipynb` | `340EF762BCE5E739212BA0097C975653834373615C4D763A72CE841425DC2C9A` | External provenance source; not copied into the repository |

The audited Git revision is `775a845acdb8207713737385923cc97b25eee4e6`.
The locally cached model/tokenizer snapshot used for future pinned captures is
`c5ebcd40d208330abc697524c919956e692655cf`.
See [`docs/research_audit.md`](../docs/research_audit.md) for interpretation and
[`paper/preprint.md`](../paper/preprint.md) for the current report.

## Human label audit packet

`prepare_label_audit.py` creates a deterministic sample stratified by family and
the automated parser's current verdict.  It prefers distinct semantic tasks.
The main sheet intentionally omits the parser's verdict and every probe score;
the separate key permits unblinding after annotation.

```powershell
python prepare_label_audit.py
```

New packets are written under `outputs/label_audit/`, leaving the original
packet untouched. Current exclusions and parser fixes can change the sampled
rows; this command does not reconstruct the earlier annotation packet exactly.

The schema records `human_correct` (`yes`, `no`, `ambiguous`), a committed
answer in `human_extracted_answer` (blank when absent), and
`human_output_validity` (`clean`, `prompt_echo`, `degenerate`, `uninterpretable`).
The completed workbook is local at
`outputs/01a01549-53a9-7883-9a47-836f66cb62f1/sdq_label_audit.xlsx`.
The included CSVs are the original blank packet and key, not a complete public
export of the human annotations.

The completed first pass contains 150/150 annotations: 74 `yes`, 64 `no`, and
12 `ambiguous`, with 112 responses marked `degenerate`, 8 `prompt_echo`, and 30
`clean`. On the 138 definite verdicts, the parser agreed on 135. The audit then
identified two first-equation parsing errors and four sampled rows from a
malformed syllogistic template. See
[`label_audit_adjudication.md`](label_audit_adjudication.md) for the frozen
decision rule and row-level adjudication, and `docs/research_audit.md` for the
research interpretation.

Generated packet hashes:

- `label_audit_blind.csv`: `9092B2B5522AF709C05093B68863AF41CE01C37416B80472D8777C47A0C798FE`
- `label_audit_key.csv`: `3C4315635DFEB02CC614A212A3DBD01E29320D9F2F4592310B115D2C0982FDD4`
