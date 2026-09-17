# Human label audit adjudication

**Audit date:** 2026-09-12  
**Source:** completed probe-blind `sdq_label_audit.xlsx` workbook  
**Sampling:** 25 rows per cached task family, stratified by the original parser verdict and preferring distinct semantic tasks

## Raw first pass

- 150/150 rows annotated
- `human_correct`: 74 yes, 64 no, 12 ambiguous
- output validity: 30 clean, 112 degenerate, 8 prompt echo
- original parser agreement on definite human verdicts: 135/138

Because the sample was deliberately stratified by parser verdict, 135/138 is a
descriptive audit count, not an estimate of the parser's population error rate.

## Frozen decision rule

- `yes`: the first answer-bearing span answers the requested task and matches the reference answer.
- `no`: the first answer-bearing span is wrong, or the generation never answers the requested task.
- `ambiguous`: the prompt/reference is invalid or underspecified, or the output makes mutually incompatible answer commitments.
- Leave the extracted answer blank when there is no committed answer. Do not solve a new question that the model merely writes.
- Output validity is separate from correctness. A response may be `yes` and `degenerate`.

## Adjudication log

The raw workbook is preserved. These changes are the explicit adjudication
needed to apply the frozen rule consistently.

| Audit ID | Raw entry | Adjudicated entry | Reason |
| --- | --- | --- | --- |
| audit_006 | ambiguous; blank | no; blank | Prompt echo gives no answer to the requested task. |
| audit_007 | no; `100` | no; blank | `Calculate 10 times 10` is a new prompt, not a committed answer of 100. |
| audit_008 | ambiguous; `1` | no; blank | The generation writes new arithmetic prompts but never answers the requested addition. |
| audit_011 | ambiguous; `100` | no; blank | The generation writes new arithmetic prompts but never answers the requested multiplication. |
| audit_014 | ambiguous; blank | no; blank | Prompt echo gives no answer to the requested task. |
| audit_020 | ambiguous; blank | no; blank | Prompt echo gives no answer to the requested task. |
| audit_021 | ambiguous; `100` | no; blank | The generation writes new arithmetic prompts but never answers the requested multiplication. |
| audit_024 | ambiguous; `3` | no; blank | The generation writes new arithmetic prompts but never answers the requested addition. |
| audit_090 | ambiguous; `Mia` | yes; `Mia` | The first answer-bearing span is `Mia`, which matches the reference; later drift is degeneration. |
| audit_103 | no; `no` | yes; `no` | `human_correct` records whether the response matches the reference, not whether the response text says yes or no. |

Four rows remain ambiguous: audit_134, audit_147, audit_148, and audit_150.
All use the legacy syllogistic `because` template, which already states the
conclusion instead of asking for a completion. The new analysis excludes all 20
cached rows using that template and repairs the generator for future datasets.

## Result after adjudication

- 146 evaluable human-reviewed rows after excluding the four sampled invalid prompts
- parser agreement: 144/146
- the two disagreements are audit_001 and audit_009, where the original parser selected a later unrelated equation after a correct first equation
- the single-step arithmetic parser now selects the first answer-bearing equation

The retained reviewed sample includes 21 valid syllogistic rows and 25 rows
from each other family. The audit used one human reviewer, with AI assistance
on rubric clarification and proposed adjudications. It does not estimate
inter-rater reliability or rule out parser failures outside the sample.

## Interpretation

The audit supports using first-answer correctness as the probe target. It does
not support treating the generations as generally clean: 112/150 were marked
degenerate. First-answer correctness and subsequent generation quality are
different outcomes. The historical intervention trials did not consistently
separate parseable wrong answers from degeneration; their saved error labels
therefore support a narrower interpretation than a measure of reasoning errors.
