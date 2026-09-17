# SDQ Benchmark + Event-Anchored Model Plan

## Goal

Stop relying on tiny, underdetermined pair sets and move to a benchmark that makes SDQ **identifiable**.

The next branch should focus on:

1. **designed data**, not just more runs  
2. **event-anchored reasoning structure**, not only family labels  
3. **clean factorization** of:
   - semantic dynamics
   - surface realization / transport
   - reasoning-path variation

The core idea is:

> If SDQ is a quotient of hidden-state trajectories under surface-form transformations, then the data must expose that product structure clearly enough for the model to recover it.

---

# 1. What is missing right now

The current setup is too small and too entangled.

You are trying to learn all of this at once from very little data:

- semantic equivalence
- surface transport
- time alignment
- composition
- family transfer
- hard-negative robustness

With only a few families and a few runs per family, many different latent geometries can explain the same observations.

That is why the models keep trading off:
- semantics vs composition
- transport simplicity vs retrieval
- family fit vs transfer

## Main missing ingredients

### A. More identifiable data
You need a benchmark with explicit crossed factors:

\[
\text{semantics} \times \text{surface form} \times \text{reasoning variant}
\]

### B. Event anchors
You need at least coarse reasoning-stage labels:
- setup
- transform
- conclusion
- answer

### C. Same-answer / different-reasoning controls
You need examples where:
- final answer is the same
- reasoning path is different

### D. Same-reasoning / many-surface controls
You need many realizations of the same underlying reasoning graph.

---

# 2. New benchmark design

The benchmark should be generated around a **factorized schema**.

## 2.1 Benchmark dimensions

Each example should be labeled by:

### Semantic task ID
What underlying reasoning content is being computed.

Examples:
- arithmetic expression
- syllogistic inference
- relation transitivity
- set inclusion
- contradiction detection

### Reasoning graph ID
The actual reasoning structure.

Examples:
- two-step addition chain
- syllogism with universal premise
- transitive relation chain
- contradiction-from-negation graph

This is more specific than just “task family.”

### Surface template ID
How the problem is phrased.

Examples:
- direct statement
- reordered statement
- connective-based statement (`therefore`, `since`, `given`)
- verbose natural-language form
- compressed logical form

### Reasoning variant ID
Alternative valid path to the same answer.

Examples:
- different decomposition order
- different intermediate variable ordering
- alternate equivalent proof structure

### Answer ID
The final answer.

This is used to construct:
- same-answer / same-reasoning
- same-answer / different-reasoning
- different-answer / same-surface
controls.

---

# 3. Benchmark families to include

You want multiple reasoning families, from simple to moderately structured.

## Family 1 — Arithmetic chains
Examples:
- `5 + 3 = ?`
- `Start with 5, add 3`
- `Given 5 and then 3 more`
- reordered / worded / paraphrased forms

### Why include
- easy to generate
- controlled semantics
- easy same-surface / different-semantics negatives
- easy same-answer / different-expression cases

### Event structure
- operand read
- operator application
- result formation
- answer emission

---

## Family 2 — Multi-step arithmetic / decomposition
Examples:
- `5 + 7 + 3`
- `(5 + 7) + 3`
- `5 + (7 + 3)`
- “start with 5, add 7, then add 3”

### Why include
- gives reasoning-path variation
- same answer can come from different decomposition orders
- richer than single-step arithmetic

### Event structure
- first combine
- intermediate state
- second combine
- final answer

---

## Family 3 — Syllogistic reasoning
Examples:
- `All A are B; all B are C; therefore all A are C`
- reordered or paraphrased forms
- lexical substitutions while preserving graph

### Why include
- close to reasoning, not just arithmetic
- clear graph structure
- strong same-reasoning / many-surface setup

### Event structure
- premise 1 load
- premise 2 load
- relation compose
- conclusion

---

## Family 4 — Relational transitivity
Examples:
- `Alice is older than Bob. Bob is older than Carol. Who is oldest?`
- same graph with different names / wording / clause order

### Why include
- graph reasoning with entities
- clean transitive structure
- easy paraphrase generation

### Event structure
- entity binding
- relation binding
- transitive compose
- query answer

---

## Family 5 — Set / class inclusion
Examples:
- `All cats are mammals. All mammals are animals. Are cats animals?`
- same logic with different lexical content

### Why include
- overlaps with your current cat/bird style prompts
- easy event annotation
- easy hard negatives via relation flipping

### Event structure
- class premise load
- chain compose
- entailment answer

---

## Family 6 — Contradiction / negation
Examples:
- `All A are B. Some A are not B. Is there a contradiction?`
- paraphrased equivalents

### Why include
- different reasoning type
- helps prevent overfitting to purely constructive chains

### Event structure
- premise load
- negation identification
- conflict check
- contradiction output

---

# 4. Data schema

Each generated example should have a structured record.

## Example schema

```json
{
  "example_id": "arith_000123",
  "task_family": "multi_step_arithmetic",
  "semantic_task_id": "sum_5_7_3",
  "reasoning_graph_id": "assoc_left",
  "surface_template_id": "given_then_then",
  "reasoning_variant_id": "left_association",
  "answer_id": "15",
  "prompt_text": "Given 5, then add 7, then add 3. What is the total?",
  "event_spans": [
    {"event": "operand_1", "start_token": 1, "end_token": 2},
    {"event": "operator_1", "start_token": 3, "end_token": 4},
    {"event": "operand_2", "start_token": 5, "end_token": 6},
    {"event": "intermediate_combine", "start_token": 7, "end_token": 8},
    {"event": "operator_2", "start_token": 9, "end_token": 10},
    {"event": "operand_3", "start_token": 11, "end_token": 12},
    {"event": "final_answer", "start_token": 13, "end_token": 15}
  ],
  "same_reasoning_group": "sr_001",
  "same_answer_diff_reasoning_group": "sadr_004",
  "hard_negative_group": "hn_009"
}



Core benchmark axes

You want four main axes.

Axis 1: Semantics

This is the underlying reasoning content.

Examples:

arithmetic operation structure

syllogistic inference rule

relation chaining

conditional reasoning pattern

Axis 2: Surface form

This is wording / syntax / connective / order.

Examples:

base

reorder

since

therefore

given

active/passive

lexical paraphrase

short vs verbose

Axis 3: Reasoning path

This matters a lot.

Two prompts may:

have the same answer

but use different reasoning structure

That must be represented explicitly.

Axis 4: Answer identity

Answer equality should not be treated as reasoning equality.

You need examples where:

answer is same, reasoning differs

reasoning is same, answer differs

both are same

neither is same

Minimum task families to include

Start with a few domains, but make them systematic.

1. Arithmetic reasoning

Examples:

add5 vs add15

two-step arithmetic

operand reorder

verbalized arithmetic

same template, different operands

same answer via different decompositions

Why this helps:

easy to generate at scale

good for hard negatives

exposes same-surface/different-semantics failures

2. Syllogistic / logical reasoning

Examples:

all/some/no structure

premise reorder

connective variation

same inference rule with different entities

different inference rule with same answer label

Why this helps:

closer to explicit reasoning structure

better for event segmentation

likely more semantically meaningful than raw arithmetic alone

3. Relational chaining

Examples:

cat/bird style facts

if A relates to B and B relates to C, conclude X

reorder premises

connective changes

same relation graph, different lexical realization

Why this helps:

bridges symbolic structure and natural language

good for event anchors

directly relevant to current families

4. Same-answer / different-reasoning tasks

Examples:

reach same boolean conclusion from different chains

arithmetic same result from different operations

logic tasks where final truth label matches but inference path differs

Why this helps:

directly prevents answer-collapse

important for proving SDQ is about reasoning, not output

Event anchor design

You do not need perfect fine-grained annotations at first.

Start with coarse event labels.

Recommended event labels

Use 3 to 5 stages.

Version A

setup

transform

conclusion

Version B

premise_load

relation_binding

rule_application

conclusion

answer_commit

Version A is easier to implement first.

Why event anchors matter

They give the model and eval pipeline a stronger notion of:

where semantic transitions happen

what should align across equivalent prompts

how to compare trajectories beyond token position

How to create event labels

For synthetic and semi-synthetic tasks, generate them automatically from prompt templates.

That is much better than trying to infer everything post hoc.

Rough data target

This matters a lot.

The current dataset is too small to identify the SDQ object reliably.

Absolute minimum useful target

Aim for at least:

200 to 400 total runs

This is the minimum point where you can start testing:

multiple semantic families

multiple surface variants

basic holdout splits

Better target

Aim for:

800 to 1,500 total runs

This is a much better range for:

stable train/test behavior

factorized splits

hard negatives

event-aware supervision

transport sharing across families

Strong target

If generation is easy, a very solid benchmark would be:

2,000 to 5,000 total runs

This would let you support:

many semantic families

many transform types

multiple reasoning variants per answer class

stronger held-out combination tests

Recommended first target

A good next milestone is:

about 1,000 runs

That is large enough to be meaningful but still manageable.

Suggested benchmark composition for the first milestone

For about 1,000 runs, something like this is reasonable.

4 domains

20 to 30 semantic families per domain

4 to 6 surface variants per family

2 reasoning variants per family where possible

Example rough count:

arithmetic: 250 runs

logic: 250 runs

relational: 250 runs

same-answer/different-reasoning: 250 runs

You do not need this exact balance, but that is a good starting shape.

Recommended factorization target

Try to structure the data roughly like:

50 to 80 semantic families total

4 to 6 surface forms per family

2 to 3 reasoning variants for a useful subset

3 to 5 event stages per example

That gives the model enough repeated structure to learn:

reusable transports

reusable semantic transitions

answer-vs-reasoning distinctions

Holdout splits you should support

This is one of the biggest missing pieces right now.

Split 1: Hold out semantic families

Train on some semantic families, test on unseen ones.

Question:

does transport reuse across unseen semantics?

Split 2: Hold out surface forms

Train on some transforms, test on unseen transforms.

Question:

does the model generalize beyond specific wording patterns?

Split 3: Hold out reasoning variants

Train on one reasoning path family, test on another with similar answers.

Question:

does the model distinguish reasoning path from output?

Split 4: Hold out combinations

Train on seen semantics and seen transforms separately, but test unseen combinations.

Question:

can the model compose reusable semantic and transport structure?

This is one of the strongest SDQ tests.

Event-anchored model plan

The next model should explicitly separate:

semantic latent state

transport/operator state

event-aware alignment

Semantic latent

Represents:

reasoning state

semantic motion

event transitions

Transport/operator state

Represents:

surface realization

local coordinate transform

connective / syntax / template effects

Event-aware alignment

Uses event anchors first, then local alignment within events.

This is better than pure tokenwise alignment.

Model sketch

Keep this simple at first.

Encoder

Input:

hidden trajectory windows

optional token metadata

event-position embedding

Output:

semantic latent trajectory

transport context features

Semantic head

Learns:

latent state

latent velocity

event-aware trajectory similarity

Transport head

Learns:

transform-type prototype

local residual operator

composition structure

Alignment module

Uses:

event anchors

monotone local alignment within event blocks

Losses to emphasize

The current project likely needs fewer random losses and more structured ones.

Keep

reconstruction loss

transport consistency

hard-negative semantic loss

latent trajectory loss

composition regularizer

Add or strengthen

event alignment loss

same-answer/different-reasoning separation loss

semantic family retrieval loss

split-space regularization between semantic and transport representations

Reduce reliance on

raw within/cross latent averages as the main proof

They can stay, but should not be the main metric.

Most important new benchmark labels

If only a few new labels are added, prioritize these.

Required

semantic_family_id

surface_template_id

transform_type

answer_id

reasoning_graph_id

coarse event_sequence

Very useful

reasoning_variant_id

same_answer_class

same_reasoning_class

These labels will make evaluation much more meaningful.

Evaluation protocol for the new branch

Use a scorecard, not one number.

Semantic metrics

latent separation

retrieval accuracy

retrieval MRR

hard-negative resistance

same-answer/different-reasoning separation

event-aligned trajectory similarity

motion L2 separation

Transport metrics

train transport residual

holdout transport residual

effective rank

composition error

inversion error

transport sharing across unseen semantic families

Benchmark metrics

performance on held-out semantics

performance on held-out surface forms

performance on held-out combinations

performance by domain

performance by event stage