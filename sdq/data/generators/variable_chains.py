"""Variable arithmetic chain generators (Family 7).

Unlike single/multi-step arithmetic where the model may have memorized
"2 + 3 = 5", these problems require tracking intermediate variables:

  "Let x = 3 + 4. Let y = x * 2. What is y + 1?"

The model must:
  1. Compute x = 7
  2. Substitute x into y = 7 * 2 = 14
  3. Compute y + 1 = 15

This produces genuine multi-step reasoning trajectories because each
step depends on the previous result — there is no shortcut.

Surface form variations test that SDQ can quotient out the phrasing
while preserving the shared computational graph.
"""

from __future__ import annotations

import random
from typing import Generator

from sdq.data.generators.schema import BenchmarkExample, EventSpan

# Variable names to use (avoid x,y,z which models see constantly)
_VAR_NAMES = ["a", "b", "c", "p", "q", "r", "m", "n"]

# Chain definitions: list of (op, operand) pairs applied sequentially
# Each chain starts from a seed value, then applies operations
_CHAIN_SEEDS: list[dict] = [
    # 2-step chains
    {"seed": 3, "steps": [("+", 4), ("*", 2)], "desc": "add_then_mul"},
    {"seed": 5, "steps": [("+", 3), ("-", 2)], "desc": "add_then_sub"},
    {"seed": 2, "steps": [("*", 3), ("+", 4)], "desc": "mul_then_add"},
    {"seed": 7, "steps": [("-", 3), ("*", 2)], "desc": "sub_then_mul"},
    {"seed": 4, "steps": [("+", 6), ("-", 3)], "desc": "add_then_sub_2"},
    {"seed": 8, "steps": [("-", 5), ("+", 7)], "desc": "sub_then_add"},
    {"seed": 6, "steps": [("*", 2), ("-", 3)], "desc": "mul_then_sub"},
    {"seed": 9, "steps": [("+", 1), ("*", 3)], "desc": "add_then_mul_2"},
    # 3-step chains
    {"seed": 2, "steps": [("+", 3), ("*", 4), ("-", 5)], "desc": "3step_a"},
    {"seed": 5, "steps": [("*", 2), ("+", 3), ("*", 2)], "desc": "3step_b"},
    {"seed": 1, "steps": [("+", 8), ("-", 3), ("*", 2)], "desc": "3step_c"},
    {"seed": 4, "steps": [("*", 3), ("-", 2), ("+", 7)], "desc": "3step_d"},
    {"seed": 3, "steps": [("+", 7), ("*", 2), ("-", 4)], "desc": "3step_e"},
    {"seed": 6, "steps": [("-", 2), ("*", 3), ("+", 1)], "desc": "3step_f"},
    # 4-step chains (genuinely hard)
    {"seed": 2, "steps": [("+", 3), ("*", 2), ("-", 1), ("*", 3)], "desc": "4step_a"},
    {"seed": 1, "steps": [("+", 4), ("*", 3), ("-", 5), ("+", 2)], "desc": "4step_b"},
    {"seed": 3, "steps": [("*", 2), ("+", 4), ("*", 2), ("-", 3)], "desc": "4step_c"},
    {"seed": 5, "steps": [("+", 2), ("-", 1), ("*", 4), ("+", 3)], "desc": "4step_d"},
]

_OP_WORD = {"+": "plus", "*": "times", "-": "minus"}
_OP_VERB = {"+": "add", "*": "multiply by", "-": "subtract"}


def _compute_chain(seed: int, steps: list[tuple[str, int]]) -> list[int]:
    """Compute intermediate values for a chain. Returns [seed, v1, v2, ...]."""
    values = [seed]
    v = seed
    for op, operand in steps:
        if op == "+":
            v = v + operand
        elif op == "-":
            v = v - operand
        elif op == "*":
            v = v * operand
        values.append(v)
    return values


def _let_template(
    seed: int, steps: list[tuple[str, int]], var_names: list[str],
) -> tuple[str, list[EventSpan]]:
    """Template A: 'Let x = 3 + 4. Let y = x * 2. What is y?'"""
    parts = []
    spans = []
    pos = 0

    # First variable assignment
    v0 = var_names[0]
    first_op, first_val = steps[0]
    stmt = f"Let {v0} = {seed} {first_op} {first_val}."
    spans.append(EventSpan("setup", pos, pos + len(stmt)))
    parts.append(stmt)
    pos += len(stmt) + 1  # +1 for space

    # Subsequent assignments
    for i, (op, val) in enumerate(steps[1:], 1):
        prev = var_names[i - 1]
        cur = var_names[i]
        stmt = f"Let {cur} = {prev} {op} {val}."
        spans.append(EventSpan("transform", pos, pos + len(stmt)))
        parts.append(stmt)
        pos += len(stmt) + 1

    # Question
    last_var = var_names[len(steps) - 1]
    q = f"What is {last_var}?"
    spans.append(EventSpan("conclusion", pos, pos + len(q)))
    parts.append(q)

    return " ".join(parts), spans


def _define_template(
    seed: int, steps: list[tuple[str, int]], var_names: list[str],
) -> tuple[str, list[EventSpan]]:
    """Template B: 'Define a as 3 plus 4. Define b as a times 2. Compute b.'"""
    parts = []
    spans = []
    pos = 0

    v0 = var_names[0]
    first_op, first_val = steps[0]
    stmt = f"Define {v0} as {seed} {_OP_WORD[first_op]} {first_val}."
    spans.append(EventSpan("setup", pos, pos + len(stmt)))
    parts.append(stmt)
    pos += len(stmt) + 1

    for i, (op, val) in enumerate(steps[1:], 1):
        prev = var_names[i - 1]
        cur = var_names[i]
        stmt = f"Define {cur} as {prev} {_OP_WORD[op]} {val}."
        spans.append(EventSpan("transform", pos, pos + len(stmt)))
        parts.append(stmt)
        pos += len(stmt) + 1

    last_var = var_names[len(steps) - 1]
    q = f"Compute {last_var}."
    spans.append(EventSpan("conclusion", pos, pos + len(q)))
    parts.append(q)

    return " ".join(parts), spans


def _set_template(
    seed: int, steps: list[tuple[str, int]], var_names: list[str],
) -> tuple[str, list[EventSpan]]:
    """Template C: 'Set a = 3 + 4, then b = a * 2. Find b.'"""
    parts = []
    pos = 0

    v0 = var_names[0]
    first_op, first_val = steps[0]
    chain = [f"{v0} = {seed} {first_op} {first_val}"]

    for i, (op, val) in enumerate(steps[1:], 1):
        prev = var_names[i - 1]
        cur = var_names[i]
        chain.append(f"{cur} = {prev} {op} {val}")

    body = "Set " + ", then ".join(chain) + "."
    spans = [EventSpan("setup", 0, len(body))]
    pos = len(body) + 1

    last_var = var_names[len(steps) - 1]
    q = f"Find {last_var}."
    spans.append(EventSpan("conclusion", pos, pos + len(q)))

    return body + " " + q, spans


def _if_template(
    seed: int, steps: list[tuple[str, int]], var_names: list[str],
) -> tuple[str, list[EventSpan]]:
    """Template D: 'If a equals 3 plus 4, and b equals a times 2, what is b?'"""
    clauses = []

    v0 = var_names[0]
    first_op, first_val = steps[0]
    clauses.append(f"{v0} equals {seed} {_OP_WORD[first_op]} {first_val}")

    for i, (op, val) in enumerate(steps[1:], 1):
        prev = var_names[i - 1]
        cur = var_names[i]
        clauses.append(f"{cur} equals {prev} {_OP_WORD[op]} {val}")

    body = "If " + ", and ".join(clauses)
    last_var = var_names[len(steps) - 1]
    text = body + f", what is {last_var}?"
    spans = [
        EventSpan("setup", 0, len(body)),
        EventSpan("conclusion", len(body), len(text)),
    ]
    return text, spans


_TEMPLATES = [
    ("let_assign", _let_template),
    ("define_worded", _define_template),
    ("set_chain", _set_template),
    ("if_conditional", _if_template),
]


def generate_variable_chains() -> Generator[BenchmarkExample, None, None]:
    """Generate Family 7 (variable arithmetic chain) examples."""
    counter = 0
    rng = random.Random(42)

    for chain_def in _CHAIN_SEEDS:
        seed = chain_def["seed"]
        steps = chain_def["steps"]
        desc = chain_def["desc"]
        n_steps = len(steps)

        values = _compute_chain(seed, steps)
        answer = values[-1]
        sem_id = f"varchain_{desc}"
        reasoning_id = f"varchain_{n_steps}step"
        same_reason = f"sr_varchain_{desc}"

        # Pick variable names for this chain
        var_names = _VAR_NAMES[:n_steps]

        for surf_id, template_fn in _TEMPLATES:
            text, spans = template_fn(seed, steps, var_names)
            yield BenchmarkExample(
                example_id=f"varchain_{counter:04d}",
                task_family="variable_chain",
                semantic_task_id=sem_id,
                reasoning_graph_id=reasoning_id,
                surface_template_id=surf_id,
                reasoning_variant_id="canonical",
                answer_id=str(answer),
                prompt_text=text,
                event_spans=tuple(spans),
                same_reasoning_group=same_reason,
            )
            counter += 1

        # Also generate with shuffled variable names (same semantics, different surface)
        alt_vars = list(var_names)
        rng.shuffle(alt_vars)
        if alt_vars != var_names:
            text, spans = _let_template(seed, steps, alt_vars)
            yield BenchmarkExample(
                example_id=f"varchain_{counter:04d}",
                task_family="variable_chain",
                semantic_task_id=sem_id,
                reasoning_graph_id=reasoning_id,
                surface_template_id="let_assign_altvar",
                reasoning_variant_id="canonical",
                answer_id=str(answer),
                prompt_text=text,
                event_spans=tuple(spans),
                same_reasoning_group=same_reason,
            )
            counter += 1
