"""Arithmetic prompt generators (Family 1: single-step, Family 2: multi-step).

Generates prompts with crossed factors:
  - Multiple operand sets (semantic_task_id)
  - Multiple surface templates (phrasing styles)
  - Multiple reasoning variants (decomposition orders for multi-step)
  - Event span annotations per template
"""

from __future__ import annotations

import itertools
from typing import Generator

from sdq.data.generators.schema import BenchmarkExample, EventSpan

# ---------------------------------------------------------------------------
# Family 1: Single-step arithmetic
# ---------------------------------------------------------------------------

_SINGLE_STEP_PROBLEMS: list[tuple[int, str, int, int]] = [
    # (answer, op_symbol, a, b)
    (5, "+", 2, 3),
    (8, "+", 5, 3),
    (11, "+", 4, 7),
    (9, "+", 6, 3),
    (13, "+", 8, 5),
    (17, "+", 9, 8),
    (10, "+", 1, 9),
    (14, "+", 7, 7),
    (12, "+", 3, 9),
    (15, "+", 6, 9),
    (20, "*", 4, 5),
    (18, "*", 3, 6),
    (24, "*", 8, 3),
    (35, "*", 5, 7),
    (42, "*", 6, 7),
    (27, "*", 9, 3),
    (16, "*", 4, 4),
    (36, "*", 6, 6),
    (21, "*", 3, 7),
    (48, "*", 8, 6),
    (7, "-", 10, 3),
    (4, "-", 9, 5),
    (6, "-", 15, 9),
    (8, "-", 20, 12),
    (3, "-", 11, 8),
]

_OP_WORDS = {"+": "add", "*": "multiply", "-": "subtract"}
_OP_PHRASES = {"+": "plus", "*": "times", "-": "minus"}


def _single_step_templates(
    a: int, b: int, op: str, answer: int,
) -> list[tuple[str, str, list[EventSpan]]]:
    """Return (prompt_text, surface_template_id, event_spans) tuples."""
    word = _OP_WORDS[op]
    phrase = _OP_PHRASES[op]
    results: list[tuple[str, str, list[EventSpan]]] = []

    # Template: equation
    text = f"{a} {op} {b} ="
    spans = [
        EventSpan("operand_load", 0, text.index(f" {op}")),
        EventSpan("operator", text.index(f" {op}"), text.index(f" {op}") + len(f" {op}")),
        EventSpan("operand_load", text.index(f"{op} ") + len(f"{op} "), text.index(" =")),
        EventSpan("answer", text.index(" ="), len(text)),
    ]
    results.append((text, "equation", spans))

    # Template: operand reorder for commutative operations; an equivalent
    # "subtract from" phrasing for subtraction.  Swapping subtraction
    # operands changes the answer and is not a surface paraphrase.
    if op == "-":
        text = f"Subtract {b} from {a}."
        op_start = text.index(str(b))
        op_end = text.index(" from")
        second_start = text.index(str(a), op_end)
        spans = [
            EventSpan("operator", 0, op_start),
            EventSpan("operand_load", op_start, op_end),
            EventSpan("operand_load", second_start, text.index(".")),
            EventSpan("answer", text.index("."), len(text)),
        ]
        results.append((text, "subtract_from", spans))
    else:
        text = f"{b} {op} {a} ="
        spans = [
            EventSpan("operand_load", 0, text.index(f" {op}")),
            EventSpan("operator", text.index(f" {op}"), text.index(f" {op}") + len(f" {op}")),
            EventSpan("operand_load", text.index(f"{op} ") + len(f"{op} "), text.index(" =")),
            EventSpan("answer", text.index(" ="), len(text)),
        ]
        results.append((text, "reorder", spans))

    # Template: question
    text = f"What is {a} {phrase} {b}?"
    p_start = text.index(f"{a}")
    results.append((text, "question", [
        EventSpan("setup", 0, p_start),
        EventSpan("operand_load", p_start, text.index("?")),
        EventSpan("answer", text.index("?"), len(text)),
    ]))

    # Template: imperative
    text = f"Calculate {a} {phrase} {b}."
    p_start = text.index(f"{a}")
    results.append((text, "imperative", [
        EventSpan("setup", 0, p_start),
        EventSpan("operand_load", p_start, text.index(".")),
        EventSpan("answer", text.index("."), len(text)),
    ]))

    # Template: worded
    text = f"Start with {a}, {word} {b}."
    results.append((text, "worded", [
        EventSpan("operand_load", 0, text.index(",")),
        EventSpan("transform", text.index(","), text.index(".")),
        EventSpan("answer", text.index("."), len(text)),
    ]))

    # Template: given
    text = f"Given {a} and {b}, compute {a} {phrase} {b}."
    results.append((text, "given", [
        EventSpan("setup", 0, text.index(", compute")),
        EventSpan("transform", text.index(", compute"), text.index(".")),
        EventSpan("answer", text.index("."), len(text)),
    ]))

    return results


def generate_single_step() -> Generator[BenchmarkExample, None, None]:
    """Generate Family 1 (single-step arithmetic) examples."""
    counter = 0
    # Group problems by answer for same_answer_diff_reasoning_group
    answer_groups: dict[int, list[str]] = {}

    examples: list[BenchmarkExample] = []
    for answer, op, a, b in _SINGLE_STEP_PROBLEMS:
        sem_id = f"{_OP_WORDS[op]}_{a}_{b}"
        reasoning_id = f"single_{op}"
        same_reason = f"sr_arith1_{sem_id}"

        if answer not in answer_groups:
            answer_groups[answer] = []
        answer_groups[answer].append(sem_id)

        for text, surf_id, spans in _single_step_templates(a, b, op, answer):
            ex = BenchmarkExample(
                example_id=f"arith1_{counter:04d}",
                task_family="arithmetic",
                semantic_task_id=sem_id,
                reasoning_graph_id=reasoning_id,
                surface_template_id=surf_id,
                reasoning_variant_id="canonical",
                answer_id=str(answer),
                prompt_text=text,
                event_spans=tuple(spans),
                same_reasoning_group=same_reason,
            )
            examples.append(ex)
            counter += 1

    # Assign same_answer_diff_reasoning groups
    sadr_map: dict[int, str] = {}
    sadr_counter = 0
    for ans, sems in answer_groups.items():
        if len(sems) > 1:
            gid = f"sadr_arith1_{sadr_counter:03d}"
            sadr_map[ans] = gid
            sadr_counter += 1

    for ex in examples:
        ans = int(ex.answer_id)
        if ans in sadr_map:
            yield BenchmarkExample(
                example_id=ex.example_id,
                task_family=ex.task_family,
                semantic_task_id=ex.semantic_task_id,
                reasoning_graph_id=ex.reasoning_graph_id,
                surface_template_id=ex.surface_template_id,
                reasoning_variant_id=ex.reasoning_variant_id,
                answer_id=ex.answer_id,
                prompt_text=ex.prompt_text,
                event_spans=ex.event_spans,
                same_reasoning_group=ex.same_reasoning_group,
                same_answer_diff_reasoning_group=sadr_map[ans],
            )
        else:
            yield ex


# ---------------------------------------------------------------------------
# Family 2: Multi-step arithmetic
# ---------------------------------------------------------------------------

_MULTI_STEP_PROBLEMS: list[tuple[tuple[int, ...], str]] = [
    # (operands, operator)
    ((5, 7, 3), "+"),
    ((2, 8, 4), "+"),
    ((9, 1, 6), "+"),
    ((3, 4, 5), "+"),
    ((6, 2, 7), "+"),
    ((8, 3, 9), "+"),
    ((4, 6, 1), "+"),
    ((7, 5, 8), "+"),
    ((1, 9, 2), "+"),
    ((10, 3, 7), "+"),
    ((2, 5, 4, 3), "+"),
    ((1, 6, 8, 2), "+"),
    ((3, 7, 1, 9), "+"),
    ((4, 2, 6, 5), "+"),
    ((8, 1, 3, 7), "+"),
    ((2, 3, 4), "*"),
    ((5, 2, 3), "*"),
    ((3, 3, 3), "*"),
    ((2, 4, 5), "*"),
    ((1, 7, 6), "*"),
]


def _multi_step_templates(
    operands: tuple[int, ...], op: str,
) -> list[tuple[str, str, str, list[EventSpan]]]:
    """Return (prompt_text, surface_template_id, reasoning_variant_id, event_spans) tuples."""
    word = _OP_WORDS[op]
    phrase = _OP_PHRASES[op]
    n = len(operands)
    results: list[tuple[str, str, str, list[EventSpan]]] = []

    # Template: equation, left-association
    expr = f" {op} ".join(str(x) for x in operands) + " ="
    results.append((expr, "equation", "left_assoc", [
        EventSpan("operand_load", 0, expr.index(f" {op}")),
        EventSpan("transform", expr.index(f" {op}"), expr.index(" =")),
        EventSpan("answer", expr.index(" ="), len(expr)),
    ]))

    # Template: equation, right-association (show with parens)
    if n >= 3:
        inner = f" {op} ".join(str(x) for x in operands[1:])
        expr2 = f"{operands[0]} {op} ({inner}) ="
        results.append((expr2, "equation_parens", "right_assoc", [
            EventSpan("operand_load", 0, expr2.index(f" {op}")),
            EventSpan("transform", expr2.index(f" {op}"), expr2.index(" =")),
            EventSpan("answer", expr2.index(" ="), len(expr2)),
        ]))

    # Template: step-by-step worded
    parts = [f"Start with {operands[0]}"]
    for x in operands[1:]:
        parts.append(f"{word} {x}")
    text = ", ".join(parts) + "."
    spans = [EventSpan("operand_load", 0, text.index(","))]
    rest_start = text.index(",")
    spans.append(EventSpan("transform", rest_start, text.index(".")))
    spans.append(EventSpan("answer", text.index("."), len(text)))
    results.append((text, "step_by_step", "left_assoc", spans))

    # Template: question form
    expr_q = f" {phrase} ".join(str(x) for x in operands)
    text = f"What is {expr_q}?"
    results.append((text, "question", "left_assoc", [
        EventSpan("setup", 0, text.index(str(operands[0]))),
        EventSpan("operand_load", text.index(str(operands[0])), text.index("?")),
        EventSpan("answer", text.index("?"), len(text)),
    ]))

    # Template: given-then chain
    chain = ", then ".join(f"{word} {x}" for x in operands[1:])
    text = f"Given {operands[0]}, {chain}. What is the total?"
    dot_idx = text.index(". What")
    results.append((text, "given_then", "left_assoc", [
        EventSpan("setup", 0, text.index(",")),
        EventSpan("transform", text.index(","), dot_idx),
        EventSpan("conclusion", dot_idx, len(text)),
    ]))

    # Template: reverse order (right-to-left)
    rev_ops = list(reversed(operands))
    rev_chain = ", then ".join(f"{word} {x}" for x in rev_ops[1:])
    text = f"Given {rev_ops[0]}, {rev_chain}. What is the result?"
    dot_idx = text.index(". What")
    results.append((text, "reverse_given", "right_assoc", [
        EventSpan("setup", 0, text.index(",")),
        EventSpan("transform", text.index(","), dot_idx),
        EventSpan("conclusion", dot_idx, len(text)),
    ]))

    return results


def generate_multi_step() -> Generator[BenchmarkExample, None, None]:
    """Generate Family 2 (multi-step arithmetic) examples."""
    counter = 0
    answer_groups: dict[int, list[str]] = {}
    examples: list[BenchmarkExample] = []

    for operands, op in _MULTI_STEP_PROBLEMS:
        if op == "+":
            answer = sum(operands)
        else:
            ans = 1
            for x in operands:
                ans *= x
            answer = ans

        sem_id = f"multi_{_OP_WORDS[op]}_{'_'.join(str(x) for x in operands)}"
        reasoning_id = f"multi_{op}_{len(operands)}step"
        same_reason = f"sr_arith2_{sem_id}"

        if answer not in answer_groups:
            answer_groups[answer] = []
        answer_groups[answer].append(sem_id)

        for text, surf_id, var_id, spans in _multi_step_templates(operands, op):
            ex = BenchmarkExample(
                example_id=f"arith2_{counter:04d}",
                task_family="multi_step_arithmetic",
                semantic_task_id=sem_id,
                reasoning_graph_id=reasoning_id,
                surface_template_id=surf_id,
                reasoning_variant_id=var_id,
                answer_id=str(answer),
                prompt_text=text,
                event_spans=tuple(spans),
                same_reasoning_group=same_reason,
            )
            examples.append(ex)
            counter += 1

    # Assign same_answer_diff_reasoning groups
    sadr_map: dict[int, str] = {}
    sadr_counter = 0
    for ans, sems in answer_groups.items():
        if len(sems) > 1:
            gid = f"sadr_arith2_{sadr_counter:03d}"
            sadr_map[ans] = gid
            sadr_counter += 1

    for ex in examples:
        ans = int(ex.answer_id)
        if ans in sadr_map:
            yield BenchmarkExample(
                example_id=ex.example_id,
                task_family=ex.task_family,
                semantic_task_id=ex.semantic_task_id,
                reasoning_graph_id=ex.reasoning_graph_id,
                surface_template_id=ex.surface_template_id,
                reasoning_variant_id=ex.reasoning_variant_id,
                answer_id=ex.answer_id,
                prompt_text=ex.prompt_text,
                event_spans=ex.event_spans,
                same_reasoning_group=ex.same_reasoning_group,
                same_answer_diff_reasoning_group=sadr_map[ans],
            )
        else:
            yield ex
