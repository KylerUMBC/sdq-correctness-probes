"""Relational transitivity prompt generator (Family 4).

Generates relational reasoning prompts with crossed factors:
  - Multiple entity sets with transitive relations (older, taller, faster, etc.)
  - Multiple surface templates (statement, question, reorder, given, etc.)
  - Event span annotations for entity binding, relation, and conclusion
"""

from __future__ import annotations

from typing import Generator

from sdq.data.generators.schema import BenchmarkExample, EventSpan

# ---------------------------------------------------------------------------
# Relation types and entity name pools
# ---------------------------------------------------------------------------

_RELATIONS: list[tuple[str, str, str]] = [
    # (relation, comparative, superlative)
    ("older", "older than", "oldest"),
    ("taller", "taller than", "tallest"),
    ("faster", "faster than", "fastest"),
    ("heavier", "heavier than", "heaviest"),
    ("stronger", "stronger than", "strongest"),
]

_NAME_SETS_3: list[tuple[str, str, str]] = [
    ("Alice", "Bob", "Carol"),
    ("Dan", "Eve", "Frank"),
    ("Grace", "Hank", "Ivy"),
    ("Jack", "Kate", "Leo"),
    ("Mia", "Noah", "Olivia"),
    ("Paul", "Quinn", "Rita"),
    ("Sam", "Tara", "Uma"),
    ("Vera", "Will", "Xena"),
]

_NAME_SETS_4: list[tuple[str, str, str, str]] = [
    ("Alice", "Bob", "Carol", "Dan"),
    ("Eve", "Frank", "Grace", "Hank"),
    ("Ivy", "Jack", "Kate", "Leo"),
    ("Mia", "Noah", "Olivia", "Paul"),
]


def _three_chain_templates(
    a: str, b: str, c: str, comp: str, superl: str,
) -> list[tuple[str, str, str, list[EventSpan]]]:
    """Surface variants for 3-entity transitive chain: A > B > C."""
    results: list[tuple[str, str, str, list[EventSpan]]] = []

    # therefore
    p1 = f"{a} is {comp} {b}."
    p2 = f" {b} is {comp} {c}."
    conc = f" Who is the {superl}?"
    text = p1 + p2 + conc
    l1, l2 = len(p1), len(p2)
    spans = [
        EventSpan("entity_bind", 0, l1),
        EventSpan("relation_bind", l1, l1 + l2),
        EventSpan("conclusion", l1 + l2, len(text)),
    ]
    results.append((text, "statement_who", "canonical", spans))

    # reorder
    p1 = f"{b} is {comp} {c}."
    p2 = f" {a} is {comp} {b}."
    conc = f" Who is the {superl}?"
    text = p1 + p2 + conc
    l1, l2 = len(p1), len(p2)
    spans = [
        EventSpan("entity_bind", 0, l1),
        EventSpan("relation_bind", l1, l1 + l2),
        EventSpan("conclusion", l1 + l2, len(text)),
    ]
    results.append((text, "reorder_who", "canonical", spans))

    # given-therefore
    text = f"Given that {a} is {comp} {b}, and {b} is {comp} {c}, therefore the {superl} is"
    g_end = text.index(", and")
    t_start = text.index(", therefore")
    spans = [
        EventSpan("entity_bind", 0, g_end),
        EventSpan("relation_bind", g_end, t_start),
        EventSpan("conclusion", t_start, len(text)),
    ]
    results.append((text, "given_therefore", "canonical", spans))

    # since
    text = f"Since {a} is {comp} {b} and {b} is {comp} {c}, the {superl} is"
    and_idx = text.index(f" and {b}")
    comma_idx = text.index(f", the")
    spans = [
        EventSpan("entity_bind", 0, and_idx),
        EventSpan("relation_bind", and_idx, comma_idx),
        EventSpan("conclusion", comma_idx, len(text)),
    ]
    results.append((text, "since", "canonical", spans))

    # if-then
    text = f"If {a} is {comp} {b}, and {b} is {comp} {c}, then the {superl} is"
    and_idx = text.index(", and")
    then_idx = text.index(", then")
    spans = [
        EventSpan("entity_bind", 0, and_idx),
        EventSpan("relation_bind", and_idx, then_idx),
        EventSpan("conclusion", then_idx, len(text)),
    ]
    results.append((text, "if_then", "canonical", spans))

    # narrative
    text = (
        f"In a group of three people, {a} is {comp} {b}, "
        f"and {b} is {comp} {c}. "
        f"The {superl} person is"
    )
    and_idx = text.index(f", and {b}")
    dot_idx = text.index(f". The")
    spans = [
        EventSpan("setup", 0, text.index(f", {a}") + 2),
        EventSpan("entity_bind", text.index(f"{a} is"), and_idx),
        EventSpan("relation_bind", and_idx, dot_idx),
        EventSpan("conclusion", dot_idx, len(text)),
    ]
    results.append((text, "narrative", "canonical", spans))

    return results


def _four_chain_templates(
    a: str, b: str, c: str, d: str, comp: str, superl: str,
) -> list[tuple[str, str, str, list[EventSpan]]]:
    """Surface variants for 4-entity transitive chain: A > B > C > D."""
    results: list[tuple[str, str, str, list[EventSpan]]] = []

    # statement
    p1 = f"{a} is {comp} {b}."
    p2 = f" {b} is {comp} {c}."
    p3 = f" {c} is {comp} {d}."
    conc = f" Who is the {superl}?"
    text = p1 + p2 + p3 + conc
    l1, l2, l3 = len(p1), len(p2), len(p3)
    spans = [
        EventSpan("entity_bind", 0, l1),
        EventSpan("relation_bind", l1, l1 + l2),
        EventSpan("relation_bind", l1 + l2, l1 + l2 + l3),
        EventSpan("conclusion", l1 + l2 + l3, len(text)),
    ]
    results.append((text, "statement_who", "canonical", spans))

    # reorder (reverse premise order)
    p1 = f"{c} is {comp} {d}."
    p2 = f" {a} is {comp} {b}."
    p3 = f" {b} is {comp} {c}."
    conc = f" Who is the {superl}?"
    text = p1 + p2 + p3 + conc
    l1, l2, l3 = len(p1), len(p2), len(p3)
    spans = [
        EventSpan("entity_bind", 0, l1),
        EventSpan("relation_bind", l1, l1 + l2),
        EventSpan("relation_bind", l1 + l2, l1 + l2 + l3),
        EventSpan("conclusion", l1 + l2 + l3, len(text)),
    ]
    results.append((text, "reorder_who", "canonical", spans))

    # given
    text = (
        f"Given that {a} is {comp} {b}, "
        f"{b} is {comp} {c}, "
        f"and {c} is {comp} {d}, "
        f"the {superl} is"
    )
    parts = text.split(", ")
    p1_end = len(parts[0]) + 2
    p2_end = p1_end + len(parts[1]) + 2
    p3_end = text.index(f"the {superl}")
    spans = [
        EventSpan("entity_bind", 0, p1_end),
        EventSpan("relation_bind", p1_end, p2_end),
        EventSpan("relation_bind", p2_end, p3_end),
        EventSpan("conclusion", p3_end, len(text)),
    ]
    results.append((text, "given", "canonical", spans))

    return results


def generate_relational() -> Generator[BenchmarkExample, None, None]:
    """Generate Family 4 (relational transitivity) examples."""
    counter = 0
    examples: list[BenchmarkExample] = []

    # 3-entity chains: 5 relations × 8 name sets × 6 surface templates
    for rel, comp, superl in _RELATIONS:
        for names in _NAME_SETS_3:
            a, b, c = names
            sem_id = f"rel3_{rel}_{a}_{b}_{c}"
            answer = a  # A is always the superlative
            same_reason = f"sr_rel3_{rel}_{a}_{b}_{c}"

            for text, surf_id, var_id, spans in _three_chain_templates(a, b, c, comp, superl):
                ex = BenchmarkExample(
                    example_id=f"rel_{counter:04d}",
                    task_family="relational",
                    semantic_task_id=sem_id,
                    reasoning_graph_id=f"transitive_3_{rel}",
                    surface_template_id=surf_id,
                    reasoning_variant_id=var_id,
                    answer_id=answer,
                    prompt_text=text,
                    event_spans=tuple(spans),
                    same_reasoning_group=same_reason,
                )
                examples.append(ex)
                counter += 1

    # 4-entity chains: 5 relations × 4 name sets × 3 surface templates
    for rel, comp, superl in _RELATIONS:
        for names in _NAME_SETS_4:
            a, b, c, d = names
            sem_id = f"rel4_{rel}_{a}_{b}_{c}_{d}"
            answer = a
            same_reason = f"sr_rel4_{rel}_{a}_{b}_{c}_{d}"

            for text, surf_id, var_id, spans in _four_chain_templates(a, b, c, d, comp, superl):
                ex = BenchmarkExample(
                    example_id=f"rel_{counter:04d}",
                    task_family="relational",
                    semantic_task_id=sem_id,
                    reasoning_graph_id=f"transitive_4_{rel}",
                    surface_template_id=surf_id,
                    reasoning_variant_id=var_id,
                    answer_id=answer,
                    prompt_text=text,
                    event_spans=tuple(spans),
                    same_reasoning_group=same_reason,
                )
                examples.append(ex)
                counter += 1

    yield from examples
