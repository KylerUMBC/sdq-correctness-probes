"""Syllogistic reasoning prompt generator (Family 3).

Generates syllogisms with crossed factors:
  - Multiple entity/class sets (semantic_task_id)
  - Multiple surface templates (therefore, since, given, because, reorder, if-then)
  - Standard and transitive chain variants
  - Event span annotations
"""

from __future__ import annotations

from typing import Generator

from sdq.data.generators.schema import BenchmarkExample, EventSpan

# ---------------------------------------------------------------------------
# Entity sets: (subject, predicate, instance)
# Each gives "All <subject> are <predicate>. <instance> is a <subject>."
# ---------------------------------------------------------------------------

_ENTITY_SETS: list[tuple[str, str, str]] = [
    ("cats", "animals", "Luna"),
    ("birds", "creatures that fly", "Robin"),
    ("fish", "aquatic animals", "Nemo"),
    ("dogs", "loyal companions", "Rex"),
    ("roses", "flowers", "the red rose"),
    ("eagles", "birds of prey", "the bald eagle"),
    ("rabbits", "fast animals", "Thumper"),
    ("bears", "strong animals", "Baloo"),
    ("dolphins", "intelligent creatures", "Flipper"),
    ("owls", "wise animals", "Hedwig"),
    ("horses", "fast runners", "Spirit"),
    ("wolves", "pack animals", "Grey"),
    ("tigers", "fierce predators", "Shere Khan"),
    ("sparrows", "small birds", "Jack"),
    ("whales", "marine mammals", "Moby"),
    ("ants", "hardworking insects", "the worker ant"),
    ("bees", "pollinators", "the honeybee"),
    ("snakes", "reptiles", "Kaa"),
    ("frogs", "amphibians", "Kermit"),
    ("salmon", "migratory fish", "the sockeye"),
]

# ---------------------------------------------------------------------------
# Transitive chain sets: (A, B, C) where "All A are B. All B are C."
# ---------------------------------------------------------------------------

_TRANSITIVE_SETS: list[tuple[str, str, str, str]] = [
    # (category_A, category_B, category_C, instance)
    ("cats", "mammals", "animals", "Luna"),
    ("roses", "flowers", "plants", "the red rose"),
    ("sparrows", "birds", "animals", "Jack"),
    ("salmon", "fish", "aquatic animals", "the sockeye"),
    ("ants", "insects", "arthropods", "the worker ant"),
    ("wolves", "canines", "mammals", "Grey"),
    ("frogs", "amphibians", "vertebrates", "Kermit"),
    ("eagles", "raptors", "birds", "the bald eagle"),
    ("tigers", "felines", "mammals", "Shere Khan"),
    ("dolphins", "cetaceans", "marine mammals", "Flipper"),
]


def _simple_syllogism_templates(
    subject: str, predicate: str, instance: str,
) -> list[tuple[str, str, list[EventSpan]]]:
    """Generate surface variants for a simple syllogism."""
    results: list[tuple[str, str, list[EventSpan]]] = []

    # therefore
    p1 = f"All {subject} are {predicate}."
    p2 = f" {instance} is a {subject.rstrip('s') if subject.endswith('s') else subject}."
    conc = f" Therefore {instance} is"
    text = p1 + p2 + conc
    spans = [
        EventSpan("premise_load", 0, len(p1)),
        EventSpan("premise_load", len(p1), len(p1) + len(p2)),
        EventSpan("conclusion", len(p1) + len(p2), len(text)),
    ]
    results.append((text, "therefore", spans))

    # since
    text = f"Since all {subject} are {predicate}, and {instance} is a {subject.rstrip('s') if subject.endswith('s') else subject}, {instance} is"
    comma1 = text.index(", and")
    comma2 = text.index(f", {instance} is")
    spans = [
        EventSpan("premise_load", 0, comma1),
        EventSpan("premise_load", comma1, comma2),
        EventSpan("conclusion", comma2, len(text)),
    ]
    results.append((text, "since", spans))

    # given
    sg = subject.rstrip('s') if subject.endswith('s') else subject
    text = f"Given that all {subject} are {predicate}, and given that {instance} is a {sg}, it follows that {instance} is"
    g2_start = text.index(", and given")
    follows_start = text.index(", it follows")
    spans = [
        EventSpan("premise_load", 0, g2_start),
        EventSpan("premise_load", g2_start, follows_start),
        EventSpan("conclusion", follows_start, len(text)),
    ]
    results.append((text, "given", spans))

    # because. Keep this as a completion task; the earlier version supplied
    # the conclusion in full and then scored the continuation as an answer.
    sg = subject.rstrip('s') if subject.endswith('s') else subject
    text = f"Because {instance} is a {sg} and all {subject} are {predicate}, {instance} is"
    conclusion_idx = text.index(f", {instance} is")
    spans = [
        EventSpan("premise_load", 0, conclusion_idx),
        EventSpan("conclusion", conclusion_idx, len(text)),
    ]
    results.append((text, "because", spans))

    # reorder (premise 2 first)
    sg = subject.rstrip('s') if subject.endswith('s') else subject
    p2 = f"{instance} is a {sg}."
    p1 = f" All {subject} are {predicate}."
    conc = f" Therefore {instance} is"
    text = p2 + p1 + conc
    spans = [
        EventSpan("premise_load", 0, len(p2)),
        EventSpan("premise_load", len(p2), len(p2) + len(p1)),
        EventSpan("conclusion", len(p2) + len(p1), len(text)),
    ]
    results.append((text, "reorder", spans))

    # if-then
    sg = subject.rstrip('s') if subject.endswith('s') else subject
    text = f"If all {subject} are {predicate}, and {instance} is a {sg}, then {instance} is"
    and_idx = text.index(", and")
    then_idx = text.index(", then")
    spans = [
        EventSpan("premise_load", 0, and_idx),
        EventSpan("premise_load", and_idx, then_idx),
        EventSpan("conclusion", then_idx, len(text)),
    ]
    results.append((text, "if_then", spans))

    return results


def _transitive_templates(
    cat_a: str, cat_b: str, cat_c: str, instance: str,
) -> list[tuple[str, str, list[EventSpan]]]:
    """Generate surface variants for transitive syllogism (All A are B, All B are C)."""
    results: list[tuple[str, str, list[EventSpan]]] = []
    sg = cat_a.rstrip('s') if cat_a.endswith('s') else cat_a

    # therefore
    p1 = f"All {cat_a} are {cat_b}."
    p2 = f" All {cat_b} are {cat_c}."
    p3 = f" {instance} is a {sg}."
    conc = f" Therefore {instance} is"
    text = p1 + p2 + p3 + conc
    l1, l2, l3 = len(p1), len(p2), len(p3)
    spans = [
        EventSpan("premise_load", 0, l1),
        EventSpan("premise_load", l1, l1 + l2),
        EventSpan("relation_bind", l1 + l2, l1 + l2 + l3),
        EventSpan("conclusion", l1 + l2 + l3, len(text)),
    ]
    results.append((text, "therefore", spans))

    # since
    text = f"Since all {cat_a} are {cat_b}, and all {cat_b} are {cat_c}, and {instance} is a {sg}, {instance} is"
    c1 = text.index(", and all")
    c2 = text.index(f", and {instance}")
    c3 = text.index(f", {instance} is", c2 + 1)
    spans = [
        EventSpan("premise_load", 0, c1),
        EventSpan("premise_load", c1, c2),
        EventSpan("relation_bind", c2, c3),
        EventSpan("conclusion", c3, len(text)),
    ]
    results.append((text, "since", spans))

    # reorder (C -> B -> A)
    p1 = f"All {cat_b} are {cat_c}."
    p2 = f" All {cat_a} are {cat_b}."
    p3 = f" {instance} is a {sg}."
    conc = f" Therefore {instance} is"
    text = p1 + p2 + p3 + conc
    l1, l2, l3 = len(p1), len(p2), len(p3)
    spans = [
        EventSpan("premise_load", 0, l1),
        EventSpan("premise_load", l1, l1 + l2),
        EventSpan("relation_bind", l1 + l2, l1 + l2 + l3),
        EventSpan("conclusion", l1 + l2 + l3, len(text)),
    ]
    results.append((text, "reorder", spans))

    # given
    text = (
        f"Given that all {cat_a} are {cat_b}, "
        f"that all {cat_b} are {cat_c}, "
        f"and that {instance} is a {sg}, "
        f"it follows that {instance} is"
    )
    t1 = text.index("that all " + cat_b)
    t2 = text.index("and that")
    t3 = text.index("it follows")
    spans = [
        EventSpan("premise_load", 0, t1),
        EventSpan("premise_load", t1, t2),
        EventSpan("relation_bind", t2, t3),
        EventSpan("conclusion", t3, len(text)),
    ]
    results.append((text, "given", spans))

    return results


def generate_syllogistic() -> Generator[BenchmarkExample, None, None]:
    """Generate Family 3 (syllogistic reasoning) examples."""
    counter = 0
    examples: list[BenchmarkExample] = []
    answer_groups: dict[str, list[str]] = {}

    # Simple syllogisms
    for subject, predicate, instance in _ENTITY_SETS:
        sg = subject.rstrip('s') if subject.endswith('s') else subject
        sem_id = f"syl_{subject}_{predicate.split()[0]}"
        answer = f"{predicate.split()[0]}"
        same_reason = f"sr_syl_{sem_id}"

        if answer not in answer_groups:
            answer_groups[answer] = []
        answer_groups[answer].append(sem_id)

        for text, surf_id, spans in _simple_syllogism_templates(subject, predicate, instance):
            ex = BenchmarkExample(
                example_id=f"syl_{counter:04d}",
                task_family="syllogistic",
                semantic_task_id=sem_id,
                reasoning_graph_id="simple_universal",
                surface_template_id=surf_id,
                reasoning_variant_id="canonical",
                answer_id=answer,
                prompt_text=text,
                event_spans=tuple(spans),
                same_reasoning_group=same_reason,
            )
            examples.append(ex)
            counter += 1

    # Transitive syllogisms
    for cat_a, cat_b, cat_c, instance in _TRANSITIVE_SETS:
        sem_id = f"syl_trans_{cat_a}_{cat_c}"
        answer = f"{cat_c.split()[0]}"
        same_reason = f"sr_syl_trans_{sem_id}"

        if answer not in answer_groups:
            answer_groups[answer] = []
        answer_groups[answer].append(sem_id)

        for text, surf_id, spans in _transitive_templates(cat_a, cat_b, cat_c, instance):
            ex = BenchmarkExample(
                example_id=f"syl_{counter:04d}",
                task_family="syllogistic",
                semantic_task_id=sem_id,
                reasoning_graph_id="transitive_universal",
                surface_template_id=surf_id,
                reasoning_variant_id="canonical",
                answer_id=answer,
                prompt_text=text,
                event_spans=tuple(spans),
                same_reasoning_group=same_reason,
            )
            examples.append(ex)
            counter += 1

    # Assign same_answer_diff_reasoning groups
    sadr_map: dict[str, str] = {}
    sadr_counter = 0
    for ans, sems in answer_groups.items():
        if len(sems) > 1:
            sadr_map[ans] = f"sadr_syl_{sadr_counter:03d}"
            sadr_counter += 1

    for ex in examples:
        if ex.answer_id in sadr_map:
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
                same_answer_diff_reasoning_group=sadr_map[ex.answer_id],
            )
        else:
            yield ex
