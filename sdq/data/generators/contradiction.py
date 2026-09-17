"""Contradiction / negation prompt generator (Family 6).

Generates contradiction-detection prompts with crossed factors:
  - Contradictory vs. consistent premise pairs
  - Multiple surface templates (direct, paraphrased, question forms)
  - Event span annotations for premise loading and conflict detection
"""

from __future__ import annotations

from typing import Generator

from sdq.data.generators.schema import BenchmarkExample, EventSpan

# ---------------------------------------------------------------------------
# Contradictory premise pairs: (statement_1, negation_of_1)
# ---------------------------------------------------------------------------

_CONTRADICTIONS: list[tuple[str, str, str]] = [
    # (premise_1, premise_2_contradicts, topic_id)
    ("All cats are mammals", "Some cats are not mammals", "cats_mammals"),
    ("All birds can fly", "Some birds cannot fly", "birds_fly"),
    ("All fish live in water", "Some fish do not live in water", "fish_water"),
    ("All dogs are loyal", "Some dogs are not loyal", "dogs_loyal"),
    ("All roses are red", "Some roses are not red", "roses_red"),
    ("All metals conduct electricity", "Some metals do not conduct electricity", "metals_conduct"),
    ("All squares have four sides", "Some squares do not have four sides", "squares_sides"),
    ("All humans are mortal", "Some humans are not mortal", "humans_mortal"),
    ("All ice is cold", "Some ice is not cold", "ice_cold"),
    ("All fire is hot", "Some fire is not hot", "fire_hot"),
    ("No reptiles are mammals", "Some reptiles are mammals", "reptiles_mammals"),
    ("No insects have bones", "Some insects have bones", "insects_bones"),
    ("No plants can move", "Some plants can move", "plants_move"),
    ("No rocks are alive", "Some rocks are alive", "rocks_alive"),
    ("No liquids are solid", "Some liquids are solid", "liquids_solid"),
]

# Consistent (non-contradictory) pairs for hard negatives
_CONSISTENT: list[tuple[str, str, str]] = [
    ("All cats are mammals", "All mammals are animals", "cats_mammals_chain"),
    ("All birds can fly", "Eagles are birds", "birds_eagles"),
    ("All dogs are loyal", "Rex is a dog", "dogs_rex"),
    ("Some roses are red", "Some roses are white", "roses_colors"),
    ("All metals conduct electricity", "Gold is a metal", "metals_gold"),
    ("All humans are mortal", "Socrates is human", "humans_socrates"),
    ("No reptiles are mammals", "Snakes are reptiles", "reptiles_snakes"),
    ("No insects have bones", "Ants are insects", "insects_ants"),
    ("All squares have four sides", "All rectangles have four sides", "shapes_sides"),
    ("All fire is hot", "The sun is hot", "hot_things"),
]


def _contradiction_templates(
    p1: str, p2: str, is_contradiction: bool,
) -> list[tuple[str, str, list[EventSpan]]]:
    """Generate surface variants for contradiction detection."""
    results: list[tuple[str, str, list[EventSpan]]] = []

    # Template: direct
    text = f"{p1}. {p2}. Is there a contradiction?"
    dot1 = p1.find(".") if "." in p1 else len(p1)
    dot1 = len(p1)
    dot2 = dot1 + 2 + len(p2)
    spans = [
        EventSpan("premise_load", 0, dot1 + 1),
        EventSpan("premise_load", dot1 + 2, dot2 + 1),
        EventSpan("conflict_check", dot2 + 2, len(text)),
    ]
    results.append((text, "direct_question", spans))

    # Template: given-detect
    text = f"Given that {p1.lower()}, and that {p2.lower()}, is there a contradiction?"
    and_idx = text.index(", and that")
    q_idx = text.index(", is there")
    spans = [
        EventSpan("premise_load", 0, and_idx),
        EventSpan("premise_load", and_idx, q_idx),
        EventSpan("conflict_check", q_idx, len(text)),
    ]
    results.append((text, "given_detect", spans))

    # Template: consider-whether
    text = f"Consider: {p1}. Also: {p2}. Do these statements contradict each other?"
    consider_end = text.index(". Also")
    also_end = text.index(". Do")
    spans = [
        EventSpan("premise_load", 0, consider_end + 1),
        EventSpan("premise_load", consider_end + 2, also_end + 1),
        EventSpan("conflict_check", also_end + 2, len(text)),
    ]
    results.append((text, "consider_whether", spans))

    # Template: true-false.  Do not insert the known class label into the
    # proposition: that made the proposition true for both classes while the
    # stored answer still encoded contradiction=yes.
    text = (
        f"{p1}. {p2}. Is it true that these statements contradict each other? "
        "Answer yes or no."
    )
    dot2_end = len(p1) + 2 + len(p2) + 1
    spans = [
        EventSpan("premise_load", 0, len(p1) + 1),
        EventSpan("premise_load", len(p1) + 2, dot2_end),
        EventSpan("conflict_check", dot2_end + 1, len(text)),
    ]
    results.append((text, "true_false", spans))

    # Keep the same polarity as the other templates: yes means contradiction.
    # The earlier "Can both be true?" wording silently reversed every label.
    text = f"Is it impossible for both of these to be true? {p1}. {p2}."
    q_end = text.index("? ") + 2
    p1_end = q_end + len(p1) + 1
    spans = [
        EventSpan("setup", 0, q_end),
        EventSpan("premise_load", q_end, p1_end),
        EventSpan("premise_load", p1_end + 1, len(text)),
    ]
    results.append((text, "can_both", spans))

    return results


def generate_contradiction() -> Generator[BenchmarkExample, None, None]:
    """Generate Family 6 (contradiction/negation) examples."""
    counter = 0
    examples: list[BenchmarkExample] = []

    # True contradictions
    for p1, p2, topic_id in _CONTRADICTIONS:
        sem_id = f"contra_{topic_id}"
        same_reason = f"sr_contra_{topic_id}"

        for text, surf_id, spans in _contradiction_templates(p1, p2, is_contradiction=True):
            ex = BenchmarkExample(
                example_id=f"contra_{counter:04d}",
                task_family="contradiction",
                semantic_task_id=sem_id,
                reasoning_graph_id="negation_conflict",
                surface_template_id=surf_id,
                reasoning_variant_id="canonical",
                answer_id="yes",
                prompt_text=text,
                event_spans=tuple(spans),
                same_reasoning_group=same_reason,
                same_answer_diff_reasoning_group="sadr_contra_yes",
            )
            examples.append(ex)
            counter += 1

    # Consistent pairs (non-contradictions) as hard negatives
    for p1, p2, topic_id in _CONSISTENT:
        sem_id = f"consist_{topic_id}"
        same_reason = f"sr_consist_{topic_id}"

        for text, surf_id, spans in _contradiction_templates(p1, p2, is_contradiction=False):
            ex = BenchmarkExample(
                example_id=f"contra_{counter:04d}",
                task_family="contradiction",
                semantic_task_id=sem_id,
                reasoning_graph_id="no_conflict",
                surface_template_id=surf_id,
                reasoning_variant_id="canonical",
                answer_id="no",
                prompt_text=text,
                event_spans=tuple(spans),
                same_reasoning_group=same_reason,
                same_answer_diff_reasoning_group="sadr_contra_no",
            )
            examples.append(ex)
            counter += 1

    yield from examples
