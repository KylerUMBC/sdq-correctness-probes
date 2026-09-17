"""Set / class inclusion prompt generator (Family 5).

Generates set-theoretic inclusion prompts with crossed factors:
  - Multiple category hierarchies (semantic_task_id)
  - Multiple surface templates
  - Inclusion vs. non-inclusion (entailment vs. non-entailment)
  - Event span annotations
"""

from __future__ import annotations

from typing import Generator

from sdq.data.generators.schema import BenchmarkExample, EventSpan

# ---------------------------------------------------------------------------
# Category hierarchies: (subset, superset) — correct inclusion
# ---------------------------------------------------------------------------

_INCLUSIONS: list[tuple[str, str]] = [
    ("cats", "mammals"),
    ("dogs", "mammals"),
    ("eagles", "birds"),
    ("sparrows", "birds"),
    ("salmon", "fish"),
    ("trout", "fish"),
    ("roses", "flowers"),
    ("tulips", "flowers"),
    ("oaks", "trees"),
    ("pines", "trees"),
    ("ants", "insects"),
    ("bees", "insects"),
    ("wolves", "canines"),
    ("tigers", "felines"),
    ("snakes", "reptiles"),
    ("frogs", "amphibians"),
    ("apples", "fruits"),
    ("carrots", "vegetables"),
    ("gold", "metals"),
    ("diamonds", "gems"),
]

# Incorrect inclusions for hard negatives (reversed or wrong)
_NON_INCLUSIONS: list[tuple[str, str]] = [
    ("mammals", "cats"),
    ("birds", "eagles"),
    ("fish", "salmon"),
    ("flowers", "roses"),
    ("trees", "oaks"),
    ("insects", "ants"),
    ("fruits", "vegetables"),
    ("metals", "gems"),
    ("reptiles", "amphibians"),
    ("canines", "felines"),
]


def _inclusion_templates(
    subset: str, superset: str, valid: bool,
) -> list[tuple[str, str, list[EventSpan]]]:
    """Generate surface variants for set inclusion queries."""
    results: list[tuple[str, str, list[EventSpan]]] = []

    if valid:
        # Correct: All subset are superset. Are subset superset? Yes.
        # Template: are_question
        p1 = f"All {subset} are {superset}."
        q = f" Are {subset} {superset}? Answer yes or no."
        text = p1 + q
        spans = [
            EventSpan("premise_load", 0, len(p1)),
            EventSpan("conclusion", len(p1), len(text)),
        ]
        results.append((text, "are_question", spans))

        # Keep every surface form on the same yes/no answer contract.  Earlier
        # versions ended several prompts as category-word completions even
        # though answer_id and the evaluator expected yes/no.
        # Template: therefore
        text = (
            f"All {subset} are {superset}. Therefore, is it true that all "
            f"{subset} are {superset}? Answer yes or no."
        )
        dot_idx = text.index(". Therefore")
        spans = [
            EventSpan("premise_load", 0, dot_idx + 1),
            EventSpan("conclusion", dot_idx + 1, len(text)),
        ]
        results.append((text, "therefore", spans))

        # Template: since
        text = (
            f"Since all {subset} are {superset}, does it follow that all "
            f"{subset} are {superset}? Answer yes or no."
        )
        comma_idx = text.index(", does it follow")
        spans = [
            EventSpan("premise_load", 0, comma_idx),
            EventSpan("conclusion", comma_idx, len(text)),
        ]
        results.append((text, "since", spans))

        # Template: given
        text = (
            f"Given that all {subset} are {superset}, are {subset} {superset}? "
            "Answer yes or no."
        )
        comma_idx = text.index(", are")
        spans = [
            EventSpan("premise_load", 0, comma_idx),
            EventSpan("conclusion", comma_idx, len(text)),
        ]
        results.append((text, "given", spans))

        # Template: boolean
        text = (
            f"Is the statement 'All {subset} are {superset}' true? "
            "Answer yes or no."
        )
        quote_start = text.index("'") + 1
        quote_end = text.index("'", quote_start)
        spans = [
            EventSpan("setup", 0, quote_start),
            EventSpan("premise_load", quote_start, quote_end),
            EventSpan("conclusion", quote_end, len(text)),
        ]
        results.append((text, "boolean", spans))

        # Template: knowing
        text = (
            f"Knowing that all {subset} are {superset}, can we conclude that all "
            f"{subset} are {superset}? Answer yes or no."
        )
        comma_idx = text.index(", can we")
        spans = [
            EventSpan("premise_load", 0, comma_idx),
            EventSpan("conclusion", comma_idx, len(text)),
        ]
        results.append((text, "knowing", spans))

    else:
        # Incorrect converse.  State the valid direction as the premise and ask
        # whether its converse follows, rather than asserting a generally false
        # premise and then asking a real-world category question.
        # Template: are_question
        text = (
            f"All {superset} are {subset}. Does it follow that all {subset} "
            f"are {superset}? Answer yes or no."
        )
        dot_idx = text.index(". Does")
        spans = [
            EventSpan("premise_load", 0, dot_idx + 1),
            EventSpan("conclusion", dot_idx + 1, len(text)),
        ]
        results.append((text, "are_question", spans))

        # Template: therefore
        text = (
            f"All {superset} are {subset}. Therefore, must all {subset} be "
            f"{superset}? Answer yes or no."
        )
        dot_idx = text.index(". Therefore")
        spans = [
            EventSpan("premise_load", 0, dot_idx + 1),
            EventSpan("conclusion", dot_idx + 1, len(text)),
        ]
        results.append((text, "therefore_invalid", spans))

        # Template: does it follow
        text = (
            f"Given that all {superset} are {subset}, does it follow that all "
            f"{subset} are {superset}? Answer yes or no."
        )
        comma_idx = text.index(", does it follow")
        spans = [
            EventSpan("premise_load", 0, comma_idx),
            EventSpan("conclusion", comma_idx, len(text)),
        ]
        results.append((text, "does_follow", spans))

        # Template: boolean
        text = (
            f"Premise: All {superset} are {subset}. Is the converse, 'All "
            f"{subset} are {superset}', entailed? Answer yes or no."
        )
        colon_idx = text.index(": ")
        spans = [
            EventSpan("setup", 0, colon_idx + 2),
            EventSpan("premise_load", colon_idx + 2, len(text)),
        ]
        results.append((text, "boolean_invalid", spans))

    return results


def generate_set_inclusion() -> Generator[BenchmarkExample, None, None]:
    """Generate Family 5 (set/class inclusion) examples."""
    counter = 0
    examples: list[BenchmarkExample] = []

    # Valid inclusions
    for subset, superset in _INCLUSIONS:
        sem_id = f"incl_{subset}_{superset}"
        same_reason = f"sr_incl_{sem_id}"

        for text, surf_id, spans in _inclusion_templates(subset, superset, valid=True):
            ex = BenchmarkExample(
                example_id=f"incl_{counter:04d}",
                task_family="set_inclusion",
                semantic_task_id=sem_id,
                reasoning_graph_id="subset_entailment",
                surface_template_id=surf_id,
                reasoning_variant_id="canonical",
                answer_id="yes",
                prompt_text=text,
                event_spans=tuple(spans),
                same_reasoning_group=same_reason,
                same_answer_diff_reasoning_group="sadr_incl_yes",
            )
            examples.append(ex)
            counter += 1

    # Invalid inclusions (reversed)
    for subset, superset in _NON_INCLUSIONS:
        sem_id = f"incl_inv_{subset}_{superset}"
        same_reason = f"sr_incl_inv_{sem_id}"

        for text, surf_id, spans in _inclusion_templates(subset, superset, valid=False):
            ex = BenchmarkExample(
                example_id=f"incl_{counter:04d}",
                task_family="set_inclusion",
                semantic_task_id=sem_id,
                reasoning_graph_id="invalid_converse",
                surface_template_id=surf_id,
                reasoning_variant_id="canonical",
                answer_id="no",
                prompt_text=text,
                event_spans=tuple(spans),
                same_reasoning_group=same_reason,
                same_answer_diff_reasoning_group="sadr_incl_no",
            )
            examples.append(ex)
            counter += 1

    yield from examples
