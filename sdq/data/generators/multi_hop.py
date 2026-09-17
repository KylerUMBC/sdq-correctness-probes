"""Multi-hop entity reasoning generator (Family 8).

Tests genuine multi-step reasoning with novel entities: the model must
chain facts about people, places, and organizations that it has never
seen during pretraining.

Example (3-hop):
  "Alex lives where Blake lives. Blake works at Acme. Acme is in Seattle.
   Where does Alex live?"
  Answer: Seattle

The model must:
  1. Resolve "where Blake lives" → need Blake's location
  2. Blake works at Acme → need Acme's location
  3. Acme is in Seattle → Seattle
  4. Therefore Alex lives in Seattle

Surface form variations test that SDQ quotients out phrasing
while preserving the shared reasoning graph.
"""

from __future__ import annotations

from typing import Generator

from sdq.data.generators.schema import BenchmarkExample, EventSpan

# Novel entity pools (deliberately unusual to avoid memorized associations)
_PEOPLE = [
    "Zara", "Kellan", "Priya", "Orin", "Maren",
    "Thane", "Liora", "Dashiel", "Fen", "Reva",
]

_ORGS = [
    "Veridian Labs", "Crestmoor Co", "Pylon Works",
    "Quartzline", "Emberfield Inc",
]

_PLACES = [
    "Ashwick", "Brindleton", "Coppervale",
    "Dunmore", "Elksrun", "Fenhollow",
]

# Relation types: (relation_verb, question_word, question_template)
# Each defines how entities chain through facts.
_RELATION_CHAINS: list[dict] = [
    # 2-hop: person → person → place
    {
        "hops": 2,
        "desc": "lives_where_person",
        "facts": [
            "{p0} lives where {p1} lives.",
            "{p1} lives in {place}.",
        ],
        "question": "Where does {p0} live?",
        "answer_key": "place",
        "question_word": "Where",
    },
    {
        "hops": 2,
        "desc": "works_where_person",
        "facts": [
            "{p0} works wherever {p1} works.",
            "{p1} works in {place}.",
        ],
        "question": "Where does {p0} work?",
        "answer_key": "place",
        "question_word": "Where",
    },
    # 3-hop: person → person → org → place
    {
        "hops": 3,
        "desc": "lives_via_org",
        "facts": [
            "{p0} lives where {p1} lives.",
            "{p1} works at {org}.",
            "{org} is in {place}.",
        ],
        "question": "Where does {p0} live?",
        "answer_key": "place",
        "question_word": "Where",
    },
    {
        "hops": 3,
        "desc": "works_via_person_org",
        "facts": [
            "{p0} works at the same company as {p1}.",
            "{p1} is employed by {org}.",
            "{org} is headquartered in {place}.",
        ],
        "question": "Where is {p0}'s company headquartered?",
        "answer_key": "place",
        "question_word": "Where",
    },
    {
        "hops": 3,
        "desc": "reports_to_chain",
        "facts": [
            "{p0} reports to {p1}.",
            "{p1} reports to {p2}.",
            "{p2} lives in {place}.",
        ],
        "question": "Where does {p0}'s boss's boss live?",
        "answer_key": "place",
        "question_word": "Where",
    },
    # 4-hop: deep chain
    {
        "hops": 4,
        "desc": "lives_deep_chain",
        "facts": [
            "{p0} lives where {p1} lives.",
            "{p1} lives where {p2} lives.",
            "{p2} works at {org}.",
            "{org} is in {place}.",
        ],
        "question": "Where does {p0} live?",
        "answer_key": "place",
        "question_word": "Where",
    },
]

# Entity assignment sets — each provides distinct entities for a chain
_ENTITY_SETS: list[dict[str, str]] = [
    {"p0": "Zara", "p1": "Kellan", "p2": "Priya", "org": "Veridian Labs", "place": "Ashwick"},
    {"p0": "Orin", "p1": "Maren", "p2": "Thane", "org": "Crestmoor Co", "place": "Brindleton"},
    {"p0": "Liora", "p1": "Dashiel", "p2": "Fen", "org": "Pylon Works", "place": "Coppervale"},
    {"p0": "Reva", "p1": "Zara", "p2": "Kellan", "org": "Quartzline", "place": "Dunmore"},
    {"p0": "Thane", "p1": "Liora", "p2": "Orin", "org": "Emberfield Inc", "place": "Elksrun"},
    {"p0": "Maren", "p1": "Reva", "p2": "Dashiel", "org": "Veridian Labs", "place": "Fenhollow"},
]


def _format_direct(facts: list[str], question: str) -> tuple[str, list[EventSpan]]:
    """Template A: facts stated directly, then question."""
    spans = []
    pos = 0
    parts = []
    for i, fact in enumerate(facts):
        event = "setup" if i == 0 else "chain"
        spans.append(EventSpan(event, pos, pos + len(fact)))
        parts.append(fact)
        pos += len(fact) + 1  # +1 for space

    spans.append(EventSpan("conclusion", pos, pos + len(question)))
    parts.append(question)
    return " ".join(parts), spans


def _format_given(facts: list[str], question: str) -> tuple[str, list[EventSpan]]:
    """Template B: 'Given that ... , <question>'"""
    # Strip trailing periods for clauses
    clauses = [f.rstrip(".") for f in facts]
    body = "Given that " + ", and ".join(clauses) + "."
    text = body + " " + question
    spans = [
        EventSpan("setup", 0, len(body)),
        EventSpan("conclusion", len(body) + 1, len(text)),
    ]
    return text, spans


def _format_reverse(facts: list[str], question: str) -> tuple[str, list[EventSpan]]:
    """Template C: facts in reverse order (harder — must reorder mentally)."""
    rev_facts = list(reversed(facts))
    spans = []
    pos = 0
    parts = []
    for i, fact in enumerate(rev_facts):
        event = "chain" if i < len(rev_facts) - 1 else "setup"
        spans.append(EventSpan(event, pos, pos + len(fact)))
        parts.append(fact)
        pos += len(fact) + 1

    spans.append(EventSpan("conclusion", pos, pos + len(question)))
    parts.append(question)
    return " ".join(parts), spans


def _format_narrative(facts: list[str], question: str) -> tuple[str, list[EventSpan]]:
    """Template D: wrapped in a narrative frame."""
    clauses = [f.rstrip(".") for f in facts]
    body = "Here is what we know: " + "; ".join(clauses) + "."
    text = body + " Based on this, " + question.lower()
    setup_end = len(body)
    spans = [
        EventSpan("setup", 0, setup_end),
        EventSpan("conclusion", setup_end + 1, len(text)),
    ]
    return text, spans


_TEMPLATES = [
    ("direct", _format_direct),
    ("given", _format_given),
    ("reverse", _format_reverse),
    ("narrative", _format_narrative),
]


def generate_multi_hop() -> Generator[BenchmarkExample, None, None]:
    """Generate Family 8 (multi-hop entity reasoning) examples."""
    counter = 0

    for chain_def in _RELATION_CHAINS:
        desc = chain_def["desc"]
        n_hops = chain_def["hops"]

        for ent_set in _ENTITY_SETS:
            # Check if this entity set has enough entities for this chain
            try:
                facts = [f.format(**ent_set) for f in chain_def["facts"]]
                question = chain_def["question"].format(**ent_set)
            except KeyError:
                continue  # entity set doesn't have required keys

            answer = ent_set[chain_def["answer_key"]]
            sem_id = f"multihop_{desc}_{ent_set['p0']}_{ent_set['p1']}"
            reasoning_id = f"multihop_{n_hops}hop"
            same_reason = f"sr_multihop_{desc}_{ent_set['p0']}"

            for surf_id, template_fn in _TEMPLATES:
                text, spans = template_fn(facts, question)
                yield BenchmarkExample(
                    example_id=f"multihop_{counter:04d}",
                    task_family="multi_hop",
                    semantic_task_id=sem_id,
                    reasoning_graph_id=reasoning_id,
                    surface_template_id=surf_id,
                    reasoning_variant_id="canonical",
                    answer_id=answer,
                    prompt_text=text,
                    event_spans=tuple(spans),
                    same_reasoning_group=same_reason,
                )
                counter += 1
