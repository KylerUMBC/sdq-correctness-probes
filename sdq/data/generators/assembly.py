"""Benchmark assembly — combine all generators, assign global IDs, build grouping links.

Primary entry point:
    assemble_benchmark() -> BenchmarkDataset
"""

from __future__ import annotations

from collections import defaultdict

from sdq.data.generators.arithmetic import generate_multi_step, generate_single_step
from sdq.data.generators.contradiction import generate_contradiction
from sdq.data.generators.multi_hop import generate_multi_hop
from sdq.data.generators.relational import generate_relational
from sdq.data.generators.schema import BenchmarkDataset, BenchmarkExample
from sdq.data.generators.set_inclusion import generate_set_inclusion
from sdq.data.generators.syllogistic import generate_syllogistic
from sdq.data.generators.variable_chains import generate_variable_chains


def assemble_benchmark() -> BenchmarkDataset:
    """Assemble the full factorized benchmark from all generators.

    Returns a BenchmarkDataset with globally unique example IDs and
    consistent grouping links.
    """
    raw: list[BenchmarkExample] = []

    # Collect from all generators
    raw.extend(generate_single_step())
    raw.extend(generate_multi_step())
    raw.extend(generate_syllogistic())
    raw.extend(generate_relational())
    raw.extend(generate_set_inclusion())
    raw.extend(generate_contradiction())
    raw.extend(generate_variable_chains())
    raw.extend(generate_multi_hop())

    # Re-assign globally unique IDs
    examples: list[BenchmarkExample] = []
    for i, ex in enumerate(raw):
        examples.append(BenchmarkExample(
            example_id=f"bench_{i:05d}",
            task_family=ex.task_family,
            semantic_task_id=ex.semantic_task_id,
            reasoning_graph_id=ex.reasoning_graph_id,
            surface_template_id=ex.surface_template_id,
            reasoning_variant_id=ex.reasoning_variant_id,
            answer_id=ex.answer_id,
            prompt_text=ex.prompt_text,
            event_spans=ex.event_spans,
            same_reasoning_group=ex.same_reasoning_group,
            same_answer_diff_reasoning_group=ex.same_answer_diff_reasoning_group,
        ))

    return BenchmarkDataset(
        examples=examples,
        version="2.0",
        description=(
            "SDQ factorized benchmark v2.0 — "
            f"{len(examples)} examples across "
            f"{len({e.task_family for e in examples})} task families"
        ),
    )


def print_benchmark_stats(dataset: BenchmarkDataset) -> str:
    """Return a human-readable summary of benchmark composition."""
    lines: list[str] = []
    lines.append(f"=== SDQ Benchmark v{dataset.version} ===")
    lines.append(f"Total examples: {len(dataset.examples)}")
    lines.append("")

    # Per-family counts
    family_counts: dict[str, int] = defaultdict(int)
    for ex in dataset.examples:
        family_counts[ex.task_family] += 1

    lines.append("By task family:")
    for fam in sorted(family_counts):
        lines.append(f"  {fam}: {family_counts[fam]}")
    lines.append("")

    # Unique counts per axis
    lines.append(f"Unique semantic_task_ids: {len(dataset.semantic_task_ids)}")
    lines.append(f"Unique surface_template_ids: {len(dataset.surface_template_ids)}")
    lines.append(f"Unique reasoning_variant_ids: {len(dataset.reasoning_variant_ids)}")
    lines.append(f"Unique answer_ids: {len(dataset.answer_ids)}")
    lines.append("")

    # Reasoning graph distribution
    graph_counts: dict[str, int] = defaultdict(int)
    for ex in dataset.examples:
        graph_counts[ex.reasoning_graph_id] += 1
    lines.append("By reasoning graph:")
    for gid in sorted(graph_counts):
        lines.append(f"  {gid}: {graph_counts[gid]}")
    lines.append("")

    # Event span coverage
    with_events = sum(1 for e in dataset.examples if e.event_spans)
    lines.append(f"Examples with event spans: {with_events}/{len(dataset.examples)}")

    # Grouping coverage
    with_sr = sum(1 for e in dataset.examples if e.same_reasoning_group)
    with_sadr = sum(1 for e in dataset.examples if e.same_answer_diff_reasoning_group)
    lines.append(f"With same_reasoning_group: {with_sr}")
    lines.append(f"With same_answer_diff_reasoning_group: {with_sadr}")

    return "\n".join(lines)
