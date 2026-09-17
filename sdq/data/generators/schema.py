"""Benchmark data schema for the factorized SDQ benchmark.

Defines the core dataclasses for benchmark examples with crossed factors:
    semantics × surface form × reasoning variant × answer identity

Also defines event span annotations for reasoning-stage anchoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EventSpan:
    """A coarse reasoning-stage annotation within a prompt."""

    event: str        # e.g. "setup", "transform", "conclusion"
    start_char: int   # character offset (inclusive)
    end_char: int     # character offset (exclusive)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.event, "start_char": self.start_char, "end_char": self.end_char}


@dataclass(frozen=True)
class BenchmarkExample:
    """A single benchmark example with full factorization metadata."""

    example_id: str
    task_family: str                # e.g. "arithmetic", "syllogistic", "relational", etc.
    semantic_task_id: str           # what reasoning content is being computed
    reasoning_graph_id: str         # the actual reasoning structure/path
    surface_template_id: str        # how the problem is phrased
    reasoning_variant_id: str       # alternative valid path to the same answer
    answer_id: str                  # the final answer
    prompt_text: str                # the actual prompt string
    event_spans: tuple[EventSpan, ...] = ()

    # Grouping IDs for contrastive controls
    same_reasoning_group: str = ""
    same_answer_diff_reasoning_group: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "example_id": self.example_id,
            "task_family": self.task_family,
            "semantic_task_id": self.semantic_task_id,
            "reasoning_graph_id": self.reasoning_graph_id,
            "surface_template_id": self.surface_template_id,
            "reasoning_variant_id": self.reasoning_variant_id,
            "answer_id": self.answer_id,
            "prompt_text": self.prompt_text,
            "event_spans": [e.to_dict() for e in self.event_spans],
        }
        if self.same_reasoning_group:
            d["same_reasoning_group"] = self.same_reasoning_group
        if self.same_answer_diff_reasoning_group:
            d["same_answer_diff_reasoning_group"] = self.same_answer_diff_reasoning_group
        return d

    @property
    def prompt_id(self) -> str:
        """ID compatible with the run storage naming convention."""
        return self.example_id


@dataclass
class BenchmarkDataset:
    """A complete benchmark dataset with all examples and metadata."""

    examples: list[BenchmarkExample]
    version: str = "1.0"
    description: str = ""

    @property
    def task_families(self) -> set[str]:
        return {e.task_family for e in self.examples}

    @property
    def semantic_task_ids(self) -> set[str]:
        return {e.semantic_task_id for e in self.examples}

    @property
    def surface_template_ids(self) -> set[str]:
        return {e.surface_template_id for e in self.examples}

    @property
    def reasoning_variant_ids(self) -> set[str]:
        return {e.reasoning_variant_id for e in self.examples}

    @property
    def answer_ids(self) -> set[str]:
        return {e.answer_id for e in self.examples}

    def filter(self, **kwargs: Any) -> list[BenchmarkExample]:
        """Filter examples by any attribute value."""
        result = self.examples
        for key, val in kwargs.items():
            result = [e for e in result if getattr(e, key) == val]
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "description": self.description,
            "num_examples": len(self.examples),
            "task_families": sorted(self.task_families),
            "examples": [e.to_dict() for e in self.examples],
        }
