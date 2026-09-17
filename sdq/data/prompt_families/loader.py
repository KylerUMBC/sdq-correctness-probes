"""Load and parse prompt family JSON files.

Supports two formats:
  1. Flat list: [{"id": ..., "text": ...}, ...]
  2. Grouped:   {"groups": [{"group_id": ..., "prompts": [...]}]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Prompt:
    """A single prompt with metadata."""

    id: str
    text: str
    form: str = ""
    surface_diff: str = ""
    valid: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptGroup:
    """A semantic equivalence class of prompts."""

    group_id: str
    prompts: list[Prompt]
    semantic_label: str = ""
    description: str = ""
    validity: str = ""
    matched_pair: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_ids(self) -> list[str]:
        return [p.id for p in self.prompts]


@dataclass
class PromptDataset:
    """A collection of prompt groups forming a dataset."""

    groups: list[PromptGroup]
    task_family: str = ""
    description: str = ""
    stage: str = ""
    train_group_ids: list[str] = field(default_factory=list)
    test_group_ids: list[str] = field(default_factory=list)
    source_path: Path | None = None

    @property
    def all_prompts(self) -> list[Prompt]:
        return [p for g in self.groups for p in g.prompts]

    @property
    def group_map(self) -> dict[str, PromptGroup]:
        return {g.group_id: g for g in self.groups}

    def prompts_by_group(self, group_id: str) -> list[Prompt]:
        return self.group_map[group_id].prompts

    @property
    def train_groups(self) -> list[PromptGroup]:
        return [g for g in self.groups if g.group_id in self.train_group_ids]

    @property
    def test_groups(self) -> list[PromptGroup]:
        return [g for g in self.groups if g.group_id in self.test_group_ids]


def _parse_prompt(raw: dict[str, Any]) -> Prompt:
    known = {"id", "text", "form", "surface_diff", "valid"}
    extra = {k: v for k, v in raw.items() if k not in known}
    return Prompt(
        id=raw["id"],
        text=raw["text"],
        form=raw.get("form", ""),
        surface_diff=raw.get("surface_diff", ""),
        valid=raw.get("valid"),
        extra=extra,
    )


def _parse_group(raw: dict[str, Any]) -> PromptGroup:
    known = {
        "group_id", "prompts", "semantic_label", "description",
        "validity", "matched_pair",
    }
    extra = {k: v for k, v in raw.items() if k not in known}
    return PromptGroup(
        group_id=raw["group_id"],
        prompts=[_parse_prompt(p) for p in raw["prompts"]],
        semantic_label=raw.get("semantic_label", ""),
        description=raw.get("description", ""),
        validity=raw.get("validity", ""),
        matched_pair=raw.get("matched_pair", ""),
        extra=extra,
    )


def load_prompt_file(path: str | Path) -> PromptDataset:
    """Load a prompt dataset from a JSON file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Flat list format (e.g. arithmetic_basic.json)
    if isinstance(data, list):
        prompts = [_parse_prompt(p) for p in data]
        group = PromptGroup(group_id="default", prompts=prompts)
        return PromptDataset(groups=[group], source_path=path)

    # Grouped format
    groups = [_parse_group(g) for g in data.get("groups", [])]
    return PromptDataset(
        groups=groups,
        task_family=data.get("task_family", data.get("task", "")),
        description=data.get("description", ""),
        stage=data.get("stage", ""),
        train_group_ids=data.get("train_groups", []),
        test_group_ids=data.get("test_groups", []),
        source_path=path,
    )
