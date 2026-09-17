"""Helpers for applying auditable benchmark-exclusion manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _matches(example: dict[str, Any], where: dict[str, Any]) -> bool:
    for field, expected in where.items():
        actual = example.get(field)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def load_excluded_example_ids(
    path: str | Path,
    examples: list[dict[str, Any]],
) -> set[str]:
    """Expand explicit IDs and declarative rules from an exclusion manifest."""
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    excluded = {
        item["example_id"] for item in manifest.get("examples", [])
    }
    for rule in manifest.get("rules", []):
        where = rule.get("where", {})
        excluded.update(
            example["example_id"] for example in examples
            if _matches(example, where)
        )

    expected_count = manifest.get("expected_excluded_count")
    if expected_count is not None and len(excluded) != expected_count:
        raise ValueError(
            f"exclusion manifest expected {expected_count} matches, got {len(excluded)}"
        )
    return excluded
