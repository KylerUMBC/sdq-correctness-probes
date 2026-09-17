#!/usr/bin/env python3
"""Create a deterministic, probe-blind human audit packet for SDQ labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from sdq.eval.intervention_data import load_benchmark, resolve_run_dirs
from sdq.eval.benchmark_audit import load_excluded_example_ids
from sdq.labels.outcome_labeler import label_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a stratified human audit of cached outcome labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", default="data/prompts/benchmark_v2.json")
    parser.add_argument("--runs-dir", default="data/runs")
    parser.add_argument(
        "--exclude-file", default="data/prompts/benchmark_v2_exclusions.json",
        help="JSON manifest of invalid cached examples to exclude; use an empty string for none",
    )
    parser.add_argument("--per-family", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/label_audit/label_audit_blind.csv")
    parser.add_argument("--key-output", default="outputs/label_audit/label_audit_key.csv")
    return parser.parse_args()


def _diverse_sample(
    records: list[dict], count: int, rng: random.Random
) -> list[dict]:
    """Prefer distinct semantic tasks, then fill remaining sample slots."""
    shuffled = records[:]
    rng.shuffle(shuffled)
    selected: list[dict] = []
    seen_groups: set[str] = set()
    for record in shuffled:
        if record["semantic_task_id"] not in seen_groups:
            selected.append(record)
            seen_groups.add(record["semantic_task_id"])
            if len(selected) == count:
                return selected
    selected_ids = {record["example_id"] for record in selected}
    for record in shuffled:
        if record["example_id"] not in selected_ids:
            selected.append(record)
            if len(selected) == count:
                break
    return selected


def select_records(
    records: list[dict], per_family: int, seed: int
) -> list[dict]:
    """Sample roughly equal parser-correct and parser-incorrect cases per family."""
    rng = random.Random(seed)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_family[record["task_family"]].append(record)

    selected: list[dict] = []
    for family in sorted(by_family):
        correct = [r for r in by_family[family] if r["parser_correct"]]
        incorrect = [r for r in by_family[family] if not r["parser_correct"]]
        n_correct = min((per_family + 1) // 2, len(correct))
        n_incorrect = min(per_family - n_correct, len(incorrect))
        if n_correct + n_incorrect < per_family:
            remaining_correct = len(correct) - n_correct
            add_correct = min(per_family - n_correct - n_incorrect, remaining_correct)
            n_correct += add_correct
            n_incorrect = min(per_family - n_correct, len(incorrect))
        family_sample = _diverse_sample(correct, n_correct, rng)
        family_sample += _diverse_sample(incorrect, n_incorrect, rng)
        rng.shuffle(family_sample)
        selected.extend(family_sample)
    return selected


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark)
    runs_dir = Path(args.runs_dir)
    examples = load_benchmark(benchmark_path)
    exclude_ids: set[str] = set()
    if args.exclude_file:
        exclusion_path = Path(args.exclude_file)
        if exclusion_path.exists():
            exclude_ids = load_excluded_example_ids(exclusion_path, examples)
    run_dirs = resolve_run_dirs(runs_dir, [ex["example_id"] for ex in examples])

    records: list[dict] = []
    for example in examples:
        if example["example_id"] in exclude_ids:
            continue
        run_dir = run_dirs.get(example["example_id"])
        if run_dir is None:
            continue
        with (run_dir / "metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        response = metadata.get("output", {}).get("new_tokens", "")
        label = label_run(response, example["answer_id"], example["task_family"])
        records.append({
            "example_id": example["example_id"],
            "task_family": example["task_family"],
            "semantic_task_id": example.get("semantic_task_id", example["example_id"]),
            "surface_template_id": example.get("surface_template_id", ""),
            "prompt_text": example["prompt_text"],
            "reference_answer": example["answer_id"],
            "model_response": response,
            "parser_correct": label.correct,
            "parser_answer": label.parsed_answer,
            "parser_confidence": label.confidence,
            "run_dir": str(run_dir),
        })

    selected = select_records(records, args.per_family, args.seed)
    output_path = Path(args.output)
    key_path = Path(args.key_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    blind_fields = [
        "audit_id",
        "example_id",
        "task_family",
        "semantic_task_id",
        "surface_template_id",
        "prompt_text",
        "reference_answer",
        "model_response",
        "human_correct",
        "human_extracted_answer",
        "human_output_validity",
        "human_notes",
    ]
    key_fields = [
        "audit_id",
        "example_id",
        "parser_correct",
        "parser_answer",
        "parser_confidence",
        "run_dir",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as blind_handle, \
            key_path.open("w", encoding="utf-8", newline="") as key_handle:
        blind_writer = csv.DictWriter(blind_handle, fieldnames=blind_fields)
        key_writer = csv.DictWriter(key_handle, fieldnames=key_fields)
        blind_writer.writeheader()
        key_writer.writeheader()
        for index, record in enumerate(selected, start=1):
            audit_id = f"audit_{index:03d}"
            blind_writer.writerow({
                "audit_id": audit_id,
                **{field: record[field] for field in blind_fields[1:8]},
                "human_correct": "",
                "human_extracted_answer": "",
                "human_output_validity": "",
                "human_notes": "",
            })
            key_writer.writerow({
                "audit_id": audit_id,
                **{field: record[field] for field in key_fields[1:]},
            })

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "parser_correct": 0})
    for record in selected:
        counts[record["task_family"]]["n"] += 1
        counts[record["task_family"]]["parser_correct"] += int(record["parser_correct"])
    print(f"Wrote {len(selected)} cases to {output_path}")
    print(f"Wrote hidden parser key to {key_path}")
    for family, count in sorted(counts.items()):
        print(f"  {family}: n={count['n']}, parser_correct={count['parser_correct']}")


if __name__ == "__main__":
    main()
