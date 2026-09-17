#!/usr/bin/env python3
"""Reanalyze saved SDQ intervention trials with prompt-clustered statistics.

The legacy summaries use a two-sample Fisher test over individual generations.
Each prompt was perturbed repeatedly, so those rows are not independent.  This
script instead pairs each designated intervention with the random directions
run on the same prompt and treats the prompt as the unit of analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from sdq.eval.paired_stats import (
    exchangeability_permutation_pvalue,
    paired_cluster_bootstrap_ci,
    paired_effect,
)
from sdq.eval.stats import holm_adjust


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prompt-clustered reanalysis of SDQ intervention trials",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs", nargs="*", default=[
            "results/interventions/mesoscale_results.json",
            "results/interventions/confirmation_results.json",
            "results/interventions/prefill_all_results.json",
        ]
    )
    p.add_argument("--control", default="within_span")
    p.add_argument("--permutations", type=int, default=50_000)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/reproduction/intervention_paired_reanalysis.json")
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_rows(
    trials: list[dict],
    control_name: str,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> list[dict]:
    control: dict[tuple[float, str], list[float]] = defaultdict(list)
    directed: dict[tuple[float, str, int | None], list[float]] = defaultdict(list)
    for trial in trials:
        key = (float(trial["mag_mult"]), trial["example_id"])
        outcome = float(bool(trial["flipped_to_incorrect"]))
        if trial["condition"] == control_name:
            control[key].append(outcome)
        elif trial["condition"] == "single":
            directed[(key[0], key[1], int(trial["component_idx"]))].append(outcome)
        elif trial["condition"] == "coordinated":
            directed[(key[0], key[1], None)].append(outcome)

    cells = sorted({(mag, comp) for mag, _, comp in directed}, key=lambda x: (x[0], x[1] is None, x[1] or -1))
    rows = []
    for mag, component in cells:
        prompt_ids = sorted({eid for m, eid, c in directed if m == mag and c == component})
        paired_ids = [eid for eid in prompt_ids if control.get((mag, eid))]
        directed_values = []
        control_values = []
        for eid in paired_ids:
            values = directed[(mag, eid, component)]
            # A designated condition should appear once per prompt.  If a
            # resumed run duplicated it, average rather than pretending the
            # duplicate is a new independent prompt.
            directed_values.append(sum(values) / len(values))
            control_values.append(control[(mag, eid)])
        d_rate, c_rate, difference = paired_effect(directed_values, control_values)
        rows.append({
            "magnitude": mag,
            "condition": "coordinated" if component is None else "single",
            "component": component,
            "n_prompts": len(paired_ids),
            "n_control_generations": sum(len(v) for v in control_values),
            "directed_rate": d_rate,
            "control_rate": c_rate,
            "rate_ratio": d_rate / c_rate if c_rate else None,
            "paired_rate_difference": difference,
            "paired_difference_95ci": paired_cluster_bootstrap_ci(
                directed_values, control_values, n_bootstrap=n_bootstrap, seed=seed
            ),
            "permutation_p_raw": exchangeability_permutation_pvalue(
                directed_values, control_values,
                n_permutations=n_permutations, seed=seed,
            ),
        })
    adjusted = holm_adjust([row["permutation_p_raw"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["permutation_p_holm"] = p_adjusted
    return rows


def main() -> None:
    args = parse_args()
    result = {
        "method": (
            "same-prompt comparison against random directions; cluster bootstrap "
            "CI and within-prompt exchangeability randomization test"
        ),
        "unit_of_analysis": "prompt",
        "control": args.control,
        "permutations": args.permutations,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "experiments": {},
    }
    for input_name in args.inputs:
        path = Path(input_name)
        saved = json.loads(path.read_text(encoding="utf-8"))
        experiment = {
            "source": str(path),
            "source_sha256": sha256(path),
            "layers": {},
        }
        for layer_name, layer in saved["per_layer"].items():
            rows = comparison_rows(
                layer["trials"], args.control, args.permutations,
                args.bootstrap, args.seed,
            )
            experiment["layers"][layer_name] = rows
            best = max(rows, key=lambda row: row["paired_rate_difference"])
            print(
                f"{path.name} {layer_name}: best paired difference "
                f"{best['paired_rate_difference']:+.3f} "
                f"(95% CI {best['paired_difference_95ci'][0]:+.3f} to "
                f"{best['paired_difference_95ci'][1]:+.3f}, "
                f"Holm p={best['permutation_p_holm']:.3f})"
            )
        result["experiments"][path.name] = experiment
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
