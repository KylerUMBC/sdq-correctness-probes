#!/usr/bin/env python3
"""Build paper tables and figures from frozen results, without rerunning models."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = [
    ("hidden_L13", "Residual state, layer 13", "#176B87"),
    ("hidden_L0", "Residual state, layer 0", "#2A9D8F"),
    ("hidden_L25", "Residual state, layer 25", "#55A89C"),
    ("input_charhash_3to5", "Hashed prompt text", "#E9C46A"),
    ("next_token_confidence", "Next-token confidence", "#F4A261"),
    ("input_surface", "Surface features", "#E76F51"),
    ("family_prior", "Family prior", "#A8A8A8"),
]


def fmt(value: float, signed: bool = False) -> str:
    """Consistent decimal rounding in tables and figures (0.0875 -> 0.088)."""
    rounded = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return format(rounded, "+.3f" if signed else ".3f")


def probe_table(grouped: dict) -> str:
    lines = ["| Signal | Pooled AUROC | Mean within-family AUROC | 95% CI |",
             "| --- | ---: | ---: | ---: |"]
    for key, label, _ in SIGNALS:
        row = grouped["results"][key]
        ci = row.get("within_family_95ci")
        interval = f"{fmt(ci[0])}–{fmt(ci[1])}" if ci else "—"
        lines.append(f"| {label} | {fmt(row['pooled_auroc'])} | "
                     f"{fmt(row['mean_within_family_auroc'])} | {interval} |")
    return "\n".join(lines) + "\n"


def dataset_table(grouped: dict) -> str:
    counts = grouped["results"]["hidden_L13"]["family_counts"]
    order = ["arithmetic", "multi_step_arithmetic", "relational", "set_inclusion",
             "syllogistic", "contradiction"]
    lines = ["| Family | Examples | Correct | Incorrect |", "| --- | ---: | ---: | ---: |"]
    for family in order:
        row = counts[family]
        lines.append(f"| {family.replace('_', ' ').title()} | {row['n']} | "
                     f"{row['n'] - row['incorrect']} | {row['incorrect']} |")
    row = grouped["dataset"]
    lines.append(f"| **Total** | **{row['n']}** | **{row['n_correct']}** | **{row['n_incorrect']}** |")
    return "\n".join(lines) + "\n"


def find_cell(data: dict, source: str, magnitude: float) -> dict:
    for row in data["experiments"][source]["layers"]["layer13"]:
        if row["condition"] == "single" and row["component"] == 0 and row["magnitude"] == magnitude:
            return row
    raise KeyError((source, magnitude))


def intervention_rows(data: dict) -> list[tuple[str, float, dict, str]]:
    return [(run, magnitude, find_cell(data, source, magnitude), color)
            for run, source, color in [
                ("Exploratory", "mesoscale_results.json", "#D97706"),
                ("Confirmation", "confirmation_results.json", "#2563EB")]
            for magnitude in (2.0, 3.0)]


def intervention_table(data: dict) -> str:
    lines = [
        "| Run | Magnitude | Prompts | Directed − within-span control | 95% CI | Holm-adjusted p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run, magnitude, row, _ in intervention_rows(data):
        ci = row["paired_difference_95ci"]
        lines.append(f"| {run} | {magnitude:g} | {row['n_prompts']} | "
                     f"{fmt(row['paired_rate_difference'], True)} | "
                     f"{fmt(ci[0], True)} to {fmt(ci[1], True)} | "
                     f"{fmt(row['permutation_p_holm'])} |")
    return "\n".join(lines) + "\n"


def forest_plot(rows: list[tuple], *, title: str, subtitle: str, xlabel: str,
                limits: tuple, ticks: list, reference: float, signed: bool = False):
    # Lazy import keeps table generation and numeric tests dependency-light.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": "#1F2937", "axes.labelcolor": "#1F2937",
                         "svg.fonttype": "none", "svg.hashsalt": "sdq-paper"}):
        fig = plt.figure(figsize=(12.6, 3.0 + 0.52 * len(rows)), facecolor="white")
        grid = fig.add_gridspec(1, 2, left=0.255, right=0.985, bottom=0.18,
                               top=0.77, width_ratios=[1.7, 1.05], wspace=0.09)
        ax = fig.add_subplot(grid[0])
        values = fig.add_subplot(grid[1], sharey=ax)
        ax.set_xlim(*limits)
        ax.set_ylim(len(rows) - 0.45, -0.55)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%+.2f" if signed else "%.1f"))
        ax.set_xlabel(xlabel, labelpad=13)
        ax.set_yticks(range(len(rows)), [row[0] for row in rows])
        ax.tick_params(axis="both", length=0, pad=10)
        ax.grid(axis="x", color="#E5E7EB", linewidth=1)
        ax.axvline(reference, color="#64748B", linestyle="--", linewidth=1.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        values.set_xlim(0, 1)
        values.axis("off")
        values.text(0.02, 1.09, "Estimate [95% CI]", transform=values.transAxes,
                    fontsize=12, fontweight="bold", va="bottom")
        for i, (label, estimate, interval, color) in enumerate(rows):
            lo, hi = interval
            ax.errorbar(estimate, i, xerr=[[estimate - lo], [hi - estimate]],
                        fmt="o", markersize=9, color=color, markeredgecolor="#374151",
                        ecolor="#475569", elinewidth=1.8, capsize=5, capthick=1.6, zorder=3)
            # Separate axes guarantee that labels cannot collide with any CI line.
            values.text(0.02, i, f"{fmt(estimate, signed)}  [{fmt(lo, signed)}, {fmt(hi, signed)}]",
                        va="center", ha="left", fontsize=12, fontfamily="DejaVu Sans Mono")
        fig.suptitle(title, x=0.5, y=0.965, fontsize=17, fontweight="bold")
        fig.text(0.5, 0.875, subtitle, ha="center", fontsize=11, color="#475569")
        return fig


def build_figures(grouped: dict, intervention: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(label, grouped["results"][key]["mean_within_family_auroc"],
             grouped["results"][key]["within_family_95ci"], color)
            for key, label, color in SIGNALS]
    probe = forest_plot(rows, title="Correctness prediction on held-out semantic problems",
                        subtitle="Mean within-family AUROC; intervals resample semantic groups",
                        xlabel="AUROC", limits=(0.47, 0.92), ticks=[0.5, 0.6, 0.7, 0.8, 0.9], reference=0.5)
    rows = [(f"{run}, magnitude {magnitude:g}", row["paired_rate_difference"],
             row["paired_difference_95ci"], color)
            for run, magnitude, row, color in intervention_rows(intervention)]
    causal = forest_plot(rows, title="Selected component: exploratory and confirmation estimates",
                         subtitle="Directed minus same-prompt within-span random-control error rate",
                         xlabel="Difference in error rate", limits=(-0.06, 0.18),
                         ticks=[-0.05, 0, 0.05, 0.10, 0.15], reference=0, signed=True)
    for name, fig in [("probe_auroc", probe), ("intervention_confirmation", causal)]:
        with matplotlib.rc_context({"svg.fonttype": "none", "svg.hashsalt": "sdq-paper"}):
            fig.savefig(out / f"{name}.svg", metadata={"Date": None})
            fig.savefig(out / f"{name}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "generated")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    grouped = json.loads((args.results_dir / "grouped_probe_results.json").read_text(encoding="utf-8"))
    intervention = json.loads((args.results_dir / "intervention_paired_reanalysis.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in [("probe_results.md", probe_table(grouped)),
                          ("dataset_counts.md", dataset_table(grouped)),
                          ("intervention_results.md", intervention_table(intervention))]:
        (args.output_dir / name).write_text(content, encoding="utf-8")
    if not args.tables_only:
        build_figures(grouped, intervention, args.output_dir)
    print(f"Wrote paper artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
