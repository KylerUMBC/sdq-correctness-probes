from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

COLORS = {
    "navy": "#223A5E",
    "blue": "#3B82C4",
    "sky": "#8EC7E8",
    "orange": "#D97732",
    "red": "#B9472F",
    "purple": "#6F4E9B",
    "green": "#2E8B57",
    "gray": "#A7A9AC",
    "dark_gray": "#3F4550",
    "light_gray": "#EEF1F5",
}


FAMILY_ORDER = [
    "arithmetic",
    "multi_step_arithmetic",
    "syllogistic",
    "relational",
    "set_inclusion",
    "contradiction",
    "variable_chain",
    "multi_hop",
]

FAMILY_LABELS = {
    "arithmetic": "Arithmetic",
    "multi_step_arithmetic": "Multi-step\narithmetic",
    "syllogistic": "Syllogistic",
    "relational": "Relational",
    "set_inclusion": "Set\ninclusion",
    "contradiction": "Contradiction",
    "variable_chain": "Variable\nchain",
    "multi_hop": "Multi-hop",
}


def load_json(name: str) -> dict:
    path = next((base / name for base in [ROOT / "archive", ROOT / "archive/legacy_experiments"]
                 if (base / name).exists()), None)
    if path is None:
        raise FileNotFoundError(name)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_first_available(*names: str) -> dict:
    for name in names:
        if any((base / name).exists() for base in [ROOT / "archive", ROOT / "archive/legacy_experiments"]):
            return load_json(name)
    raise FileNotFoundError(f"None of these result files exist: {', '.join(names)}")


def save_table(df: pd.DataFrame, name: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / name, index=False)


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.replace("_", "\n"))


def ordered_family_frame(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["family"] = pd.Categorical(df["family"], FAMILY_ORDER, ordered=True)
    return df.sort_values("family").reset_index(drop=True)


def layer19_subspace_results() -> pd.DataFrame:
    rows = []
    raw = {
        0.17: {
            "directed": (3, 60),
            "in_subspace": (17, 300),
            "rand_subspace": (17, 300),
            "complement": (17, 300),
            "full_random": (19, 300),
        },
        0.34: {
            "directed": (4, 60),
            "in_subspace": (19, 300),
            "rand_subspace": (18, 300),
            "complement": (19, 300),
            "full_random": (17, 300),
        },
        0.86: {
            "directed": (4, 60),
            "in_subspace": (20, 300),
            "rand_subspace": (18, 300),
            "complement": (19, 300),
            "full_random": (20, 300),
        },
    }
    for magnitude, conditions in raw.items():
        for condition, (flips, n) in conditions.items():
            low, high = wilson_interval(flips, n)
            rows.append(
                {
                    "layer": 19,
                    "hook_mode": "persistent",
                    "magnitude": magnitude,
                    "condition": condition,
                    "flips": flips,
                    "n": n,
                    "flip_rate": flips / n,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def subspace_specificity_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": 13,
                "mean_in_subspace_vs_rand_subspace": 0.98,
                "source": "sdq_writeup_astra.md / project context",
                "detail_available": "mean ratio only",
            },
            {
                "layer": 19,
                "mean_in_subspace_vs_rand_subspace": 1.06,
                "source": "recovered layer 19 console table",
                "detail_available": "full condition table",
            },
        ]
    )


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def plot_family_aurocs(commitment: dict, ews: dict) -> None:
    rows = []
    hue_order = ["h_0 linear", "z_0 linear", "Family prior", "Full EWS"]
    probe_specs = [
        ("h_0 linear", commitment["probes"]["h_0_linear"]["family_aurocs"]),
        ("z_0 linear", commitment["probes"]["z_0_linear"]["family_aurocs"]),
        ("Family prior", commitment["probes"]["family_prior"]["family_aurocs"]),
        ("Full EWS", ews.get("family_aurocs", {})),
    ]
    for probe, family_scores in probe_specs:
        for family, auroc in family_scores.items():
            rows.append({"probe": probe, "family": family, "auroc": auroc})
    df = ordered_family_frame(rows)
    save_table(df.assign(family=df["family"].astype(str)), "family_auroc_breakdown.csv")

    fig, ax = plt.subplots(figsize=(14.5, 6.6))
    sns.barplot(
        data=df,
        x="family",
        y="auroc",
        hue="probe",
        hue_order=hue_order,
        palette=[COLORS["navy"], COLORS["sky"], COLORS["gray"], COLORS["orange"]],
        ax=ax,
    )
    ax.axhline(0.5, color=COLORS["dark_gray"], lw=1, ls="--", alpha=0.6)
    ax.axhline(0.8, color=COLORS["dark_gray"], lw=0.8, ls=":", alpha=0.4)
    ax.set_ylim(0.25, 1.03)
    ax.set_xlabel("")
    ax.set_ylabel("Within-family AUROC")
    ax.set_title("Early commitment is not just a task-family prior")
    ax.set_xticks(range(len(FAMILY_ORDER)))
    ax.set_xticklabels([family_label(family) for family in FAMILY_ORDER])
    ax.legend(
        title="",
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        borderaxespad=0.0,
        columnspacing=1.9,
        handletextpad=0.6,
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.27, top=0.90)
    sns.despine(fig)
    save_fig(fig, "family_auroc_breakdown")


def plot_probe_summary(commitment: dict, ews: dict, ews_probe: dict) -> None:
    rows = []
    for row in commitment["comparison"]:
        rows.append({"model": row["name"], "metric": "Pooled", "auroc": row["pooled"]})
        rows.append({"model": row["name"], "metric": "Within-family", "auroc": row["within_family"]})
    rows.extend(
        [
            {"model": "ews_probe_mlp", "metric": "Pooled", "auroc": ews_probe.get("overall_auroc", np.nan)},
            {"model": "ews_probe_mlp", "metric": "Sequence", "auroc": ews_probe.get("sequence_auroc", np.nan)},
            {"model": "ews_gru", "metric": "Sequence", "auroc": ews["summary"]["sequence_auroc"]},
        ]
    )
    df = pd.DataFrame(rows)
    save_table(df, "probe_summary_aurocs.csv")

    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    plot_df = df.dropna().copy()
    label_map = {
        "h_0_linear": "h0 linear",
        "z_0_linear": "z0 linear",
        "family_prior": "family prior",
        "ews_probe_mlp": "EWS probe MLP",
        "ews_gru": "EWS GRU",
    }
    plot_df["label"] = plot_df["model"].map(label_map).fillna(plot_df["model"]) + "\n" + plot_df["metric"]
    palette = [
        COLORS["navy"] if "h0 linear" in label else
        COLORS["blue"] if "z0 linear" in label else
        COLORS["gray"] if "family prior" in label else
        COLORS["orange"] if "EWS GRU" in label else
        COLORS["green"]
        for label in plot_df["label"]
    ]
    sns.barplot(data=plot_df, y="label", x="auroc", palette=palette, hue="label", legend=False, ax=ax)
    ax.axvline(0.5, color=COLORS["dark_gray"], lw=1, ls="--", alpha=0.5)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("")
    ax.set_title("Probe and sequence model headline comparison")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=4, fontsize=9)
    sns.despine(fig)
    save_fig(fig, "probe_summary_aurocs")


def plot_family_balance(commitment: dict) -> None:
    rows = []
    h0_scores = commitment["probes"]["h_0_linear"]["family_aurocs"]
    for family, counts in commitment["meta"]["family_breakdown"].items():
        total = counts["correct"] + counts["incorrect"]
        rows.append(
            {
                "family": family,
                "correct": counts["correct"],
                "incorrect": counts["incorrect"],
                "total": total,
                "accuracy": counts["correct"] / total,
                "failure_rate": counts["incorrect"] / total,
                "h0_auroc": h0_scores[family],
            }
        )
    df = ordered_family_frame(rows)
    save_table(df.assign(family=df["family"].astype(str)), "family_balance_and_h0_auroc.csv")

    fig, ax = plt.subplots(figsize=(10.5, 6))
    sizes = 60 + 4.5 * df["total"]
    sc = ax.scatter(
        df["accuracy"],
        df["h0_auroc"],
        s=sizes,
        c=df["failure_rate"],
        cmap="mako_r",
        edgecolor="white",
        linewidth=1.5,
    )
    for _, row in df.iterrows():
        ax.annotate(
            family_label(str(row["family"])).replace("\n", " "),
            (row["accuracy"], row["h0_auroc"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.axhline(0.5, color="#333333", lw=1, ls="--", alpha=0.45)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Observed model accuracy by family")
    ax.set_ylabel("h_0 probe AUROC within family")
    ax.set_title("Probe strength is high even in families with very different base rates")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Failure rate")
    sns.despine(fig)
    save_fig(fig, "family_balance_vs_h0_auroc")


def plot_subspace(commitment: dict) -> None:
    subspace = commitment["subspace"]
    ranks = sorted(int(k) for k in subspace["rank_aurocs"].keys())
    df = pd.DataFrame(
        {
            "rank": ranks,
            "pooled_auroc": [subspace["rank_aurocs"][str(k)] for k in ranks],
            "within_family_auroc": [subspace["rank_within_family_aurocs"][str(k)] for k in ranks],
        }
    )
    save_table(df, "subspace_rank_sweep.csv")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df["rank"], df["pooled_auroc"], marker="o", lw=2.4, label="Pooled")
    ax.plot(df["rank"], df["within_family_auroc"], marker="o", lw=2.4, label="Within-family")
    ax.axhline(subspace["full_auroc"], color="#31588a", ls="--", lw=1, alpha=0.55, label="Full h_0")
    ax.axvline(subspace["rank_for_90pct"], color="#444444", ls=":", lw=1)
    ax.axvline(subspace["rank_for_95pct"], color="#444444", ls=":", lw=1)
    ax.text(subspace["rank_for_90pct"], 0.53, "90% lift\nrank 64", ha="right", va="bottom", fontsize=9)
    ax.text(subspace["rank_for_95pct"], 0.53, "95% lift\nrank 96", ha="left", va="bottom", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0.5, 0.94)
    ax.set_xlabel("PCA subspace rank")
    ax.set_ylabel("AUROC")
    ax.set_title("Commitment signal is mesoscale: useful rank is about 64-96 dimensions")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(fig)
    save_fig(fig, "subspace_rank_sweep")


def plot_cross_family_transfer(commitment: dict) -> None:
    df = pd.DataFrame(commitment["cross_family_transfer"]).rename(columns={"held_out_family": "family"})
    df["family"] = pd.Categorical(df["family"], FAMILY_ORDER, ordered=True)
    df = df.sort_values("family").reset_index(drop=True)
    save_table(df.assign(family=df["family"].astype(str)), "cross_family_transfer.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = np.where(df["auroc"] >= 0.55, "#31588a", "#b8b8b8")
    ax.bar(df["family"].astype(str), df["auroc"], color=colors)
    ax.axhline(0.5, color="#333333", lw=1, ls="--", alpha=0.55)
    ax.axhline(0.55, color="#333333", lw=0.9, ls=":", alpha=0.45)
    ax.set_ylim(0.3, 0.9)
    ax.set_xlabel("")
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Cross-family transfer: 4 of 8 held-out families above the 0.55 signal gate")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([family_label(str(family)) for family in df["family"]])
    for i, row in df.iterrows():
        ax.text(i, row["auroc"] + 0.015, f"{row['auroc']:.2f}", ha="center", fontsize=9)
    sns.despine(fig)
    save_fig(fig, "cross_family_transfer")


def plot_ews_timestep(ews: dict, ews_probe: dict) -> None:
    def curve_frame(source: dict, value_key: str, label: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestep": [int(k) for k in source[value_key].keys()],
                "auroc": [float(v) for v in source[value_key].values()],
                "series": label,
            }
        ).sort_values("timestep")

    df = pd.concat(
        [
            curve_frame(ews, "timestep_aurocs", "Full EWS GRU"),
            curve_frame(ews_probe, "auroc_curve", "Generation-state MLP probe"),
        ],
        ignore_index=True,
    )
    save_table(df, "ews_timestep_auroc_curves.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=df, x="timestep", y="auroc", hue="series", lw=2.2, ax=ax)
    ax.axhline(0.5, color="#333333", lw=1, ls="--", alpha=0.45)
    ax.axhline(0.75, color="#333333", lw=0.8, ls=":", alpha=0.35)
    ax.set_ylim(0.48, 1.01)
    ax.set_xlabel("Generated token timestep")
    ax.set_ylabel("AUROC")
    ax.set_title("Early-warning signal appears immediately and evolves during generation")
    ax.legend(title="", frameon=False)
    sns.despine(fig)
    save_fig(fig, "ews_timestep_auroc")


def plot_flip_rates(intervention: dict) -> None:
    rows = []
    for row in intervention["magnitude_results"]:
        magnitude = row["magnitude"]
        for kind in ["directed", "random"]:
            rate = row[f"{kind}_flip_rate"]
            n = row[f"n_{kind}"]
            low, high = wilson_interval(rate * n, n)
            rows.append(
                {
                    "source": "target_layer_25_prefill_json",
                    "layer": intervention["summary"].get("target_layer"),
                    "hook_mode": "prefill",
                    "magnitude": magnitude,
                    "magnitude_std": magnitude / intervention["summary"]["direction_projection_stats"]["std"],
                    "condition": kind,
                    "flip_rate": rate,
                    "n": n,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    # Extracted from the Phase 4 writeup table. The raw JSON artifact retained in
    # this repo is the original layer-25 run; this table preserves the post-fix
    # layer-13 sweep that the writeup calls out as scientifically interesting.
    layer13 = pd.DataFrame(
        [
            {"magnitude": 0.60, "magnitude_std": 0.5, "directed": 0.013, "random": 0.030, "ratio": 0.42},
            {"magnitude": 1.20, "magnitude_std": 1.0, "directed": 0.050, "random": 0.007, "ratio": 6.67},
            {"magnitude": 2.41, "magnitude_std": 2.0, "directed": 0.013, "random": 0.033, "ratio": 0.38},
            {"magnitude": 3.61, "magnitude_std": 3.0, "directed": 0.025, "random": 0.022, "ratio": 1.11},
            {"magnitude": 6.02, "magnitude_std": 5.0, "directed": 0.025, "random": 0.035, "ratio": 0.71},
        ]
    )
    for _, row in layer13.iterrows():
        for kind, n in [("directed", 80), ("random", 400)]:
            low, high = wilson_interval(row[kind] * n, n)
            rows.append(
                {
                    "source": "layer_13_persistent_writeup",
                    "layer": 13,
                    "hook_mode": "persistent",
                    "magnitude": row["magnitude"],
                    "magnitude_std": row["magnitude_std"],
                    "condition": kind,
                    "flip_rate": row[kind],
                    "n": n,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    df = pd.DataFrame(rows)
    save_table(df, "flip_rates_by_magnitude.csv")
    save_table(layer13, "layer13_persistent_flip_ratio_table.csv")

    fig, (ax_rate, ax_ratio) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.8),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    x = np.arange(len(layer13))
    width = 0.36
    ax_rate.bar(
        x - width / 2,
        100 * layer13["directed"],
        width=width,
        color=COLORS["navy"],
        label="Directed",
    )
    ax_rate.bar(
        x + width / 2,
        100 * layer13["random"],
        width=width,
        color=COLORS["gray"],
        label="Random control",
    )
    for i, row in layer13.iterrows():
        ax_rate.text(i - width / 2, 100 * row["directed"] + 0.22, f"{100 * row['directed']:.1f}%", ha="center", fontsize=8.5)
        ax_rate.text(i + width / 2, 100 * row["random"] + 0.22, f"{100 * row['random']:.1f}%", ha="center", fontsize=8.5)
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels([f"{v:g}x" for v in layer13["magnitude_std"]])
    ax_rate.set_ylim(0, 6.2)
    ax_rate.set_xlabel("Perturbation magnitude")
    ax_rate.set_ylabel("Answer flip rate")
    ax_rate.set_title("Layer 13 persistent hook: flip rates")
    ax_rate.legend(frameon=False, loc="upper left")
    ax_rate.grid(axis="y", alpha=0.18)

    ax_ratio.plot(layer13["magnitude_std"], layer13["ratio"], marker="o", color=COLORS["purple"], lw=2.6)
    ax_ratio.scatter([1.0], [6.67], s=130, color=COLORS["red"], zorder=3)
    ax_ratio.axhline(1, color=COLORS["dark_gray"], ls="--", lw=1, alpha=0.55)
    ax_ratio.axhline(2, color=COLORS["dark_gray"], ls=":", lw=1, alpha=0.5)
    ax_ratio.annotate(
        "Only passing point\n6.67x at 1x",
        (1.0, 6.67),
        xytext=(1.55, 6.2),
        arrowprops={"arrowstyle": "->", "color": COLORS["dark_gray"], "lw": 1},
        fontsize=9,
    )
    ax_ratio.set_ylim(0, 7.4)
    ax_ratio.set_xticks(layer13["magnitude_std"])
    ax_ratio.set_xlabel("Perturbation magnitude")
    ax_ratio.set_ylabel("Directed / random ratio")
    ax_ratio.set_title("Directional advantage")
    ax_ratio.grid(axis="y", alpha=0.18)

    fig.suptitle("Rank-1 intervention effect is narrow, not robust", y=1.03, fontsize=15, fontweight="semibold")
    fig.text(
        0.5,
        -0.02,
        "Raw layer-25 prefill artifact is 0% flips at every magnitude; the interpretable post-fix signal is the layer-13 persistent sweep shown here.",
        ha="center",
        fontsize=9,
        color=COLORS["dark_gray"],
    )
    sns.despine(fig)
    save_fig(fig, "flip_rates_by_magnitude")

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.plot(layer13["magnitude_std"], layer13["ratio"], marker="o", color=COLORS["purple"], lw=2.4)
    ax1.axhline(1, color=COLORS["dark_gray"], ls="--", lw=1, alpha=0.55)
    ax1.axhline(2, color=COLORS["dark_gray"], ls=":", lw=1, alpha=0.5)
    ax1.scatter([1.0], [6.67], s=140, color=COLORS["red"], zorder=3)
    ax1.annotate("6.67x at 1x std", (1.0, 6.67), xytext=(14, -8), textcoords="offset points")
    ax1.set_xlabel("Perturbation magnitude (x projection std)")
    ax1.set_ylabel("Directed / random flip ratio")
    ax1.set_ylim(0, 7.4)
    ax1.set_title("The rank-1 causal effect is sharp and non-monotonic")
    sns.despine(fig)
    save_fig(fig, "layer13_flip_ratio")


def plot_layer19_subspace_specificity() -> None:
    df = layer19_subspace_results()
    save_table(df, "layer19_subspace_specificity.csv")

    condition_order = ["directed", "in_subspace", "rand_subspace", "complement", "full_random"]
    labels = {
        "directed": "Directed",
        "in_subspace": "In subspace",
        "rand_subspace": "Random subspace",
        "complement": "Complement",
        "full_random": "Full random",
    }
    palette = {
        "directed": COLORS["navy"],
        "in_subspace": COLORS["purple"],
        "rand_subspace": COLORS["gray"],
        "complement": COLORS["sky"],
        "full_random": COLORS["orange"],
    }

    fig, (ax_rate, ax_ratio) = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.8),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    plot_df = df.copy()
    plot_df["condition_label"] = plot_df["condition"].map(labels)
    plot_df["flip_rate_pct"] = 100 * plot_df["flip_rate"]
    sns.lineplot(
        data=plot_df,
        x="magnitude",
        y="flip_rate_pct",
        hue="condition",
        hue_order=condition_order,
        palette=palette,
        marker="o",
        lw=2.2,
        ax=ax_rate,
    )
    handles, _ = ax_rate.get_legend_handles_labels()
    ax_rate.legend(
        handles,
        [labels[c] for c in condition_order],
        title="",
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=8.5,
    )
    ax_rate.set_ylim(4.2, 7.4)
    ax_rate.set_xticks([0.17, 0.34, 0.86])
    ax_rate.set_xlabel("Perturbation magnitude")
    ax_rate.set_ylabel("Answer flip rate")
    ax_rate.set_title("Layer 19 persistent hook: all conditions cluster near 6%")
    ax_rate.grid(axis="y", alpha=0.18)

    pivot = df.pivot(index="magnitude", columns="condition", values="flip_rate").reset_index()
    ratio_df = pd.DataFrame(
        {
            "magnitude": pivot["magnitude"],
            "in / random subspace": pivot["in_subspace"] / pivot["rand_subspace"],
            "in / complement": pivot["in_subspace"] / pivot["complement"],
            "directed / in": pivot["directed"] / pivot["in_subspace"],
        }
    )
    save_table(ratio_df, "layer19_subspace_specificity_ratios.csv")
    ratio_long = ratio_df.melt(id_vars="magnitude", var_name="comparison", value_name="ratio")
    sns.lineplot(
        data=ratio_long,
        x="magnitude",
        y="ratio",
        hue="comparison",
        marker="o",
        lw=2.2,
        palette=[COLORS["purple"], COLORS["blue"], COLORS["green"]],
        ax=ax_ratio,
    )
    ax_ratio.axhline(1, color=COLORS["dark_gray"], ls="--", lw=1, alpha=0.55)
    ax_ratio.axhline(2, color=COLORS["dark_gray"], ls=":", lw=1, alpha=0.4)
    ax_ratio.set_ylim(0.82, 1.18)
    ax_ratio.set_xticks([0.17, 0.34, 0.86])
    ax_ratio.set_xlabel("Perturbation magnitude")
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_title("Key ratios stay near 1x")
    ax_ratio.legend(title="", frameon=False, fontsize=8.5, loc="upper left")
    ax_ratio.text(
        0.98,
        0.08,
        "Mean in/random\nsubspace ratio: 1.06x",
        transform=ax_ratio.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["light_gray"]},
    )
    ax_ratio.grid(axis="y", alpha=0.18)

    fig.suptitle("Layer 19 supports the negative result: no subspace specificity", y=1.03, fontsize=15, fontweight="semibold")
    sns.despine(fig)
    save_fig(fig, "layer19_subspace_specificity")


def plot_layer13_layer19_subspace_specificity() -> None:
    summary = subspace_specificity_summary()
    layer19 = layer19_subspace_results()
    save_table(summary, "layer13_layer19_subspace_specificity_summary.csv")

    fig, (ax_summary, ax_detail) = plt.subplots(
        1,
        2,
        figsize=(14.0, 5.7),
        gridspec_kw={"width_ratios": [0.9, 1.45]},
    )

    bars = ax_summary.bar(
        summary["layer"].astype(str),
        summary["mean_in_subspace_vs_rand_subspace"],
        color=[COLORS["navy"], COLORS["purple"]],
        width=0.58,
    )
    ax_summary.axhspan(0.9, 1.1, color=COLORS["light_gray"], alpha=0.85, zorder=0)
    ax_summary.axhline(1, color=COLORS["dark_gray"], ls="--", lw=1.1, alpha=0.7)
    ax_summary.axhline(2, color=COLORS["red"], ls=":", lw=1.2, alpha=0.55)
    ax_summary.set_ylim(0, 2.15)
    ax_summary.set_xlabel("Layer")
    ax_summary.set_ylabel("In-subspace / random-subspace")
    ax_summary.set_title("No layer shows privileged subspace causality")
    ax_summary.bar_label(bars, labels=[f"{v:.2f}x" for v in summary["mean_in_subspace_vs_rand_subspace"]], padding=4, fontsize=10)
    ax_summary.text(
        0.04,
        0.93,
        "Null band: roughly 1x",
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color=COLORS["dark_gray"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": COLORS["light_gray"]},
    )
    ax_summary.grid(axis="y", alpha=0.16)

    condition_order = ["in_subspace", "rand_subspace", "complement", "full_random", "directed"]
    labels = {
        "in_subspace": "In subspace",
        "rand_subspace": "Random subspace",
        "complement": "Complement",
        "full_random": "Full random",
        "directed": "Directed",
    }
    palette = {
        "in_subspace": COLORS["purple"],
        "rand_subspace": COLORS["gray"],
        "complement": COLORS["sky"],
        "full_random": COLORS["orange"],
        "directed": COLORS["navy"],
    }
    plot_df = layer19.copy()
    plot_df["flip_rate_pct"] = 100 * plot_df["flip_rate"]
    sns.lineplot(
        data=plot_df,
        x="magnitude",
        y="flip_rate_pct",
        hue="condition",
        hue_order=condition_order,
        palette=palette,
        marker="o",
        lw=2.2,
        ax=ax_detail,
    )
    handles, _ = ax_detail.get_legend_handles_labels()
    ax_detail.legend(
        handles,
        [labels[c] for c in condition_order],
        title="",
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=8.5,
    )
    ax_detail.set_ylim(4.2, 7.4)
    ax_detail.set_xticks([0.17, 0.34, 0.86])
    ax_detail.set_xlabel("Layer 19 perturbation magnitude")
    ax_detail.set_ylabel("Answer flip rate (%)")
    ax_detail.set_title("Layer 19 detail: all conditions cluster near 6%")
    ax_detail.grid(axis="y", alpha=0.18)

    fig.suptitle("Layer 13 and 19 subspace specificity: negative control result", y=1.03, fontsize=15, fontweight="semibold")
    fig.text(
        0.5,
        -0.02,
        "Layer 13 full table was not archived, but the saved mean in/random-subspace ratio is 0.98x; layer 19 detail comes from the recovered console table.",
        ha="center",
        fontsize=9,
        color=COLORS["dark_gray"],
    )
    sns.despine(fig)
    save_fig(fig, "layer13_layer19_subspace_specificity")


def plot_layer_sweep_summary() -> None:
    df = pd.DataFrame(
        [
            {
                "layer": 6,
                "best_directed_flip_rate": 0.075,
                "directional_evidence": "No clean directional edge",
                "best_ratio": "Random logit shift larger",
                "best_magnitude_std": "not archived",
                "summary": "Most flips, but not direction-specific",
            },
            {
                "layer": 13,
                "best_directed_flip_rate": 0.050,
                "directional_evidence": "6.67x directed/random",
                "best_ratio": "6.67x",
                "best_magnitude_std": "1x",
                "summary": "Best narrow directional signal",
            },
            {
                "layer": 19,
                "best_directed_flip_rate": 4 / 60,
                "directional_evidence": "No subspace specificity",
                "best_ratio": "mean in/random = 1.06x",
                "best_magnitude_std": "0.34-0.86",
                "summary": "All conditions cluster near 6%",
            },
            {
                "layer": 25,
                "best_directed_flip_rate": 0.000,
                "directional_evidence": "No effect",
                "best_ratio": "0.00x",
                "best_magnitude_std": "all tested",
                "summary": "Late residual stream is frozen",
            },
        ]
    )
    save_table(df, "layer_sweep_summary_from_writeup.csv")

    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    plot_df = df.copy()
    known = plot_df.dropna(subset=["best_directed_flip_rate"])
    ax.plot(known["layer"], 100 * known["best_directed_flip_rate"], color=COLORS["blue"], lw=2.2, alpha=0.8)
    point_colors = [COLORS["blue"], COLORS["navy"], COLORS["purple"], COLORS["gray"]]
    ax.scatter(
        known["layer"],
        100 * known["best_directed_flip_rate"],
        s=180,
        color=point_colors[: len(known)],
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )
    for _, row in known.iterrows():
        ax.text(
            row["layer"],
            100 * row["best_directed_flip_rate"] + 0.45,
            f"{100 * row['best_directed_flip_rate']:.1f}%",
            ha="center",
            fontsize=10,
            fontweight="semibold",
        )
    ax.set_xticks([6, 13, 19, 25])
    ax.set_xlim(3.5, 27.5)
    ax.set_ylim(-0.2, 8.8)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Best directed flip rate")
    ax.set_title("Layer sweep: effects are mid-stack and mostly not direction-specific")
    ax.grid(axis="y", alpha=0.18)

    table_rows = []
    for _, row in plot_df.iterrows():
        rate = "not archived" if pd.isna(row["best_directed_flip_rate"]) else f"{100 * row['best_directed_flip_rate']:.1f}%"
        table_rows.append([f"L{int(row['layer'])}", rate, row["directional_evidence"], row["summary"]])
    table = ax.table(
        cellText=table_rows,
        colLabels=["Layer", "Best directed flips", "Directional evidence", "Interpretation"],
        cellLoc="left",
        colLoc="left",
        loc="bottom",
        bbox=[0.0, -0.58, 1.0, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor(COLORS["navy"])
            cell.set_text_props(color="white", weight="semibold")
        else:
            cell.set_facecolor(COLORS["light_gray"] if r % 2 else "white")
    fig.subplots_adjust(bottom=0.38)
    sns.despine(fig)
    save_fig(fig, "layer_sweep_summary")


def plot_pca_spectrum(commitment: dict) -> None:
    vals = np.array(commitment["subspace"]["pca_explained_variance_top20"], dtype=float)
    df = pd.DataFrame(
        {
            "component": np.arange(1, len(vals) + 1),
            "explained_variance": vals,
            "share_top20": vals / vals.sum(),
            "cumulative_share_top20": np.cumsum(vals) / vals.sum(),
        }
    )
    save_table(df, "pca_explained_variance_top20.csv")

    fig, ax1 = plt.subplots(figsize=(10.5, 5.6))
    ax1.bar(df["component"], df["share_top20"], color=COLORS["sky"], label="Individual PC")
    ax1.set_xlabel("PCA component")
    ax1.set_ylabel("Share of top-20 variance only")
    ax2 = ax1.twinx()
    ax2.plot(df["component"], df["cumulative_share_top20"], color=COLORS["navy"], marker="o", lw=2.0, label="Cumulative")
    ax2.set_ylabel("Cumulative share of top-20 variance")
    ax1.set_title("PCA spectrum: variance is front-loaded, but prediction is not rank-1")
    ax1.set_xticks(df["component"])
    ax1.text(
        0.98,
        0.78,
        "Takeaway:\nPC1 dominates raw variance,\nbut AUROC needs a much\nlarger 64-96D subspace.",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": COLORS["light_gray"]},
    )
    ax1.text(
        0.02,
        -0.19,
        "This is a scree plot over the top 20 PCs, not the full 2304D variance budget.",
        transform=ax1.transAxes,
        fontsize=9,
        color=COLORS["dark_gray"],
    )
    sns.despine(fig, right=False)
    save_fig(fig, "pca_spectrum_top20")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D5DAE0",
            "grid.color": "#D9DEE6",
            "grid.linewidth": 0.8,
            "axes.titleweight": "semibold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    commitment = load_json("commitment_results.json")
    intervention = load_json("commitment_intervention_results.json")
    ews = load_first_available("ews_results2.json", "ews_results.json")
    ews_probe = load_first_available("ews_probe_results2.json", "ews_probe_results.json")

    plot_family_aurocs(commitment, ews)
    plot_probe_summary(commitment, ews, ews_probe)
    plot_family_balance(commitment)
    plot_subspace(commitment)
    plot_cross_family_transfer(commitment)
    plot_ews_timestep(ews, ews_probe)
    plot_flip_rates(intervention)
    plot_layer19_subspace_specificity()
    plot_layer13_layer19_subspace_specificity()
    plot_layer_sweep_summary()
    plot_pca_spectrum(commitment)

    print(f"Wrote visualizations to {FIG_DIR.relative_to(ROOT)}")
    print(f"Wrote source tables to {TABLE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
