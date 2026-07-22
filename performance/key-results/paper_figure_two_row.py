#!/usr/bin/env python3
"""
Paper figure: 2-row all-metrics comparison.
Top row = Gemma, bottom row = Nemotron.
Four panels per row, one per F1 metric.
Bars are grouped by condition (All Tools / SPARQL / BSchema).
"""

from __future__ import annotations

import csv
import statistics
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent

GEMMA_FILES = {
    "All Tools": HERE / "gemma_all_tools_unknown_model_20260415_172220.csv",
    "SPARQL":    HERE / "gemma-sparql_unknown_model_20260415_230150.csv",
    "BSchema":   HERE / "gemma-bschema7_unknown_model_20260415_203102.csv",
}

NEMOTRON_FILES = {
    "All Tools": HERE / "nemotron-super-all-tools_unknown_model_20260511_235503.csv",
    "SPARQL":    HERE / "nemotron-super-sparql_unknown_model_20260512_223748.csv",
    "BSchema":   HERE / "nemotron-super-bschema7_unknown_model_20260512_114006.csv",
}

# ── Metrics ──────────────────────────────────────────────────────────────────
F1_METRICS = [
    "arity_matching_f1",
    "exact_match_f1",
    "row_matching_f1",
    "entity_set_f1",
]
METRIC_TITLES = [
    "Arity Matching F1",
    "Exact Match F1",
    "Row Matching F1",
    "Entity Set F1",
]

BUILDING_ORDER = ["DFLEXLIBS", "LBNL", "MORTAR", "TUC"]

# ── Colour palette (muted, colorblind-safe) ──────────────────────────────────
CONDITION_COLORS = {
    "All Tools": "#5B8DB8",  # steel blue
    "BSchema":   "#6AAB7A",  # sage green
    "SPARQL":    "#C47B6A",  # terracotta
}
CONDITION_ORDER = ["All Tools", "BSchema", "SPARQL"]

# ── I/O helpers ──────────────────────────────────────────────────────────────
def _load(path: Path) -> list[dict]:
    csv.field_size_limit(10 * 1024 * 1024)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(val, default=0.0) -> float:
    try:
        return float(val) if val and val.strip() else default
    except (ValueError, AttributeError):
        return default


# ── Metric computation ───────────────────────────────────────────────────────
def building_means(data: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {building: {metric: (mean, ci)}} for all F1 metrics."""
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: {m: [] for m in F1_METRICS})
    for row in data:
        qid = row.get("query_id", "")
        building = qid.split("_")[0] if "_" in qid else qid
        for m in F1_METRICS:
            buckets[building][m].append(_to_float(row.get(m)))

    result = {}
    for building, vals in buckets.items():
        result[building] = {}
        for m in F1_METRICS:
            v_list = vals[m]
            if not v_list:
                result[building][m] = (0.0, 0.0)
                continue
            
            mean = statistics.mean(v_list)
            sem = 0.0
            if len(v_list) >= 2:
                stdev = statistics.stdev(v_list)
                sem = stdev / math.sqrt(len(v_list))
            result[building][m] = (mean, sem)
    return result


def load_row(files: dict[str, Path]) -> dict[str, dict[str, dict[str, float]]]:
    """Return {condition: {building: {metric: mean}}}."""
    return {cond: building_means(_load(path)) for cond, path in files.items()}


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_row(
    ax_row: list,
    row_data: dict[str, dict[str, dict[str, float]]],
    buildings: list[str],
    show_titles: bool,
    show_xlabel: bool,
    show_legend: bool,
    show_error_bars: bool = False,
):
    """Fill one horizontal row of axes."""
    n_conditions = len(CONDITION_ORDER)
    x = np.arange(len(buildings))
    width = 0.72 / n_conditions

    for col_idx, (metric, title) in enumerate(zip(F1_METRICS, METRIC_TITLES)):
        ax = ax_row[col_idx]

        for bar_idx, cond in enumerate(CONDITION_ORDER):
            data_points = [
                row_data[cond].get(b, {}).get(metric, (0.0, 0.0))
                for b in buildings
            ]
            means = [p[0] for p in data_points]
            sems = [p[1] for p in data_points]
            
            # Clip upper error bars at 1.0
            lower_err = sems
            upper_err = [min(s, 1.0 - m) if m < 1.0 else 0.0 for m, s in zip(means, sems)]
            asym_err = [lower_err, upper_err]
            
            offset = width * (bar_idx - n_conditions / 2 + 0.5)
            err_kwargs = dict(yerr=asym_err, capsize=3, ecolor="#666666", error_kw={"elinewidth": 1.4}) if show_error_bars else {}
            ax.bar(
                x + offset, means, width,
                **err_kwargs,
                color=CONDITION_COLORS[cond],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.92,
                zorder=3,
            )

            # Draw a thin stub at y=0 for explicitly zero bars
            for i, m in enumerate(means):
                if m == 0.0:
                    ax.bar(
                        x[i] + offset, 0.012, width,
                        color=CONDITION_COLORS[cond],
                        edgecolor="none",
                        alpha=0.85,
                        zorder=4,
                    )

        ax.set_xlim(-0.55, len(buildings) - 0.45)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=35, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if show_titles:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=8)

        if col_idx == 0:
            ax.set_ylabel("F1 Score", fontsize=12)
        else:
            ax.set_yticklabels([])

        if show_xlabel:
            ax.set_xlabel("Building", fontsize=12)

    if show_legend:
        handles = [
            mpatches.Patch(color=CONDITION_COLORS[c], label=c)
            for c in CONDITION_ORDER
        ]
        ax_row[0].legend(
            handles=handles,
            fontsize=11,
            framealpha=0.9,
            edgecolor="#cccccc",
            loc="upper left",
        )


def main():
    gemma_data    = load_row(GEMMA_FILES)
    nemotron_data = load_row(NEMOTRON_FILES)

    # Collect buildings present in any file
    all_buildings: set[str] = set()
    for row_data in (gemma_data, nemotron_data):
        for cond_data in row_data.values():
            all_buildings.update(cond_data.keys())
    buildings = [b for b in BUILDING_ORDER if b in all_buildings]

    # ── Figure layout ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 4,
        figsize=(22, 9),
        gridspec_kw={"hspace": 0.42, "wspace": 0.06},
    )

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.facecolor": "#f8f8f8",
        "figure.facecolor": "white",
    })

    show_error_bars = False
    plot_row(axes[0], gemma_data,    buildings, show_titles=True,  show_xlabel=False, show_legend=False, show_error_bars=show_error_bars)
    plot_row(axes[1], nemotron_data, buildings, show_titles=False, show_xlabel=True,  show_legend=True,  show_error_bars=show_error_bars)

    # Row labels on the left margin
    for ax_row, label in zip(axes, ["Gemma", "Nemotron Super"]):
        ax_row[0].annotate(
            label,
            xy=(-0.28, 0.5), xycoords="axes fraction",
            fontsize=15, fontweight="bold",
            ha="center", va="center",
            rotation=90,
        )

    out = HERE / "paper_two_row_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
