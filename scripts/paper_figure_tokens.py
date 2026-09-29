#!/usr/bin/env python3
"""
Paper figure: token usage comparison (saved as figs/tokens.png).
Gemma panel on the left, Nemotron panel on the right.
Each bar is a single stacked column per (building, condition): a solid
segment for prompt tokens with a hatched/shaded segment stacked on top for
completion tokens, so the total bar height reads as total tokens per query.
Both panels share the same absolute y-axis (0-275k tokens) for direct
comparison.
"""

from __future__ import annotations

import csv
import statistics
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
RESULTS = HERE.parent / "results" / "paper-run-3"

GEMMA_FILES = {
    "Tool-based Method":      RESULTS / "gemma-all-zeroshot.csv",
    "BSchema":                RESULTS / "gemma-bschema7-zeroshot.csv",
    "SPARQL Validation Only": RESULTS / "gemma-sparql-zeroshot.csv",
}

NEMOTRON_FILES = {
    "Tool-based Method":      RESULTS / "nemotron-all-zeroshot.csv",
    "BSchema":                RESULTS / "nemotron-bschema7-zeroshot.csv",
    "SPARQL Validation Only": RESULTS / "nemotron-sparql-zeroshot.csv",
}

TOKEN_COLS = ["prompt_tokens", "completion_tokens"]

BUILDING_ORDER = ["DFLEXLIBS", "LBNL", "MORTAR", "TUC"]

# ── Colour palette (matches figs/f1-scores.png; Okabe-Ito colorblind-safe) ──
CONDITION_COLORS = {
    "Tool-based Method":      "#0072B2",  # blue
    "BSchema":                "#009E73",  # bluish green
    "SPARQL Validation Only": "#D55E00",  # vermillion
}
CONDITION_ORDER = ["Tool-based Method", "BSchema", "SPARQL Validation Only"]

# ── Shared absolute y-axis (thousands of tokens) ─────────────────────────────
Y_MAX = 275

# ── Font sizes (uniform across titles, axis text, and ticks) ────────────────
FONT_TITLE = 19
FONT_AXIS_LABEL = 19
FONT_TICK = 17
FONT_LEGEND = 17
FONT_ROW_LABEL = 19

# ── I/O helpers ──────────────────────────────────────────────────────────────
def _load(path: Path) -> list:
    csv.field_size_limit(10 * 1024 * 1024)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(val, default=0.0) -> float:
    try:
        return float(val) if val and val.strip() else default
    except (ValueError, AttributeError):
        return default


# ── Metric computation ───────────────────────────────────────────────────────
def building_token_means(data: list) -> dict:
    """Return {building: {token_col: mean_in_thousands}}."""
    buckets = defaultdict(lambda: {t: [] for t in TOKEN_COLS})
    for row in data:
        qid = row.get("query_id", "")
        building = qid.split("_")[0] if "_" in qid else qid
        for t in TOKEN_COLS:
            buckets[building][t].append(_to_float(row.get(t)))

    result = {}
    for building, vals in buckets.items():
        result[building] = {}
        for t in TOKEN_COLS:
            v_list = vals[t]
            if not v_list:
                result[building][t] = 0.0
                continue
            result[building][t] = statistics.mean(v_list) / 1_000
    return result


def load_panel(files: dict) -> dict:
    """Return {condition: {building: {token_col: mean_in_thousands}}}."""
    return {cond: building_token_means(_load(path)) for cond, path in files.items()}


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_panel(
    ax,
    panel_data: dict,
    buildings: list,
    title: str,
    show_ylabel: bool,
):
    n_conditions = len(CONDITION_ORDER)
    x = np.arange(len(buildings))
    width = 0.72 / n_conditions

    for bar_idx, cond in enumerate(CONDITION_ORDER):
        prompt_means = [panel_data[cond].get(b, {}).get("prompt_tokens", 0.0) for b in buildings]
        completion_means = [panel_data[cond].get(b, {}).get("completion_tokens", 0.0) for b in buildings]
        offset = width * (bar_idx - n_conditions / 2 + 0.5)
        color = CONDITION_COLORS[cond]

        # Prompt tokens: solid base segment
        ax.bar(
            x + offset, prompt_means, width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.96,
            zorder=3,
        )
        # Completion tokens: shaded/hatched segment stacked on top
        ax.bar(
            x + offset, completion_means, width,
            bottom=prompt_means,
            color=color,
            alpha=0.55,
            hatch="////",
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_xlim(-0.55, len(buildings) - 0.45)
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks(x)
    ax.set_xticklabels(buildings, rotation=35, ha="right", fontsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.set_xlabel("Building", fontsize=FONT_AXIS_LABEL)
    if show_ylabel:
        ax.set_ylabel("Mean Tokens per Query (thousands)", fontsize=FONT_AXIS_LABEL)


def main():
    gemma_data    = load_panel(GEMMA_FILES)
    nemotron_data = load_panel(NEMOTRON_FILES)

    all_buildings: set = set()
    for panel_data in (gemma_data, nemotron_data):
        for cond_data in panel_data.values():
            all_buildings.update(cond_data.keys())
    buildings = [b for b in BUILDING_ORDER if b in all_buildings]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.facecolor": "#f8f8f8",
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"wspace": 0.1})

    plot_panel(axes[0], gemma_data,    buildings, "Gemma 4 31B Thinking", show_ylabel=True)
    plot_panel(axes[1], nemotron_data, buildings, "Nemotron Super 3",     show_ylabel=False)
    axes[1].set_yticklabels([])

    # ── Legend, placed above the panels so it never covers bars ────────────
    condition_handles = [
        mpatches.Patch(color=CONDITION_COLORS[c], label=c, alpha=0.96)
        for c in CONDITION_ORDER
    ]
    segment_handles = [
        mpatches.Patch(facecolor="#888888", edgecolor="white", alpha=0.96, label="Prompt tokens"),
        mpatches.Patch(facecolor="#888888", edgecolor="white", alpha=0.55, hatch="////", label="Completion tokens"),
    ]
    fig.legend(
        handles=condition_handles + segment_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=5,
        fontsize=FONT_LEGEND,
        framealpha=0.9,
        edgecolor="#cccccc",
        handletextpad=0.6,
        columnspacing=1.4,
    )

    out = HERE.parent / "figs" / "tokens.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
