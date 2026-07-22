#!/usr/bin/env python3
"""
Paper figure: 2-row token usage comparison.
Top row = Gemma, bottom row = Nemotron.
Three panels per row: prompt, completion, and total tokens.
Bars grouped by condition (All Tools / BSchema / SPARQL).

Rows where prompt=completion=0 but total>0 are caused by the agent hitting
the token limit mid-run (logging bug). These rows are excluded from the
prompt and completion means (their totals are still correct, so they remain
in the total panel).
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
    "BSchema":   HERE / "gemma-bschema7_unknown_model_20260415_203102.csv",
    "SPARQL":    HERE / "gemma-sparql_unknown_model_20260415_230150.csv",
}

NEMOTRON_FILES = {
    "All Tools": HERE / "nemotron-super-all-tools_unknown_model_20260511_235503.csv",
    "BSchema":   HERE / "nemotron-super-bschema7_unknown_model_20260512_114006.csv",
    "SPARQL":    HERE / "nemotron-super-sparql_unknown_model_20260512_223748.csv",
}

TOKEN_COLS   = ["prompt_tokens", "completion_tokens"]
TOKEN_TITLES = ["Prompt Tokens", "Completion Tokens"]

BUILDING_ORDER = ["DFLEXLIBS", "LBNL", "MORTAR", "TUC"]

CONDITION_COLORS = {
    "All Tools": "#5B8DB8",
    "BSchema":   "#6AAB7A",
    "SPARQL":    "#C47B6A",
}
CONDITION_ORDER = ["All Tools", "BSchema", "SPARQL"]

STUB_HEIGHT_FRAC = 0.018  # fraction of final ylim used for zero-value stub bars

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


def _is_bad_row(row: dict) -> bool:
    """True when the token limit was exceeded and the prompt/completion split wasn't logged."""
    return (
        _to_float(row.get("prompt_tokens")) == 0
        and _to_float(row.get("completion_tokens")) == 0
        and _to_float(row.get("total_tokens")) > 0
    )


# ── Metric computation ───────────────────────────────────────────────────────
def building_token_means(data: list) -> dict:
    """
    Return {building: {token_col: mean_in_thousands}}.

    prompt_tokens / completion_tokens: bad rows (limit-exceeded) excluded.
    total_tokens: all rows included (total is correctly logged even on limit).
    """
    buckets_all  = defaultdict(lambda: {t: [] for t in TOKEN_COLS})
    buckets_good = defaultdict(lambda: {t: [] for t in TOKEN_COLS})

    for row in data:
        qid = row.get("query_id", "")
        building = qid.split("_")[0] if "_" in qid else qid
        bad = _is_bad_row(row)
        for t in TOKEN_COLS:
            v = _to_float(row.get(t))
            buckets_all[building][t].append(v)
            if not bad:
                buckets_good[building][t].append(v)

    result = {}
    for building in buckets_all:
        result[building] = {}
        for t in TOKEN_COLS:
            if t == "total_tokens":
                vals = buckets_all[building][t]
            else:
                vals = buckets_good[building][t]
            
            if not vals:
                result[building][t] = (0.0, 0.0)
                continue

            mean = statistics.mean(vals) / 1_000
            sem = 0.0
            if len(vals) >= 2:
                stdev = statistics.stdev(vals)
                sem = stdev / math.sqrt(len(vals)) / 1_000
            result[building][t] = (mean, sem)

    return result


def load_row(files: dict) -> dict:
    """Return {condition: {building: {token_col: mean_in_thousands}}}."""
    return {cond: building_token_means(_load(path)) for cond, path in files.items()}


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_row(
    ax_row: list,
    row_data: dict,
    buildings: list,
    show_titles: bool,
    show_xlabel: bool,
    show_legend: bool,
    show_error_bars: bool = False,
) -> dict:
    """
    Draw bars for one model row.  Returns zero_positions so stubs can be
    added later (after axis alignment) with the correct ylim.
    """
    n_conditions = len(CONDITION_ORDER)
    x = np.arange(len(buildings))
    width = 0.72 / n_conditions

    # zeros[(col_idx, bar_idx, building_idx)] = x_centre for stub
    zeros: dict = {}

    for col_idx, (token_col, title) in enumerate(zip(TOKEN_COLS, TOKEN_TITLES)):
        ax = ax_row[col_idx]

        for bar_idx, cond in enumerate(CONDITION_ORDER):
            data_points = [
                row_data[cond].get(b, {}).get(token_col, (0.0, 0.0))
                for b in buildings
            ]
            means = [p[0] for p in data_points]
            sems = [p[1] for p in data_points]
            offset = width * (bar_idx - n_conditions / 2 + 0.5)
            err_kwargs = dict(yerr=sems, capsize=3, ecolor="#666666", error_kw={"elinewidth": 1.4}) if show_error_bars else {}
            ax.bar(
                x + offset, means, width,
                **err_kwargs,
                color=CONDITION_COLORS[cond],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.92,
                zorder=3,
            )
            for i, m in enumerate(means):
                if m == 0.0:
                    zeros[(col_idx, bar_idx, i)] = (x[i] + offset, cond)

        ax.set_xlim(-0.55, len(buildings) - 0.45)
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=35, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if show_titles:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=8)

        # Each column has its own scale, so show y labels on every panel
        ax.set_ylabel("Mean Tokens (thousands)", fontsize=11)

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

    return zeros


def draw_stubs(ax_row: list, zeros: dict, width: float) -> None:
    """Draw zero-value indicator stubs using the final (aligned) ylim."""
    for (col_idx, bar_idx, _), (x_pos, cond) in zeros.items():
        ax = ax_row[col_idx]
        stub_h = ax.get_ylim()[1] * STUB_HEIGHT_FRAC
        ax.bar(
            x_pos, stub_h, width,
            color=CONDITION_COLORS[cond],
            edgecolor="none",
            alpha=0.85,
            zorder=4,
        )


def main():
    gemma_data    = load_row(GEMMA_FILES)
    nemotron_data = load_row(NEMOTRON_FILES)

    all_buildings: set = set()
    for row_data in (gemma_data, nemotron_data):
        for cond_data in row_data.values():
            all_buildings.update(cond_data.keys())
    buildings = [b for b in BUILDING_ORDER if b in all_buildings]

    n_conditions = len(CONDITION_ORDER)
    bar_width = 0.72 / n_conditions

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.facecolor": "#f8f8f8",
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        gridspec_kw={"hspace": 0.42, "wspace": 0.32},
    )

    show_error_bars = False
    zeros_gemma    = plot_row(axes[0], gemma_data,    buildings, show_titles=True,  show_xlabel=False, show_legend=False, show_error_bars=show_error_bars)
    zeros_nemotron = plot_row(axes[1], nemotron_data, buildings, show_titles=False, show_xlabel=True,  show_legend=True,  show_error_bars=show_error_bars)

    # Align y-axes within each column across rows, then draw stubs with correct height
    for col_idx in range(2):
        ymax = max(axes[0][col_idx].get_ylim()[1], axes[1][col_idx].get_ylim()[1])
        for row_idx in range(2):
            axes[row_idx][col_idx].set_ylim(0, ymax)

    draw_stubs(axes[0], zeros_gemma,    bar_width)
    draw_stubs(axes[1], zeros_nemotron, bar_width)

    # Row labels
    for ax_row, label in zip(axes, ["Gemma", "Nemotron (Super)"]):
        ax_row[0].annotate(
            label,
            xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=15, fontweight="bold",
            ha="center", va="center",
            rotation=90,
        )

    out = HERE / "paper_token_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
