#!/usr/bin/env python3
"""
Visualize Performance by Question Source

Merges a result CSV with a reference CSV on the 'question' column to obtain
the question source (human, LLM_1, …, LLM_5), then plots mean F1 scores
broken down by source.
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

csv.field_size_limit(sys.maxsize)

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})

F1_METRICS = [
    ("row_matching_f1",   "Row Matching F1"),
    ("entity_set_f1",     "Entity Set F1"),
    ("exact_match_f1",    "Exact Match F1"),
    ("arity_matching_f1", "Arity Matching F1"),
]

SOURCE_ORDER = ["human", "LLM_1", "LLM_2", "LLM_3", "LLM_4", "LLM_5"]

SOURCE_COLORS = {
    "human": "#ff7f0e",
    "LLM_1": "#1f77b4",
    "LLM_2": "#4a9fd4",
    "LLM_3": "#6ab7e8",
    "LLM_4": "#90cef5",
    "LLM_5": "#b3deff",
}


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def load_csv(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(val, default=0.0) -> float:
    try:
        return float(val) if val and str(val).strip() else default
    except (ValueError, AttributeError):
        return default


# ----------------------------------------------------------------------
# Merge + group
# ----------------------------------------------------------------------

def build_source_map(reference_rows: List[dict]) -> Dict[str, str]:
    """Map question text → source label from the reference CSV."""
    return {row["question"].strip(): row["source"].strip()
            for row in reference_rows
            if row.get("question") and row.get("source")}


def compute_source_metrics(result_rows: List[dict],
                           source_map: Dict[str, str]) -> Dict[str, dict]:
    """
    Group F1 scores by question source.

    Returns:
        { source: { metric: [scores] } }
    """
    by_source: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    unmatched = 0

    for row in result_rows:
        question = row.get("question", "").strip()
        source = source_map.get(question)
        if source is None:
            unmatched += 1
            continue
        for metric, _ in F1_METRICS:
            by_source[source][metric].append(to_float(row.get(metric)))

    if unmatched:
        print(f"  Warning: {unmatched} rows had no matching source in reference")

    # Compute mean ± SEM per source
    stats: dict[str, dict] = {}
    for source, metric_scores in by_source.items():
        stats[source] = {}
        for metric, _ in F1_METRICS:
            scores = metric_scores[metric]
            n = len(scores)
            mean = statistics.mean(scores) if n else 0.0
            sem = (statistics.stdev(scores) / n ** 0.5) if n > 1 else 0.0
            stats[source][metric] = {"mean": mean, "sem": sem, "n": n}

    return stats


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def plot_by_source(source_stats: Dict[str, dict],
                   output_path: str = "question_source_f1.png",
                   title_suffix: str = "") -> None:
    """4-panel bar chart: one panel per F1 metric, bars = question sources."""

    sources = [s for s in SOURCE_ORDER if s in source_stats]
    if not sources:
        print("No matching sources to plot.")
        return

    n_sources = len(sources)
    x = np.arange(n_sources)
    colors = [SOURCE_COLORS.get(s, "#cccccc") for s in sources]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, (metric, title) in zip(axes, F1_METRICS):
        means = [source_stats[s][metric]["mean"] for s in sources]
        sems  = [source_stats[s][metric]["sem"]  for s in sources]
        ns    = [source_stats[s][metric]["n"]    for s in sources]

        bars = ax.bar(x, means, color=colors, edgecolor="black",
                      linewidth=0.5, alpha=0.85, width=0.6)
        ax.errorbar(x, means, yerr=sems,
                    fmt="none", ecolor="black", capsize=3,
                    capthick=0.8, linewidth=0.8)

        # sample-size annotation
        for xi, (mean, n) in enumerate(zip(means, ns)):
            ax.text(xi, mean + 0.01, f"n={n}", ha="center",
                    va="bottom", fontsize=6.5, color="#444444")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean F1 Score")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.axhline(y=np.mean(means), color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.6, label="overall mean")

    if title_suffix:
        fig.suptitle(f"Performance by Question Source — {title_suffix}",
                     fontsize=12, y=1.02)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Question-source plot saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot F1 scores broken down by question source"
    )
    parser.add_argument("csv_file",
                        help="Result CSV file to analyse")
    parser.add_argument("--reference-csv", required=True,
                        help="Reference CSV containing question→source mapping")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save output plots")
    args = parser.parse_args()

    stem = Path(args.csv_file).stem
    idx = stem.find("_unknown")
    label = stem[:idx] if idx != -1 else stem

    print(f"Loading result CSV: {args.csv_file}")
    result_rows = load_csv(args.csv_file)
    print(f"  {len(result_rows)} rows")

    print(f"Loading reference CSV: {args.reference_csv}")
    ref_rows = load_csv(args.reference_csv)
    source_map = build_source_map(ref_rows)
    print(f"  {len(source_map)} question→source mappings")

    source_stats = compute_source_metrics(result_rows, source_map)

    matched_sources = [s for s in SOURCE_ORDER if s in source_stats]
    for src in matched_sources:
        parts = []
        for metric, _ in F1_METRICS:
            m = source_stats[src][metric]["mean"]
            parts.append(f"{metric}={m:.3f}")
        print(f"  {src}: " + ", ".join(parts))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_by_source(
        source_stats,
        output_path=str(output_dir / "question_source_f1.png"),
        title_suffix=label,
    )


if __name__ == "__main__":
    main()
