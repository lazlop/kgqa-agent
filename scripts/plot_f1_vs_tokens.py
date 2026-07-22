#!/usr/bin/env python3
"""
Plot row_matching_f1 vs total_tokens for a results CSV.

Usage:
    python scripts/plot_f1_vs_tokens.py results/some_file.csv
    python scripts/plot_f1_vs_tokens.py results/*.csv   # overlay multiple files
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(filepath):
    csv.field_size_limit(10 * 1024 * 1024)
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_xy(rows):
    xs, ys = [], []
    for row in rows:
        tokens = parse_float(row.get("total_tokens"))
        f1 = parse_float(row.get("row_matching_f1"))
        if tokens is not None and f1 is not None:
            xs.append(tokens)
            ys.append(f1)
    return xs, ys


def main(paths):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, path in enumerate(paths):
        rows = load_rows(path)
        xs, ys = extract_xy(rows)
        if not xs:
            print(f"No valid data in {path}", file=sys.stderr)
            continue

        label = Path(path).stem
        color = colors[i % len(colors)]

        ax.scatter(xs, ys, alpha=0.5, s=30, color=color, label=label)

        # trend line
        coeffs = np.polyfit(xs, ys, 1)
        x_range = np.linspace(min(xs), max(xs), 200)
        ax.plot(x_range, np.polyval(coeffs, x_range), color=color, linewidth=1.5, linestyle="--")

        print(f"{label}: n={len(xs)}, mean_f1={np.mean(ys):.3f}, mean_tokens={np.mean(xs):.0f}")

    ax.set_xlabel("Total Tokens", fontsize=12)
    ax.set_ylabel("Row Matching F1", fontsize=12)
    ax.set_title("Row F1 vs Tokens Spent", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = Path(paths[0]).parent / "f1_vs_tokens.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_f1_vs_tokens.py <file.csv> [file2.csv ...]")
        sys.exit(1)
    main(sys.argv[1:])
