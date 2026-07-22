#!/usr/bin/env python3
"""
Visualize Tool Usage Metrics

This script creates visualizations for:
- Tool impact on performance (delta F1) broken down by building
- Correlation between tool count and performance
"""

import csv
import re
import statistics
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import sys
from scipy import stats

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# Set matplotlib parameters for publication quality
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
    'font.sans-serif': ['Arial', 'DejaVu Sans']
})


# ----------------------------------------------------------------------
# Tool extraction
# ----------------------------------------------------------------------
def extract_tool_calls(message_history: str) -> List[str]:
    """Extract tool names from ToolCallPart entries in the message history."""
    pattern = r"ToolCallPart\(tool_name='([^']+)'"
    matches = re.findall(pattern, message_history)
    return matches


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_csv_data(filepath):
    """Load CSV data and return a list of row-dictionaries."""
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_to_numeric(value, default=0.0):
    """Convert a CSV cell to float – fallback to default on failure."""
    try:
        return float(value) if value and value.strip() else default
    except (ValueError, AttributeError):
        return default


# ----------------------------------------------------------------------
# Compute metrics
# ----------------------------------------------------------------------
def compute_tool_success_correlation(data):
    """Compute per-query tool usage and success metrics."""
    query_data = []
    
    for row in data:
        query_id = row.get("query_id", "unknown")
        building = query_id.split("_")[0] if "_" in query_id else "unknown"
        message_history = row.get("message_history", "")
        row_f1 = convert_to_numeric(row.get("row_matching_f1", 0))
        
        tools = extract_tool_calls(message_history)
        tools = [t for t in tools if t != "final_result"]
        
        query_data.append({
            "building": building,
            "tool_count": len(tools),
            "row_f1": row_f1,
            "tools": tools
        })
    
    return query_data


def compute_tool_impact_stats_by_building(query_data, min_occurrences=5):
    """Compute impact statistics for each tool broken down by building."""
    tool_building_stats = defaultdict(lambda: defaultdict(lambda: {
        'with_tool_f1': [],
        'without_tool_f1': [],
        'usage_count': 0
    }))
    
    # Get all unique tools and buildings
    all_tools = set()
    all_buildings = set()
    for query in query_data:
        all_tools.update(query['tools'])
        all_buildings.add(query['building'])
    
    # For each tool and building combination, track F1 scores when present vs absent
    for tool in all_tools:
        for building in all_buildings:
            building_queries = [q for q in query_data if q['building'] == building]
            for query in building_queries:
                if tool in query['tools']:
                    tool_building_stats[tool][building]['with_tool_f1'].append(query['row_f1'])
                    tool_building_stats[tool][building]['usage_count'] += 1
                else:
                    tool_building_stats[tool][building]['without_tool_f1'].append(query['row_f1'])
    
    # Compute statistics
    tool_metrics = defaultdict(dict)
    for tool, building_stats in tool_building_stats.items():
        # Check if tool meets minimum occurrence threshold across all buildings
        total_usage = sum(stats['usage_count'] for stats in building_stats.values())
        if total_usage >= min_occurrences:
            for building, stats in building_stats.items():
                if stats['with_tool_f1']:  # Only include if tool was used with this building
                    with_f1 = statistics.mean(stats['with_tool_f1'])
                    without_f1 = statistics.mean(stats['without_tool_f1']) if stats['without_tool_f1'] else 0
                    
                    # Calculate standard error for delta F1
                    # Using bootstrap-style approximation for SEM of difference
                    with_std = statistics.stdev(stats['with_tool_f1']) if len(stats['with_tool_f1']) > 1 else 0
                    without_std = statistics.stdev(stats['without_tool_f1']) if len(stats['without_tool_f1']) > 1 else 0
                    
                    with_sem = with_std / np.sqrt(len(stats['with_tool_f1'])) if len(stats['with_tool_f1']) > 0 else 0
                    without_sem = without_std / np.sqrt(len(stats['without_tool_f1'])) if len(stats['without_tool_f1']) > 0 else 0
                    
                    # Standard error of difference (assuming independence)
                    delta_sem = np.sqrt(with_sem**2 + without_sem**2)
                    
                    tool_metrics[tool][building] = {
                        'with_f1': with_f1,
                        'without_f1': without_f1,
                        'delta_f1': with_f1 - without_f1,
                        'delta_sem': delta_sem,
                        'usage_count': stats['usage_count'],
                        'with_count': len(stats['with_tool_f1']),
                        'without_count': len(stats['without_tool_f1'])
                    }
    
    return tool_metrics


# ----------------------------------------------------------------------
# Visualization 1: Tool impact on performance (by building)
# ----------------------------------------------------------------------
def plot_tool_impact(tool_metrics, output_path="tool_impact.png"):
    """Create visualization showing tool impact on performance broken down by building."""
    
    if not tool_metrics:
        print("No tool metrics to plot")
        return
    
    # Get all buildings
    all_buildings = set()
    for tool_data in tool_metrics.values():
        all_buildings.update(tool_data.keys())
    buildings = sorted(all_buildings)
    
    # Calculate average delta F1 across buildings for sorting
    tool_avg_delta = {}
    for tool, building_data in tool_metrics.items():
        deltas = [data['delta_f1'] for data in building_data.values()]
        tool_avg_delta[tool] = statistics.mean(deltas) if deltas else 0
    
    # Sort tools by average delta F1
    sorted_tools = sorted(tool_avg_delta.keys(), 
                         key=lambda x: tool_avg_delta[x], 
                         reverse=True)
    
    # Prepare data for grouped bar chart
    n_tools = len(sorted_tools)
    n_buildings = len(buildings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5,4))
    
    # Set up bar positions
    bar_height = 0.8 / n_buildings
    y_positions = np.arange(n_tools)
    
    # Color palette for buildings
    colors = plt.cm.Set3(np.linspace(0, 1, n_buildings))
    
    # Plot bars for each building
    for i, building in enumerate(buildings):
        deltas = []
        errors = []
        for tool in sorted_tools:
            if building in tool_metrics[tool]:
                deltas.append(tool_metrics[tool][building]['delta_f1'])
                errors.append(tool_metrics[tool][building]['delta_sem'])
            else:
                deltas.append(np.nan)  # No data for this building/tool combo
                errors.append(0)
        
        y_pos = y_positions + (i - n_buildings/2 + 0.5) * bar_height
        bars = ax.barh(y_pos, deltas, bar_height, 
                      label=building, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
        # Add error bars (inconspicuous style)
        for j, (y, delta, error) in enumerate(zip(y_pos, deltas, errors)):
            if not np.isnan(delta) and error > 0:
                ax.errorbar(delta, y, xerr=error, 
                           fmt='none', ecolor='gray', 
                           capsize=2, capthick=0.5, 
                           linewidth=0.8, alpha=0.5, zorder=10)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    
    # Compute total sample size (queries where tool was used) across all buildings
    tool_sample_sizes = {
        tool: sum(tool_metrics[tool][b]['with_count'] for b in tool_metrics[tool])
        for tool in sorted_tools
    }

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{t}  (n={tool_sample_sizes[t]})" for t in sorted_tools],
        fontsize=8
    )
    ax.set_xlabel('ΔF1 Score\n(With Tool − Without Tool)', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    # Set x-axis limits
    all_deltas = [data['delta_f1'] for tool_data in tool_metrics.values() 
                  for data in tool_data.values()]
    if all_deltas:
        x_min = min(min(all_deltas), -0.1)
        x_max = max(max(all_deltas), 0.1)
        ax.set_xlim(x_min - 0.2, x_max + 0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tool impact plot saved to: {output_path}")
    plt.close()


# ----------------------------------------------------------------------
# Visualization 2: Correlation heatmap
# ----------------------------------------------------------------------
def plot_correlation_heatmap(query_data, output_path="correlation_heatmap.png"):
    """Create a heatmap showing correlation between tool count and F1 score for each building."""
    
    # Group data by building
    by_building = defaultdict(list)
    for query in query_data:
        by_building[query['building']].append(query)
    
    buildings = sorted(by_building.keys())
    
    # Calculate correlations
    correlations = []
    p_values = []
    sample_sizes = []
    
    for building in buildings:
        building_queries = by_building[building]
        tool_counts = [q['tool_count'] for q in building_queries]
        f1_scores = [q['row_f1'] for q in building_queries]
        
        sample_sizes.append(len(building_queries))
        
        # Calculate correlation if we have variation
        if len(set(tool_counts)) > 1 and len(tool_counts) > 1:
            corr, p_val = stats.pearsonr(tool_counts, f1_scores)
            correlations.append(corr)
            p_values.append(p_val)
        else:
            correlations.append(0.0)
            p_values.append(1.0)
    
    # Create figure - sized for single column
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Create 2D array for heatmap (single column)
    corr_array = np.array(correlations).reshape(-1, 1)
    
    im = ax.imshow(corr_array, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(buildings)))
    ax.set_yticklabels(buildings, fontsize=9)
    ax.set_xticks([0])
    ax.set_xticklabels(['Tool Count vs. F1'], fontsize=9)
    
    # Add correlation values as text with sample size
    for i, (corr, p_val, n) in enumerate(zip(correlations, p_values, sample_sizes)):
        color = 'white' if abs(corr) > 0.5 else 'black'
        significance = '*' if p_val < 0.05 else ''
        ax.text(0, i, f'{corr:.2f}{significance}\n(n={n})', 
                ha='center', va='center', color=color, fontsize=8)
    
    # Add colorbar with better sizing for narrow column
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson r', rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to: {output_path}")
    plt.close()
    
    # Print summary statistics
    print("\nCorrelation Summary:")
    print("-" * 60)
    for building, corr, p_val, n in zip(buildings, correlations, p_values, sample_sizes):
        sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{building:20s}: r={corr:+.3f} (p={p_val:.3f}{sig:2s}, n={n})")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize tool usage metrics"
    )
    parser.add_argument(
        "csv_file",
        help="CSV file to analyze"
    )
    parser.add_argument(
        "--min-tool-occurrences",
        type=int,
        default=5,
        help="Minimum occurrences for a tool to be included in analysis (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output plots"
    )
    args = parser.parse_args()
    
    print(f"Loading data from {args.csv_file}...")
    data = load_csv_data(args.csv_file)
    print(f"  Loaded {len(data)} records")
    
    # Compute metrics
    print("\nComputing metrics...")
    query_data = compute_tool_success_correlation(data)
    tool_impact = compute_tool_impact_stats_by_building(query_data, args.min_tool_occurrences)
    
    print(f"  Analyzed {len(tool_impact)} tools (min {args.min_tool_occurrences} occurrences)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_tool_impact(
        tool_impact,
        output_path=str(output_dir / "tool_impact.png")
    )
    
    plot_correlation_heatmap(
        query_data,
        output_path=str(output_dir / "correlation_heatmap.png")
    )
    
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()