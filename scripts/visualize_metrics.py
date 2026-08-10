#!/usr/bin/env python3
"""
Visualize Performance Metrics Across Multiple CSV Files

This script creates visualizations comparing model performance across multiple
CSV files, including:
- Bar charts comparing mean performance by building (distinguishing baselines from test runs)
- Scatter plots comparing mean performance vs token count
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_csv_data(filepath):
    """Load CSV data and return a list of row‑dictionaries."""
    csv.field_size_limit(10 * 1024 * 1024)
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_to_numeric(value, default=0.0):
    """Convert a CSV cell to float – fallback to *default* on failure."""
    try:
        return float(value) if value and value.strip() else default
    except (ValueError, AttributeError):
        return default


# ----------------------------------------------------------------------
# Building-level metrics computation
# ----------------------------------------------------------------------
def compute_oracle_building_metrics(all_file_metrics: Dict[str, Dict[str, dict]]):
    """
    Compute oracle (maximum) scores per building by taking the best score
    from any model for each question.
    
    Returns:
        Dict with building-level oracle statistics (same format as compute_building_metrics)
    """
    # Organize by building
    by_building = defaultdict(lambda: {
        "arity_matching_f1": [],
        "exact_match_f1": [],
        "entity_set_f1": [],
        "row_matching_f1": []
    })
    
    # Get all unique questions
    all_questions = set()
    for file_metrics in all_file_metrics.values():
        all_questions.update(file_metrics.keys())
    
    # For each question, find max score and assign to building
    for question in all_questions:
        # Find which building this question belongs to
        building = None
        for file_metrics in all_file_metrics.values():
            if question in file_metrics:
                query_id = file_metrics[question]["query_id"]
                building = query_id.split("_")[0] if "_" in query_id else query_id
                break
        
        if not building:
            continue
        
        # Collect max scores across all models for this question
        max_scores = {
            "arity_matching_f1": 0.0,
            "exact_match_f1": 0.0,
            "entity_set_f1": 0.0,
            "row_matching_f1": 0.0
        }
        
        for file_metrics in all_file_metrics.values():
            if question in file_metrics:
                for metric in max_scores.keys():
                    max_scores[metric] = max(max_scores[metric], 
                                            file_metrics[question][metric])
        
        # Add to building
        for metric in max_scores.keys():
            by_building[building][metric].append(max_scores[metric])
    
    # Compute statistics per building
    oracle_stats = {}
    for building, metrics in by_building.items():
        oracle_stats[building] = {
            "count": len(metrics["row_matching_f1"]),
            "f1_scores": {}
        }
        
        for f1_type in ["arity_matching_f1", "exact_match_f1", "entity_set_f1", 
                        "row_matching_f1"]:
            values = metrics[f1_type]
            mean_val = statistics.mean(values) if values else 0.0
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
            cv_val = (stdev_val / mean_val) if mean_val > 0 else 0.0
            
            oracle_stats[building]["f1_scores"][f1_type] = {
                "mean": mean_val,
                "stdev": stdev_val,
                "cv": cv_val
            }
    
    return oracle_stats

def compute_building_metrics(data):
    """Compute building-level metrics from CSV data."""
    by_building = defaultdict(lambda: {
        "count": 0,
        "arity_matching_f1": [],
        "exact_match_f1": [],
        "entity_set_f1": [],
        "row_matching_f1": [],
        "total_tokens": []
    })
    
    for row in data:
        query_id = row.get("query_id", "unknown")
        building = query_id.split("_")[0] if "_" in query_id else query_id
        
        # F1 scores
        arity_f1 = convert_to_numeric(row.get("arity_matching_f1"))
        exact_f1 = convert_to_numeric(row.get("exact_match_f1"))
        entity_f1 = convert_to_numeric(row.get("entity_set_f1"))
        row_f1 = convert_to_numeric(row.get("row_matching_f1"))
        total_tokens = convert_to_numeric(row.get("total_tokens"))
        
        bd = by_building[building]
        bd["count"] += 1
        bd["arity_matching_f1"].append(arity_f1)
        bd["exact_match_f1"].append(exact_f1)
        bd["entity_set_f1"].append(entity_f1)
        bd["row_matching_f1"].append(row_f1)
        bd["total_tokens"].append(total_tokens)
    
    # Compute statistics
    building_stats = {}
    for building, metrics in by_building.items():
        building_stats[building] = {
            "count": metrics["count"],
            "f1_scores": {},
            "total_tokens": {
                "mean": statistics.mean(metrics["total_tokens"]) if metrics["total_tokens"] else 0.0
            }
        }
        
        for f1_type in ["arity_matching_f1", "exact_match_f1", "entity_set_f1", 
                        "row_matching_f1"]:
            values = metrics[f1_type]
            mean_val = statistics.mean(values) if values else 0.0
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
            # Compute coefficient of variation (avoid division by zero)
            cv_val = (stdev_val / mean_val) if mean_val > 0 else 0.0
            
            building_stats[building]["f1_scores"][f1_type] = {
                "mean": mean_val,
                "stdev": stdev_val,
                "cv": cv_val
            }
    
    return building_stats


# ----------------------------------------------------------------------
# Single-row multi-panel visualization with selected metrics (MEAN)
# ----------------------------------------------------------------------
def plot_all_f1_metrics_comparison(
    baseline_dict: Dict[str, dict],
    test_dict: Dict[str, dict],
    oracle_dict: Dict[str, dict] = None,  # NEW: Add oracle parameter
    output_path: str = "all_metrics_comparison.png"
):
    """Create a single-row figure comparing selected F1 metrics (mean values)."""
    # Metrics in specified order: arity, exact match, row match, entity set
    f1_metrics = ["arity_matching_f1", "exact_match_f1", "row_matching_f1", "entity_set_f1"]
    metric_titles = ["Arity Matching F1", "Exact Match F1", "Row Matching F1", "Entity Set F1"]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Get all unique buildings
    all_buildings = set()
    for stats in list(baseline_dict.values()) + list(test_dict.values()):
        all_buildings.update(stats.keys())
    buildings = sorted(all_buildings)
    
    x = np.arange(len(buildings))
    total_bars = len(baseline_dict) + len(test_dict)
    if oracle_dict:
        total_bars += 1  # Add one for oracle
    width = 0.8 / total_bars
    
    # Baseline colors and patterns
    baseline_colors = ['#A9A9A9', '#808080', '#696969', '#5C5C5C']
    hatches = ['///', '\\\\\\', '|||', '---']
    
    # Test colors
    test_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Oracle color (gold/yellow to stand out)
    oracle_color = '#FFD700'
    
    for idx, (f1_metric, title) in enumerate(zip(f1_metrics, metric_titles)):
        ax = axes[idx]
        bar_idx = 0
        
        # Plot baselines
        for i, (name, stats) in enumerate(baseline_dict.items()):
            means = []
            for building in buildings:
                if building in stats:
                    means.append(stats[building]["f1_scores"][f1_metric]["mean"])
                else:
                    means.append(0.0)
            
            offset = width * (bar_idx - total_bars / 2 + 0.5)
            color = baseline_colors[i % len(baseline_colors)]
            hatch = hatches[i % len(hatches)]
            label = f"{name}" if idx == 0 else None
            ax.bar(x + offset, means, width, label=label, 
                   alpha=0.7, color=color, hatch=hatch, edgecolor='black', linewidth=0.5)
            bar_idx += 1
        
        # Plot test runs
        for i, (name, stats) in enumerate(test_dict.items()):
            means = []
            for building in buildings:
                if building in stats:
                    means.append(stats[building]["f1_scores"][f1_metric]["mean"])
                else:
                    means.append(0.0)
            
            offset = width * (bar_idx - total_bars / 2 + 0.5)
            color = test_colors[i % len(test_colors)]
            label = name if idx == 0 else None
            ax.bar(x + offset, means, width, label=label, 
                   alpha=0.8, color=color)
            bar_idx += 1
        
        # NEW: Plot oracle
        if oracle_dict:
            means = []
            for building in buildings:
                if building in oracle_dict:
                    means.append(oracle_dict[building]["f1_scores"][f1_metric]["mean"])
                else:
                    means.append(0.0)
            
            offset = width * (bar_idx - total_bars / 2 + 0.5)
            label = "Oracle" if idx == 0 else None
            ax.bar(x + offset, means, width, label=label, 
                   alpha=0.9, color=oracle_color, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Building', fontsize=16)
        if idx == 0:
            ax.set_ylabel('F1 Score', fontsize=16)
        else:
            ax.set_yticklabels([])  # Remove y-axis labels for non-leftmost plots
        ax.set_title(title, fontsize=18, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(buildings, rotation=45, ha='right', fontsize=15)
        
        # Increase y-axis tick label size
        if idx == 0:
            ax.tick_params(axis='y', which='major', labelsize=14)
            ax.legend(fontsize=13, loc='upper left')
        
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)  # Common y-axis for all F1 scores
    
    plt.subplots_adjust(wspace=0.05)  # Make plots touch each other
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Multi-panel comparison saved to: {output_path}")
    plt.close()
# ----------------------------------------------------------------------
# Question-source metrics
# ----------------------------------------------------------------------
SOURCE_ORDER = ["human", "LLM_1", "LLM_2", "LLM_3", "LLM_4", "LLM_5"]


def build_source_map(reference_path: str) -> dict:
    """Return {question_text: source_label} from the reference CSV."""
    csv.field_size_limit(10 * 1024 * 1024)
    source_map = {}
    with open(reference_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row.get("question", "").strip()
            s = row.get("source", "").strip()
            if q and s:
                source_map[q] = s
    return source_map


# ----------------------------------------------------------------------
# Token comparison plots
# ----------------------------------------------------------------------
TOKEN_TYPES = ["prompt_tokens", "completion_tokens", "total_tokens"]
TOKEN_TITLES = ["Prompt Tokens", "Completion Tokens", "Total Tokens"]


def compute_building_token_metrics(data: list) -> dict:
    """Return per-building mean token counts for all three token columns."""
    by_building = defaultdict(lambda: {t: [] for t in TOKEN_TYPES})
    for row in data:
        query_id = row.get("query_id", "unknown")
        building = query_id.split("_")[0] if "_" in query_id else query_id
        for t in TOKEN_TYPES:
            by_building[building][t].append(convert_to_numeric(row.get(t)))

    stats = {}
    for building, tokens in by_building.items():
        stats[building] = {}
        for t in TOKEN_TYPES:
            values = tokens[t]
            mean_val = statistics.mean(values) if values else 0.0
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
            stats[building][t] = {"mean": mean_val, "stdev": stdev_val}
    return stats


def compute_source_token_metrics(data: list, source_map: dict) -> dict:
    """Return per-source mean token counts for all three token columns."""
    by_source = defaultdict(lambda: {t: [] for t in TOKEN_TYPES})
    for row in data:
        source = source_map.get(row.get("question", "").strip())
        if source is None:
            continue
        for t in TOKEN_TYPES:
            by_source[source][t].append(convert_to_numeric(row.get(t)))

    stats = {}
    for source, tokens in by_source.items():
        stats[source] = {}
        for t in TOKEN_TYPES:
            values = tokens[t]
            mean_val = statistics.mean(values) if values else 0.0
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
            stats[source][t] = {"mean": mean_val, "stdev": stdev_val}
    return stats


def _plot_token_panels(
    baseline_dict: Dict[str, dict],
    test_dict: Dict[str, dict],
    categories: List[str],
    x_label: str,
    output_path: str,
    rotate: int = 45,
):
    """Shared renderer: 3-panel grouped bar chart comparing token counts."""
    x = np.arange(len(categories))
    total_bars = len(baseline_dict) + len(test_dict)
    width = 0.8 / total_bars

    baseline_colors = ['#A9A9A9', '#808080', '#696969', '#5C5C5C']
    hatches = ['///', '\\\\\\', '|||', '---']
    test_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (token_type, title) in enumerate(zip(TOKEN_TYPES, TOKEN_TITLES)):
        ax = axes[idx]
        bar_idx = 0

        for i, (name, stats) in enumerate(baseline_dict.items()):
            means = [stats[c][token_type]["mean"] / 1000 if c in stats else 0.0
                     for c in categories]
            offset = width * (bar_idx - total_bars / 2 + 0.5)
            label = name if idx == 0 else None
            ax.bar(x + offset, means, width, label=label, alpha=0.7,
                   color=baseline_colors[i % len(baseline_colors)],
                   hatch=hatches[i % len(hatches)],
                   edgecolor='black', linewidth=0.5)
            bar_idx += 1

        for i, (name, stats) in enumerate(test_dict.items()):
            means = [stats[c][token_type]["mean"] / 1000 if c in stats else 0.0
                     for c in categories]
            offset = width * (bar_idx - total_bars / 2 + 0.5)
            label = name if idx == 0 else None
            ax.bar(x + offset, means, width, label=label, alpha=0.8,
                   color=test_colors[i % len(test_colors)])
            bar_idx += 1

        ax.set_xlabel(x_label, fontsize=16)
        if idx == 0:
            ax.set_ylabel('Mean Tokens (thousands)', fontsize=16)
            ax.tick_params(axis='y', which='major', labelsize=14)
            ax.legend(fontsize=13, loc='upper left')
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=18, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=rotate, ha='right', fontsize=13)
        ax.grid(axis='y', alpha=0.3)

    plt.subplots_adjust(wspace=0.05)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Token comparison plot saved to: {output_path}")
    plt.close()


def plot_token_comparison(
    baseline_dict: Dict[str, dict],
    test_dict: Dict[str, dict],
    output_path: str = "token_comparison.png",
):
    """3-panel bar chart: mean token counts per building."""
    all_buildings = set()
    for stats in list(baseline_dict.values()) + list(test_dict.values()):
        all_buildings.update(stats.keys())
    buildings = sorted(all_buildings)
    _plot_token_panels(baseline_dict, test_dict, buildings,
                       x_label='Building', output_path=output_path, rotate=45)


def plot_token_by_source(
    baseline_dict: Dict[str, dict],
    test_dict: Dict[str, dict],
    output_path: str = "token_by_source.png",
):
    """3-panel bar chart: mean token counts per question source."""
    all_sources = set()
    for stats in list(baseline_dict.values()) + list(test_dict.values()):
        all_sources.update(stats.keys())
    sources = [s for s in SOURCE_ORDER if s in all_sources]
    _plot_token_panels(baseline_dict, test_dict, sources,
                       x_label='Question Source', output_path=output_path, rotate=30)


# ----------------------------------------------------------------------
# Per-question comparison across files
# ----------------------------------------------------------------------
def compute_per_question_metrics(data, file_label):
    """Compute metrics for each unique question."""
    question_metrics = {}
    
    for row in data:
        query_id = row.get("query_id", "unknown")
        question = row.get("question", "")
        
        # Use question text as the unique key (query_id can repeat)
        unique_key = question.strip()
        
        if not unique_key:
            continue
            
        row_f1 = convert_to_numeric(row.get("row_matching_f1"))
        
        question_metrics[unique_key] = {
            "query_id": query_id,
            "file": file_label,
            "row_matching_f1": row_f1,
            "arity_matching_f1": convert_to_numeric(row.get("arity_matching_f1")),
            "exact_match_f1": convert_to_numeric(row.get("exact_match_f1")),
            "entity_set_f1": convert_to_numeric(row.get("entity_set_f1"))
        }
    
    return question_metrics


def compute_similarity_metrics(all_file_metrics: Dict[str, Dict[str, dict]]):
    """
    Compute single metrics to assess whether models perform similarly per question.
    
    Returns:
        Dict with overall similarity metrics
    """
    # Get all unique questions
    all_questions = set()
    for file_metrics in all_file_metrics.values():
        all_questions.update(file_metrics.keys())
    
    # Collect per-question statistics
    per_question_cvs = []
    per_question_ranges = []
    per_question_stdevs = []
    
    for question in all_questions:
        scores = []
        
        # Collect scores from each file
        for file_name, metrics in all_file_metrics.items():
            if question in metrics:
                score = metrics[question]["row_matching_f1"]
                scores.append(score)
        
        # Only analyze if question appears in multiple files
        if len(scores) > 1:
            mean_score = statistics.mean(scores)
            stdev_score = statistics.stdev(scores)
            cv_score = (stdev_score / mean_score) if mean_score > 0 else 0.0
            score_range = max(scores) - min(scores)
            
            per_question_cvs.append(cv_score)
            per_question_ranges.append(score_range)
            per_question_stdevs.append(stdev_score)
    
    # Compute overall similarity metrics
    if not per_question_cvs:
        return None
    
    # Primary metric: Mean Coefficient of Variation across all questions
    # Lower = more similar performance across models
    mean_cv = statistics.mean(per_question_cvs)
    median_cv = statistics.median(per_question_cvs)
    
    # Secondary metric: Mean absolute difference (range) across questions
    mean_range = statistics.mean(per_question_ranges)
    median_range = statistics.median(per_question_ranges)
    
    # Tertiary metric: Mean standard deviation
    mean_stdev = statistics.mean(per_question_stdevs)
    
    # Concordance metric: Percentage of questions with low variance
    # CV < 0.1 means models agree closely on that question
    low_variance_threshold = 0.1
    num_concordant = sum(1 for cv in per_question_cvs if cv < low_variance_threshold)
    concordance_rate = num_concordant / len(per_question_cvs)
    
    # High variance questions
    high_variance_threshold = 0.5
    num_discordant = sum(1 for cv in per_question_cvs if cv > high_variance_threshold)
    discordance_rate = num_discordant / len(per_question_cvs)
    
    return {
        "mean_cv": mean_cv,
        "median_cv": median_cv,
        "mean_range": mean_range,
        "median_range": median_range,
        "mean_stdev": mean_stdev,
        "concordance_rate": concordance_rate,
        "discordance_rate": discordance_rate,
        "num_questions": len(per_question_cvs),
        "num_concordant": num_concordant,
        "num_discordant": num_discordant
    }



def identify_discordant_questions(
    all_file_metrics: Dict[str, Dict[str, dict]],
    top_n: int = 10
):
    """
    Identify questions where models disagree most.
    
    Returns list of (question, cv, scores_dict) tuples
    """
    all_questions = set()
    for file_metrics in all_file_metrics.values():
        all_questions.update(file_metrics.keys())
    
    question_variance = []
    
    for question in all_questions:
        scores_dict = {}
        scores = []
        
        for file_name, metrics in all_file_metrics.items():
            if question in metrics:
                score = metrics[question]["row_matching_f1"]
                scores.append(score)
                scores_dict[file_name] = score
        
        if len(scores) > 1:
            mean_score = statistics.mean(scores)
            stdev_score = statistics.stdev(scores)
            cv_score = (stdev_score / mean_score) if mean_score > 0 else 0.0
            
            question_variance.append((question, cv_score, scores_dict))
    
    # Sort by CV (highest variance first)
    question_variance.sort(key=lambda x: x[1], reverse=True)
    
    return question_variance[:top_n]


def print_discordant_questions(discordant_questions: List[Tuple], max_question_len: int = 80):
    """Print questions where models disagree most."""
    print("\n" + "="*70)
    print("TOP DISCORDANT QUESTIONS (Highest Disagreement)")
    print("="*70)
    
    for i, (question, cv, scores_dict) in enumerate(discordant_questions, 1):
        # Truncate long questions
        display_question = question[:max_question_len] + "..." if len(question) > max_question_len else question
        
        print(f"\n{i}. Question: {display_question}")
        print(f"   CV: {cv:.4f}")
        print(f"   Scores by file:")
        for file_name, score in sorted(scores_dict.items()):
            print(f"     {file_name}: {score:.4f}")
    
    print("="*70 + "\n")

def compute_oracle_scores(all_file_metrics: Dict[str, Dict[str, dict]]):
    """
    Compute the maximum possible F1 scores by taking the best score 
    from any model for each question (oracle/upper bound).
    
    Returns:
        Dict with oracle scores for each metric
    """
    # Get all unique questions
    all_questions = set()
    for file_metrics in all_file_metrics.values():
        all_questions.update(file_metrics.keys())
    
    # For each metric, collect the max score across models for each question
    oracle_scores = {
        "row_matching_f1": [],
        "arity_matching_f1": [],
        "exact_match_f1": [],
        "entity_set_f1": []
    }
    
    for question in all_questions:
        # Collect all scores for this question across models
        question_scores = {
            "row_matching_f1": [],
            "arity_matching_f1": [],
            "exact_match_f1": [],
            "entity_set_f1": []
        }
        
        for file_name, metrics in all_file_metrics.items():
            if question in metrics:
                for metric_name in oracle_scores.keys():
                    score = metrics[question][metric_name]
                    question_scores[metric_name].append(score)
        
        # Take the max score for each metric for this question
        for metric_name in oracle_scores.keys():
            if question_scores[metric_name]:  # If we have scores for this question
                max_score = max(question_scores[metric_name])
                oracle_scores[metric_name].append(max_score)
    
    # Compute mean of the max scores
    oracle_means = {}
    for metric_name, scores in oracle_scores.items():
        if scores:
            oracle_means[metric_name] = statistics.mean(scores)
        else:
            oracle_means[metric_name] = 0.0
    
    return oracle_means


def print_oracle_scores(oracle_scores: dict, all_file_metrics: Dict[str, Dict[str, dict]]):
    """Print the oracle (maximum possible) scores."""
    print("\n" + "="*70)
    print("ORACLE SCORES (Maximum Achievable by Taking Best per Question)")
    print("="*70)
    print("These scores represent what you could achieve by selecting the best")
    print("model's prediction for each individual question.")
    print()
    
    # Also compute individual model means for comparison
    individual_means = {}
    for file_name, metrics in all_file_metrics.items():
        individual_means[file_name] = {}
        for metric_type in ["row_matching_f1", "arity_matching_f1", 
                           "exact_match_f1", "entity_set_f1"]:
            scores = [m[metric_type] for m in metrics.values()]
            individual_means[file_name][metric_type] = statistics.mean(scores) if scores else 0.0
    
    # Print oracle scores with comparison
    metric_labels = {
        "arity_matching_f1": "Arity Matching F1",
        "exact_match_f1": "Exact Match F1",
        "row_matching_f1": "Row Matching F1",
        "entity_set_f1": "Entity Set F1"
    }
    
    for metric_name, label in metric_labels.items():
        oracle_score = oracle_scores[metric_name]
        
        # Find best individual model for this metric
        best_individual = max(individual_means.items(), 
                             key=lambda x: x[1][metric_name])
        best_name = best_individual[0]
        best_score = best_individual[1][metric_name]
        
        improvement = oracle_score - best_score
        improvement_pct = (improvement / best_score * 100) if best_score > 0 else 0
        
        print(f"{label}:")
        print(f"  Oracle (max per question): {oracle_score:.4f}")
        print(f"  Best individual model ({best_name}): {best_score:.4f}")
        print(f"  Potential improvement: +{improvement:.4f} ({improvement_pct:+.1f}%)")
        print()
    
    print("="*70 + "\n")


def print_similarity_assessment(similarity_metrics: dict):
    """Print a clear assessment of whether models perform similarly."""
    if similarity_metrics is None:
        print("\nInsufficient data for similarity analysis")
        return
    
    print("\n" + "="*70)
    print("MODEL SIMILARITY ASSESSMENT (Per-Question Performance)")
    print("="*70)
    print(f"Total questions analyzed: {similarity_metrics['num_questions']}")
    print()
    
    # Primary metric interpretation
    mean_cv = similarity_metrics['mean_cv']
    print("PRIMARY METRIC: Mean Coefficient of Variation (CV)")
    print(f"  Value: {mean_cv:.4f}")
    print(f"  Median CV: {similarity_metrics['median_cv']:.4f}")
    
    if mean_cv < 0.15:
        assessment = "VERY SIMILAR - Models perform consistently across questions"
        color = "✓"
    elif mean_cv < 0.30:
        assessment = "MODERATELY SIMILAR - Some variance but generally consistent"
        color = "~"
    elif mean_cv < 0.50:
        assessment = "MODERATELY DIFFERENT - Noticeable variance across questions"
        color = "!"
    else:
        assessment = "VERY DIFFERENT - High variance in per-question performance"
        color = "✗"
    
    print(f"  Assessment: {color} {assessment}")
    print()
    
    # Secondary metrics
    print("SUPPORTING METRICS:")
    print(f"  Mean score range per question: {similarity_metrics['mean_range']:.4f}")
    print(f"  Median score range per question: {similarity_metrics['median_range']:.4f}")
    print(f"  Mean standard deviation: {similarity_metrics['mean_stdev']:.4f}")
    print()
    
    # Concordance analysis
    concordance_pct = similarity_metrics['concordance_rate'] * 100
    discordance_pct = similarity_metrics['discordance_rate'] * 100
    
    print("CONCORDANCE ANALYSIS:")
    print(f"  Questions with high agreement (CV < 0.1): {similarity_metrics['num_concordant']} ({concordance_pct:.1f}%)")
    print(f"  Questions with high disagreement (CV > 0.5): {similarity_metrics['num_discordant']} ({discordance_pct:.1f}%)")
    print()
    
    # Overall interpretation
    print("INTERPRETATION:")
    if mean_cv < 0.15 and concordance_pct > 60:
        print("  → Models are performing SIMILARLY on most questions")
        print("  → Differences are primarily due to noise/randomness")
    elif mean_cv < 0.30 and concordance_pct > 40:
        print("  → Models show MODERATE similarity")
        print("  → Some systematic differences exist but overall trends align")
    elif discordance_pct > 30:
        print("  → Models are performing DIFFERENTLY on many questions")
        print("  → Substantial systematic differences in approach or capability")
    else:
        print("  → Models show MIXED performance patterns")
        print("  → Neither consistently similar nor consistently different")
    
    print("="*70 + "\n")
# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def _clean_name(stem: str) -> str:
    """Strip '_unknown...' suffix from CSV stem names."""
    idx = stem.find("_unknown")
    return stem[:idx] if idx != -1 else stem


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize performance metrics across multiple CSV files"
    )
    parser.add_argument(
        "--baseline",
        action='append',
        default=[],
        help="Baseline CSV files (format: name:path or just path). Can be specified multiple times."
    )
    parser.add_argument(
        "--test",
        action='append',
        default=[],
        help="Test CSV files (format: name:path or just path). Can be specified multiple times."
    )
    parser.add_argument(
        "--f1-metric",
        default="exact_match_f1",
        choices=["arity_matching_f1", "exact_match_f1", "entity_set_f1", 
                 "row_matching_f1"],
        help="F1 metric to use for primary visualizations"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--with-oracle",
        default = False,
        help="Print detailed processing information"
    )
    parser.add_argument(
        "--reference-csv",
        default=None,
        help="Reference CSV with question→source mapping for source-level comparison"
    )
    args = parser.parse_args()

    if not args.baseline and not args.test:
        parser.error("Must specify at least one --baseline or --test CSV file")

    def parse_csv_arg(csv_arg):
        if ':' in csv_arg:
            name, path = csv_arg.split(':', 1)
        else:
            path = csv_arg
            name = Path(path).stem
        return _clean_name(name), path

    # Parse baseline CSV files
    baseline_dict = {}
    for csv_arg in args.baseline:
        name, path = parse_csv_arg(csv_arg)
        print(f"Loading baseline {name} from {path}...")
        data = load_csv_data(path)
        baseline_dict[name] = compute_building_metrics(data)
        print(f"  Loaded {len(data)} records, {len(baseline_dict[name])} buildings")

    # Parse test CSV files
    test_dict = {}
    for csv_arg in args.test:
        name, path = parse_csv_arg(csv_arg)
        print(f"Loading test {name} from {path}...")
        data = load_csv_data(path)
        test_dict[name] = compute_building_metrics(data)
        print(f"  Loaded {len(data)} records, {len(test_dict[name])} buildings")

    # Per-question metrics, keyed by question text (query_id is not unique
    # per question, so it is never used as a dedup/grouping key here).
    all_file_metrics = {}
    for csv_arg in args.baseline + args.test:
        name, path = parse_csv_arg(csv_arg)
        all_file_metrics[name] = compute_per_question_metrics(load_csv_data(path), name)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    oracle_dict = compute_oracle_building_metrics(all_file_metrics) if args.with_oracle else None

    plot_all_f1_metrics_comparison(
        baseline_dict,
        test_dict,
        oracle_dict=oracle_dict,
        output_path=str(output_dir / "all_metrics_comparison.png")
    )

    # Token comparison by building
    print("Generating token comparison plot...")
    baseline_tok_dict = {}
    test_tok_dict = {}
    for csv_arg in args.baseline:
        name, path = parse_csv_arg(csv_arg)
        baseline_tok_dict[name] = compute_building_token_metrics(load_csv_data(path))
    for csv_arg in args.test:
        name, path = parse_csv_arg(csv_arg)
        test_tok_dict[name] = compute_building_token_metrics(load_csv_data(path))
    plot_token_comparison(
        baseline_tok_dict, test_tok_dict,
        output_path=str(output_dir / "token_comparison.png"),
    )

    if args.reference_csv:
        print("\nGenerating question-source token comparison...")
        source_map = build_source_map(args.reference_csv)

        baseline_tok_src_dict = {}
        test_tok_src_dict = {}
        for csv_arg in args.baseline:
            name, path = parse_csv_arg(csv_arg)
            baseline_tok_src_dict[name] = compute_source_token_metrics(
                load_csv_data(path), source_map)
        for csv_arg in args.test:
            name, path = parse_csv_arg(csv_arg)
            test_tok_src_dict[name] = compute_source_token_metrics(
                load_csv_data(path), source_map)
        plot_token_by_source(
            baseline_tok_src_dict, test_tok_src_dict,
            output_path=str(output_dir / "token_by_source.png"),
        )

    print("\nAll visualizations complete!")

    # Per-question similarity analysis across files
    print("\nAnalyzing per-question similarity across files...")
    similarity_metrics = compute_similarity_metrics(all_file_metrics)
    oracle_scores = compute_oracle_scores(all_file_metrics)

    print_similarity_assessment(similarity_metrics)
    print_oracle_scores(oracle_scores, all_file_metrics)

    discordant_questions = identify_discordant_questions(all_file_metrics, top_n=10)
    print_discordant_questions(discordant_questions)


if __name__ == "__main__":
    main()