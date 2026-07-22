#!/usr/bin/env python3
"""
Aggregate Performance Metrics Calculator

This script computes aggregate metrics for model performance based on:
- Token usage (prompt_tokens, completion_tokens, total_tokens)
- F1 scores (arity_matching_f1, exact_match_f1, entity_set_f1,
             row_matching_f1, best_subset_column_f1)

In addition, it now produces **building‑level** mean F1 scores.  The building
name is taken as the prefix before the first “_” in ``query_id`` (e.g.
``MORTAR_001`` → ``MORTAR``).
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_csv_data(filepath):
    """Load CSV data and return a list of row‑dictionaries."""
    csv.field_size_limit(10 * 1024 * 1024)   # allow very large fields
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_to_numeric(value, default=0.0):
    """Convert a CSV cell to float – fallback to *default* on failure."""
    try:
        return float(value) if value and value.strip() else default
    except (ValueError, AttributeError):
        return default


# ----------------------------------------------------------------------
# Core aggregation
# ----------------------------------------------------------------------
def compute_aggregate_metrics(data):
    """Collect overall, per‑query, per‑model and per‑building statistics."""
    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------
    metrics = {
        "total_queries": len(data),
        "token_usage": {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": []
        },
        "f1_scores": {
            "arity_matching_f1": [],
            "exact_match_f1": [],
            "entity_set_f1": [],
            "row_matching_f1": [],
            "best_subset_column_f1": []
        },
        "by_query_id": defaultdict(lambda: {
            "count": 0,
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": [],
            "arity_matching_f1": [],
            "exact_match_f1": [],
            "entity_set_f1": [],
            "row_matching_f1": [],
            "best_subset_column_f1": []
        }),
        "by_model": defaultdict(lambda: {
            "count": 0,
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": [],
            "arity_matching_f1": [],
            "exact_match_f1": [],
            "entity_set_f1": [],
            "row_matching_f1": [],
            "best_subset_column_f1": []
        }),
        # ---- NEW: per‑building aggregates (sums only) ----
        "by_building": defaultdict(lambda: {
            "count": 0,
            "arity_matching_f1": 0.0,
            "exact_match_f1": 0.0,
            "entity_set_f1": 0.0,
            "row_matching_f1": 0.0,
            "best_subset_column_f1": 0.0
        })
    }

    # ------------------------------------------------------------------
    # Row‑wise accumulation
    # ------------------------------------------------------------------
    for row in data:
        query_id = row.get("query_id", "unknown")
        model = row.get("model", "unknown")

        # ----- token metrics -----
        prompt_tokens = convert_to_numeric(row.get("prompt_tokens"))
        completion_tokens = convert_to_numeric(row.get("completion_tokens"))
        total_tokens = convert_to_numeric(row.get("total_tokens"))

        # ----- F1 scores -----
        arity_f1 = convert_to_numeric(row.get("arity_matching_f1"))
        exact_f1 = convert_to_numeric(row.get("exact_match_f1"))
        entity_f1 = convert_to_numeric(row.get("entity_set_f1"))
        row_f1 = convert_to_numeric(row.get("row_matching_f1"))
        best_subset_f1 = convert_to_numeric(row.get("best_subset_column_f1"))

        # ----- overall lists -----
        metrics["token_usage"]["prompt_tokens"].append(prompt_tokens)
        metrics["token_usage"]["completion_tokens"].append(completion_tokens)
        metrics["token_usage"]["total_tokens"].append(total_tokens)

        metrics["f1_scores"]["arity_matching_f1"].append(arity_f1)
        metrics["f1_scores"]["exact_match_f1"].append(exact_f1)
        metrics["f1_scores"]["entity_set_f1"].append(entity_f1)
        metrics["f1_scores"]["row_matching_f1"].append(row_f1)
        metrics["f1_scores"]["best_subset_column_f1"].append(best_subset_f1)

        # ----- by query id -----
        qd = metrics["by_query_id"][query_id]
        qd["count"] += 1
        qd["prompt_tokens"].append(prompt_tokens)
        qd["completion_tokens"].append(completion_tokens)
        qd["total_tokens"].append(total_tokens)
        qd["arity_matching_f1"].append(arity_f1)
        qd["exact_match_f1"].append(exact_f1)
        qd["entity_set_f1"].append(entity_f1)
        qd["row_matching_f1"].append(row_f1)
        qd["best_subset_column_f1"].append(best_subset_f1)

        # ----- by model -----
        md = metrics["by_model"][model]
        md["count"] += 1
        md["prompt_tokens"].append(prompt_tokens)
        md["completion_tokens"].append(completion_tokens)
        md["total_tokens"].append(total_tokens)
        md["arity_matching_f1"].append(arity_f1)
        md["exact_match_f1"].append(exact_f1)
        md["entity_set_f1"].append(entity_f1)
        md["row_matching_f1"].append(row_f1)
        md["best_subset_column_f1"].append(best_subset_f1)

        # ----- by building -----
        building = query_id.split("_")[0] if "_" in query_id else query_id
        bd = metrics["by_building"][building]
        bd["count"] += 1
        bd["arity_matching_f1"] += arity_f1
        bd["exact_match_f1"] += exact_f1
        bd["entity_set_f1"] += entity_f1
        bd["row_matching_f1"] += row_f1
        bd["best_subset_column_f1"] += best_subset_f1

    return metrics


# ----------------------------------------------------------------------
# Helper: basic statistics on a list of numbers
# ----------------------------------------------------------------------
def compute_statistics(values):
    """Return mean, median, min, max, std_dev and sum for *values*."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std_dev": 0.0,
            "sum": 0.0
        }

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "sum": sum(values)
    }


# ----------------------------------------------------------------------
# Human‑readable text report
# ----------------------------------------------------------------------
def format_report(metrics):
    report = []

    # ------------------------------------------------------------------
    # header
    # ------------------------------------------------------------------
    report.append("=" * 80)
    report.append("AGGREGATE PERFORMANCE METRICS REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal Queries Analyzed: {metrics['total_queries']}")

    # ------------------------------------------------------------------
    # overall token usage
    # ------------------------------------------------------------------
    report.append("\n" + "=" * 80)
    report.append("OVERALL TOKEN USAGE")
    report.append("=" * 80)

    for token_type in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        s = compute_statistics(metrics["token_usage"][token_type])
        report.append(f"\n{token_type.replace('_', ' ').title()}:")
        report.append(f"  Mean:     {s['mean']:,.2f}")
        report.append(f"  Median:   {s['median']:,.2f}")
        report.append(f"  Min:      {s['min']:,.0f}")
        report.append(f"  Max:      {s['max']:,.0f}")
        report.append(f"  Std Dev:  {s['std_dev']:,.2f}")
        report.append(f"  Total:    {s['sum']:,.0f}")

    # ------------------------------------------------------------------
    # overall F1 scores
    # ------------------------------------------------------------------
    report.append("\n" + "=" * 80)
    report.append("OVERALL F1 SCORES")
    report.append("=" * 80)

    for f1_type in [
        "arity_matching_f1",
        "exact_match_f1",
        "entity_set_f1",
        "row_matching_f1",
        "best_subset_column_f1",
    ]:
        s = compute_statistics(metrics["f1_scores"][f1_type])
        report.append(f"\n{f1_type.replace('_', ' ').title()}:")
        report.append(f"  Mean:     {s['mean']:.4f}")
        report.append(f"  Median:   {s['median']:.4f}")
        report.append(f"  Min:      {s['min']:.4f}")
        report.append(f"  Max:      {s['max']:.4f}")
        report.append(f"  Std Dev:  {s['std_dev']:.4f}")

    # ------------------------------------------------------------------
    # per‑query breakdown
    # ------------------------------------------------------------------
    report.append("\n" + "=" * 80)
    report.append("METRICS BY QUERY ID")
    report.append("=" * 80)

    for qid, d in sorted(metrics["by_query_id"].items()):
        report.append(f"\n{qid} ({d['count']} queries):")
        # token averages
        p = compute_statistics(d["prompt_tokens"])
        c = compute_statistics(d["completion_tokens"])
        t = compute_statistics(d["total_tokens"])
        report.append(
            f"  Tokens - Prompt (avg): {p['mean']:,.1f}, "
            f"Completion (avg): {c['mean']:,.1f}, "
            f"Total (avg): {t['mean']:,.1f}"
        )
        # f1 averages
        a = compute_statistics(d["arity_matching_f1"])
        e = compute_statistics(d["exact_match_f1"])
        ent = compute_statistics(d["entity_set_f1"])
        r = compute_statistics(d["row_matching_f1"])
        b = compute_statistics(d["best_subset_column_f1"])
        report.append(
            f"  F1 Scores - Arity: {a['mean']:.3f}, "
            f"Exact: {e['mean']:.3f}, "
            f"Entity: {ent['mean']:.3f}, "
            f"Row: {r['mean']:.3f}, "
            f"Best Subset: {b['mean']:.3f}"
        )

    # ------------------------------------------------------------------
    # per‑model breakdown
    # ------------------------------------------------------------------
    report.append("\n" + "=" * 80)
    report.append("METRICS BY MODEL")
    report.append("=" * 80)

    for model, d in sorted(metrics["by_model"].items()):
        report.append(f"\n{model} ({d['count']} queries):")
        p = compute_statistics(d["prompt_tokens"])
        c = compute_statistics(d["completion_tokens"])
        t = compute_statistics(d["total_tokens"])
        report.append(
            f"  Tokens - Prompt (avg): {p['mean']:,.1f}, "
            f"Completion (avg): {c['mean']:,.1f}, "
            f"Total (avg): {t['mean']:,.1f}"
        )
        a = compute_statistics(d["arity_matching_f1"])
        e = compute_statistics(d["exact_match_f1"])
        ent = compute_statistics(d["entity_set_f1"])
        r = compute_statistics(d["row_matching_f1"])
        report.append(
            f"  F1 Scores - Arity: {a['mean']:.3f}, "
            f"Exact: {e['mean']:.3f}, "
            f"Entity: {ent['mean']:.3f}, "
            f"Row: {r['mean']:.3f}"
        )

    # ------------------------------------------------------------------
    # **NEW** – per‑building mean F1 scores
    # ------------------------------------------------------------------
    report.append("\n" + "=" * 80)
    report.append("METRICS BY BUILDING")
    report.append("=" * 80)

    for building, d in sorted(metrics["by_building"].items()):
        cnt = d["count"]
        if cnt == 0:
            continue
        # helper to avoid division‑by‑zero
        mean = lambda v: v / cnt if cnt else 0.0
        ar = mean(d["arity_matching_f1"])
        ex = mean(d["exact_match_f1"])
        en = mean(d["entity_set_f1"])
        ro = mean(d["row_matching_f1"])
        bs = mean(d["best_subset_column_f1"])
        report.append(f"\n{building} ({cnt} queries):")
        report.append(
            f"  F1 Scores - Arity: {ar:.3f}, Exact: {ex:.3f}, "
            f"Entity: {en:.3f}, Row: {ro:.3f}, Best Subset: {bs:.3f}"
        )

    report.append("\n" + "=" * 80)
    return "\n".join(report)


# ----------------------------------------------------------------------
# JSON export (adds ``by_building``)
# ----------------------------------------------------------------------
def export_json(metrics, output_path):
    json_data = {
        "total_queries": metrics["total_queries"],
        "overall": {"token_usage": {}, "f1_scores": {}},
        "by_query_id": {},
        "by_model": {},
        # ---- NEW section ----
        "by_building": {}
    }

    # overall token usage & f1 scores
    for tt in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        json_data["overall"]["token_usage"][tt] = compute_statistics(
            metrics["token_usage"][tt]
        )
    for ft in [
        "arity_matching_f1",
        "exact_match_f1",
        "entity_set_f1",
        "row_matching_f1",
        "best_subset_column_f1",
    ]:
        json_data["overall"]["f1_scores"][ft] = compute_statistics(
            metrics["f1_scores"][ft]
        )

    # per‑query
    for qid, d in metrics["by_query_id"].items():
        json_data["by_query_id"][qid] = {
            "count": d["count"],
            "token_usage": {
                "prompt_tokens": compute_statistics(d["prompt_tokens"]),
                "completion_tokens": compute_statistics(d["completion_tokens"]),
                "total_tokens": compute_statistics(d["total_tokens"]),
            },
            "f1_scores": {
                "arity_matching_f1": compute_statistics(d["arity_matching_f1"]),
                "exact_match_f1": compute_statistics(d["exact_match_f1"]),
                "entity_set_f1": compute_statistics(d["entity_set_f1"]),
                "row_matching_f1": compute_statistics(d["row_matching_f1"]),
                "best_subset_column_f1": compute_statistics(d["best_subset_column_f1"]),
            },
        }

    # per‑model
    for model, d in metrics["by_model"].items():
        json_data["by_model"][model] = {
            "count": d["count"],
            "token_usage": {
                "prompt_tokens": compute_statistics(d["prompt_tokens"]),
                "completion_tokens": compute_statistics(d["completion_tokens"]),
                "total_tokens": compute_statistics(d["total_tokens"]),
            },
            "f1_scores": {
                "arity_matching_f1": compute_statistics(d["arity_matching_f1"]),
                "exact_match_f1": compute_statistics(d["exact_match_f1"]),
                "entity_set_f1": compute_statistics(d["entity_set_f1"]),
                "row_matching_f1": compute_statistics(d["row_matching_f1"]),
                "best_subset_column_f1": compute_statistics(d["best_subset_column_f1"]),
            },
        }

    # **NEW** – per‑building (only means & sums for the five F1 metrics)
    for building, d in metrics["by_building"].items():
        cnt = d["count"]
        if cnt == 0:
            continue
        json_data["by_building"][building] = {
            "count": cnt,
            "f1_scores": {
                "arity_matching_f1": {
                    "mean": d["arity_matching_f1"] / cnt,
                    "sum": d["arity_matching_f1"],
                },
                "exact_match_f1": {
                    "mean": d["exact_match_f1"] / cnt,
                    "sum": d["exact_match_f1"],
                },
                "entity_set_f1": {
                    "mean": d["entity_set_f1"] / cnt,
                    "sum": d["entity_set_f1"],
                },
                "row_matching_f1": {
                    "mean": d["row_matching_f1"] / cnt,
                    "sum": d["row_matching_f1"],
                },
                "best_subset_column_f1": {
                    "mean": d["best_subset_column_f1"] / cnt,
                    "sum": d["best_subset_column_f1"],
                },
            },
        }

    # write file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    return output_path


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute aggregate performance metrics from a CSV file"
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument(
        "--output-json", help="Path to output JSON file (optional)", default=None
    )
    parser.add_argument(
        "--output-txt", help="Path to output text report file (optional)", default=None
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # load + compute
    # ------------------------------------------------------------------
    print(f"Loading data from {args.input_file}...")
    data = load_csv_data(args.input_file)
    print(f"Loaded {len(data)} records")

    print("Computing aggregate metrics...")
    metrics = compute_aggregate_metrics(data)

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    report = format_report(metrics)
    print("\n" + report)

    if args.output_txt:
        Path(args.output_txt).write_text(report, encoding="utf-8")
        print(f"\nText report saved to: {args.output_txt}")

    if args.output_json:
        json_path = export_json(metrics, args.output_json)
        print(f"JSON metrics saved to: {json_path}")


if __name__ == "__main__":
    main()