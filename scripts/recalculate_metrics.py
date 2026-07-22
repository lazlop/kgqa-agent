#!/usr/bin/env python3
"""
recalculate_metrics.py

Re-executes the ground-truth and generated SPARQL queries stored in a result
CSV against their corresponding local RDF graphs, then recomputes all five F1
metrics and writes an updated CSV.

The script derives the correct graph file from the query_id prefix using either:
  (a) a JSON mapping file (--building-map), or
  (b) a BuildingQA-style directory layout (--buildingqa-dir).

Usage examples
--------------
# Using the bundled data/ directory (mirrors how benchmark.py works; auto-detected
# if --buildingqa-dir etc. are omitted entirely):
python3 scripts/recalculate_metrics.py results/gemma/gemma-bschema3_*.csv \\
    --buildingqa-dir ../data

# Using an explicit JSON map  {"LBNL": "path/to/b59.ttl", ...}:
python3 scripts/recalculate_metrics.py results/gemma/gemma-bschema3_*.csv \\
    --building-map my_map.json

# Override graph directories directly:
python3 scripts/recalculate_metrics.py results/gemma/gemma-bschema3_*.csv \\
    --with-ontology-dir ../data/eval_buildings \\
    --without-ontology-dir ../data/eval_buildings/without-ontology

Output: <input_stem>_recalc.csv  (in the same directory as the input, unless
        --output is given)
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Allow importing from the repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.metrics import (
    get_arity_matching_f1,
    get_entity_set_f1,
    get_row_matching_f1,
    get_exact_match_f1,
    get_best_subset_column_f1,
)


csv.field_size_limit(10 * 1024 * 1024)

# ---------------------------------------------------------------------------
# Known mapping: query_id prefix → TTL filename in eval_buildings/
# (derived from BuildingQA/Benchmark_QA_pairs/*.json)
# ---------------------------------------------------------------------------
DEFAULT_PREFIX_TO_TTL = {
    "LBNL":      "b59.ttl",
    "MORTAR":    "bldg11.ttl",
    "DFLEXLIBS": "dflexlibs_multizone.ttl",
    "TUC":       "TUC_building.ttl",
}


# ---------------------------------------------------------------------------
# RDF graph helpers
# ---------------------------------------------------------------------------

def _load_graph(path: Path):
    """Load a TTL file into an rdflib Oxigraph graph."""
    from rdflib import Graph
    g = Graph(store="Oxigraph")
    g.parse(str(path))
    return g


# Cache so we don't reload the same graph repeatedly
_graph_cache: dict = {}

def get_graph(ttl_path: Path):
    key = str(ttl_path.resolve())
    if key not in _graph_cache:
        print(f"  Loading graph: {ttl_path} ...", end=" ", flush=True)
        t0 = time.time()
        _graph_cache[key] = _load_graph(ttl_path)
        print(f"({time.time()-t0:.1f}s, {len(_graph_cache[key])} triples)")
    return _graph_cache[key]


def run_sparql(graph, query: str, result_length: int = 100_000):
    """Execute a SPARQL query on a graph; return agent-style result dict."""
    import re
    from rdflib import URIRef, Literal, BNode

    def _format_term(term):
        if isinstance(term, URIRef):
            return {"type": "uri", "value": str(term)}
        elif isinstance(term, Literal):
            return {"type": "literal", "value": str(term)}
        elif isinstance(term, BNode):
            return {"type": "bnode", "value": str(term)}
        return {"type": "unknown", "value": str(term)}

    # Add a LIMIT if not already present
    def _add_limit(q, limit):
        if re.search(r'\bLIMIT\b', q, re.IGNORECASE):
            return q
        return q.rstrip().rstrip('}').rstrip() + f"\n}} LIMIT {limit}"

    try:
        limited_query = _add_limit(query, result_length)
        qres = graph.query(limited_query)
        variables = [str(v) for v in qres.vars] if qres.vars else []
        bindings = []
        for row in qres:
            binding = {}
            for i, var in enumerate(variables):
                val = row[i] if i < len(row) else None
                if val is not None:
                    binding[var] = _format_term(val)
            bindings.append(binding)
        bindings = bindings[:result_length]
        return {
            "syntax_ok": True,
            "results": bindings,
            "row_count": len(bindings),
            "col_count": len(variables),
            "error_message": None,
        }
    except Exception as exc:
        return {
            "syntax_ok": False,
            "results": [],
            "row_count": 0,
            "col_count": 0,
            "error_message": str(exc),
        }


# ---------------------------------------------------------------------------
# Building → graph file resolution
# ---------------------------------------------------------------------------

def build_prefix_map(
    buildingqa_dir: Optional[Path],
    with_ontology_dir: Optional[Path],
    without_ontology_dir: Optional[Path],
    building_map_path: Optional[Path],
) -> Dict[str, Tuple[Path, Path]]:
    """
    Returns a dict: prefix → (with_ontology_ttl_path, without_ontology_ttl_path)
    """
    mapping: Dict[str, Tuple[Path, Path]] = {}

    if building_map_path:
        import json
        raw = json.loads(building_map_path.read_text())
        # Accepts {"PREFIX": "with_ontology_path"} or {"PREFIX": {"with_ontology": ..., "without_ontology": ...}}
        for prefix, val in raw.items():
            if isinstance(val, str):
                wp = Path(val)
                mapping[prefix] = (wp, wp)   # same file for both if only one given
            else:
                mapping[prefix] = (Path(val["with_ontology"]), Path(val["without_ontology"]))
        return mapping

    # Derive from directory layout
    if buildingqa_dir:
        with_ontology_dir = buildingqa_dir / "eval_buildings"
        without_ontology_dir = buildingqa_dir / "eval_buildings" / "without-ontology"

    if not with_ontology_dir or not without_ontology_dir:
        raise ValueError(
            "Provide --buildingqa-dir, --building-map, or both "
            "--with-ontology-dir and --without-ontology-dir"
        )

    with_ontology_dir = Path(with_ontology_dir)
    without_ontology_dir = Path(without_ontology_dir)

    for prefix, ttl_name in DEFAULT_PREFIX_TO_TTL.items():
        wp = with_ontology_dir / ttl_name
        np = without_ontology_dir / ttl_name
        if wp.exists():
            mapping[prefix] = (wp, np if np.exists() else wp)
        else:
            print(f"  Warning: graph file not found for prefix '{prefix}': {wp}")

    return mapping


# ---------------------------------------------------------------------------
# Main recalculation logic
# ---------------------------------------------------------------------------

def recalculate(input_path: Path, prefix_map: dict, output_path: Path):
    # Read input
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("  No rows found.")
        return

    # Columns we will rewrite
    RECOMPUTED_COLS = {
        "syntax_ok", "returns_results", "perfect_match",
        "gt_num_rows", "gt_num_cols", "gen_num_rows", "gen_num_cols",
        "arity_matching_f1", "exact_match_f1", "entity_set_f1",
        "row_matching_f1", "best_subset_column_f1", "less_columns_flag",
    }

    updated_rows = []
    errors = []

    # Pre-load graphs needed by this file
    needed_prefixes = {
        row.get("query_id", "").split("_")[0]
        for row in rows
        if "_" in row.get("query_id", "")
    }
    graphs: Dict[str, tuple] = {}  # prefix → (with_ontology_graph, without_ontology_graph)
    for prefix in sorted(needed_prefixes):
        if prefix not in prefix_map:
            print(f"  Warning: no graph mapping for prefix '{prefix}', rows will keep original scores.")
            continue
        with_ontology_path, without_ontology_path = prefix_map[prefix]
        graphs[prefix] = (get_graph(with_ontology_path), get_graph(without_ontology_path))

    total = len(rows)
    for i, row in enumerate(rows, 1):
        query_id = row.get("query_id", "unknown")
        prefix = query_id.split("_")[0] if "_" in query_id else query_id

        if prefix not in graphs:
            updated_rows.append(row)
            continue

        with_ontology_graph, without_ontology_graph = graphs[prefix]

        gt_sparql = row.get("ground_truth_sparql", "")
        gen_sparql = row.get("generated_sparql", "")

        if not gen_sparql.strip():
            # Nothing generated – keep zeros
            updated_rows.append(row)
            continue

        print(f"  [{i}/{total}] {query_id} ...", end=" ", flush=True)

        # Execute both queries on the with-ontology graph (same graph agents/agent.py's
        # sparql_query() scores against — see WITH_ONTOLOGY_GRAPH_FILE in agents/kgqa.py)
        gt_res = run_sparql(with_ontology_graph, gt_sparql) if gt_sparql.strip() else None
        gen_res = run_sparql(with_ontology_graph, gen_sparql)

        arity_f1 = exact_f1 = entity_f1 = row_f1 = best_subset_f1 = 0.0
        less_columns_flag = False

        if gt_res and gt_res["syntax_ok"] and gen_res["syntax_ok"]:
            gold_rows = gt_res["results"]
            pred_rows = gen_res["results"]

            arity_f1      = get_arity_matching_f1(gen_sparql, gt_sparql)
            entity_f1     = get_entity_set_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            row_f1        = get_row_matching_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            exact_f1      = get_exact_match_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            best_subset_f1 = get_best_subset_column_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            less_columns_flag = gen_res["col_count"] < gt_res["col_count"]
        elif not gen_res["syntax_ok"]:
            errors.append((query_id, gen_res["error_message"]))

        new_row = dict(row)
        new_row["syntax_ok"]            = gen_res["syntax_ok"]
        new_row["returns_results"]      = gen_res["row_count"] > 0
        new_row["perfect_match"]        = row_f1 == 1.0
        new_row["gt_num_rows"]          = gt_res["row_count"] if gt_res else 0
        new_row["gt_num_cols"]          = gt_res["col_count"] if gt_res else 0
        new_row["gen_num_rows"]         = gen_res["row_count"]
        new_row["gen_num_cols"]         = gen_res["col_count"]
        new_row["arity_matching_f1"]    = arity_f1
        new_row["exact_match_f1"]       = exact_f1
        new_row["entity_set_f1"]        = entity_f1
        new_row["row_matching_f1"]      = row_f1
        new_row["best_subset_column_f1"] = best_subset_f1
        new_row["less_columns_flag"]    = less_columns_flag

        print(f"row_f1={row_f1:.3f}  exact={exact_f1:.3f}")
        updated_rows.append(new_row)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_cols = list(fieldnames)  # preserve original column order
    for col in RECOMPUTED_COLS:
        if col not in all_cols:
            all_cols.append(col)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"\n  Written {len(updated_rows)} rows to: {output_path}")

    if errors:
        print(f"\n  {len(errors)} query/queries had syntax errors:")
        for qid, msg in errors[:10]:
            print(f"    {qid}: {msg[:120]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recompute F1 metrics for result CSV(s) by re-running SPARQL queries."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help=(
            "One or more result CSV files or directories to reprocess. "
            "Directories are expanded to all *.csv files they contain (non-recursively), "
            "skipping files that already end in '_recalc.csv'."
        ),
    )
    parser.add_argument(
        "--buildingqa-dir",
        default=None,
        help=(
            "Root of the bundled data/ directory (or a BuildingQA repo clone). Expects "
            "subdirectories 'eval_buildings/' (with ontology) and "
            "'eval_buildings/without-ontology/'. Relative paths are resolved from the "
            "script's repo root."
        ),
    )
    parser.add_argument(
        "--with-ontology-dir",
        default=None,
        help=(
            "Directory containing the full reasoned TTL files (overrides --buildingqa-dir). "
            "This is the graph queries are actually scored against, same as agents/agent.py."
        ),
    )
    parser.add_argument(
        "--without-ontology-dir",
        default=None,
        help="Directory containing lean, instance-only TTL files (overrides --buildingqa-dir).",
    )
    parser.add_argument(
        "--building-map",
        default=None,
        help=(
            "JSON file mapping query_id prefix to TTL path(s). "
            'Format: {"LBNL": "path/b59.ttl"} or '
            '{"LBNL": {"with_ontology": "...", "without_ontology": "..."}}.'
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Only valid when a single input file is given. "
            "Defaults to <input_stem>_recalc.csv next to the input."
        ),
    )
    args = parser.parse_args()

    # Resolve buildingqa-dir relative to repo root if needed
    buildingqa_dir = None
    if args.buildingqa_dir:
        buildingqa_dir = Path(args.buildingqa_dir)
        if not buildingqa_dir.is_absolute():
            buildingqa_dir = (REPO_ROOT / buildingqa_dir).resolve()

    with_ontology_dir = Path(args.with_ontology_dir) if args.with_ontology_dir else None
    without_ontology_dir = Path(args.without_ontology_dir) if args.without_ontology_dir else None
    map_path = Path(args.building_map) if args.building_map else None

    # If none of the graph sources are specified, try the bundled data/ directory
    if buildingqa_dir is None and with_ontology_dir is None and map_path is None:
        default_bqa = (REPO_ROOT / "data").resolve()
        if default_bqa.exists():
            buildingqa_dir = default_bqa
            print(f"Auto-detected bundled BuildingQA data at: {buildingqa_dir}")
        else:
            parser.error(
                "Could not auto-detect the bundled data/ directory. "
                "Provide --buildingqa-dir, --building-map, or "
                "--with-ontology-dir + --without-ontology-dir."
            )

    prefix_map = build_prefix_map(buildingqa_dir, with_ontology_dir, without_ontology_dir, map_path)
    print(f"Graph mapping: { {k: str(v[0]) for k, v in prefix_map.items()} }")

    # Expand directories to their CSV files
    resolved_inputs = []
    for entry in args.input_files:
        p = Path(entry)
        if p.is_dir():
            csvs = sorted(f for f in p.glob("*.csv") if not f.name.endswith("_recalc.csv"))
            if not csvs:
                print(f"No CSV files found in directory: {p}")
            resolved_inputs.extend(csvs)
        else:
            resolved_inputs.append(p)

    if args.output and len(resolved_inputs) > 1:
        parser.error("--output can only be used with a single input file.")

    for input_path in resolved_inputs:
        if not input_path.exists():
            print(f"Skipping (not found): {input_path}")
            continue

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_name(input_path.stem + "_recalc.csv")

        print(f"\n{'='*60}")
        print(f"Processing: {input_path.name}")
        print(f"Output:     {output_path}")
        print('='*60)

        recalculate(input_path, prefix_map, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
