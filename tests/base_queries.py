"""
Script to execute SPARQL queries from benchmark QA pairs and report result counts.
Loads queries and their corresponding TTL files, executes queries, and highlights zero results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from specified path."""
    path = Path(config_path)
    if not path.is_file():
        print(f"Configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    with open(path) as f:
        return json.load(f)


def load_json_file(filepath: Path) -> Dict:
    """Load and parse JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_ttl_path(building_id: str, eval_buildings_dir: Path) -> Path:
    """Get the corresponding TTL file path for a building ID."""
    ttl_path = eval_buildings_dir / building_id
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path}")
    return ttl_path


def load_graph(ttl_path: Path) -> Graph:
    """Load RDF graph from TTL file."""
    g = Graph(store = 'Oxigraph')
    g.parse(ttl_path, format='turtle')
    return g


def execute_sparql_query(graph: Graph, query: str) -> int:
    """
    Execute SPARQL query on graph and return result count.
    
    Args:
        graph: RDF graph to query
        query: SPARQL query string
        
    Returns:
        Number of results returned by the query
    """
    try:
        results = graph.query(query)
        return len(list(results))
    except Exception as e:
        print(f"    ⚠️  Query execution error: {str(e)}")
        return -1  # Return -1 to indicate error


def process_building_queries(
    json_data: List[Dict],
    eval_buildings_dir: Path,
) -> Dict:
    """
    Process all queries for a single building.

    Args:
        json_data: List containing building data with queries.
        eval_buildings_dir: Directory containing evaluation building files.
        
    Returns:
        Dictionary with statistics about the run.
    """
    if not json_data:
        print("Warning: Empty JSON data")
        return {}

    building_data = json_data[0]
    building_id = building_data.get("building_id")
    if not building_id:
        print("Warning: No building_id found in JSON")
        return {}

    print("\n" + "=" * 80)
    print(f"Processing building: {building_id}")
    print("=" * 80)

    # Load TTL graph
    try:
        ttl_path = get_ttl_path(building_id, eval_buildings_dir)
        print(f"Loading graph from: {ttl_path}")
        graph = load_graph(ttl_path)
        print(f"✅ Graph loaded successfully ({len(graph)} triples)")
    except Exception as e:
        print(f"❌ Error loading graph: {str(e)}")
        return {}

    queries_list = building_data.get("queries", [])
    total_questions = sum(len(q.get("questions", [])) for q in queries_list)

    print(f"Total query groups: {len(queries_list)}")
    print(f"Total questions to process: {total_questions}")

    stats = {
        "total_queries": 0,
        "zero_results": 0,
        "errors": 0,
        "successful": 0
    }

    question_counter = 0

    for query_group in queries_list:
        query_id = query_group.get("query_id", "unknown")
        sparql_query = query_group.get("sparql_query", "")
        description = query_group.get("description", "")

        print(f"\n{'─' * 80}")
        print(f"Query Group: {query_id}")
        print(f"Description: {description}")
        print(f"{'─' * 80}")

        if not sparql_query:
            print("  ⚠️  No SPARQL query found in this group")
            continue

        for question_data in query_group.get("questions", []):
            question_counter += 1
            stats["total_queries"] += 1
            
            q_num = question_data.get("question_number", question_counter)
            q_text = question_data.get("text", "")
            q_source = question_data.get("source", "unknown")

            print(f"\n  Question {question_counter}/{total_questions} (Q{q_num}):")
            print(f"    Text: {q_text}")
            print(f"    Source: {q_source}")
            
            # Execute the SPARQL query
            result_count = execute_sparql_query(graph, sparql_query)
            
            if result_count == -1:
                stats["errors"] += 1
                print(f"    ❌ ERROR executing query")
            elif result_count == 0:
                stats["zero_results"] += 1
                stats["successful"] += 1
                print(f"    🔴 ZERO RESULTS RETURNED")
            else:
                stats["successful"] += 1
                print(f"    ✅ Results: {result_count}")

    return stats


def main(config_path: str) -> None:
    """
    Main entry point for the query execution script.
    
    Args:
        config_path: Path to benchmark configuration file.
    """
    config = load_config(config_path)
    exclude_buildings = config.get("exclude-buildings", [])
    benchmark_dir = Path(config.get("buildingqa-dir")) / "Benchmark_QA_pairs"
    eval_buildings_dir = Path(config.get("buildingqa-dir")) / "eval_buildings"
    
    print("\n" + "=" * 80)
    print("SPARQL Query Execution Report")
    print(f"Configuration: {config_path}")
    print("=" * 80)

    if not benchmark_dir.is_dir():
        print(f"Benchmark directory not found: {benchmark_dir}", file=sys.stderr)
        return

    json_files = list(benchmark_dir.glob("*_combined.json"))
    if not json_files:
        print(f"No JSON files found in {benchmark_dir}", file=sys.stderr)
        return

    print(f"\nFound {len(json_files)} building files to process\n")
    
    overall_stats = {
        "total_queries": 0,
        "zero_results": 0,
        "errors": 0,
        "successful": 0,
        "buildings_processed": 0
    }
    
    for idx, json_path in enumerate(json_files, start=1):
        print("\n" + "#" * 80)
        print(f"Building {idx}/{len(json_files)}: {json_path.name}")
        print("#" * 80)
        
        if any(exc in json_path.name for exc in exclude_buildings):
            print(f"⏭️  Skipping excluded building {json_path.name}")
            continue
            
        try:
            json_data = load_json_file(json_path)
            building_stats = process_building_queries(json_data, eval_buildings_dir)
            
            # Aggregate stats
            for key in ["total_queries", "zero_results", "errors", "successful"]:
                overall_stats[key] += building_stats.get(key, 0)
            overall_stats["buildings_processed"] += 1
            
        except Exception as e:
            print(f"❌ Unexpected error while processing {json_path.name}: {e}")
            import traceback
            print(traceback.format_exc())
            continue

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Buildings processed: {overall_stats['buildings_processed']}")
    print(f"Total queries executed: {overall_stats['total_queries']}")
    print(f"Successful executions: {overall_stats['successful']}")
    print(f"Queries with ZERO results: {overall_stats['zero_results']} 🔴")
    print(f"Query errors: {overall_stats['errors']}")
    if overall_stats['successful'] > 0:
        zero_pct = (overall_stats['zero_results'] / overall_stats['successful']) * 100
        print(f"Zero results percentage: {zero_pct:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute SPARQL queries from benchmark QA pairs and report result counts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/benchmark-config.json",
        help="Path to benchmark configuration file (default: ../configs/benchmark-config.json)"
    )
    
    args = parser.parse_args()
    main(args.config)