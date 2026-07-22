"""
Script to run SPARQL query generation agent on all benchmark QA pairs.
Processes all JSON files in Benchmark_QA_pairs directory against their corresponding TTL files.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import agent and utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent import SimpleSparqlAgentMCP
# from agents.react_agent import SimpleSparqlAgentMCP

from scripts.utils import CsvLogger


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from specified path."""
    path = Path(config_path)
    if not path.is_file():
        print(f"Configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    with open(path) as f:
        return json.load(f)


def get_model_name_from_config(config_path: str) -> str:
    """Extract model name from config file."""
    config = load_config(config_path)
    model_name = config.get("model", ["unknown_model"])[0]
    # Clean up model name for use in filename (replace slashes, colons, etc.)
    model_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    return model_name


def create_timestamped_logger(results_dir: Path, model_name: str, base_name: str = "sparql_agent_run") -> tuple[CsvLogger, str]:
    """Create a logger with timestamp and model name in filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_{model_name}_{timestamp}.csv"
    logger = CsvLogger(results_dir / log_filename)
    print(f"Created logger: {log_filename}")
    return logger, log_filename


def load_json_file(filepath: Path) -> Dict:
    """Load and parse JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_ttl_path(building_id: str, with_ontology_dir: Path, without_ontology_dir: Path) -> tuple[Path, Path]:
    """Get the corresponding with-ontology and without-ontology TTL file paths for a building ID."""
    with_ontology_path = with_ontology_dir / building_id
    without_ontology_path = without_ontology_dir / building_id
    if not with_ontology_path.exists():
        raise FileNotFoundError(f"With-ontology TTL file not found: {with_ontology_path}")
    if not without_ontology_path.exists():
        raise FileNotFoundError(f"Without-ontology TTL file not found: {without_ontology_path}")
    return with_ontology_path, without_ontology_path


def load_kg_content(ttl_path: Path) -> str:
    """Load the knowledge graph content from TTL file."""
    with open(ttl_path, "r") as f:
        return f.read()


# Standard prefixes for SPARQL queries
STANDARD_PREFIXES = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2001/XMLSchema#>
"""


async def process_building_queries(
    json_data: List[Dict],
    logger: CsvLogger,
    with_ontology_dir: Path,
    without_ontology_dir: Path,
    config_file: str,
    completed_set: Optional[set] = None,
) -> None:
    """
    Process all queries for a single building.

    Args:
        json_data: List containing building data with queries.
        logger: CSV logger instance.
        with_ontology_dir: Directory containing full reasoned building files (eval_buildings/).
        without_ontology_dir: Directory containing lean, instance-only building files
            (eval_buildings/without-ontology/).
        config_file: Path to configuration file.
        completed_set: If provided, skip (query_id, question) pairs already in this set.
    """
    if not json_data:
        print("Warning: Empty JSON data")
        return

    building_data = json_data[0]
    building_id = building_data.get("building_id")
    if not building_id:
        print("Warning: No building_id found in JSON")
        return

    print("\n" + "=" * 60)
    print(f"Processing building: {building_id}")
    print("=" * 60)

    # Load TTL graphs
    with_ontology_path, without_ontology_path = get_ttl_path(building_id, with_ontology_dir, without_ontology_dir)

    # Initialise agent
    agent = SimpleSparqlAgentMCP(
        with_ontology_graph_file=str(with_ontology_path),
        without_ontology_graph_file=str(without_ontology_path),
        config_file=config_file,
    )

    queries_list = building_data.get("queries", [])
    total_questions = sum(len(q.get("questions", [])) for q in queries_list)

    print(f"Total query groups: {len(queries_list)}")
    print(f"Total questions to process: {total_questions}")

    question_counter = 0

    for query_group in queries_list:
        query_id = query_group.get("query_id", "unknown")
        ground_truth_sparql = query_group.get("sparql_query", "")
        description = query_group.get("description", "")

        print(f"\n--- Query Group: {query_id} ---")
        print(f"Description: {description}")

        for question_data in query_group.get("questions", []):
            question_counter += 1
            q_num = question_data.get("question_number", question_counter)
            q_text = question_data.get("text", "")
            q_source = question_data.get("source", "unknown")

            print(f"\nQuestion {question_counter}/{total_questions} (Q{q_num}):")
            print(f"  Text: {q_text}")
            print(f"  Source: {q_source}")

            if completed_set is not None and (query_id, q_text) in completed_set:
                print("  ⏭️  Skipping (already completed)")
                continue

            eval_data = {
                "query_id": query_id,
                "question": q_text,
                "ground_truth_sparql": ground_truth_sparql,
            }

            try:
                await agent.generate_query(
                    eval_data=eval_data,
                    logger=logger,
                    prefixes=STANDARD_PREFIXES,
                )
                print("  ✅ Completed")
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                import traceback

                print(f"  Traceback: {traceback.format_exc()}")


def load_csv_data(csv_path: str) -> List[Dict]:
    """Load data from CSV file."""
    import csv
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def load_completed_questions(csv_path: str) -> set:
    """Return a set of (query_id, question) pairs that completed successfully (non-empty message_history)."""
    import csv
    completed = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('message_history', '').strip():
                completed.add((row['query_id'], row['question']))
    print(f"Found {len(completed)} already-completed questions in {csv_path}")
    return completed


def compute_aggregate_metrics(data: List[Dict]) -> Dict:
    """Compute aggregate metrics from benchmark data."""
    # Placeholder - implement your actual metrics computation
    total_queries = len(data)
    # Add your specific metric calculations here
    metrics = {
        "total_queries": total_queries,
        # Add more metrics as needed
    }
    return metrics


async def process_single_config(config_path: Path, results_dir: Path, resume_csv: Optional[Path] = None) -> Path:
    """
    Process a single configuration file.

    Args:
        config_path: Path to the benchmark configuration file.
        results_dir: Directory where results should be saved.
        resume_csv: If provided, only re-run questions with empty message_history in this CSV,
                    appending results to the same file.

    Returns:
        Path to the generated CSV log file.
    """
    config = load_config(str(config_path))
    exclude_buildings = config.get("exclude-buildings", [])
    benchmark_dir = Path(config.get("buildingqa-dir")) / "Benchmark_QA_pairs"
    with_ontology_dir = Path(config.get("buildingqa-dir")) / "eval_buildings"
    without_ontology_dir = with_ontology_dir / "without-ontology"

    # Get model name from config
    model_name = get_model_name_from_config(str(config_path))
    run_base_name = config.get('run-name', 'sparql_agent_run')

    completed_set = None
    if resume_csv is not None:
        completed_set = load_completed_questions(str(resume_csv))
        # retry_set is the inverse: passed to process_building_queries to skip completed ones

    print("\n" + "=" * 80)
    print("Resuming SPARQL Agent Benchmark Run" if resume_csv else "Starting SPARQL Agent Benchmark Run")
    print(f"Configuration: {config_path}")
    print(f"Model: {model_name}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if resume_csv:
        print(f"Resuming from: {resume_csv}  ({len(completed_set)} questions already done, rest will be retried)")
    print("=" * 80)

    if resume_csv is not None:
        logger = CsvLogger(resume_csv)
        log_path = resume_csv
    else:
        logger, log_filename = create_timestamped_logger(results_dir, model_name, run_base_name)
        log_path = results_dir / log_filename

    if not benchmark_dir.is_dir():
        print(f"Benchmark directory not found: {benchmark_dir}", file=sys.stderr)
        return log_path

    json_files = list(benchmark_dir.glob("*_combined.json"))
    if not json_files:
        print(f"No JSON files found in {benchmark_dir}", file=sys.stderr)
        return log_path

    print(f"\nFound {len(json_files)} building files to process")
    
    for idx, json_path in enumerate(json_files, start=1):
        print("\n" + "#" * 60)
        print(f"Building {idx}/{len(json_files)}: {json_path.name}")
        print("#" * 60)
        if any(exc in json_path.name for exc in exclude_buildings):
            print(f"Skipping excluded building {json_path.name}")
            continue
        try:
            json_data = load_json_file(json_path)
            await process_building_queries(
                json_data,
                logger,
                with_ontology_dir,
                without_ontology_dir,
                config_file=str(config_path),
                completed_set=completed_set,
            )
        except Exception as e:
            print(f"❌ Unexpected error while processing {json_path.name}: {e}")
            import traceback

            print(traceback.format_exc())
            continue

    print("\n" + "=" * 60)
    print(f"Config {config_path.name} completed!")
    print(f"Results logged to: {log_filename}")
    print("=" * 60)
    
    return log_path


async def main_async(config_path: str, resume_csv: Optional[str] = None) -> None:
    """
    Async entry point for the benchmark run.

    Args:
        config_path: Path to a config file or directory of config files.
        resume_csv: Optional path to a CSV from a previous run; only failed rows are retried.
    """
    config_input = Path(config_path)

    # Determine results directory from first config
    if config_input.is_dir():
        config_files = sorted(list(config_input.glob("*.json")))
        if not config_files:
            print(f"No JSON config files found in {config_input}", file=sys.stderr)
            return
        first_config = load_config(str(config_files[0]))
    else:
        config_files = [config_input]
        first_config = load_config(str(config_input))

    results_dir = Path(first_config.get("results-dir", "../results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    resume_path = Path(resume_csv) if resume_csv else None
    if resume_path and not resume_path.is_file():
        print(f"Resume CSV not found: {resume_path}", file=sys.stderr)
        return

    print(f"\n{'=' * 80}")
    print(f"Processing {len(config_files)} configuration(s)")
    print(f"Results directory: {results_dir}")
    if resume_path:
        print(f"Resume mode: {resume_path}")
    print(f"{'=' * 80}\n")

    # Process each config and collect log paths
    log_paths = []
    for config_file in config_files:
        log_path = await process_single_config(config_file, results_dir, resume_csv=resume_path)
        log_paths.append(log_path)
    
    # Compute aggregate metrics for each config
    print("\n" + "=" * 80)
    print("Computing Aggregate Metrics")
    print("=" * 80)
    
    for log_path in log_paths:
        if not log_path.exists():
            print(f"Log file not found: {log_path}")
            continue
            
        print(f"\nLoading data from {log_path}...")
        data = load_csv_data(str(log_path))
        print(f"Loaded {len(data)} records")
        
        print("Computing aggregate metrics...")
        metrics = compute_aggregate_metrics(data)
        
        print(f"Metrics for {log_path.name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("All benchmark runs completed!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SPARQL query generation agent on benchmark QA pairs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/benchmark-config.json",
        help="Path to a benchmark configuration file or directory of config files (default: ../configs/benchmark-config.json)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a previous run's CSV; only rows with empty message_history are retried and appended to that file"
    )

    args = parser.parse_args()

    asyncio.run(main_async(args.config, resume_csv=args.resume))