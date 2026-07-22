# Project Organization

This document describes the file structure of the rdf-mcp-relax benchmark project. See
`README.md` for the sparql-relax integration and how it differs from `rdf-mcp`.

## Directory Structure

```
kgqa-agent/
├── agents/                # Agent + MCP tool definitions
│   ├── agent.py            # SimpleSparqlAgentMCP: single-pass execution + evaluation/logging
│   └── kgqa.py              # FastMCP toolsets (graph nav + sparql-relax-backed query tools)
├── scripts/               # Executable scripts
│   ├── benchmark.py                   # Main workflow script
│   ├── recalculate_metrics.py         # Re-score an existing results CSV
│   ├── aggregate_metrics.py           # Aggregate metrics across a run
│   ├── analyze_failures.py            # Failure classification
│   ├── analyze_tool_usage.py          # Tool-usage analysis
│   ├── visualize_metrics.py, visualize_tool_usage.py, visualize_question_source.py
│   ├── plot_f1_vs_tokens.py
│   ├── metrics.py                     # F1 metric implementations
│   ├── namespaces.py                  # RDF namespace definitions
│   ├── utils.py                       # CsvLogger + shared helpers
│   └── run_all_analysis.sh            # aggregate + visualize + tool-usage over a CSV directory
├── example_configs/                # Run configurations (model × toolset), api-key left as a placeholder
├── data/                   # Bundled benchmark buildings/QA pairs/B-Schemas (see README.md)
├── tests/                  # Test files + a local fixture graph (test-building.ttl)
├── performance/            # Selected files showing performance of the agents
└── results/                # Benchmark results and analysis outputs (gitignored)
```

## Core Files

- `agents/agent.py` - `SimpleSparqlAgentMCP`: runs the tool-using agent in a single execution
  pass (no separate planning phase or critique/retry loop), then scores and logs the result.
- `agents/kgqa.py` - MCP tool definitions and the five ablation toolsets (`all_tools`,
  `graph_tools`, `bschema_tools`, `sparql_only`, `sparql_no_validation`).
- `scripts/benchmark.py` - Main benchmark runner.
- `scripts/analyze_failures.py` - Failure analysis tool.
- `scripts/aggregate_metrics.py` - Metrics aggregation tool.
- `scripts/recalculate_metrics.py` - Re-execute stored queries and recompute F1 metrics.
- `scripts/metrics.py` - Metrics calculation.
- `scripts/utils.py` - Utility functions (`CsvLogger`, etc).
- `scripts/namespaces.py` - RDF namespace definitions.

## Usage

### Running the benchmark

```bash
cd scripts
uv run python benchmark.py --config ../configs/gemma/gemma-sparql.json
```

Resuming a partially-completed run (only re-runs rows with empty `message_history`):

```bash
uv run python benchmark.py --config ../configs/flash/gemma-sparql.json \
    --resume ../results/flash/flash-sparql_unknown_model_20260416_215704.csv
```

### Running individual components

```bash
# Recompute metrics for an existing results CSV
uv run python scripts/recalculate_metrics.py results/gemma/gemma-bschema3_*.csv

# Analyze failures from existing results
uv run python scripts/analyze_failures.py results/benchmark_run_YYYYMMDD_HHMMSS.csv

# Compute aggregate metrics
uv run python scripts/aggregate_metrics.py results/benchmark_run_YYYYMMDD_HHMMSS.csv \
    --output-json results/metrics.json \
    --output-txt results/metrics_report.txt
```

### Running all analysis over a directory of results

```bash
bash ./scripts/run_all_analysis.sh results/gemma-planning/
```

## Output Files

All benchmark outputs are stored in the `results/` directory (gitignored):

- `*_run_*.csv` - Raw benchmark results with all metrics
- `*_metrics.json` - Aggregated metrics in JSON format
- `*_metrics_report.txt` - Human-readable metrics report
- `*_failure_analysis.csv` - Failure classification results

## Configuration

Configuration files are stored in `example_configs/` (templates) / `configs/` (your local,
gitignored copies with real credentials filled in), one per (model, toolset) run:

- `buildingqa-dir` - points at `../data` (the bundled subset of BuildingQA)
- `bschema_dir` - points at `../data/bschema/threshold-{0,30,70}` for `bschema_tools` runs
- `toolset` - one of `all_tools`, `graph_tools`, `bschema_tools`, `sparql_only`,
  `sparql_no_validation` (see `agents/kgqa.py`'s `TOOLSETS`)
- `api-key` / `base-url` - model provider credentials; `api-key` ships as a placeholder
  (`REPLACE_WITH_YOUR_API_KEY`) — fill in your own before running
