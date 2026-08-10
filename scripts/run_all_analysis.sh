#!/usr/bin/env bash
# run_all_analysis.sh
#
# Runs aggregate_metrics.py, visualize_metrics.py, and visualize_tool_usage.py
# on every CSV file found in a directory.
#
# Usage:
#   ./scripts/run_all_analysis.sh <csv_dir> [output_dir] [reference_csv]
#
# Arguments:
#   csv_dir       Directory containing CSV result files (required)
#   output_dir    Where to write all outputs (default: <csv_dir>/analysis)
#   reference_csv CSV with question→source mapping for source-level analysis (optional)
#
# Each CSV file gets its own subdirectory under output_dir named after the
# file stem (without extension).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <csv_dir> [output_dir]"
    exit 1
fi

CSV_DIR="$1"
OUTPUT_DIR="${2:-${CSV_DIR}/analysis}"
REFERENCE_CSV="${3:-}"

if [[ ! -d "$CSV_DIR" ]]; then
    echo "Error: '$CSV_DIR' is not a directory"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --------------------------------------------------------------------------
# Collect CSV files
# --------------------------------------------------------------------------
mapfile -t CSV_FILES < <(find "$CSV_DIR" -maxdepth 1 -name "*.csv" | sort)

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    echo "No CSV files found in '$CSV_DIR'"
    exit 1
fi

echo "Found ${#CSV_FILES[@]} CSV file(s) in '$CSV_DIR'"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# --------------------------------------------------------------------------
# Per-file analysis
# --------------------------------------------------------------------------
for csv_file in "${CSV_FILES[@]}"; do
    stem="$(basename "$csv_file" .csv)"
    file_out="${OUTPUT_DIR}/${stem}"
    mkdir -p "$file_out"

    echo ""
    echo ">>> $stem"

    # aggregate_metrics.py
    echo "  [1/2] aggregate_metrics..."
    python3 "${SCRIPT_DIR}/aggregate_metrics.py" \
        "$csv_file" \
        --output-json "${file_out}/metrics.json" \
        --output-txt  "${file_out}/metrics.txt" \
        > /dev/null

    # visualize_tool_usage.py
    echo "  [2/2] visualize_tool_usage..."
    python3 "${SCRIPT_DIR}/visualize_tool_usage.py" \
        "$csv_file" \
        --output-dir "$file_out" \
        2>&1 | grep -v "^$" | sed 's/^/    /'

    echo "  Done → $file_out"
done

# --------------------------------------------------------------------------
# Cross-file comparison (visualize_metrics.py)
# Uses the first CSV as baseline and the rest as test runs.
# --------------------------------------------------------------------------
echo ""
echo "========================================"
echo ">>> Cross-file comparison (visualize_metrics.py)"

BASELINE_ARG=""
TEST_ARGS=()
first=true

for csv_file in "${CSV_FILES[@]}"; do
    stem="$(basename "$csv_file" .csv)"
    if $first; then
        BASELINE_ARG="--baseline ${stem}:${csv_file}"
        first=false
    else
        TEST_ARGS+=("--test" "${stem}:${csv_file}")
    fi
done

COMPARISON_OUT="${OUTPUT_DIR}/comparison"
mkdir -p "$COMPARISON_OUT"

python3 "${SCRIPT_DIR}/visualize_metrics.py" \
    $BASELINE_ARG \
    "${TEST_ARGS[@]}" \
    --output-dir "$COMPARISON_OUT" \
    --reference-csv "$REFERENCE_CSV"

echo ""
echo "========================================"
echo "All done."
echo "  Per-file outputs : $OUTPUT_DIR/<stem>/"
echo "  Comparison plots : $COMPARISON_OUT/"
