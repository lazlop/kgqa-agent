#!/usr/bin/env python3
"""
Analyze tool usage frequency from CSV files containing message history with ModelRequest/ModelResponse data.

This script parses the message_history column to extract tool names from ToolCallPart entries
and generates frequency statistics.
"""

import csv
import re
from collections import Counter
from typing import Dict, List, Tuple
import argparse
import json
import sys

# Increase CSV field size limit to handle large message_history fields
csv.field_size_limit(sys.maxsize)


def extract_tool_calls(message_history: str) -> List[str]:
    """
    Extract tool names from ToolCallPart entries in the message history.
    
    Args:
        message_history: String containing ModelRequest/ModelResponse snippets
        
    Returns:
        List of tool names found in the message history
    """
    tool_names = []
    
    # Pattern to match ToolCallPart with tool_name parameter
    # Example: ToolCallPart(tool_name='final_result', args=...
    pattern = r"ToolCallPart\(tool_name='([^']+)'"
    
    matches = re.findall(pattern, message_history)
    tool_names.extend(matches)
    
    return tool_names


def analyze_tool_usage(csv_file: str) -> Tuple[Counter, Dict[str, List[str]]]:
    """
    Analyze tool usage from a CSV file containing message history.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Tuple of (tool frequency counter, dict mapping query_id to tool names used)
    """
    tool_counter = Counter()
    query_tools = {}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, start=1):
            query_id = row.get('query_id', 'unknown')
            question_number = row.get('question_number', '')
            question = row.get('question', '')
            message_history = row.get('message_history', '')
            
            # Create a unique key for this query
            # Use row number to ensure uniqueness since query_id may not be unique
            if question_number:
                query_key = f"{query_id}_{question_number}"
            else:
                # Include row number for uniqueness
                query_key = f"{query_id}_row{row_num}"
            
            # Extract tool calls from message history
            tools = extract_tool_calls(message_history)
            
            # Update counters
            tool_counter.update(tools)
            
            # Store tools per query with additional metadata
            if tools:
                query_tools[query_key] = {
                    'tools': tools,
                    'question': question[:100] if question else '',  # Store first 100 chars of question
                    'query_id': query_id,
                    'row': row_num
                }
    
    return tool_counter, query_tools


def print_analysis(tool_counter: Counter, query_tools: Dict[str, Dict]):
    """
    Print the tool usage analysis results.
    
    Args:
        tool_counter: Counter of tool usage frequencies
        query_tools: Dictionary mapping query keys to query metadata including tools
    """
    print("=" * 80)
    print("TOOL USAGE FREQUENCY ANALYSIS")
    print("=" * 80)
    print()
    
    # Overall statistics
    total_queries = len(query_tools)
    total_tool_calls = sum(tool_counter.values())
    unique_tools = len(tool_counter)
    
    print(f"Total queries analyzed: {total_queries}")
    print(f"Total tool calls: {total_tool_calls}")
    print(f"Unique tools used: {unique_tools}")
    print()
    
    # Tool frequency table
    print("-" * 80)
    print("TOOL FREQUENCY TABLE")
    print("-" * 80)
    print(f"{'Tool Name':<40} {'Count':>10} {'Percentage':>10}")
    print("-" * 80)
    
    for tool_name, count in tool_counter.most_common():
        percentage = (count / total_tool_calls * 100) if total_tool_calls > 0 else 0
        print(f"{tool_name:<40} {count:>10} {percentage:>9.2f}%")
    
    print("-" * 80)
    print()
    
    # Queries with multiple tool calls
    multi_tool_queries = {k: v for k, v in query_tools.items() if len(v.get('tools', [])) > 1}
    if multi_tool_queries:
        print("-" * 80)
        print(f"QUERIES WITH MULTIPLE TOOL CALLS ({len(multi_tool_queries)} queries)")
        print("-" * 80)
        for query_key, data in sorted(multi_tool_queries.items()):
            tools = data.get('tools', [])
            question = data.get('question', '')
            print(f"{query_key}: {', '.join(tools)}")
            if question:
                print(f"  Question: {question}")
        print()


def save_json_report(tool_counter: Counter, query_tools: Dict[str, Dict], output_file: str):
    """
    Save the analysis results to a JSON file.
    
    Args:
        tool_counter: Counter of tool usage frequencies
        query_tools: Dictionary mapping query keys to query metadata including tools
        output_file: Path to output JSON file
    """
    total_tool_calls = sum(tool_counter.values())
    
    report = {
        "summary": {
            "total_queries": len(query_tools),
            "total_tool_calls": total_tool_calls,
            "unique_tools": len(tool_counter)
        },
        "tool_frequencies": [
            {
                "tool_name": tool_name,
                "count": count,
                "percentage": round(count / total_tool_calls * 100, 2) if total_tool_calls > 0 else 0
            }
            for tool_name, count in tool_counter.most_common()
        ],
        "query_details": {
            query_key: {
                "tools": data.get('tools', []),
                "question": data.get('question', ''),
                "query_id": data.get('query_id', ''),
                "row": data.get('row', 0)
            }
            for query_key, data in sorted(query_tools.items())
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"JSON report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze tool usage frequency from message history in CSV files'
    )
    parser.add_argument(
        'csv_file',
        help='Path to the CSV file containing message_history column'
    )
    parser.add_argument(
        '--json',
        dest='json_output',
        help='Save results to JSON file',
        metavar='OUTPUT_FILE'
    )
    
    args = parser.parse_args()
    
    # Analyze the CSV file
    tool_counter, query_tools = analyze_tool_usage(args.csv_file)
    
    # Print analysis
    print_analysis(tool_counter, query_tools)
    
    # Save JSON report if requested
    if args.json_output:
        save_json_report(tool_counter, query_tools, args.json_output)


if __name__ == '__main__':
    main()
