"""
analyze_failures.py

Analyzes SPARQL query generation failures using a pydantic_ai agent to classify
failure types based on message history and query comparison.
"""
import asyncio
from datetime import datetime
import json
import os
import sys
from pathlib import Path
from typing import Optional, List
import yaml

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


# ============================================================================
# Pydantic Models for Failure Classification
# ============================================================================

class FailureClassification(BaseModel):
    """Model for classifying SPARQL query generation failures."""
    
    primary_category: str = Field(
        ...,
        description=(
            "Primary failure category. Must be one of: "
            "'Vocabulary Hallucination', 'Topological Mismatch', "
            "'Identifier Guessing', 'Misunderstanding User Intent'"
        )
    )
    evidence: str = Field(
        ...,
        description=(
            "Specific evidence from the query or message history that supports "
            "this classification. Include concrete examples."
        )
    )
    
    secondary_category: Optional[str] = Field(
        None,
        description=(
            "Optional secondary failure category if multiple failure types are present. "
            "Must be one of: 'Vocabulary Hallucination', 'Topological Mismatch', "
            "'Identifier Guessing', 'Misunderstanding User Intent', or None"
        )
    )
    
# ============================================================================
# Utility Functions
# ============================================================================

def truncate_message_history(
    message_history: str,
    max_chars: int = 8000,
    keep_first_chars: int = 2000,
    keep_last_chars: int = 6000,
) -> str:
    """
    Truncate message history to prevent token overflow while preserving context.
    
    Args:
        message_history: The full message history string
        max_chars: Maximum characters to keep (approximate 2:1 char:token ratio)
        keep_first_chars: Number of characters to keep from the beginning
        keep_last_chars: Number of characters to keep from the end
    
    Returns:
        Truncated message history with ellipsis indicator if truncated
    """
    if len(message_history) <= max_chars:
        return message_history
    
    # Keep first part (initial context) and last part (most recent errors)
    first_part = message_history[:keep_first_chars]
    last_part = message_history[-keep_last_chars:]
    
    omitted_chars = len(message_history) - keep_first_chars - keep_last_chars
    truncation_notice = (
        f"\n\n... [TRUNCATED: {omitted_chars} characters omitted to stay within token limits] ...\n\n"
    )
    
    return first_part + truncation_notice + last_part


# ============================================================================
# Failure Analysis Agent
# ============================================================================

class FailureAnalyzer:
    """Analyzes SPARQL query generation failures using an LLM agent."""
    
    SYSTEM_PROMPT = """You are an expert in SPARQL query generation and knowledge graph analysis,
specifically for Brick Schema and building metadata. Your task is to analyze failed SPARQL query 
generation attempts and classify the type of failure.

There are 4 possible failure categories:

1. **Vocabulary Hallucination**: The LLM uses incorrect or invented ontology terms.
   - Example: Using `brick:feedsAir` instead of the correct `brick:feeds`
   - Root cause: Probabilistic token prediction overrides strict ontology adherence
   - What's missing: Metacognition - the ability to look up and verify valid vocabulary (T-Box) before using it

2. **Topological Mismatch**: The LLM assumes direct relationships where indirect ones exist.
   - Example: Assuming Room → AHU directly when the actual path is Room → VAV → AHU
   - Root cause: Lack of spatial/structural awareness of the graph structure
   - What's missing: Perception - the ability to explore the local neighborhood of entities (A-Box) to understand connections

3. **Identifier Guessing**: The LLM invents URIs instead of using actual instance identifiers.
   - Example: Creating `ex:Building_1` instead of the actual URI like `urn:bldg:site_main`
   - Root cause: Inaccessibility of instance data - the model doesn't have the database in its weights
   - What's missing: Entity Resolution - the ability to search and map natural language names to precise URIs

4. **Misunderstanding User Intent**: The LLM misinterprets what the user is asking for.
   - Example: Returning entity labels when the user asked for entity types
   - Root cause: Ambiguous natural language or incorrect interpretation of the question
   - What's missing: Better question understanding or clarification

Analyze the provided information carefully and classify the failure with evidence."""

    def __init__(
        self,
        model_name: Optional[str] = "lbl/cborg-coder",
        api_key_file: Optional[str] = "../configs/analysis-config.json",
        max_message_history_chars: int = 8000,
    ):
        """
        Initialize the Failure Analyzer.
        
        Args:
            model_name: Name of the model to use for analysis
            api_key_file: Path to JSON file containing 'api-key', 'base-url', and optionally 'model'
            max_message_history_chars: Maximum characters to include from message history
        """
        self.max_message_history_chars = max_message_history_chars
        # Load API credentials
        with open(api_key_file, 'r') as file:
            config = json.load(file)
        self.api_key = config.get('api-key')
        self.base_url = config.get('base-url')
        self.model_name = config.get('model', model_name)
        self._model_name = self.model_name
        self._api_key_file = api_key_file

        
        # Set up the model
        self.model = OpenAIModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key),
        )
        
        # Create the agent with retries
        self.agent = Agent(
            self.model,
            result_type=FailureClassification,
            system_prompt=self.SYSTEM_PROMPT,
            retries=1,
        )
    
    async def analyze_failure(
        self,
        question: str,
        generated_sparql: str,
        ground_truth_sparql: str,
        error_messages: str,
    ) -> FailureClassification:
        """
        Analyze a single query failure.
        
        Args:
            question: The natural language question
            generated_sparql: The SPARQL query generated by the agent
            ground_truth_sparql: The correct SPARQL query
            error_messages: The error messages from the agent
        
        Returns:
            FailureClassification with the analysis results
        """
        # Truncate message history to prevent token overflow
        truncated_messages = truncate_message_history(
            error_messages,
            max_chars=self.max_message_history_chars
        )
        
        user_message = f"""Analyze this SPARQL query generation failure:

**Question:** {question}

**Generated SPARQL Query:**
```sparql
{generated_sparql}
```

**Ground Truth SPARQL Query:**
```sparql
{ground_truth_sparql}
```

**Message History (Agent Conversation):**
{truncated_messages}

Based on the above information, classify the primary failure type and provide evidence.

You MUST respond with a valid JSON object containing all required fields:
- primary_category (one of: 'Vocabulary Hallucination', 'Topological Mismatch', 'Identifier Guessing', 'Misunderstanding User Intent')
- evidence (string with specific examples)
- secondary_category (optional, same options as primary_category or null)"""

        result = await self.agent.run(user_message)
        return result.data
    
    async def analyze_batch(
        self,
        failures_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze a batch of failures sequentially (one at a time).
        
        Args:
            failures_df: DataFrame with failed queries
            max_concurrent: Ignored, kept for API compatibility
        
        Returns:
            DataFrame with added classification columns
        """
        results = []
        
        # Process rows one at a time
        for idx, row in failures_df.iterrows():
            try:
                print(f"Analyzing {idx + 1}/{len(failures_df)}: query_id {row.get('query_id', 'unknown')}")
                temp_analyzer = FailureAnalyzer(
                    model_name=self._model_name,
                    api_key_file=self._api_key_file,
                    max_message_history_chars=self.max_message_history_chars
                )
                classification = await temp_analyzer.analyze_failure(
                    question=row['question'],
                    generated_sparql=row['generated_sparql'],
                    ground_truth_sparql=row['ground_truth_sparql'],
                    error_messages=row['message_history'],
                )
                result = {
                    'failure_category': classification.primary_category,
                    'failure_evidence': classification.evidence,
                    'secondary_category': classification.secondary_category,
                }
                print(f"  ✓ Classified as: {classification.primary_category} ")
            except Exception as e:
                print(f"  ✗ Error analyzing query_id {row.get('query_id', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                result = {
                    'failure_category': 'ERROR',
                    'failure_evidence': str(e),
                    'secondary_category': None,
                }
            results.append(result)
        
        # Add results to dataframe
        for col in ['failure_category', 'failure_evidence', 
                    'secondary_category']:
            failures_df[col] = [r[col] for r in results]
        
        return failures_df


# ============================================================================
# Main Processing Functions
# ============================================================================

def load_and_filter_failures(
    csv_path: str,
    row_matching_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Load CSV and filter for failed queries.
    
    Args:
        csv_path: Path to the CSV file
        row_matching_threshold: Threshold for row_matching_f1 (queries below this are failures)
    
    Returns:
        DataFrame with only failed queries
    """
    print(f"📂 Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total rows: {len(df)}")
    
    # Filter for failures
    failures = df[df['row_matching_f1'] < row_matching_threshold].copy()
    print(f"   Failed queries (row_matching_f1 < {row_matching_threshold}): {len(failures)}")
    
    return failures


async def analyze_failures_async(
    csv_path: str,
    output_path: Optional[str] = None,
    model_name: str = "lbl/cborg-coder",
    api_key_file: Optional[str] = "../configs/analysis-config.json",
    row_matching_threshold: float = 1.0,
    max_message_history_chars: int = 8000,
) -> pd.DataFrame:
    """
    Analyze failures from a CSV file asynchronously.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output CSV file (defaults to input_path with '_analyzed' suffix)
        model_name: LLM model to use for analysis
        api_key_file: Path to JSON file with API credentials
        row_matching_threshold: Threshold for considering a query as failed
        max_message_history_chars: Maximum characters to include from message history (default: 8000)
    
    Returns:
        DataFrame with failure classifications
    """
    # Load and filter failures
    failures_df = load_and_filter_failures(csv_path, row_matching_threshold)
    
    if len(failures_df) == 0:
        print("✅ No failures found to analyze!")
        return failures_df
    
    # Initialize analyzer
    print(f"\n🤖 Initializing failure analyzer with model: {model_name}")
    print(f"   Max message history chars: {max_message_history_chars}")
    analyzer = FailureAnalyzer(
        model_name=model_name,
        api_key_file=api_key_file,
        max_message_history_chars=max_message_history_chars,
    )
    
    # Analyze failures
    print(f"\n🔍 Analyzing {len(failures_df)} failures...")
    analyzed_df = await analyzer.analyze_batch(failures_df)
    
    # Select only the required columns
    required_columns = [
        'query_id',
        'question_number',
        'question',
        'model',
        'failure_category',
        'failure_evidence',
        'secondary_category'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_columns = [col for col in required_columns if col in analyzed_df.columns]
    if len(available_columns) < len(required_columns):
        missing = set(required_columns) - set(available_columns)
        print(f"⚠️  Warning: Missing columns in input CSV: {missing}")
    
    output_df = analyzed_df[available_columns]
    
    # Save results
    if output_path is None:
        base_path = Path(csv_path)
        output_path = base_path.parent / f"{base_path.stem}_analyzed{base_path.suffix}"
    
    output_df.to_csv(output_path, index=False)
    print(f"\n✅ Analysis complete! Results saved to: {output_path}")
    
    # Print summary statistics
    print("\n📊 Failure Category Summary:")
    category_counts = analyzed_df['failure_category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(analyzed_df)) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")

    
    return analyzed_df

def create_timestamped_logger(base_name: str = "failure_analysis"):
    """Create a logger with timestamp in filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_{timestamp}.csv"
    return log_filename

def analyze_failures(
    csv_path: str,
    output_path: Optional[str] = None,
    model_name: str = "lbl/cborg-coder",
    api_key_file: Optional[str] = "../configs/analysis-config.json",
    row_matching_threshold: float = 1.0,
    max_message_history_chars: int = 8000,
) -> pd.DataFrame:
    """
    Synchronous wrapper for analyze_failures_async.
    
    See analyze_failures_async for parameter documentation.
    """
    return asyncio.run(
        analyze_failures_async(
            csv_path=csv_path,
            output_path=output_path,
            model_name=model_name,
            api_key_file=api_key_file,
            row_matching_threshold=row_matching_threshold,
            max_message_history_chars=max_message_history_chars,
        )
    )


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze SPARQL query generation failures using LLM classification"
    )
    parser.add_argument(
        "csv_path",
        help="Path to CSV file with query results"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for analyzed CSV (default: input_analyzed.csv)",
        default = create_timestamped_logger()
    )
    parser.add_argument(
        "-m", "--model",
        help="LLM model to use (default: gpt-oss)",
        default = "lbl/cborg-coder"
    )
    parser.add_argument(
        "-f", "--api-key-file",
        help="Path to YAML file with API credentials",
        default="../configs/analysis-config.json"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=1.0,
        help="row_matching_f1 threshold for failures (default: 1.0)"
    )
    parser.add_argument(
        "--max-message-chars",
        type=int,
        default=4000,
        help="Maximum characters to include from message history (default: 4000, ~2000 tokens)"
    )
    
    args = parser.parse_args()
    
    analyze_failures(
        csv_path=args.csv_path,
        output_path=args.output,
        model_name=args.model,
        api_key_file=args.api_key_file,
        row_matching_threshold=args.threshold,
        max_message_history_chars=args.max_message_chars,
    )
