"""
simple_sparql_agent_mcp.py

Simplified SPARQL agent that uses tool calls up to a maximum limit.
"""
from pprint import pprint
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonschema import ValidationError
import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits, UsageLimitExceeded
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from httpx import AsyncClient, HTTPStatusError, TransportError
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential


from pyparsing import ParseException
from rdflib import BNode, Graph, Literal, URIRef

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.metrics import (
    get_arity_matching_f1,
    get_entity_set_f1,
    get_row_matching_f1,
    get_exact_match_f1,

)

from scripts.utils import CsvLogger

from agents.kgqa import sparql_query, TOOLSETS


def create_retrying_client():
    """Create a client with smart retry handling for multiple error types."""

    def should_retry_status(response):
        """Raise exceptions for retryable HTTP status codes.

        500 is included alongside the usual 429/502/503/504: this gateway (litellm, fronting
        cborg.lbl.gov) returns a generic 500 -- not 502/503/504 -- when the backend vllm host
        it's routing to is unreachable (seen directly in practice: "Cannot connect to host
        vllm-h100-4x-1:8000 ... Connect call failed"). That's exactly the kind of transient,
        worth-retrying failure this function exists to catch, so excluding 500 defeats the
        purpose for this particular API.
        """
        if response.status_code in (429, 500, 502, 503, 504):
            response.raise_for_status()  # This will raise HTTPStatusError

    transport = AsyncTenacityTransport(
        config=RetryConfig(
            # Retry on HTTP errors and connection issues. TransportError (not the builtin
            # ConnectionError, which httpx never actually raises) is what httpx raises for a
            # failed TCP connection -- ConnectError, ReadTimeout, PoolTimeout, etc. are all
            # subclasses of it. The openai SDK wraps these in its own APIConnectionError
            # before this transport sees them, but the retry hook here operates at the httpx
            # layer (below that wrapping), so TransportError is the type that's actually
            # raised at the point tenacity intercepts it.
            retry=retry_if_exception_type((HTTPStatusError, TransportError)),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=60),
                max_wait=300
            ),
            # Stop after 5 attempts
            stop=stop_after_attempt(5),
            # Re-raise the last exception if all retries fail
            reraise=True
        ),
        validate_response=should_retry_status
    )
    return AsyncClient(transport=transport)


class SparqlQuery(BaseModel):
    """Model for the agent's output."""
    sparql_query: str = Field(..., description="The generated SPARQL query.")


class SimpleSparqlAgentMCP:
    """
    A simplified SPARQL agent that uses MCP tools to generate queries in a single
    execution pass (no separate planning phase, no critique/retry loop).
    """

    def __init__(
        self,
        with_ontology_graph_file: str,
        without_ontology_graph_file: str,
        model_name: str = "lbl/cborg-coder",
        max_tool_calls: int = 100,
        total_tokens_limit: int = 200000,
        config_file: Optional[str] = None,
        toolset: Optional[str] = 'mcp',
        mcp_server_script: str = "../agents/kgqa.py",
        reasoning_model: bool = False,
    ):
        """
        Initialize the Simple SPARQL Agent with MCP support.

        Args:
            with_ontology_graph_file: Path to the full reasoned TTL file (ontology + inferred
                triples included) that sparql_validator/sparql_snapshot/sparql_query run
                against. Registered with sparql-relax and exposed to kgqa.py as
                WITH_ONTOLOGY_GRAPH_FILE.
            without_ontology_graph_file: Path to the lean, instance-only TTL file (no ontology
                schema triples, no inferred classes) used by tools that walk the graph directly
                (describe_entity, get_relationship_between_classes). Exposed to kgqa.py as
                WITHOUT_ONTOLOGY_GRAPH_FILE.
            model_name: Name of the model to use
            max_tool_calls: Maximum number of tool calls allowed
            config_file: Path to a JSON run config (required; see configs/ for examples)
            mcp_server_script: Name of the MCP server script
        """
        self.with_ontology_graph_file = with_ontology_graph_file
        self.without_ontology_graph_file = without_ontology_graph_file
        self.model_name = model_name
        self.max_tool_calls = max_tool_calls
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.messages = []
        self.total_tokens_limit = total_tokens_limit

        if config_file:
            with open(config_file, 'r') as file:
                config = json.load(file)
                self.provider = config.get('provider', 'openai')
                self.model_name = config.get('models', model_name)[0]
                self.total_tokens_limit = config.get('total_tokens_limit', total_tokens_limit)
                self.max_tool_calls = config.get('max_tool_calls', max_tool_calls)
                self.mcp_server_script = config.get('mcp_server_script', mcp_server_script)
                self.reasoning_model = config.get('reasoning_model', reasoning_model)
                self.toolset_name = config.get('toolset', toolset)
                self.toolset = MCPToolset(TOOLSETS.get(self.toolset_name))
                self.bschema_dir = config.get('bschema_dir', None)

                if self.provider == 'ollama':
                    self.base_url = config.get('base-url', 'http://localhost:11434/v1')
                    self.api_key = config.get('api-key', 'ollama')
                else:
                    self.api_key = config.get('api-key')
                    self.base_url = config.get('base-url')

                print(config)

        print(f"🗂️ Local TTL file mode activated. Loading graph from: {self.without_ontology_graph_file}")
        if not os.path.exists(self.without_ontology_graph_file):
            print(f"   -> ❌ ERROR: File not found at {self.without_ontology_graph_file}. Queries will fail.")
            return
        try:
            self.graph = Graph(store = "Oxigraph")
            self.graph.parse(self.without_ontology_graph_file, format="turtle")
            print(f"   -> ✅ Graph loaded successfully with {len(self.graph)} triples.")
        except Exception as e:
            print(f"   -> ❌ ERROR: Failed to load or parse the TTL file: {e}")
            self.graph = None

        self.bschema = ""
        if self.bschema_dir:
            bschema_path = Path(self.bschema_dir) / Path(self.without_ontology_graph_file).name
            if not bschema_path.exists():
                print(f"   -> ❌ ERROR: B-Schema file not found at {bschema_path}.")
            else:
                self.bschema = bschema_path.read_text()
                print(f"   -> ✅ B-Schema loaded from {bschema_path} ({len(self.bschema)} chars).")

        # Pass both graph files as environment variables to the MCP server (agents/kgqa.py)
        os.environ['WITHOUT_ONTOLOGY_GRAPH_FILE'] = self.without_ontology_graph_file
        os.environ['WITH_ONTOLOGY_GRAPH_FILE'] = self.with_ontology_graph_file
        # Set up the model
        client = create_retrying_client()

        self.model = OpenAIChatModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key, http_client = client),
        )


        self.limits = UsageLimits(total_tokens_limit = self.total_tokens_limit, request_limit = self.max_tool_calls)

        # sparql_no_validation and bschema_no_validation's only tool is sparql_snapshot (no
        # diagnosis); every other toolset carries sparql_validator instead.
        NO_VALIDATION_TOOLSETS = ('sparql_no_validation', 'bschema_no_validation')
        BSCHEMA_TOOLSETS = ('bschema_tools', 'bschema_no_validation')
        self.is_bschema_variant = self.toolset_name in BSCHEMA_TOOLSETS
        self.enforce_final_validation = self.toolset_name not in NO_VALIDATION_TOOLSETS
        self.validator_tool_name = 'sparql_validator' if self.enforce_final_validation else 'sparql_snapshot'
        final_check_rule = (
            f"4. Final check: Before returning your answer, call {self.validator_tool_name} on the "
            f"exact, complete query you are about to submit — not just the schema fragments you used "
            f"to build it up.\n\n"
        )

        if self.is_bschema_variant:
            system_prompt = (
                f"You are an expert SPARQL developer specializing in Brick Schema and ASHRAE 223p.\n"
                f"Generate complete, validated SPARQL queries to answer user questions.\n\n"

                f"You are provided with the B-Schema of the building model — a structural summary that "
                f"shows class types and the predicates that connect them, with all repetitious instance "
                f"detail removed. Use it to understand the graph topology and construct correct queries.\n\n"

                f"QUERY CONSTRUCTION RULES:\n"
                f"1. Prefixes: Always define standard prefixes (brick:, rdf:, rdfs:, unit:, s223:)\n"
                f"2. Projections: When writing SPARQL queries return more columns rather than fewer\n"
                f"3. {'Validation' if self.enforce_final_validation else 'Verification'}: Use the "
                f"{self.validator_tool_name} tool to test your query before finalising it\n"
                f"{final_check_rule}"

                f"REASONING APPROACH:\n"
                f"- Read the B-Schema carefully to identify the relevant class types and predicates\n"
                f"- Derive the triple patterns directly from the B-Schema\n"
                f"- Use {self.validator_tool_name} to confirm the query returns results before returning it\n"
                + (
                    "- If validation fails, adjust the query based on the validator's feedback, then re-validate the adjusted query itself\n"
                    if self.enforce_final_validation
                    else "- If the query returns nothing, adjust it and check again -- this tool won't explain why it failed\n"
                )
            )
        else:
            system_prompt = (
                f"You are an expert SPARQL developer specializing in Brick Schema and ASHRAE 223p.\n"
                f"Generate complete, validated SPARQL queries to answer user questions.\n\n"

                f"QUERY CONSTRUCTION RULES:\n"
                f"1. Prefixes: Always define standard prefixes (brick:, rdf:, rdfs:, unit:, s223:)\n"
                f"2. Projections: When writing SPARQL queries return more columns rather than fewer\n"
                f"3. Verification: Never guess entity names or relationships - use tools to verify\n"
                f"{final_check_rule}"

                f"REASONING APPROACH:\n"
                f"- Before each tool call, briefly state: (1) what you know, (2) what you need to verify to answer the user request, (3) which tool to use\n"
                f"- Keep reasoning concise - 2-3 bullet points maximum\n"
                f"- After gathering information, assemble the full query, then run that exact assembled query through {self.validator_tool_name} before returning it — validating individual pieces while exploring the schema is not the same as validating the query you submit\n"
            )

        self.agent = Agent(
            self.model,
            output_type=SparqlQuery,
            toolsets = [self.toolset],
            system_prompt=system_prompt,
            retries=10
        )

        print('✅ SimpleSparqlAgentMCP initialized successfully.')

    async def generate_query(
        self,
        eval_data: Dict[str, Any],
        logger: CsvLogger,
        prefixes: str,
    ) -> None:
        """Generate and evaluate a SPARQL query in a single execution pass."""
        self.prompt_tokens = self.completion_tokens = self.total_tokens = 0

        generated_query = ""
        tool_calls_exceeded = False
        actual_tool_calls = 0

        nl_question = eval_data['question']
        ground_truth_sparql = eval_data.get('ground_truth_sparql')

        print(f"\n🚀 Generating query for: '{nl_question}'")

        self.all_previous_messages = []
        messages = []

        try:
            # =========================================================================
            # EXECUTION - Generate the query in a single pass
            # =========================================================================
            final_step = (
                f"3. Assemble the full query, then run that exact, complete query (not just the "
                f"pieces you used to explore the schema) through {self.validator_tool_name} and "
                f"confirm it returns results\n"
                f"4. Return that exact query, the one {self.validator_tool_name} just confirmed, as "
                f"your final answer\n\n"
            )
            if self.is_bschema_variant:
                execution_prompt = (
                    f"B-Schema (structural summary of the building model. It replaces references to instances with different versions of classes."
                    f"This indicates different predicates connecting these instances to other things:\n"
                    f"```turtle\n{self.bschema}\n```\n\n"
                    f"Question: {nl_question}\n\n"
                    f"Using the B-Schema above:\n"
                    f"1. Identify the relevant class types and predicates that answer the question\n"
                    f"2. Construct a SPARQL query using those class types and predicates\n"
                    f"{final_step}"
                )
            else:
                execution_prompt = (
                    f"Question: {nl_question}\n"
                    f"Using the available tools:\n"
                    f"1. Identify the relevant class types and predicates that answer the question\n"
                    f"2. Construct a SPARQL query using those class types and predicates\n"
                    f"{final_step}"
                )

            print(f"🔧 Executing...")

            with capture_run_messages() as messages:
                result = await self.agent.run(
                    execution_prompt,
                    usage_limits=self.limits,
                )

                # Track tokens
                if hasattr(result, 'usage'):
                    usage = result.usage
                    if usage:
                        self.prompt_tokens += usage.input_tokens
                        self.completion_tokens += usage.output_tokens
                        self.total_tokens += usage.total_tokens

                # Count tool calls
                for msg in messages:
                    if hasattr(msg, 'parts'):
                        for part in msg.parts:
                            if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                                actual_tool_calls += 1
                            elif type(part).__name__ == 'ToolCallPart':
                                actual_tool_calls += 1

                if actual_tool_calls == 0:
                    print("⚠️ No tools called")

                # Check tool call limit
                if actual_tool_calls > self.max_tool_calls:
                    tool_calls_exceeded = True
                    print(f"⚠️ Tool call limit exceeded: {actual_tool_calls}/{self.max_tool_calls}")
                    generated_query = ""
                else:
                    generated_query = result.output.sparql_query
                    print(f"✅ Generated query (used {actual_tool_calls}/{self.max_tool_calls} tools):\n{generated_query}")

        except UsageLimitExceeded as e:
            # Capture the usage information from the exception
            print(f"❌ Token limit exceeded: {e}")
            token_limit_exceeded = True

            # Parse token count from exception message.
            # pydantic_ai raises UsageLimitExceeded with only total_tokens in the message;
            # the input/output split is not exposed. Any unaccounted gap (tokens from the
            # in-flight request that raised the exception) is attributed to prompt_tokens
            # since prompt tokens dominate in large contexts.
            import re
            match = re.search(r'total_tokens=(\d+)', str(e))
            if match:
                tokens_at_limit = int(match.group(1))
                gap = tokens_at_limit - (self.prompt_tokens + self.completion_tokens)
                if gap > 0:
                    self.prompt_tokens += gap
                self.total_tokens = tokens_at_limit
                print(f"📊 Token usage at limit: {tokens_at_limit}/{self.total_tokens_limit}")
            else:
                print(f"⚠️ Could not parse token count from exception: {e}")

            # Try to extract any partial query that was generated
            last_sparql_query = os.getenv('LAST_SPARQL_QUERY', '')
            if last_sparql_query:
                generated_query = last_sparql_query
                print("♻️ Using LAST_SPARQL_QUERY from environment.")

        except Exception as e:
            print(f"❌ Error during query generation: {e}")
            import traceback
            traceback.print_exc()
            generated_query = ""
            tool_calls_exceeded = False

        finally:
            # Capture whatever messages were recorded, even on UsageLimitExceeded
            # or any other exception raised mid-run (messages is populated live
            # by capture_run_messages(), not just on successful completion).
            self.all_previous_messages += [str(msg) for msg in messages]

        if not generated_query:
            last_sparql_query = os.getenv('LAST_SPARQL_QUERY', '')
            if last_sparql_query != '':
                generated_query = last_sparql_query
                print("♻️ Using LAST_SPARQL_QUERY from environment.")
            else:
                print("💔 Could not generate a query")

        # -----------------------------------------------------------------
        # Evaluate with exponential backoff
        # unnecessary, but also don't need to remove.
        # -----------------------------------------------------------------

        print("Evaluating generated query...")
        gen_results_obj = sparql_query(generated_query)

        print("Evaluating ground truth query...")
        if ground_truth_sparql:
            gt_results_obj = sparql_query(ground_truth_sparql)
        # Calculate metrics
        print("Calculating evaluation metrics...")
        arity_f1, entity_set_f1, row_matching_f1, exact_match_f1 = 0.0, 0.0, 0.0, 0.0
        less_columns_flag = False

        if gt_results_obj and gt_results_obj["syntax_ok"] and gen_results_obj["syntax_ok"]:
            gold_rows = gt_results_obj["results"]
            pred_rows = gen_results_obj["results"]

            arity_f1 = get_arity_matching_f1(generated_query, ground_truth_sparql)
            print('calculated arity f1:', arity_f1)
            entity_set_f1 = get_entity_set_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            print('calculated entity set f1:', entity_set_f1)
            row_matching_f1 = get_row_matching_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            print('calculated row matching f1:', row_matching_f1)
            exact_match_f1 = get_exact_match_f1(gold_rows=gold_rows, pred_rows=pred_rows)
            print('calculated exact match f1:', exact_match_f1)
            less_columns_flag = gen_results_obj['col_count'] < gt_results_obj['col_count']

        log_entry = {
            **eval_data,
            'model': self.model_name,
            'generated_sparql': generated_query,
            'message_history': "\n".join(self.all_previous_messages),
            'syntax_ok': gen_results_obj['syntax_ok'],
            'returns_results': gen_results_obj['row_count'] > 0,
            'perfect_match': row_matching_f1 == 1.0,
            'gt_num_rows': gt_results_obj['row_count'] if gt_results_obj else 0,
            'gt_num_cols': gt_results_obj['col_count'] if gt_results_obj else 0,
            'gen_num_rows': gen_results_obj['row_count'],
            'gen_num_cols': gen_results_obj['col_count'],
            'arity_matching_f1': arity_f1,
            'entity_set_f1': entity_set_f1,
            'row_matching_f1': row_matching_f1,
            'exact_match_f1': exact_match_f1,
            'less_columns_flag': less_columns_flag,
            'tool_calls_exceeded': tool_calls_exceeded,
            'actual_tool_calls': actual_tool_calls,
            'max_tool_calls': self.max_tool_calls,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        }

        logger.log(log_entry)
        print(f"📊 Logged results for query_id: {eval_data['query_id']}")
        pprint({'entity_set_f1': entity_set_f1,
            'row_matching_f1': row_matching_f1,
            'exact_match_f1': exact_match_f1,
            'total_tokens': self.total_tokens}
            )
