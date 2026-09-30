"""
simple_sparql_agent_mcp.py

Simplified SPARQL agent that uses tool calls up to a maximum limit.
"""
from pprint import pprint
import asyncio
import copy
import dataclasses
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
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits, UsageLimitExceeded
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport
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


# Replaces the (large, static, never-changing) B-Schema block once a run has burned
# through half its token budget -- the B-Schema is embedded once in the first user
# message but gets re-billed in full on every subsequent turn (pydantic-ai's usage
# tracking sums input_tokens across every turn), so for bschema-toolset runs it can
# account for the majority of tokens spent on later turns.
_BSCHEMA_PRUNE_PLACEHOLDER = (
    "[B-Schema omitted here to conserve the remaining token budget -- you already "
    "reviewed it above; continue from what you've learned so far.]"
)


# toolset name that runs the agent against an external kgqa-tools (sparql-relax-mcp) server
# over stdio instead of one of agents/kgqa.py's in-process TOOLSETS. Both building graphs are
# preloaded before any question runs, and the harness fills in every tool's `dataset` argument
# itself (it's removed from the tool schemas the agent sees): summarize_schema always runs on
# the instance-only graph, so the summary isn't swamped by ontology triples, and every other
# tool on the reasoned graph (the one ground truth is scored against).
KGQA_TOOLS_TOOLSET = 'kgqa_tools'
KGQA_TOOLS_QUERY_DATASET = 'building_with_ontology'
KGQA_TOOLS_SCHEMA_DATASET = 'building_only'


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
        example: Optional[Dict[str, str]] = None,
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
            example: Optional one-shot worked example for this building, as
                {'question': ..., 'sparql_query': ...}. Prepended to every execution prompt.
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

        self.example = example
        self.example_block = ""
        if self.example:
            self.example_block = (
                f"WORKED EXAMPLE (a different question on this same building, illustrating its "
                f"schema and query conventions — the actual question below may need a different "
                f"structure, but follow the same conventions where they apply):\n"
                f"Question: {self.example['question']}\n"
                f"SPARQL:\n```sparql\n{self.example['sparql_query']}\n```\n\n"
            )

        self.kgqa_tools_server = None
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
                if self.toolset_name == KGQA_TOOLS_TOOLSET:
                    self.kgqa_tools_config = config.get('kgqa_tools', {})
                    self.toolset = self._build_kgqa_tools_toolset(self.kgqa_tools_config)
                else:
                    self.toolset = MCPToolset(TOOLSETS.get(self.toolset_name))
                self.bschema_dir = config.get('bschema_dir', None)
                self.bschema_dir_overrides = config.get('bschema_dir_overrides', {})

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
        building_filename = Path(self.without_ontology_graph_file).name
        effective_bschema_dir = self.bschema_dir_overrides.get(building_filename, self.bschema_dir)
        if effective_bschema_dir:
            bschema_path = Path(effective_bschema_dir) / building_filename
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
        if self.toolset_name == KGQA_TOOLS_TOOLSET:
            self.validator_tool_name = self.kgqa_tools_config.get('validator_tool', 'run_query')
        shared_rules = (
            "3. Reference literals: In addition to unique identifiers, return human-readable name "
            "properties and labels. When in doubt, return more labels and IDs rather than fewer.\n"
            "4. Qualifying conditions: If the question states a qualifying condition (e.g. a status, "
            "category, or named subtype), that condition must appear as an explicit constraint in your "
            "WHERE clause (a triple pattern or FILTER) — not just be implied by which columns you "
            "selected.\n"
            "5. Compound questions: If the question asks for multiple distinct pieces of information, "
            "include one SELECT variable for every requested item rather than collapsing them into a "
            "single generic (type, value) pair.\n"
            "6. Simple paths over broad filters: Prefer the most direct, specific graph path implied by "
            "the class names in the question (e.g. `?sp a brick:Zone_Air_Temperature_Setpoint`) over a "
            "broad `regex()`/`FILTER(... IN (...))` match against a generic type variable. Only widen to "
            "a regex/substring/multi-type filter when the question itself is asking about a broad or "
            "unnamed category.\n"
            "7. Use SELECT DISTINCT unless the question specifically calls for duplicate/repeated rows.\n"
            "8. Identifying literals must bind: When a question asks for a single identifying value (a "
            "name, ID, or reference for an entity) and the entity exposes more than one literal that "
            "could serve as it, prefer the human-readable name over an opaque ID/GUID — unless the "
            "question names a specific kind of value (e.g. a global/unique identifier, an endpoint "
            "address, a database name) that only one candidate literal satisfies, in which case use "
            "that one. This is separate from rule 5: if the question asks for multiple distinct values "
            "(e.g. both a label and an endpoint), include a column for each rather than choosing between "
            "them. Whichever literal you use, project it through a required triple pattern, not inside "
            "OPTIONAL — an unbound identifier produces a row that can't match anything.\n\n"
        )

        if self.is_bschema_variant:
            system_prompt = (
                f"You are an expert SPARQL developer specializing in Brick Schema and ASHRAE 223p.\n"
                f"Generate complete, validated SPARQL queries to answer user questions.\n\n"

                f"You are provided with the B-Schema of the building model — a structural summary that "
                f"shows class types and the predicates that connect them, with all repetitious instance "
                f"detail removed. Use it to understand the graph topology and construct correct queries.\n\n"

                f"QUERY CONSTRUCTION RULES:\n"
                f"1. Projections: When writing SPARQL queries return more columns rather than fewer. "
                f"When querying for data about an entity, include the entity's URI as well as any "
                f"relevant labels for it — extra columns are safe, missing ones aren't.\n"
                f"2. {'Validation' if self.enforce_final_validation else 'Verification'}: Use the "
                f"{self.validator_tool_name} tool to test your query before finalising it\n"
                f"{shared_rules}"

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
                f"1. Projections: When writing SPARQL queries return more columns rather than fewer. "
                f"When querying for data about an entity, include the entity's URI as well as any "
                f"relevant labels for it — extra columns are safe, missing ones aren't.\n"
                f"2. Verification: Never guess entity names or relationships - use tools to verify\n"
                f"{shared_rules}"

                f"REASONING APPROACH:\n"
                f"- Before each tool call, briefly state: (1) what you know, (2) what you need to verify to answer the user request, (3) which tool to use\n"
                f"- Keep reasoning concise - 2-3 bullet points maximum\n"
                f"- After gathering information, assemble the full query, then run that exact assembled query through {self.validator_tool_name} before returning it — validating individual pieces while exploring the schema is not the same as validating the query you submit\n"
            )

        capabilities = []
        if self.is_bschema_variant and self.bschema:
            capabilities.append(Hooks(before_model_request=self._prune_bschema_from_history))

        self.agent = Agent(
            self.model,
            output_type=SparqlQuery,
            toolsets = [self.toolset],
            system_prompt=system_prompt,
            retries=10,
            capabilities=capabilities,
        )

        print('✅ SimpleSparqlAgentMCP initialized successfully.')

    def _build_kgqa_tools_toolset(self, cfg: Dict[str, Any]):
        """
        Stdio toolset for an external kgqa-tools (sparql-relax-mcp) server, by default the
        local checkout at cfg['dir'] run via `uv --directory <dir> run sparql-relax-mcp`, so
        uncommitted changes there are what gets benchmarked. The server's own tool
        descriptions and (by default) its instructions are passed through unchanged, except
        that the harness supplies some arguments itself and removes them from the tool schemas
        the agent sees: every tool's `dataset` (see KGQA_TOOLS_TOOLSET) and any configured
        summarize_schema_args. The dataset-management tools in cfg['hidden_tools'] are withheld
        entirely, since the harness loads the graphs itself (see __aenter__).

        cfg keys (all optional):
            dir: kgqa-tools checkout (default ../../kgqa-tools, relative to the cwd)
            server_toolset: the server's --toolset, 'extended' or 'core' (default 'extended')
            command/args: override the launch command entirely (e.g. uvx from GitHub)
            hidden_tools: tool names withheld from the agent
                (default ['load_dataset', 'list_datasets'])
            include_instructions: send the server's instructions to the agent (default True)
            validator_tool: the tool the prompt tells the agent to check its final query with
                (default 'run_query')
            summarize_schema_args: summarize_schema arguments fixed by the harness, e.g.
                {"similarity_threshold": 0.3} (default {}: the agent/server defaults apply)
            summarize_schema_args_overrides: per-building replacements for those, keyed by
                building filename like bschema_dir_overrides, e.g.
                {"b59.ttl": {"similarity_threshold": 0.3}}
            init_timeout: seconds to wait for the server to start (default 120)
        """
        tools_dir = Path(cfg.get('dir', '../../kgqa-tools')).resolve()
        command = cfg.get('command', 'uv')
        args = cfg.get('args', [
            '--directory', str(tools_dir), 'run', 'sparql-relax-mcp',
            '--toolset', cfg.get('server_toolset', 'extended'),
        ])
        # `uv --directory` changes the server's cwd, so the paths must be absolute
        self.kgqa_tools_datasets = {
            KGQA_TOOLS_QUERY_DATASET: str(Path(self.with_ontology_graph_file).resolve()),
            KGQA_TOOLS_SCHEMA_DATASET: str(Path(self.without_ontology_graph_file).resolve()),
        }

        print(f"🔌 kgqa-tools server: {command} {' '.join(args)}")
        # keep_alive=False: the subprocess (and its loaded graph) lives exactly as long as
        # this agent's `async with` block, rather than lingering after each building
        self.kgqa_tools_server = MCPToolset(
            StdioTransport(command=command, args=args, keep_alive=False),
            include_instructions=cfg.get('include_instructions', True),
            init_timeout=cfg.get('init_timeout', 120),
        )
        building_filename = Path(self.without_ontology_graph_file).name
        schema_args = {
            **cfg.get('summarize_schema_args', {}),
            **cfg.get('summarize_schema_args_overrides', {}).get(building_filename, {}),
        }
        if schema_args:
            print(f"📌 summarize_schema args fixed for {building_filename}: {schema_args}")

        def pinned_args(tool_name: str) -> Dict[str, Any]:
            if tool_name == 'summarize_schema':
                return {'dataset': KGQA_TOOLS_SCHEMA_DATASET, **schema_args}
            return {'dataset': KGQA_TOOLS_QUERY_DATASET}

        async def pin_args(ctx, call_tool, name, args):
            return await call_tool(name, {**args, **pinned_args(name)})

        self.kgqa_tools_server.process_tool_call = pin_args

        def hide_pinned_params(ctx, tool_defs):
            prepared = []
            for tool_def in tool_defs:
                pinned = pinned_args(tool_def.name)
                schema = copy.deepcopy(tool_def.parameters_json_schema)
                for key in pinned:
                    schema.get('properties', {}).pop(key, None)
                if 'required' in schema:
                    schema['required'] = [k for k in schema['required'] if k not in pinned]
                prepared.append(dataclasses.replace(tool_def, parameters_json_schema=schema))
            return prepared

        hidden = set(cfg.get('hidden_tools', ['load_dataset', 'list_datasets']))
        return (
            self.kgqa_tools_server
            .filtered(lambda ctx, tool_def: tool_def.name not in hidden)
            .prepared(hide_pinned_params)
        )

    async def __aenter__(self):
        """Start the kgqa-tools server (if this run uses one) and load both building graphs
        into it. A no-op for the in-process agents/kgqa.py toolsets."""
        if self.kgqa_tools_server is not None:
            await self.kgqa_tools_server.__aenter__()
            for name, path in self.kgqa_tools_datasets.items():
                result = await self.kgqa_tools_server.direct_call_tool(
                    'load_dataset', {'name': name, 'path': path, 'format': 'turtle'},
                )
                print(f"   -> ✅ Loaded '{name}' into kgqa-tools ({result['triple_count']} triples)")
        return self

    async def __aexit__(self, *exc_info):
        if self.kgqa_tools_server is not None:
            await self.kgqa_tools_server.__aexit__(*exc_info)

    def _prune_bschema_from_history(self, ctx, request_context):
        """
        Before-model-request hook (bschema toolsets only): once cumulative usage for
        this run has crossed half of total_tokens_limit, strip the B-Schema block out
        of the resent first user message so later turns stop re-paying for it. Runs on
        every request past that point, but str.replace is naturally a no-op once the
        block has already been removed, so it only edits anything on the first call
        after crossing the threshold.

        An alternative approach -- injecting a one-time warning near the budget limit
        instead of removing content -- was tried and compared against this on LBNL (the
        one building large enough to threaten the budget at all) and lost on both F1 and
        near-budget rate on 3 of 4 model/shot configs, so pruning is the one attached.
        """
        limit = ctx.usage_limits.total_tokens_limit if ctx.usage_limits else None
        used = ctx.usage.total_tokens if ctx.usage else 0
        if not limit or used < limit / 2:
            return request_context

        for message in request_context.messages:
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str) \
                        and self.bschema in part.content:
                    part.content = part.content.replace(self.bschema, _BSCHEMA_PRUNE_PLACEHOLDER)
                    print(f"✂️  Pruned B-Schema from history ({used}/{limit} tokens used)")

        return request_context

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
                    f"{self.example_block}"
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
                    f"{self.example_block}"
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
