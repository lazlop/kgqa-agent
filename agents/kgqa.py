from mcp.server.fastmcp import FastMCP
from rdflib import Graph, URIRef, Literal, Namespace, BRICK, RDFS, RDF, BNode, SH
from rdflib.term import Variable
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from collections import deque
import os
import sys

from scripts.namespaces import bind_prefixes, get_prefixes, S223, BRICK, convert_to_prefixed

sys.path.insert(0, str(Path(__file__).parent))

# Query diagnosis/execution is delegated to sparql-relax-mcp (github.com/lazlop/sparql-relax)
# instead of the local greedy-pruning implementation rdf-mcp used to carry in
# agents/sparql_ablation.py + agents/sparql_prune.py. We import its FastMCP tool functions
# directly (in-process) rather than reimplementing the diagnosis logic here.
from sparql_relax_mcp.server import (
    load_dataset as _relax_load_dataset,
    diagnose as _relax_diagnose,
    query as _relax_query,
)

# Fixed name under which the current building graph is registered with sparql-relax.
# Hidden from the agent-facing tools below (they hardcode it) so their call signatures
# stay identical to the query(...) / validator(...) tools this file exposed before the swap.
RELAX_DATASET_NAME = "building"

all_tools = FastMCP("GraphDemo")
graph_tools = FastMCP("t2")
bschema_tools = FastMCP("t3")
sparql_only = FastMCP("t4")
sparql_no_validation = FastMCP("t5")
TOOLSETS = {'all_tools': all_tools,'graph_tools': graph_tools,'bschema_tools': bschema_tools,'sparql_only':sparql_only, 'sparql_no_validation': sparql_no_validation}

print("loaded mcp")

# Define namespaces to exclude from shortest path search
QUDT = Namespace("http://qudt.org/schema/qudt/")
EXCLUDED_NAMESPACES = [
    str(RDF),
    str(RDFS),
    str(SH),
    str(QUDT)
]

# Specific predicates to exclude
EXCLUDED_PREDICATES = [
    BRICK.aliasOf,
    BRICK.hasAssociatedTag
]

def _expand_uri(uri, g):
    try:
        uri = g.namespace_manager.expand_curie(uri)
    except Exception as e:
        print("couldn't expand uri due to error, treating as uri: ", e)
        uri = URIRef(uri)
    return uri

def _is_excluded_predicate(pred: URIRef) -> bool:
    """Check if a predicate belongs to an excluded namespace or is a specific excluded predicate."""
    # Check if it's a specifically excluded predicate
    if pred in EXCLUDED_PREDICATES:
        return True
    # Check if it belongs to an excluded namespace
    pred_str = str(pred)
    return any(pred_str.startswith(ns) for ns in EXCLUDED_NAMESPACES)

# ontology can be brick or 223 
ontology_brick = Graph(store = "Oxigraph").parse("https://brickschema.org/schema/1.4/Brick.ttl")
ontology_s223 = Graph(store = "Oxigraph").parse("https://open223.info/223p.ttl")
ontology = ontology_brick + ontology_s223
print('loaded_ontologies')

graph = None
parsed_graph = None
def _ensure_graph_loaded():
    """Lazy load the graphs from the WITHOUT_ONTOLOGY_GRAPH_FILE / WITH_ONTOLOGY_GRAPH_FILE
    environment variables. `graph` (without ontology) is the lean instance-only graph used by
    tools that walk the graph directly (describe_entity, get_relationship_between_classes);
    `parsed_graph` (with ontology) is the full reasoned graph used for actual query execution."""
    print("loading graph...")
    global graph
    global parsed_graph
    if graph is None:
        graph_file = os.getenv("WITHOUT_ONTOLOGY_GRAPH_FILE")
        parsed_graph_file = os.getenv("WITH_ONTOLOGY_GRAPH_FILE")
        if not graph_file:
            # Attempt to fall back to a default test graph if the environment variable is not set
            default_path = os.path.join(os.path.dirname(__file__), "test-building.ttl")
            if os.path.exists(default_path):
                graph_file = default_path
            else:
                raise ValueError("WITHOUT_ONTOLOGY_GRAPH_FILE environment variable not set and default test graph not found")
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        
        graph = Graph(store='Oxigraph')
        graph.parse(graph_file)
        bind_prefixes(graph)
        parsed_graph = Graph(store='Oxigraph')
        parsed_graph.parse(parsed_graph_file)
        bind_prefixes(parsed_graph)
        print(f"✅ Loaded graph from {graph_file} ({len(graph)} triples)")

        # Register the same graph with sparql-relax under a fixed name so the
        # diagnose/query-backed tools below can run against it without the agent
        # ever having to call load_dataset itself.
        _relax_load_dataset(name=RELAX_DATASET_NAME, path=parsed_graph_file, format="turtle")
        print(f"✅ Registered '{RELAX_DATASET_NAME}' with sparql-relax")
    return graph, parsed_graph

def _format_term(term, graph=None) -> Dict[str, Any]:
    """Format an RDF term for JSON serialization with prefixed URIs"""
    if isinstance(term, URIRef):
        if graph is not None:
            prefixed_value = convert_to_prefixed(str(term), graph)
        else:
            prefixed_value = str(term)
        return {'type': 'uri', 'value': prefixed_value}
    elif isinstance(term, Literal):
        result = {'type': 'literal', 'value': str(term)}
        if term.datatype:
            if graph is not None:
                result['datatype'] = convert_to_prefixed(str(term.datatype), graph)
            else:
                result['datatype'] = str(term.datatype)
        if term.language:
            result['language'] = term.language
        return result
    elif isinstance(term, BNode):
        return {'type': 'bnode', 'value': str(term)}
    else:
        return {'type': 'unknown', 'value': str(term)}

@graph_tools.tool()
@all_tools.tool()
def describe_entity(
    entity: str | URIRef
) -> str:
    """
    Get detailed information about a specific entity including all its properties, 
    relationships, and type information. Use this when you need to understand what 
    an entity is, what properties it has, or how it connects to other entities.
    
    When to use:
    - After finding an entity URI and needing to see its complete details
    - To correct identifiers in a SPARQL query after it fails
    
    Args:
        entity: The full URI of the entity
    
    Returns:
        Turtle-formatted RDF graph showing the entity and its immediate connections
    """

    g, parsed_graph = _ensure_graph_loaded() 

    # originally were args
    num_hops = 1 # may want to make an arg again, but it just retrieves so much info
    get_classes = False
    
    # Convert string to URIRef
    if isinstance(entity, str):
        central_uri = _expand_uri(entity, g)
    
    subgraph = Graph(store='Oxigraph')
    
    visited_nodes = set()
    current_layer = {central_uri}
    
    for class_uri in g.objects(central_uri, RDF.type): 
        subgraph.add((central_uri, RDF.type, class_uri))
    
    for hop in range(num_hops):
        next_layer = set()
        
        for node in current_layer:
            if node in visited_nodes:
                continue
            visited_nodes.add(node)
            
            for p, o in g.predicate_objects(node): 
                subgraph.add((node, p, o))
                if isinstance(o, URIRef):
                    next_layer.add(o)
                    for class_uri in g.objects(o, RDF.type): 
                        subgraph.add((o, RDF.type, class_uri))
            
            for s, p in g.subject_predicates(node): 
                subgraph.add((s, p, node))
                if isinstance(s, URIRef):
                    next_layer.add(s)
                    for class_uri in g.objects(s, RDF.type): 
                        subgraph.add((s, RDF.type, class_uri))
        
        current_layer = next_layer
    
    if get_classes:
        for s, p, o in subgraph:
            for class_uri in g.objects(s, RDF.type): 
                subgraph.add((s, RDF.type, class_uri))
            for class_uri in g.objects(o, RDF.type): 
                subgraph.add((o, RDF.type, class_uri))
    
    return subgraph.serialize(format="turtle")

@graph_tools.tool()
@all_tools.tool()
def get_building_summary() -> Dict[str, Any]:
    """
    Get an overview of frequent types of entities, literals and relationships exist in the building model.
    USE THIS FIRST when starting to work with an unfamiliar building model or dataset.
    
    When to use:
    - At the start of a conversation to understand what data is available
    
    Returns:
        Frequent entity types (classes), relationships, and literals
    """
    g, parsed_graph = _ensure_graph_loaded()

    # switching to taking top 50 of each 
    top_n = 50
    percentile = 0  # Exclude bottom 50%
    # Validate percentile
    if not 0.0 <= percentile <= 1.0:
        percentile = 0.0
    
    # Count entities by class using SPARQL
    class_query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    SELECT ?class (COUNT(?entity) as ?count)
    WHERE {
        ?entity rdf:type ?class .
    }
    GROUP BY ?class
    ORDER BY DESC(?count)
    """
    
    class_counts = {}
    for row in g.query(class_query):
        class_name = convert_to_prefixed(row['class'], g)
        class_counts[class_name] = int(row['count'])
    
    # Count relationships by predicate type using SPARQL
    relationship_query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?predicate (COUNT(*) as ?count)
    WHERE {
        ?subject ?predicate ?object .
        FILTER(?predicate != rdf:type)
    }
    GROUP BY ?predicate
    ORDER BY DESC(?count)
    """
    
    relationship_counts = {}
    for row in g.query(relationship_query):
        pred_str = str(row.predicate)
        pred_name = convert_to_prefixed(row.predicate, g)
        relationship_counts[pred_name] = int(row['count'])
    
    # Count literals by predicate and datatype using SPARQL
    literal_query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    SELECT ?object ?datatype (COUNT(*) as ?count)
    WHERE {
        ?subject ?predicate ?object .
        FILTER(isLiteral(?object))
        BIND(DATATYPE(?object) AS ?datatype)
    }
    GROUP BY ?object ?datatype
    ORDER BY DESC(?count)
    """
    
    literal_counts = {}
    for row in g.query(literal_query):
        pred_str = str(row.object)
        datatype_str = convert_to_prefixed(row.datatype, g) if row.datatype else "plain"
        key = f"{pred_str} ({datatype_str})"
        literal_counts[key] = int(row['count'])
    
    # Apply percentile filtering if requested
    prefixes = []
    if percentile > 0.0:
        # Filter classes by percentile
        if class_counts:
            class_values = sorted(class_counts.values())
            percentile_index = int(len(class_values) * percentile)
            class_threshold = class_values[percentile_index] if percentile_index < len(class_values) else 0
            class_counts = {k: v for k, v in class_counts.items() if v > class_threshold}
        
        # Filter relationships by percentile
        if relationship_counts:
            rel_values = sorted(relationship_counts.values())
            percentile_index = int(len(rel_values) * percentile)
            rel_threshold = rel_values[percentile_index] if percentile_index < len(rel_values) else 0
            relationship_counts = {k: v for k, v in relationship_counts.items() if v > rel_threshold}

        # Filter literals by percentile
        if literal_counts:
            lit_values = sorted(literal_counts.values())
            percentile_index = int(len(lit_values) * percentile)
            lit_threshold = lit_values[percentile_index] if percentile_index < len(lit_values) else 0
            literal_counts = {k: v for k, v in literal_counts.items() if v > lit_threshold}
        
    classes = [k for k in class_counts.keys()][:top_n]
    relationships = [k for k in relationship_counts.keys()][:top_n]
    literals = [k for k in literal_counts.keys()][:top_n]

    for k in classes + relationships:
        prefixes.append(str(k).split(":")[0])

    namespaces = {pre: str(ns) for pre, ns in parsed_graph.namespace_manager.namespaces() if str(pre) in prefixes}

    return {
        'prefixes': namespaces,
        "classes": classes,
        "relationships": relationships,
        "literals": literals,
    }

def add_limit_to_sparql(query: str, limit: int = 1000) -> str:
    """Add LIMIT clause to a SPARQL query string."""
    query = query.strip()
    
    # Remove trailing semicolon if present
    if query.endswith(';'):
        query = query[:-1].strip()
    
    # Check if LIMIT already exists (case-insensitive)
    if 'LIMIT' in query.upper():
        # Replace existing LIMIT
        import re
        query = re.sub(r'\bLIMIT\s+\d+\b', f'LIMIT {limit}', query, flags=re.IGNORECASE)
    else:
        # Add new LIMIT
        query = f"{query}\nLIMIT {limit}"
    
    return query

def add_prefixes_to_sparql(query: str, graph) -> str:
    """Add PREFIX declarations before the SELECT portion of a SPARQL query string.
    
    Args:
        query: SPARQL query string (may or may not already have prefixes)
        graph: RDF graph with namespace bindings
    
    Returns:
        Query string with prefixes added after any existing prefixes
    """
    import re
    
    query = query.strip()
    
    # Get prefixes from the graph
    prefixes = get_prefixes(graph)
    
    # Check if query already has PREFIX declarations
    # Match PREFIX lines at the start of the query
    existing_prefix_pattern = r'^(\s*PREFIX\s+\w+:\s*<[^>]+>\s*\n?)+'
    match = re.match(existing_prefix_pattern, query, re.IGNORECASE | re.MULTILINE)
    
    if match:
        # Query already has prefixes, add new ones after them
        existing_prefixes = query[:match.end()]
        query_remainder = query[match.end():].strip()
        return f"{existing_prefixes}{prefixes}\n{query_remainder}"
    else:
        # No existing prefixes, add them before the query
        return f"{prefixes}\n{query}"

@graph_tools.tool()
@all_tools.tool()
def find_entities_by_type(klass: str | URIRef) -> Dict[str, Any]:
    """
    Find all entities of a specific type (class). Use this to discover what specific 
    entities exist in the building model before querying details about them.
    
    When to use:
    - After get_building_summary() to find actual instances of a class
    - To get entity URIs needed for describe_entity()
    - Before writing complex SPARQL queries to verify entities exist
    
    Args:
        klass: Full URI of the class 
    
    Returns:
        List of matching entities with their URIs and labels
    """
    top_n = 20
    g, parsed_graph = _ensure_graph_loaded()
    class_uri = _expand_uri(klass, g)
    include_subclasses = False # may not want this all the time
    # Check if the class exists in the ontology
    if (class_uri, RDF.type, None) not in ontology and \
       (class_uri, RDFS.subClassOf, None) not in ontology:
        return {
            "summary": f"Warning: '{klass}' may not be a valid class in the ontology.",
            "class_searched": convert_to_prefixed(klass, parsed_graph),
            "entities": [],
            "count": 0,
            "include_subclasses": include_subclasses
        }
    
    entities = []
    
    if include_subclasses:
        # Get all subclasses of the target class from the ontology
        subclasses = set([class_uri])
        
        # Query for transitive subclasses
        for subclass in ontology.transitive_subjects(RDFS.subClassOf, class_uri):
            subclasses.add(subclass)
        
        # Find all entities that are instances of any of these classes
        for target_class in subclasses:
            for entity in parsed_graph.subjects(RDF.type, target_class):
                entity_info = {
                    "uri": convert_to_prefixed(str(entity), parsed_graph),
                    # "class": convert_to_prefixed(str(target_class), parsed_graph)
                }
                
                # Try to get a label
                label = None
                for lbl in parsed_graph.objects(entity, RDFS.label):
                    label = str(lbl)
                    break
                
                if label:
                    entity_info["label"] = label
                
                entities.append(entity_info)
    else:
        # Only find direct instances
        for entity in parsed_graph.subjects(RDF.type, class_uri):
            entity_info = {
                "uri": convert_to_prefixed(str(entity), parsed_graph),
                # "class": convert_to_prefixed(klass, parsed_graph)
            }
            
            # Try to get a label
            label = None
            for lbl in g.objects(entity, RDFS.label):
                label = str(lbl)
                break
            
            if label:
                entity_info["label"] = label
            
            entities.append(entity_info)
    
    # Remove duplicates (can happen with multiple class assertions)
    seen_uris = set()
    unique_entities = []
    for entity in entities:
        if entity["uri"] not in seen_uris:
            seen_uris.add(entity["uri"])
            unique_entities.append(entity)
    
    summary = (
        f"Found {len(unique_entities)} entities of type {klass}"
        f"{' (including subclasses)' if include_subclasses else ''}"
    )
    
    return {
        # "summary": summary,
        "class_searched": convert_to_prefixed(klass, parsed_graph),
        "entities": unique_entities[:top_n],
        # "count": len(unique_entities),
        # "include_subclasses": include_subclasses
    }

def _format_rdflib_results(qres) -> Dict[str, Any]:
    """Converts rdflib QueryResult to the same dict format as SPARQLWrapper with prefixed URIs."""
    g, parsed_graph = _ensure_graph_loaded()
    variables = [str(v) for v in qres.vars]
    bindings = []
    for row in qres:
        binding_row = {}
        for var_name in variables:
            term = row[var_name]
            if term is None:
                continue
            
            term_dict = {}
            if isinstance(term, URIRef):
                if parsed_graph is not None:
                    term_dict = {'type': 'uri', 'value': convert_to_prefixed(str(term), parsed_graph)}
                else:
                    term_dict = {'type': 'uri', 'value': str(term)}
            elif isinstance(term, Literal):
                term_dict = {'type': 'literal', 'value': str(term)}
                if term.datatype:
                    if graph is not None:
                        term_dict['datatype'] = convert_to_prefixed(str(term.datatype), graph)
                    else:
                        term_dict['datatype'] = str(term.datatype)
                if term.language:
                    term_dict['xml:lang'] = term.language
            elif isinstance(term, BNode):
                term_dict = {'type': 'bnode', 'value': str(term)}
            
            binding_row[var_name] = term_dict
        bindings.append(binding_row)
    
    return {"results": bindings, "variables": variables}

# @mcp.tool()
def get_sparql_prefixes() -> str:
    """
    Retrieve all prefixes and their associated namespaces for the current graph.
    
    When to use:
    - To understand what prefixes are available for SPARQL queries
    - To help construct SPARQL queries with correct prefixes
    - To understand the ontology structure
    
    Returns:
        Dictionary of prefixes and their namespaces
    """
    g, parsed_graph = _ensure_graph_loaded()
    prefixes = []
    for s,p,o in g:
        if isinstance(s, URIRef):
            prefixes.append(convert_to_prefixed(s,g).split(":")[0])
        if isinstance(p, URIRef):
            prefixes.append(convert_to_prefixed(p,g).split(":")[0])
        if isinstance(o, URIRef):
            prefixes.append(convert_to_prefixed(o,g).split(":")[0])

    pre_ns_list = []
    for pre, ns in parsed_graph.namespace_manager.namespaces():
        if pre in prefixes:
            pre_ns_list.append((pre, ns))

    return "\n".join(
        f"PREFIX {prefix}: <{namespace}>"
        for prefix, namespace in pre_ns_list
    )

def _relax_term_to_binding(term: Optional[Dict[str, Any]], graph_for_prefixes) -> Optional[Dict[str, Any]]:
    """Reshape a sparql-relax term dict ({'type','value','datatype','lang'}) into the
    {'type','value','datatype','language'} shape the rest of this file's tools use, prefixing
    URIs the same way _format_rdflib_results does."""
    if term is None:
        return None
    out: Dict[str, Any] = {"type": term.get("type"), "value": term.get("value")}
    if term.get("type") == "uri":
        out["value"] = convert_to_prefixed(term["value"], graph_for_prefixes)
    if term.get("datatype"):
        out["datatype"] = convert_to_prefixed(term["datatype"], graph_for_prefixes)
    if term.get("lang"):
        out["language"] = term["lang"]
    return out


def _relax_rows_to_bindings(rows: List[Dict[str, Any]], graph_for_prefixes) -> List[Dict[str, Any]]:
    return [
        {var: _relax_term_to_binding(term, graph_for_prefixes) for var, term in row.items()}
        for row in rows
    ]


@sparql_no_validation.tool()
def sparql_snapshot(query: str) -> Dict[str, Any]:
    """
    Retrieves the top 10 results of a SPARQL query, with no diagnosis of why a query
    returns nothing. Only use this for multi-step reasoning, filtering, or complex graph
    pattern analysis.

    When to use:
    - After using simpler tools to better understand complex topologies
    - After using simpler tools to identify specific elements of information needed for a query

    When NOT to use:
    - Simple "list all X" questions → use find_entities_by_type() instead
    - Getting details about one entity → use describe_entity() instead

    Args:
        query: SPARQL SELECT query (prefixes will be added automatically)

    Returns:
        Top 10 query results with bindings for each variable
    """
    g, parsed_graph = _ensure_graph_loaded()
    try:
        print("\n🔎 Getting SPARQL Snapshot")
        query_with_limit = add_limit_to_sparql(query, limit=10000)
        prefixes = get_prefixes(parsed_graph)
        full_query = prefixes + "\n" + query_with_limit

        result = _relax_query(dataset=RELAX_DATASET_NAME, query=full_query, row_limit=10000)
        rows = result.get("rows", [])
        variables = result.get("variables", [])
        bindings = _relax_rows_to_bindings(rows, parsed_graph)

        if bindings:
            summary = f"Query executed successfully on local graph. Found {len(bindings)} results."
            print(f"   -> Retrieved {len(bindings)} results.")
        else:
            summary = "The query executed successfully on the local graph but returned no results."
            print(summary)

        return {
            "summary_string": summary,
            "results": bindings[:10],
            "row_count": len(bindings),
            "col_count": len(variables),
            "syntax_ok": True,
            "error_message": None,
        }
    except Exception as e:
        print(f"   -> SPARQL Query (local) Failed: {e}")
        error_msg = f"The query failed to parse with the following error: {str(e)}"
        return {
            "summary_string": error_msg,
            "results": [],
            "row_count": 0,
            "col_count": 0,
            "syntax_ok": False,
            "error_message": str(e),
        }


@sparql_only.tool()
@graph_tools.tool()
@all_tools.tool()
@bschema_tools.tool()
def sparql_validator(query: str, relax: bool = False) -> Dict[str, Any]:
    """
    Diagnoses a SPARQL query with sparql-relax before returning results, so a query that
    returns nothing comes back with an explanation of *why* (which triple pattern or FILTER
    is broken) instead of a bare empty result. Only use this for checking query syntax or
    complex graph pattern analysis.

    When to use:
    - After using simpler tools to better understand complex topologies
    - After using simpler tools to identify specific elements of information needed for a query

    When NOT to use:
    - Simple "list all X" questions → use find_entities_by_type() instead
    - Getting details about one entity → use describe_entity() instead

    Args:
        query: SPARQL SELECT query (prefixes will be added automatically)
        relax: If True, also search the graph's real edges for a corrected version of each
            broken triple (see 'relaxed_query' in the response). Usually it's more effective
            to leave this False and fix the query yourself based on the diagnosis.

    Returns:
        Top 5 query results with bindings for each variable if the query works; otherwise a
        diagnosis of which triple pattern(s) or FILTER(s) are responsible for the empty result.
    """
    g, parsed_graph = _ensure_graph_loaded()
    try:
        print("\n🔍 Validating SPARQL query...")
        query_with_limit = add_limit_to_sparql(query, limit=10000)
        prefixes = get_prefixes(parsed_graph)
        full_query = prefixes + "\n" + query_with_limit

        report = _relax_diagnose(dataset=RELAX_DATASET_NAME, query=full_query, relax=relax)

        # ── success path ────────────────────────────────────────────────────
        if report["ok"]:
            try:
                q_result = _relax_query(dataset=RELAX_DATASET_NAME, query=full_query, row_limit=5)
                bindings = _relax_rows_to_bindings(q_result.get("rows", []), parsed_graph)
                variables = q_result.get("variables", [])
            except Exception:
                bindings, variables = [], []
            summary = f"Query executed successfully. Found {report['row_count']} results."
            print(f"   -> {summary}")
            return {
                "summary_string": summary,
                "results": bindings,
                "row_count": report["row_count"],
                "col_count": len(variables),
                "syntax_ok": True,
                "error_message": None,
            }

        # ── failure path: translate sparql-relax's diagnosis into a readable summary ──
        print("   -> Query is broken; see diagnosis")
        lines = []
        for i, culprit in enumerate(report["culprits"], 1):
            triples_text = " AND ".join(t["triple"] for t in culprit["triples"])
            fix_note = ""
            if culprit.get("relaxed_query"):
                fix_note = (
                    f" -- possible fix found (would return {culprit['row_count_with_fix']} row(s)); "
                    "call again with relax=true to see it"
                )
            lines.append(f"  {i}. {triples_text}{fix_note}")
        for j, f in enumerate(report["filter_issues"], len(report["culprits"]) + 1):
            lines.append(f"  {j}. FILTER({f['expression']}) -- removing it returns {f['row_count_without_filter']} row(s)")

        summary = report["message"]
        if lines:
            summary += "\n\nBroken triple patterns / filters:\n" + "\n".join(lines)

        return {
            "summary_string": summary,
            "results": [],
            "row_count": report["row_count"],
            "col_count": 0,
            "syntax_ok": True,
            "error_message": None,
        }

    except Exception as e:
        print(f"   -> SPARQL Validator failed: {e}")
        error_msg = f"The query failed to parse with the following error: {str(e)}"
        return {
            "summary_string": error_msg,
            "results": [],
            "row_count": 0,
            "col_count": 0,
            "syntax_ok": False,
            "error_message": str(e),
        }


# Not exposed as an MCP tool: called directly by agents/agent.py's evaluation harness to
# score a generated query against the ground truth, independent of whichever toolset the
# agent itself used.
def sparql_query(query: str, result_length: int = 100000) -> Dict[str, Any]:
    """
    Execute a SPARQL query and return its full results (up to result_length rows).
    Used internally for evaluation; not exposed to agents as an MCP tool (they get
    sparql_validator / sparql_snapshot instead).

    Args:
        query: SPARQL SELECT query (prefixes will be added automatically)
        result_length: Maximum results to return (default: 100000)

    Returns:
        Query results with bindings for each variable
    """
    g, parsed_graph = _ensure_graph_loaded()
    try:
        print("\n🔎 Running SPARQL query...")
        limit = result_length if result_length >= 1 else 100000
        query_with_limit = add_limit_to_sparql(query, limit=limit)
        prefixes = get_prefixes(parsed_graph)
        full_query = prefixes + "\n" + query_with_limit

        result = _relax_query(dataset=RELAX_DATASET_NAME, query=full_query, row_limit=limit)
        rows = result.get("rows", [])
        variables = result.get("variables", [])
        bindings = _relax_rows_to_bindings(rows, parsed_graph)[:result_length]

        summary = f"Query executed successfully on local graph. Found {len(bindings)} results."
        print(summary)
        if not bindings:
            summary = "The query executed successfully on the local graph but returned no results."

        return {
            "summary_string": summary,
            "results": bindings,
            "row_count": len(bindings),
            "col_count": len(variables),
            "syntax_ok": True,
            "error_message": None,
        }
    except Exception as e:
        print(f"   -> SPARQL Query (local) Failed: {e}")
        error_msg = f"The query failed to parse with the following error: {str(e)}"
        return {
            "summary_string": error_msg,
            "results": [],
            "row_count": 0,
            "col_count": 0,
            "syntax_ok": False,
            "error_message": str(e),
        }

@graph_tools.tool()
@all_tools.tool()
def get_relationship_between_classes(
    start_class_uri: str,
    end_class_uri: str
) -> Dict[str, Any]:
    """
    Find the shortest path between instances of two classes in the graph.
    The path starts from an instance of start_class (via inverse rdf:type) and 
    ends at an instance of end_class (via rdf:type).

    When to use:
    - Finding the relationship between two types of entities
    - Questions about paths or indirect relationships between entities
    
    Args:
        start_class_uri: The URI of the starting class
        end_class_uri: The URI of the ending class
    
    Returns:
        Dictionary containing the shortest path information including:
        - path: List of nodes in the path (class -> instance -> ... -> instance -> class)
        - predicates: List of predicates connecting the nodes
        - length: Length of the path
        - found: Whether a path was found
    """
    g, parsed_graph = _ensure_graph_loaded()

    max_depth: int = 7
    
    # Convert strings to URIRefs
    start_class = _expand_uri(start_class_uri, g)
    end_class = _expand_uri(end_class_uri, g)
    
    # Get all instances of the start class
    start_instances = list(g.subjects(RDF.type, start_class))
    if not start_instances:
        return {
            "summary": f"No instances found for start class: {start_class_uri}",
            "found": False,
            "path": [],
            "predicates": [],
            "length": 0,
            "error": "no_start_instances"
        }
    
    # Get all instances of the end class
    end_instances = set(g.subjects(RDF.type, end_class))
    if not end_instances:
        return {
            "summary": f"No instances found for end class: {end_class_uri}",
            "found": False,
            "path": [],
            "predicates": [],
            "length": 0,
            "error": "no_end_instances"
        }
    
    # Check if any instance belongs to both classes
    common_instances = set(start_instances) & end_instances
    if common_instances:
        instance = list(common_instances)[0]
        return {
            "summary": f"Found common instance of both classes",
            "found": True,
            "path": [str(start_class), str(instance), str(end_class)],
            "predicates": ["^rdf:type", "rdf:type"],
            "length": 2
        }
    
    # Find shortest path from any start instance to any end instance
    result = _find_instance_to_instance_path(g, start_instances, end_instances, max_depth)
    
    if result["found"]:
        # Prepend start class and append end class to the path
        full_path = [convert_to_prefixed(start_class, parsed_graph)] + [convert_to_prefixed(p, parsed_graph) for p in result["path"]] + [convert_to_prefixed(end_class, parsed_graph)]
        full_predicates = ["^rdf:type"] + [convert_to_prefixed(p, parsed_graph) for p in result["predicates"]] + ["rdf:type"]
        
        path_str = " -> ".join(full_path)
        pred_str = " -> ".join(full_predicates)
        summary = (
            f"Path: {path_str}\n"
            f"Predicates: {pred_str}"
        )
        
        return {
            "summary": summary,
            "found": True,
            "path": full_path,
            "predicates": full_predicates,
            "length": len(full_path) - 1
        }
    else:
        summary = f"No path found between instances of {start_class_uri} and {end_class_uri} within {max_depth} hops"
        return {
            "summary": summary,
            "found": False,
            "path": [],
            "predicates": [],
            "length": 0
        }

def _find_instance_to_instance_path(
    g: Graph,
    start_instances: List[URIRef],
    end_instances: set,
    max_depth: int
) -> Dict[str, Any]:
    """
    Find shortest path from any instance in start_instances to any instance in end_instances.
    Uses BFS to explore from all start instances simultaneously.
    """
    # Queue stores: (current_node, path_nodes, path_predicates)
    queue = deque()
    visited = set()
    
    # Initialize queue with all start instances
    for instance in start_instances:
        queue.append((instance, [instance], []))
        visited.add(instance)
    
    while queue:
        current, path_nodes, path_predicates = queue.popleft()
        
        # Check if we've exceeded max depth
        if len(path_nodes) > max_depth:
            continue
        
        # Check if current node is an end instance
        if current in end_instances:
            return {
                "found": True,
                "path": [str(node) for node in path_nodes],
                "predicates": [str(pred) for pred in path_predicates],
                "length": len(path_nodes) - 1
            }
        
        # Explore outgoing edges (current as subject)
        for pred, obj in g.predicate_objects(current):
            # Skip predicates from excluded namespaces (RDF, RDFS, SHACL, QUDT)
            if _is_excluded_predicate(pred):
                continue
            if isinstance(obj, URIRef):
                if obj in end_instances:
                    # Found a target instance
                    return {
                        "found": True,
                        "path": [str(node) for node in path_nodes + [obj]],
                        "predicates": [str(pred) for pred in path_predicates + [pred]],
                        "length": len(path_nodes)
                    }
                
                if obj not in visited:
                    visited.add(obj)
                    queue.append((obj, path_nodes + [obj], path_predicates + [pred]))
        
        # Explore incoming edges (current as object)
        for subj, pred in g.subject_predicates(current):
            # Skip predicates from excluded namespaces (RDF, RDFS, SHACL, QUDT)
            if _is_excluded_predicate(pred):
                continue
            if isinstance(subj, URIRef):
                if subj in end_instances:
                    # Found a target instance
                    return {
                        "found": True,
                        "path": [str(node) for node in path_nodes + [subj]],
                        "predicates": [str(pred) for pred in path_predicates + [f"^{pred}"]],
                        "length": len(path_nodes)
                    }
                
                if subj not in visited:
                    visited.add(subj)
                    queue.append((subj, path_nodes + [subj], path_predicates + [f"^{pred}"]))
    
    return {
        "found": False,
        "path": [],
        "predicates": [],
        "length": 0
    }

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Literal as typeLiteral
from rdflib import Namespace

DEFAULT_EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"

# Common namespaces
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")


class GraphURIFinder:
    """Fuzzy search for classes and predicates in an RDF graph."""
    
    def __init__(self, graph, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the URI finder with a graph.
        
        Args:
            graph: rdflib.Graph instance
            embedding_model: Name of the sentence transformer model to use
        """
        self.graph = graph
        self.embedding_model = SentenceTransformer(embedding_model)
        self.metadatas = []
        self.embeddings = None
        self._build_index()
    
    def _extract_local_name(self, uri: str) -> str:
        """Extract the local name from a URI."""
        uri_str = str(uri)
        if '#' in uri_str:
            return uri_str.split('#')[-1]
        elif '/' in uri_str:
            return uri_str.split('/')[-1]
        return uri_str
    
    def _extract_classes(self) -> List[Dict]:
        """Extract class information from the graph."""
        query = """
        SELECT DISTINCT ?klass ?label ?comment
        WHERE {
            ?subject a ?klass .
            
            OPTIONAL { ?klass rdfs:label ?label }
            OPTIONAL { ?klass rdfs:comment ?comment }
        }
        ORDER BY ?klass
        """
        
        classes = []
        try:
            results = self.graph.query(query)
            
            for row in results:
                class_uri = str(row['klass'])
                label = str(row['label']) if row['label'] else self._extract_local_name(class_uri)
                comment = str(row['comment']) if row['comment'] else ""
                
                # Get parent classes
                parents = []
                for parent in self.graph.objects(row['klass'], RDFS['subClassOf']):
                    parent_name = self._extract_local_name(str(parent))
                    if parent_name not in ['Class', 'Resource']:  # Skip generic parents
                        parents.append(parent_name)
                
                class_info = {
                    'uri': class_uri,
                    'label': label,
                    'local_name': self._extract_local_name(class_uri),
                    'comment': comment,
                    'parents': ', '.join(parents),
                    'type': 'class'
                }
                classes.append(class_info)
                
        except Exception as e:
            print(f"Failed to extract classes: {e}")
            import traceback
            traceback.print_exc()
            
        return classes
    
    def _extract_predicates(self) -> List[Dict]:
        """Extract predicate/property information from the graph."""
        query = """
        SELECT DISTINCT ?predicate ?label ?comment
        WHERE {
            ?s ?predicate ?o .
            
            OPTIONAL { ?predicate rdfs:label ?label }
            OPTIONAL { ?predicate rdfs:comment ?comment }
        }
        ORDER BY ?predicate
        """
        
        predicates = []
        try:
            results = self.graph.query(query)
            
            for row in results:
                pred_uri = str(row['predicate'])
                label = str(row['label']) if row['label'] else self._extract_local_name(pred_uri)
                comment = str(row['comment']) if row['comment'] else ""
                
                pred_info = {
                    'uri': pred_uri,
                    'label': label,
                    'local_name': self._extract_local_name(pred_uri),
                    'comment': comment,
                    'parents': '',
                    'type': 'predicate'
                }
                predicates.append(pred_info)
                
        except Exception as e:
            print(f"Failed to extract predicates: {e}")
            import traceback
            traceback.print_exc()
            
        return predicates
    
    def _build_index(self):
        """Build the search index from the graph."""
        # Extract classes and predicates
        classes = self._extract_classes()
        predicates = self._extract_predicates()
        items = classes + predicates
        
        if not items:
            print("No classes or predicates found in graph")
            return
        
        # Build searchable documents
        documents = []
        self.metadatas = []
        
        for item in items:
            # Create searchable text
            searchable_parts = [
                item['local_name'],
                item['label'],
                item['comment'],
                item['parents']
            ]
            searchable_text = ' '.join(filter(None, searchable_parts))
            
            documents.append(searchable_text)
            self.metadatas.append(item)
        
        # Generate embeddings
        if documents:
            self.embeddings = self.embedding_model.encode(documents)
            print(f"Indexed {len(documents)} items ({len(classes)} classes, {len(predicates)} predicates)")
    
    def find_similar(
        self, 
        query: str, 
        search_type: typeLiteral["both", "class", "predicate"] = "both",
        n_results: int = 5
    ) -> List[Dict]:
        """
        Find URIs similar to the given query string.
        
        Args:
            query: Search query string
            search_type: What to search for - "both", "classes", or "predicates"
            n_results: Number of results to return
            
        Returns:
            List of dictionaries containing similar URIs and metadata
        """
        if self.embeddings is None or len(self.metadatas) == 0:
            return []
        try:
            # Filter by type if requested
            if search_type == "both":
                filtered_embeddings = self.embeddings
                filtered_indices = list(range(len(self.metadatas)))
            else:
                filtered_indices = [
                    i for i, meta in enumerate(self.metadatas)
                    if meta['type'] == search_type  # "class" or "predicate"
                ]

                if not filtered_indices:
                    return []
                filtered_embeddings = self.embeddings[filtered_indices]
            
            # Compute similarity
            query_embedding = self.embedding_model.encode(query)
            similarities = self.embedding_model.similarity(
                filtered_embeddings, 
                query_embedding
            ).squeeze(1)
            # Get top k results
            n_results = min(n_results, len(filtered_indices))
            topk_indices = similarities.topk(n_results).indices.tolist()

            # Convert topk_indices to integers if they're strings
            topk_indices = [int(i) if isinstance(i, str) else i for i in topk_indices]
            # Map back to original indices and return matching metadata
            original_indices = [filtered_indices[i] for i in topk_indices]
            # Convert original_indices to integers if they're strings
            original_indices = [int(i) for i in original_indices]
            matches = [self.metadatas[i] for i in original_indices]

            return matches
            
        except Exception as e:
            print(f"Failed to find similar URIs: {e}")
            import traceback
            traceback.print_exc()
            return []


# Global URI finder instance
_uri_finder: Optional[GraphURIFinder] = None

def _ensure_uri_finder_loaded():
    """
    Ensure the URI finder is loaded and initialized with the current graph.
    Rebuilds the index if the graph has changed.
    
    Returns:
        Initialized GraphURIFinder instance
    """
    global _uri_finder
    
    g, parsed_graph = _ensure_graph_loaded()
    
    # Rebuild if graph changed or not yet initialized
    if _uri_finder is None or _uri_finder.graph != g:
        print("Building fuzzy search index from graph...")
        _uri_finder = GraphURIFinder(g)
    
    return _uri_finder

@all_tools.tool()
def find_similar_class_in_graph(
    input_string: str,
    concept_type: typeLiteral["both", "classes", "predicates"] = "both",
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Find classes or predicates that are similar to the search input_string using simple semantic similarity.
    These classes or predicates are used within the model of the building.
    Default behavior retrieves 5 likely classes.
    
    When to use:
    - When you want to make sure of the the exact class or property name to use
    
    Args:
        concept_string: Natural language description or partial name to find similar classes for
        search_type: 
                    - "both": Search both classes and predicates (default)
                    - "classes": Search only classes
                    - "predicates": Search only predicates
        n_results: Number of similar results to return (default: 5)
    
    Returns:
        Dictionary containing:
        - matches: List of matching items with uri

    """
    finder = _ensure_uri_finder_loaded()
    
        # Validate search_type
    if concept_type == "classes":
        concept_type = "class"
    elif concept_type == "predicates":
        concept_type = "predicate"

    if concept_type not in ["both", "class", "predicate"]:
        return {
            "error": f"Invalid concept_type '{concept_type}'. Must be 'both', 'classes', or 'predicates'",
        }
    
    try:
        finder = _ensure_uri_finder_loaded()
        
        matches = finder.find_similar(
            query=input_string, 
            search_type=concept_type,
            n_results=n_results
        )
        match_uri_list = [convert_to_prefixed(dct['uri'], parsed_graph) for dct in matches]
        return {
            "matches": match_uri_list,
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"Search failed: {str(e)}",
        }
    
@all_tools.tool()
def describe_class(
    class_uri: str
) -> Dict[str, Any]:
    """
    Get detailed information about a class from the ontology including its definition,
    parents, children, and related predicates 
    
    When to use:
    - To understand what a class represents and how it relates to other classes
    - To discover what properties can be used with instances of this class
    - To understand the class hierarchy around a particular class
    
    Args:
        class_uri: The full URI of the class to describe
    
    Returns:
        Dictionary containing:
        - comment: rdfs:comment defining the class
        - parents: List of parent classes 
        - children: List of child classes (
        - from_predicates: Predicates that can relate entities of this class to other entities
        - to_predicates: Predicates that can relate other entities to entities of this class
    """
    g, parsed_graph = _ensure_graph_loaded()
    
    class_uri = _expand_uri(class_uri, g)

    result = {
        "class": convert_to_prefixed(class_uri, parsed_graph),
        "comment": None,
        "parents": [],
        "children": [],
        "from_predicates": [],
        "to_predicates": []
    }
    # testing with parsed_graph instead
    ontology = parsed_graph
    # Get rdfs:comment
    for comment in ontology.objects(class_uri, RDFS.comment):
        result["comment"] = str(comment)
        break  # Take first comment
    
    # Get parent classes (rdfs:subClassOf)
    for parent in ontology.objects(class_uri, RDFS.subClassOf):
        if isinstance(parent, URIRef):
            result["parents"].append(convert_to_prefixed(str(parent), parsed_graph))
    
    # Get child classes (inverse rdfs:subClassOf)
    for child in ontology.subjects(RDFS.subClassOf, class_uri):
        if isinstance(child, URIRef):
            result["children"].append(convert_to_prefixed(str(child), parsed_graph))
    
    # Track predicates we've already added to avoid duplicates
    from_predicates_seen = set()
    to_predicates_seen = set()
    
    # Helper function to get all ancestor classes
    def get_ancestors(cls):
        ancestors = set()
        to_visit = [cls]
        while to_visit:
            current = to_visit.pop()
            for parent in ontology.objects(current, RDFS.subClassOf):
                if isinstance(parent, URIRef) and parent not in ancestors:
                    ancestors.add(parent)
                    to_visit.append(parent)
        return ancestors
    
    # Get all classes to check (this class + all ancestors)
    classes_to_check = [class_uri] + list(get_ancestors(class_uri))
    
    # Helper to add predicate info
    def add_from_predicate(pred_str, source, property_shape=None):
        if pred_str not in from_predicates_seen:
            from_predicates_seen.add(pred_str)
            pred_info = {
                "predicate": convert_to_prefixed(pred_str, parsed_graph),
                "comment": None,
                # "source": source
            }
            predicate = URIRef(pred_str)
            # Try to get comment/description
            if property_shape:
                for name in ontology.objects(property_shape, SH.name):
                    pred_info["comment"] = str(name)
                    break
                if not pred_info["comment"]:
                    for desc in ontology.objects(property_shape, SH.description):
                        pred_info["comment"] = str(desc)
                        break
            if not pred_info["comment"]:
                for comment in ontology.objects(predicate, RDFS.comment):
                    pred_info["comment"] = str(comment)
                    break
            result["from_predicates"].append(pred_info)
    
    def add_to_predicate(pred_str, source, property_shape=None):
        if pred_str not in to_predicates_seen:
            to_predicates_seen.add(pred_str)
            pred_info = {
                "predicate": convert_to_prefixed(pred_str, parsed_graph),
                "comment": None,
                # "source": source
            }
            predicate = URIRef(pred_str)
            # Try to get comment/description
            if property_shape:
                for name in ontology.objects(property_shape, SH.name):
                    pred_info["comment"] = str(name)
                    break
                if not pred_info["comment"]:
                    for desc in ontology.objects(property_shape, SH.description):
                        pred_info["comment"] = str(desc)
                        break
            if not pred_info["comment"]:
                for comment in ontology.objects(predicate, RDFS.comment):
                    pred_info["comment"] = str(comment)
                    break
            result["to_predicates"].append(pred_info)
    
    # Check each class (original + ancestors)
    for check_class in classes_to_check:
        # Get predicates where this class is the domain (from rdfs:domain)
        for predicate in ontology.subjects(RDFS.domain, check_class):
            if isinstance(predicate, URIRef):
                add_from_predicate(str(predicate), "rdfs:domain")
        
        # Get predicates where this class is the range (from rdfs:range)
        for predicate in ontology.subjects(RDFS.range, check_class):
            if isinstance(predicate, URIRef):
                add_to_predicate(str(predicate), "rdfs:range")
        
        # SHACL: Check if this class is a NodeShape with properties
        # This handles the pattern where a class defines sh:property directly (Brick style)
        for prop_shape in ontology.objects(check_class, SH.property):
            # Get sh:path (the predicate)
            for path in ontology.objects(prop_shape, SH.path):
                if isinstance(path, URIRef):
                    add_from_predicate(str(path), "shacl:property", prop_shape)
                    
                    # Also check for range constraints on this property
                    for target_class in ontology.objects(prop_shape, SH['class']):
                        if isinstance(target_class, URIRef):
                            # This predicate points TO the target_class
                            pass  # We'll handle this when checking that class
                    
                    for node_ref in ontology.objects(prop_shape, SH.node):
                        # Check if this node reference has a targetClass
                        for target_class in ontology.objects(node_ref, SH.targetClass):
                            if isinstance(target_class, URIRef):
                                pass  # We'll handle this when checking that class
        
        # SHACL: Check for NodeShapes that target this class (223 style)
        for node_shape in ontology.subjects(SH.targetClass, check_class):
            # Get properties defined in this NodeShape
            for prop in ontology.objects(node_shape, SH.property):
                # Get sh:path (the predicate)
                for path in ontology.objects(prop, SH.path):
                    if isinstance(path, URIRef):
                        add_from_predicate(str(path), "shacl:targetClass", prop)
        
        # SHACL: Look for property shapes where sh:class points to this class
        for property_shape in ontology.subjects(SH['class'], check_class):
            # Get sh:path from this property shape
            for path in ontology.objects(property_shape, SH.path):
                if isinstance(path, URIRef):
                    add_to_predicate(str(path), "shacl:class", property_shape)
        
        # SHACL: Look for property shapes where sh:node points to a NodeShape targeting this class
        for node_ref in ontology.subjects(SH.targetClass, check_class):
            for property_shape in ontology.subjects(SH.node, node_ref):
                # Get sh:path from this property shape
                for path in ontology.objects(property_shape, SH.path):
                    if isinstance(path, URIRef):
                        add_to_predicate(str(path), "shacl:node", property_shape)
    
    return result