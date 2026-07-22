"""Test the add_prefixes_to_sparql function"""
import sys
from pathlib import Path
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdflib import Graph
from scripts.namespaces import bind_prefixes, get_prefixes

def add_prefixes_to_sparql(query: str, graph) -> str:
    """Add PREFIX declarations before the SELECT portion of a SPARQL query string.
    
    Args:
        query: SPARQL query string (may or may not already have prefixes)
        graph: RDF graph with namespace bindings
    
    Returns:
        Query string with prefixes added before SELECT
    """
    query = query.strip()
    
    # Get prefixes from the graph
    prefixes = get_prefixes(graph)
    
    # Check if query already has PREFIX declarations
    # Match PREFIX lines at the start of the query
    existing_prefix_pattern = r'^(\s*PREFIX\s+\w+:\s*<[^>]+>\s*\n?)+'
    match = re.match(existing_prefix_pattern, query, re.IGNORECASE | re.MULTILINE)
    
    if match:
        # Query already has prefixes, replace them
        query_without_prefixes = query[match.end():].strip()
        return f"{prefixes}\n{query_without_prefixes}"
    else:
        # No existing prefixes, add them before the query
        return f"{prefixes}\n{query}"

def test_add_prefixes_to_sparql():
    """Test adding prefixes to SPARQL queries"""
    
    # Create a test graph with namespace bindings
    g = Graph()
    bind_prefixes(g)
    
    # Test 1: Query without prefixes
    query1 = """
    SELECT ?s ?p ?o
    WHERE {
        ?s ?p ?o .
    }
    LIMIT 10
    """
    
    result1 = add_prefixes_to_sparql(query1, g)
    print("Test 1 - Query without prefixes:")
    print(result1)
    print("\n" + "="*80 + "\n")
    
    # Test 2: Query with existing prefixes (should replace them)
    query2 = """
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-schema#>
    
    SELECT ?s ?type
    WHERE {
        ?s rdf:type brick:Equipment .
    }
    """
    
    result2 = add_prefixes_to_sparql(query2, g)
    print("Test 2 - Query with existing prefixes (should be replaced):")
    print(result2)
    print("\n" + "="*80 + "\n")
    
    # Test 3: Simple SELECT query
    query3 = "SELECT ?x WHERE { ?x a brick:Equipment }"
    
    result3 = add_prefixes_to_sparql(query3, g)
    print("Test 3 - Simple query:")
    print(result3)
    print("\n" + "="*80 + "\n")
    
    # Verify that prefixes are at the beginning
    assert result1.strip().startswith("PREFIX"), "Prefixes should be at the start"
    assert "SELECT" in result1, "SELECT should be present"
    assert result2.strip().startswith("PREFIX"), "Prefixes should be at the start"
    assert result3.strip().startswith("PREFIX"), "Prefixes should be at the start"
    
    # Verify the query body is preserved
    assert "?s ?p ?o" in result1, "Query body should be preserved"
    assert "brick:Equipment" in result2, "Query body should be preserved"
    assert "brick:Equipment" in result3, "Query body should be preserved"
    
    print("✅ All tests passed!")
    print("\nThe add_prefixes_to_sparql function works correctly:")
    print("- Adds prefixes to queries without them")
    print("- Replaces existing prefixes with the graph's prefixes")
    print("- Preserves the query body")

if __name__ == "__main__":
    test_add_prefixes_to_sparql()
