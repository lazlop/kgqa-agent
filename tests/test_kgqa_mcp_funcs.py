"""
test_mcp.py - Basic test script for GraphDemo MCP tools
"""
import os
from pprint import pprint
import sys
from rdflib import URIRef
# Set up environment variables for graph files
# Update this path to point to your actual Brick TTL file
# GRAPH_FILE = '../data/eval_buildings/without-ontology/b59.ttl'
GRAPH_FILE = "test-building.ttl"
os.environ["WITHOUT_ONTOLOGY_GRAPH_FILE"] = GRAPH_FILE
os.environ["WITH_ONTOLOGY_GRAPH_FILE"] = GRAPH_FILE
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the MCP server functions
from scripts.namespaces import BRICK, S223
from agents.kgqa import (
    describe_entity,
    sparql_query,
    get_building_summary,
    find_entities_by_type,
    get_relationship_between_classes,
    find_similar_class_in_graph,
    get_sparql_prefixes,
    describe_class,
    sparql_validator,
    _ensure_graph_loaded
)

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_graph_loading():
    """Test 1: Verify graph loads correctly"""
    print_section("TEST 1: Graph Loading")
    try:
        graph, pg = _ensure_graph_loaded()
        print(f"✅ Graph loaded successfully")
        print(f"   Total triples: {len(graph)}")
        print(" get sparql prefixes: ")
        prefixes = get_sparql_prefixes()
        print(type(prefixes))
        pprint(prefixes)
        return True
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        return False

def test_building_summary():
    """Test 2: Get building summary"""
    print_section("TEST 2: Building Summary")
    try:
        result = get_building_summary()
        print(f"✅ {result}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_find_entities_by_type():
    """Test 3: Find entities by type"""
    print_section("TEST 3: Find Entities by Type")
    
    # Test with common Brick classes - modify based on what's in your model
    test_classes = [
        "VAV",
        "Temperature_Sensor", 
        "Air_Handling_Unit",
        "Zone",
        "Equipment"
    ]
    
    for brick_class in test_classes:
        try:
            result = find_entities_by_type(BRICK[brick_class])
            print(list(result.values())[:5])
        except Exception as e:
            print(f"❌ Failed for {brick_class}: {e}")
    
    # Test without subclasses
    print("\n🔍 Testing without subclasses (Equipment only):")
    try:
        result = find_entities_by_type(BRICK["Equipment"])
        print(list(result.values())[:5])
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_entity_info():
    """Test 4: Get entity info"""
    print_section("TEST 4: Entity Info")
    
    # First, find an entity to query
    try:
        result = find_entities_by_type(BRICK["VAV"])
        entity_uri = result['entities'][0]['uri']
        print(f"📍 Testing with entity: {entity_uri}")
        
        # Get 1-hop info
        info = describe_entity(entity_uri)
        print(f"✅ Retrieved entity info (1 hop)")
        print(f"   Turtle output length: {len(info)} characters")
        print(f"\n   First 500 characters:")
        print(f"   {info[:500]}...")
        # print(info)
        # print('with 2 hops:')
        # info = describe_entity(entity_uri)
        # print(f"✅ Retrieved entity info (2 hops)")
        # print(f"   Turtle output length: {len(info)} characters")
        # print(f"\n   First 500 characters:")
        # print(f"   {info[:500]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

def test_sparql_query():
    """Test 5: SPARQL queries"""
    print_section("TEST 5: SPARQL Queries")
    
    # Test 1: Count all entities
    query1 = """
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    SELECT (COUNT(?entity) as ?count)
    WHERE {
        ?entity rdf:type ?class .
        FILTER(STRSTARTS(STR(?class), STR(brick:)))
    }
    """
    
    try:
        result = sparql_query(query1)
        print(f"✅ Query 1: {result['summary_string']}")
        if result['results']:
            print(f"   Total entities: {result['results'][0].get('count', {}).get('value', 'N/A')}")
    except Exception as e:
        print(f"❌ Query 1 failed: {e}")
    
    # Test 2: Get all VAVs with their labels
    query2 = """
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?vav ?label
    WHERE {
        ?vav rdf:type brick:VAV .
        OPTIONAL { ?vav rdfs:label ?label }
    }
    LIMIT 5
    """
    
    try:
        result = sparql_query(query2)
        print(f"\n✅ Query 2: {result['summary_string']}")
        # if result['results']:
        #     print(f"   Found {len(result['results'])} VAVs (limited to 5):")
        #     for row in result['results']:
        #         vav = row.get('vav', {}).get('value', 'N/A')
        #         label = row.get('label', {}).get('value', 'No label')
        #         print(f"   - {label}")
    except Exception as e:
        print(f"❌ Query 2 failed: {e}")
    
    # Test 3: Incorrect overspecified
    query2 = """
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?vav ?label
    WHERE {
        ?vav rdf:type brick:VAV .
        ?vav rdf:type brick:Air_Handling_Unit .
        OPTIONAL { ?vav rdfs:label ?label }
    }
    LIMIT 5
    """
    
    try:
        result = sparql_validator(query2)
        print(f"\n✅ Query 3: {result['summary_string']}")
        # if result['results']:
        #     print(f"   Found {len(result['results'])} VAVs (limited to 5):")
        #     for row in result['results']:
        #         vav = row.get('vav', {}).get('value', 'N/A')
        #         label = row.get('label', {}).get('value', 'No label')
        #         print(f"   - {label}")
    except Exception as e:
        print(f"❌ Query 2 failed: {e}")

def test_shortest_path():
    """Test 6: Find shortest path between entities"""
    print_section("TEST 6: Shortest Path Finding")
    entity1_uri = BRICK['VAV']
    entity2_uri = BRICK['Valve_Command']
    # First, find two entities to test with
    try:
        # Test 1: Bidirectional search
        print("\n🔍 Test 1: Bidirectional BFS")
        path_result = get_relationship_between_classes(entity1_uri, entity2_uri)
        print(f"   {path_result['summary']}")
        if path_result['found']:
            print(f"   Path length: {path_result['length']}")
            print(f"   predicates: {path_result['predicates']}")
            if path_result['length'] <= 3:  # Only print full path if short
                print(f"   Full path:")
                for i, node in enumerate(path_result['path']):
                    print(f"     {i}. {node}")
                    if i < len(path_result['predicates']):
                        print(f"        --[{path_result['predicates'][i]}]-->")
        
        # Test 2: Unidirectional search
        print("\n🔍 Test 2: Unidirectional BFS")
        path_result = get_relationship_between_classes(entity1_uri, entity2_uri)
        print(f"   {path_result['summary']}")
        if path_result['found']:
            print(f"   Path length: {path_result['length']}")
        
        # Test 3: Same start and end
        print("\n🔍 Test 3: Same start and end URI")
        path_result = get_relationship_between_classes(entity1_uri, entity1_uri)
        print(f"   {path_result['summary']}")
        print(f"   Found: {path_result['found']}, Length: {path_result['length']}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test 7: Error handling"""
    print_section("TEST 7: Error Handling")
    
    # Test invalid class
    try:
        result = find_entities_by_type("InvalidClassName")
        print(f"⚠️  Invalid class test: {result['summary']}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test invalid SPARQL
    try:
        result = sparql_query("INVALID SPARQL QUERY")
        print(f"⚠️  Invalid SPARQL test: {result['summary_string']}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_fuzzy_search():
    """Test 8: Fuzzy search"""
    print_section("TEST 8: Fuzzy Search")
    try:
        results = find_similar_class_in_graph("heating coil", search_type="classes", n_results=3)
        print(f"✅ Fuzzy search results for 'heating coil':")
        pprint(results)
        results = find_similar_class_in_graph("heating coil", search_type="predicates", n_results=3)
        print(f"✅ Fuzzy search results for 'heating coil':")
        pprint(results)
        results = find_similar_class_in_graph("heating coil", search_type="both", n_results=3)
        print(f"✅ Fuzzy search results for 'heating coil':")
        pprint(results)
    except Exception as e:
        print(f"❌ Fuzzy search failed: {e}")

def test_describe_class():
    """Test 9: Describe class from ontology"""
    print_section("TEST 9: Describe Class")
    
    # Test with various Brick classes
    test_classes = [
        ('brick:VAV', "VAV"),
        ('brick:Temperature_Sensor', "Temperature_Sensor"),
        ("brick:Air_Handling_Unit", "Air_Handling_Unit"),
        ("brick:Equipment", "223:Equipment"),
    ]
    
    for class_uri, class_name in test_classes:
        try:
            print(f"\n🔍 Testing class: {class_name}")
            result = describe_class(str(class_uri))
            
            print(f"✅ Class: {result['class']}")
            print(result)
                
        except Exception as e:
            print(f"❌ Failed for {class_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test with an invalid class
    print(f"\n🔍 Testing invalid class:")
    try:
        result = describe_class("http://example.com/InvalidClass")
        print(f"⚠️  Result for invalid class:")
        pprint(result)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Run all tests"""
    print("\n" + "🧪"*40)
    print("  MCP GraphDemo Test Suite")
    print("🧪"*40)
    
    # Verify graph file exists
    if not os.path.exists(GRAPH_FILE):
        print(f"\n❌ ERROR: Graph file not found: {GRAPH_FILE}")
        print("Please update GRAPH_FILE path in this test script")
        return
    
    print(f"\n📁 Using graph file: {GRAPH_FILE}")
    
    # Run tests
    tests = [
        # test_graph_loading,
        # test_building_summary,
        # test_find_entities_by_type,
        # test_entity_info,
        test_sparql_query,
        # test_shortest_path,
        # test_error_handling,
        # test_fuzzy_search,
        # test_describe_class,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

if __name__ == "__main__":
    main()