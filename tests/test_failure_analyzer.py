"""
test_failure_analyzer.py

Quick test script to verify the failure analyzer works correctly.
"""
import pandas as pd
from analyze_failures import FailureClassification, FailureAnalyzer
import asyncio


async def test_classification_model():
    """Test that the Pydantic model works correctly."""
    print("Testing FailureClassification model...")
    
    classification = FailureClassification(
        primary_category="Vocabulary Hallucination",
        confidence=0.95,
        evidence="Used brick:feedsAir instead of brick:feeds",
        secondary_category=None,
        explanation="The model hallucinated a non-existent property"
    )
    
    print(f"✅ Model created successfully")
    print(f"   Category: {classification.primary_category}")
    print(f"   Confidence: {classification.confidence}")
    print(f"   Evidence: {classification.evidence[:50]}...")
    return True


async def test_analyzer_initialization():
    """Test that the analyzer can be initialized."""
    print("\nTesting FailureAnalyzer initialization...")
    
    try:
        # This will use environment variables if available
        analyzer = FailureAnalyzer(model_name="gpt-4o-mini")
        print(f"✅ Analyzer initialized successfully")
        print(f"   Model: {analyzer.model.model_name}")
        return True
    except Exception as e:
        print(f"⚠️  Analyzer initialization failed (expected if no API key): {e}")
        return False


def test_csv_loading():
    """Test loading and filtering the CSV."""
    print("\nTesting CSV loading and filtering...")
    
    from analyze_failures import load_and_filter_failures
    
    try:
        failures_df = load_and_filter_failures(
            "sparql_agent_run_20260119_150453.csv",
            row_matching_threshold=1.0
        )
        
        print(f"✅ CSV loaded successfully")
        print(f"   Total failures: {len(failures_df)}")
        print(f"   Columns: {list(failures_df.columns)}")
        
        if len(failures_df) > 0:
            print(f"\n   Sample failure:")
            sample = failures_df.iloc[0]
            print(f"   - Query ID: {sample['query_id']}")
            print(f"   - Question: {sample['question'][:60]}...")
            print(f"   - row_matching_f1: {sample['row_matching_f1']}")
        
        return True
    except FileNotFoundError:
        print(f"⚠️  CSV file not found (expected if file doesn't exist)")
        return False
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False


def create_sample_csv():
    """Create a small sample CSV for testing."""
    print("\nCreating sample CSV for testing...")
    
    sample_data = {
        'query_id': ['TEST_001', 'TEST_002'],
        'question_number': ['', ''],
        'source': ['TEST', 'TEST'],
        'question': [
            'What are the brick points in this graph?',
            'Find all VAV units'
        ],
        'model': ['test-model', 'test-model'],
        'message_history': [
            'ModelRequest(parts=[UserPromptPart(content="Question: What are the brick points?")])',
            'ModelRequest(parts=[UserPromptPart(content="Question: Find all VAV units")])'
        ],
        'ground_truth_sparql': [
            'SELECT ?point WHERE { ?point rdf:type brick:Point }',
            'SELECT ?vav WHERE { ?vav rdf:type brick:VAV }'
        ],
        'generated_sparql': [
            'SELECT ?point WHERE { ?point rdf:type brick:PointDevice }',
            'SELECT ?vav WHERE { ?vav rdf:type brick:VariableAirVolume }'
        ],
        'syntax_ok': [True, True],
        'returns_results': [False, False],
        'perfect_match': [False, False],
        'gt_num_rows': [100, 50],
        'gt_num_cols': [1, 1],
        'gen_num_rows': [0, 0],
        'gen_num_cols': [1, 1],
        'arity_matching_f1': [1.0, 1.0],
        'exact_match_f1': [0.0, 0.0],
        'entity_set_f1': [0.0, 0.0],
        'row_matching_f1': [0.0, 0.0],
        'less_columns_flag': [False, False],
        'prompt_tokens': [100, 100],
        'completion_tokens': [50, 50],
        'total_tokens': [150, 150]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('test_sample.csv', index=False)
    print(f"✅ Created test_sample.csv with {len(df)} rows")
    return True


async def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("RUNNING FAILURE ANALYZER TESTS")
    print("="*80)
    
    results = []
    
    # Test 1: Pydantic model
    results.append(await test_classification_model())
    
    # Test 2: Analyzer initialization
    results.append(await test_analyzer_initialization())
    
    # Test 3: CSV loading
    results.append(test_csv_loading())
    
    # Test 4: Create sample CSV
    results.append(create_sample_csv())
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed (may be expected if API key not configured)")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Run: python analyze_failures.py test_sample.csv")
    print("3. Or run: python example_analyze_failures.py")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
