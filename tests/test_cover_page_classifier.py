import pytest
from classifiers.cover_page_classifier import CoverPageClassifier

def test_classifier():
    """Test the enhanced classifier with sample content"""
    classifier = CoverPageClassifier()

    # Sample test blocks - ADD 'type' field and ensure indices allow for sequential testing
    test_blocks_scenario_1 = [
        {"index": 0, "content": "Some irrelevant intro paragraph.", "type": "paragraph"},
        {"index": 1, "content": "BESTCO LTD. (formerly GoodCo Ltd.)", "type": "paragraph"},
        {"index": 2, "content": "Consolidated Financial Statements", "type": "paragraph"},
        {"index": 3, "content": "For the years ended December 31, 2023 and 2022", "type": "paragraph"},
        {"index": 4, "content": "(Expressed in thousands of US dollars)", "type": "paragraph"},
        {"index": 5, "content": "This is an unrelated paragraph, window should have ended.", "type": "paragraph"},
        {"index": 6, "content": "FINANCIAL HIGHLIGHTS", "type": "table"}, # This will break the window
        {"index": 47, "content": "ASSETS Current Cash and cash equivalents...", "type": "table"}
    ]

    test_blocks_scenario_2 = [ # Test case where cover page starts a bit later
        {"index": 8, "content": "MegaCorp Global Holdings INC.", "type": "paragraph"},
        {"index": 9, "content": "ANNUAL REPORT", "type": "paragraph"},
        {"index": 10, "content": "For the Year Ended MARCH 31, 2024", "type": "paragraph"},
        {"index": 11, "content": "Amounts in millions of EUR", "type": "paragraph"},
        {"index": 12, "content": "Auditor's Report", "type": "paragraph"}, # Potentially ends the cover page section
    ]

    test_blocks_scenario_3 = [ # Test no cover page
        {"index": 0, "content": "Chapter 1: Introduction", "type": "paragraph"},
        {"index": 1, "content": "This document describes various things.", "type": "paragraph"},
        {"index": 2, "content": "SECTION A", "type": "paragraph"},
    ]

    test_blocks_scenario_4 = [ # Test a single block cover page if strong enough
        {"index": 0, "content": "ULTIMATE REPORT CORP. CONSOLIDATED FINANCIAL STATEMENTS FOR THE YEAR ENDED DECEMBER 31, 2023 (IN THOUSANDS OF USD)", "type": "paragraph"},
        {"index": 1, "content": "Next Section...", "type": "paragraph"},
    ]


    print("Testing Enhanced Cover Page Classifier")
    print("=" * 70)

    for i, test_blocks in enumerate([test_blocks_scenario_1, test_blocks_scenario_2, test_blocks_scenario_3, test_blocks_scenario_4]):
        print(f"\n--- SCENARIO {i+1} ---")
        # Using a slightly lower threshold for testing to see more candidates.
        # For production, confidence_threshold=0.6 or higher is typical.
        result = classifier.classify_cover_page(test_blocks, confidence_threshold=0.5, debug=True)

        if result:
            print(f"\nCLASSIFICATION RESULT (SCENARIO {i+1}):")
            print(f"  Section: {result['section_name']}")
            print(f"  Blocks: {result['start_block_index']}-{result['end_block_index']} (Indices: {result['block_indices']})")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Raw Score: {result['raw_score']}")
            print(f"  Breakdown: {result['breakdown']}")
            print(f"  Num Blocks in Cover Page: {result['num_blocks']}")
            # print(f"  Content Preview: {result['content'][:200].strip()}...") # Full content can be long
        else:
            print(f"\nNo cover page found for Scenario {i+1}")
        print("-" * 70)
