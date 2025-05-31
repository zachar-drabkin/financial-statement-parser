import pytest
import os
import json
from typing import List, Dict, Any
from classifiers.cover_page_classifier import CoverPageClassifier

TEST_RULES_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TEST_COVER_PAGE_RULES_FILE = os.path.join(TEST_RULES_DIR, "test_cover_page_rules.json")

def create_doc_blocks(contents: List[str], start_index: int = 0, block_type: str = "paragraph") -> List[Dict[str, Any]]:
    """Creates a list of doc_block dictionaries."""
    return [
        {"index": i + start_index, "content": content, "type": block_type}
        for i, content in enumerate(contents)
    ]

@pytest.fixture(scope="module")
def cover_page_classifier_instance():
    """Fixture to provide an instance of CoverPageClassifier with test rules."""
    if not os.path.exists(TEST_COVER_PAGE_RULES_FILE):
        os.makedirs(TEST_RULES_DIR, exist_ok=True)
        dummy_rules = {
            "max_score_exemplar": 10, "title_phrases": {"keywords": ["DUMMY"], "score": 1},
            "date_phrases": {"keywords": [], "patterns": [], "score": 0},
            "currency_phrases": {"keywords": [], "iso_codes": [], "symbols": [], "score": 0, "single_indicator_score": 0},
            "company_indicators": {"suffixes": [], "score": 0}, "year_pattern_score": 0,
            "block_bonus_max_index": 1, "block_bonus_score": 0,
            "combination_bonus_min_strong_indicators": 1, "combination_bonus_score": 0
        }
        with open(TEST_COVER_PAGE_RULES_FILE, 'w') as f:
            json.dump(dummy_rules, f)
        print(f"Warning: Created dummy test rules file at {TEST_COVER_PAGE_RULES_FILE}. Ensure it's correctly set up.")

    return CoverPageClassifier(rules_file_path=TEST_COVER_PAGE_RULES_FILE)

# --- Tests for _calculate_score ---
def test_calculate_score_perfect_match_at_start(cover_page_classifier_instance: CoverPageClassifier):
    classifier = cover_page_classifier_instance
    text = "TEST REPORT TITLE. FOR THE YEAR 2023. AMOUNTS IN TESTDOLLARS. TESTCORP."
    first_block_index = 0
    expected_total_score = 11

    result = classifier._calculate_score(text, first_block_index)
    assert result["total"] == expected_total_score
    assert result["breakdown"]["title"] == 4
    assert result["breakdown"]["date"] == 2
    assert result["breakdown"]["currency"] == 2
    assert result["breakdown"]["company"] == 1
    assert result["breakdown"]["year"] == 0
    assert result["breakdown"]["block_bonus"] == 1
    assert result["breakdown"]["combination_bonus"] == 1

def test_calculate_score_no_match(cover_page_classifier_instance: CoverPageClassifier):
    classifier = cover_page_classifier_instance
    text = "This is some unrelated text."
    first_block_index = 10
    expected_total_score = 0

    result = classifier._calculate_score(text, first_block_index)
    assert result["total"] == expected_total_score

def test_calculate_score_partial_match_with_bonuses(cover_page_classifier_instance: CoverPageClassifier):
    classifier = cover_page_classifier_instance
    text = "FINANCIAL OVERVIEW. AS AT DECEMBER 31."
    first_block_index = 1

    expected_total_score = 4 + 2 + 1 + 1

    result = classifier._calculate_score(text, first_block_index)
    assert result["total"] == expected_total_score
    assert result["breakdown"]["title"] == 4
    assert result["breakdown"]["date"] == 2
    assert result["breakdown"]["block_bonus"] == 1
    assert result["breakdown"]["combination_bonus"] == 1


def test_calculate_score_single_currency_indicator(cover_page_classifier_instance: CoverPageClassifier):
    classifier = cover_page_classifier_instance
    text = "Report amount TSD only."
    first_block_index = 10

    expected_total_score = 1

    result = classifier._calculate_score(text, first_block_index)
    assert result["total"] == expected_total_score
    assert result["breakdown"]["currency"] == 1
    assert result["breakdown"]["combination_bonus"] == 0


# --- Tests for classify (main public method) ---
def test_classify_clear_cover_page_at_start(cover_page_classifier_instance: CoverPageClassifier):
    classifier = cover_page_classifier_instance
    doc_blocks = create_doc_blocks([
        "TEST REPORT TITLE",
        "FOR THE YEAR 2023",
        "AMOUNTS IN TESTDOLLARS",
        "TESTCORP",
        "Some other introductory text."
    ])

    result = classifier.classify(doc_blocks, confidence_threshold=0.7)
    assert result is not None
    assert result["section_name"] == "Cover Page"
    assert result["start_block_index"] == 0
    assert result["end_block_index"] == 4
    assert result["confidence"] == pytest.approx(1.0)

def test_classify_no_cover_page(cover_page_classifier_instance: CoverPageClassifier):
    classifier = cover_page_classifier_instance
    doc_blocks = create_doc_blocks([
        "This is an introduction.",
        "Chapter 1: The Beginning.",
        "Some detailed content here."
    ])
    result = classifier.classify(doc_blocks, confidence_threshold=0.5)
    assert result is None

def test_classify_cover_page_below_threshold(cover_page_classifier_instance: CoverPageClassifier):
    classifier = cover_page_classifier_instance

    doc_blocks = create_doc_blocks(["TEST REPORT TITLE", "Other text"])

    result_fails = classifier.classify(doc_blocks, confidence_threshold=0.6)
    assert result_fails is None

    result_passes = classifier.classify(doc_blocks, confidence_threshold=0.49)
    assert result_passes is not None
    if result_passes:
        assert result_passes["confidence"] == pytest.approx(0.5)
        assert result_passes["start_block_index"] == 0
        assert result_passes["end_block_index"] == 1


def test_classify_respects_max_start_block_index(cover_page_classifier_instance: CoverPageClassifier):
    """
    Test that classification does not initiate a new window search
    at or beyond max_start_block_index_to_check.
    """
    classifier = cover_page_classifier_instance

    doc_blocks_scenario = (
        create_doc_blocks(["Irrelevant Section Title"], start_index=0, block_type="heading1") + # Index 0 (non-paragraph)
        create_doc_blocks(["Some intro text, not a cover page."], start_index=1) +             # Index 1 (paragraph, but low score)
        create_doc_blocks(["Table of contents placeholder"], start_index=2, block_type="table") + # Index 2 (non-paragraph, stops expansion from block 1)
        create_doc_blocks(
            ["TEST REPORT TITLE",
             "FOR THE YEAR 2023",
             "AMOUNTS IN TESTDOLLARS",
             "TESTCORP"],
            start_index=3
        ) +
        create_doc_blocks(["Further irrelevant text"], start_index=7)
    )

    result_misses = classifier.classify(
        doc_blocks_scenario,
        confidence_threshold=0.7,
        max_start_block_index_to_check=3
    )
    assert result_misses is None, \
        "Should not find CP if max_start_block_index_to_check prevents initiating search at the CP's actual start index (index 3)"

    result_finds = classifier.classify(
        doc_blocks_scenario,
        confidence_threshold=0.7,
        max_start_block_index_to_check=4
    )
    assert result_finds is not None, \
        "Should find CP if max_start_block_index_to_check allows initiating search at the CP's actual start index (index 3)"
    if result_finds:
        assert result_finds["start_block_index"] == 3, \
            f"Found CP should start at index 3, but was {result_finds['start_block_index']}"

        assert result_finds["block_indices"] == [3, 4, 5, 6, 7], \
             f"Found CP block indices incorrect, got {result_finds['block_indices']}"

        assert result_finds["raw_score"] == 11
        assert result_finds["confidence"] == pytest.approx(1.0)


def test_classify_window_expansion_stops_at_non_paragraph(cover_page_classifier_instance: CoverPageClassifier):
    classifier = cover_page_classifier_instance
    doc_blocks = [
        {"index": 0, "content": "TEST REPORT TITLE", "type": "paragraph"},
        {"index": 1, "content": "FOR THE YEAR 2023", "type": "paragraph"},
        {"index": 2, "content": "A TABLE HERE", "type": "table"},
        {"index": 3, "content": "AMOUNTS IN TESTDOLLARS", "type": "paragraph"}
    ]

    result = classifier.classify(doc_blocks, confidence_threshold=0.7)
    assert result is not None
    if result:
        assert result["start_block_index"] == 0
        assert result["end_block_index"] == 1
        assert result["block_indices"] == [0, 1]
        assert result["confidence"] == pytest.approx(0.8)
        assert result["raw_score"] == 8
