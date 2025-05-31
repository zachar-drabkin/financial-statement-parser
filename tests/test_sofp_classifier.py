import unittest
import os
import sys
import logging
from typing import List, Dict, Any

# Ensure the classifiers and utils are in the Python path
# Adjust this path if your project structure is different
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Assuming tests are in a 'tests' subdirectory
sys.path.insert(0, project_root)

from classifiers.sofp_classifier import SoFPClassifier
from utils.text_utils import normalize_text # Assuming this is used by the classifier

# Configure basic logging for tests (optional, but can be helpful)
# logging.basicConfig(level=logging.DEBUG) # Uncomment to see classifier debug logs

# Define a path to the test rules file (relative to this test file)
TEST_RULES_PATH = os.path.join(current_dir, "test_sofp_rules.json")

def create_doc_blocks(block_contents: List[str], block_types: List[str] = None, start_index: int = 0) -> List[Dict[str, Any]]:
    """Helper function to create mock doc_blocks."""
    blocks = []
    if block_types is None:
        block_types = ['paragraph'] * len(block_contents)
    for i, content in enumerate(block_contents):
        blocks.append({
            'index': start_index + i,
            'type': block_types[i],
            'content': content
        })
    return blocks

class TestSoFPClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the classifier instance once for all tests."""
        cls.test_rules_content_as_string = """
{
    "max_score_exemplar": 30,
    "section_name": "Test Statement of Financial Position",
    "title_phrases": {
        "keywords": ["STATEMENT OF FINANCIAL POSITION", "BALANCE SHEET"],
        "score": 5,
        "length_check_buffer": 25
    },
    "major_section_keywords": {
        "assets": ["ASSETS"],
        "liabilities": ["LIABILITIES"],
        "equity": ["EQUITY", "SHAREHOLDERS EQUITY"],
        "combined_liabilities_equity": ["LIABILITIES AND EQUITY"],
        "score_per_section": 2,
        "max_score": 6
    },
    "asset_keywords": {
        "keywords": ["CASH", "INVENTORY", "PPE", "RECEIVABLES"],
        "max_score": 3,
        "count_for_max_score": 3,
        "score_per_item": 1
    },
    "liability_keywords": {
        "keywords": ["PAYABLES", "DEBT"],
        "max_score": 2,
        "count_for_max_score": 2,
        "score_per_item": 1
    },
    "equity_keywords": {
        "keywords": ["SHARE CAPITAL", "RETAINED EARNINGS"],
        "max_score": 2,
        "count_for_max_score": 2,
        "score_per_item": 1
    },
    "total_indicator_keywords": {
        "keywords": ["TOTAL ASSETS", "TOTAL LIABILITIES", "TOTAL EQUITY", "TOTAL LIABILITIES AND EQUITY"],
        "max_score": 4,
        "count_for_max_score": 2,
        "score_per_item": 2
    },
    "structural_cues": {
        "comparative_year_pattern": "\\\\b(2022|2023|2024)\\\\b",
        "note_column_keywords_regex": ["\\\\bNOTE\\\\b"],
        "currency_indicators_regex": ["\\\\$"],
        "score_comparative_years": 2,
        "score_single_year": 1,
        "score_notes_or_currency": 1,
        "max_score": 3
    },
    "balancing_equation": {
        "score": 10,
        "relative_tolerance": 1e-3,
        "absolute_tolerance": 0.51,
        "num_columns_to_check_for_balance": 2,
        "enable_note_ref_heuristic": true,
        "max_len_for_note_ref_heuristic": 1,
        "max_value_for_note_ref_heuristic": 5,
        "total_assets_labels": ["TOTAL ASSETS"],
        "total_liabilities_labels": ["TOTAL LIABILITIES"],
        "total_equity_labels": ["TOTAL EQUITY", "TOTAL SHAREHOLDERS EQUITY"],
        "total_liabilities_and_equity_labels": ["TOTAL LIABILITIES AND EQUITY"]
    },
    "hard_termination_section_starters": {
        "keywords": [
            "STATEMENT OF INCOME",
            "NOTES TO FINANCIAL STATEMENTS"
        ],
        "regex_patterns": [
            "STATEMENTS?\\\\s+OF\\\\s+COMPREHENSIVE\\\\s+INCOME\\\\b",
            "^\\\\s*(?:(?:NOTE|Note)\\\\s+([A-Z0-9]+(?:\\\\.[A-Z0-9]+)*(?:\\\\([a-zA-Z0-9]+\\\\))?\\\\.?)|([A-Z0-9]+(?:\\\\.[A-Z0-9]+)*(?:\\\\([a-zA-Z0-9]+\\\\))?\\\\.))\\\\s*(?=[A-Z]{2,})[A-Z][A-Za-z0-9\\\\s,'&\\\\(\\\\)\\\\/\\\\-\\\\.]{5,}"
        ],
        "content_indicator_regex_for_period_statements": [
            "FOR\\\\s+THE\\\\s+YEAR\\\\s+ENDED"
        ],
        "strong_period_header_phrases_standalone_regex": [
            "YEAR\\\\s+ENDED\\\\s+DECEMBER\\\\s+31"
        ],
        "min_other_statement_keywords_for_table_termination": 1,
        "other_statement_termination_keywords": {
            "line_items": ["REVENUE", "NET INCOME"]
        }
    },
    "block_bonus_score": 2,
    "block_bonus_full_score_max_index": 5,
    "block_bonus_max_index": 10,
    "block_bonus_partial_factor": 0.5,
    "combination_bonus_score": 3,
    "min_score_factor_for_strong_indicator": 0.5,
    "min_strong_indicators_for_combo": 3,
    "min_avg_items_score_for_strong_combo": 1.0,
    "strong_table_combination_bonus": 4,
    "max_blocks_in_core_sofp_window": 10,
    "min_content_blocks_after_title_for_core": 1,
    "core_confidence_factor_for_expansion": 0.80,
    "max_blocks_for_sofp_expansion": 5
}
"""
        if not os.path.exists(TEST_RULES_PATH):
            try:
                with open(TEST_RULES_PATH, "w") as f:
                    f.write(cls.test_rules_content_as_string)
                print(f"Created dummy {TEST_RULES_PATH} for testing using internal string.")
            except IOError as e:
                print(f"Could not create dummy test rules file: {e}. Tests might fail to load rules.")

        cls.classifier = SoFPClassifier(rules_file_path=TEST_RULES_PATH)
        if not cls.classifier.rules:
            import json
            try:
                rules_dict = json.loads(cls.test_rules_content_as_string)
                cls.classifier.rules = rules_dict
                cls.classifier.max_score_exemplar = rules_dict.get("max_score_exemplar", 1)
                print("Loaded rules from internal string due to file issue.")
            except json.JSONDecodeError as jde:
                raise unittest.SkipTest(f"Could not load rules from {TEST_RULES_PATH} or parse internal string: {jde}. Skipping SoFP tests.")

        if not cls.classifier.rules:
             raise unittest.SkipTest(f"Failed to load rules definitively. Skipping SoFP tests.")


    def test_empty_document(self):
        doc_blocks = []
        result = self.classifier.classify(doc_blocks, confidence_threshold=0.1)
        self.assertIsNone(result, "Should return None for an empty document")

    def test_simple_sofp_title_only(self):
        doc_blocks = create_doc_blocks(
            ["STATEMENT OF FINANCIAL POSITION", "Some other content"],
            block_types=['paragraph', 'paragraph']
        )
        result = self.classifier.classify(doc_blocks, confidence_threshold=0.01)

        self.assertIsNotNone(result, "Title followed by generic content should be identified with low threshold")
        if result:
            self.assertEqual(result['start_block_index'], 0)
            self.assertEqual(result['end_block_index'], 1, "Should include the subsequent non-terminating paragraph")


    def test_sofp_with_major_sections_and_items(self):
        doc_blocks = create_doc_blocks(
            [
                "STATEMENT OF FINANCIAL POSITION",
                "ASSETS",
                "Cash $100",
                "Inventory $200",
                "TOTAL ASSETS $300",
                "LIABILITIES",
                "Payables $50",
                "TOTAL LIABILITIES $50",
                "EQUITY",
                "Share Capital $250",
                "TOTAL EQUITY $250",
                "TOTAL LIABILITIES AND EQUITY $300"
            ],
            block_types=['paragraph'] + ['table']*11
        )
        result = self.classifier.classify(doc_blocks, confidence_threshold=0.5)
        self.assertIsNotNone(result, "SoFP section should be identified")
        if result:
            self.assertEqual(result['start_block_index'], 0)
            self.assertEqual(result['end_block_index'], 11)
            expected_raw_score = 36
            self.assertAlmostEqual(result['raw_score'], expected_raw_score, places=1, msg=f"Raw score incorrect. Breakdown: {result.get('breakdown')}")
            self.assertIn("balancing_equation", result['breakdown'])
            self.assertEqual(result['breakdown']['balancing_equation'], 10)


    def test_balancing_equation_correct(self):
        content = "TOTAL ASSETS $1,000.50\nTOTAL LIABILITIES $600.00\nTOTAL EQUITY $400.50"
        score = self.classifier._check_balancing_equation(content)
        self.assertEqual(score, self.classifier.rules.get("balancing_equation", {}).get("score", 0))

    def test_balancing_equation_incorrect(self):
        content = "TOTAL ASSETS $1,000\nTOTAL LIABILITIES $600\nTOTAL EQUITY $390"
        score = self.classifier._check_balancing_equation(content)
        self.assertEqual(score, 0)

    def test_balancing_equation_with_parentheses_and_columns(self):
        content_balanced_direct = (
            "Description 2023 2022\n"
            "TOTAL ASSETS $1,234.56 $1,000.00\n"
            "TOTAL LIABILITIES AND EQUITY $1,234.56 $600.00"
        )
        score_direct = self.classifier._check_balancing_equation(content_balanced_direct)
        self.assertEqual(score_direct, self.classifier.rules.get("balancing_equation", {}).get("score", 0), "Should balance using direct L+E label for first column")

        content_balanced_separate = (
            "Description 2023 2022\n"
            "TOTAL ASSETS $1,234.56 $1,000.00\n"
            "TOTAL LIABILITIES $ (234.56) $ (200.00)\n"
            "TOTAL EQUITY $1,469.12 $1,200.00\n"
        )
        score_separate = self.classifier._check_balancing_equation(content_balanced_separate)
        self.assertEqual(score_separate, self.classifier.rules.get("balancing_equation", {}).get("score", 0), "Should balance using separate L and E labels for first column")


    def test_hard_termination_by_other_statement_title(self):
        self.assertTrue(self.classifier._is_hard_termination_block("STATEMENT OF INCOME", "paragraph"))
        self.assertTrue(self.classifier._is_hard_termination_block("STATEMENT OF COMPREHENSIVE INCOME", "paragraph"))

    def test_hard_termination_by_table_header_for_other_statement(self):
        table_content_soci_header = "| | YEAR ENDED DECEMBER 31, 2023 | YEAR ENDED DECEMBER 31, 2022 \n Revenue $100 $90"
        self.assertTrue(self.classifier._is_hard_termination_block(table_content_soci_header, "table"))

        table_content_sofp_dates = "| | AS AT DECEMBER 31, 2023 | AS AT DECEMBER 31, 2022 \n ASSETS ..."
        self.assertFalse(self.classifier._is_hard_termination_block(table_content_sofp_dates, "table"))

    def test_sofp_followed_by_soci(self):
        doc_blocks = create_doc_blocks(
            [
                "STATEMENT OF FINANCIAL POSITION",
                "ASSETS",
                "Cash $100",
                "TOTAL ASSETS $100",
                "LIABILITIES AND EQUITY",
                "TOTAL LIABILITIES AND EQUITY $100",
                "Approved by the board",
                "STATEMENT OF COMPREHENSIVE INCOME",
                "Revenue $500"
            ],
            block_types=['paragraph', 'table', 'table', 'table', 'table', 'table', 'paragraph', 'paragraph', 'table']
        )
        result = self.classifier.classify(doc_blocks, confidence_threshold=0.5)
        self.assertIsNotNone(result)
        if result:
            self.assertEqual(result['start_block_index'], 0)
            self.assertEqual(result['end_block_index'], 6, "SoFP should end before SoCI title (keyword termination)")

    def test_sofp_with_trailing_note_references_not_headings(self):
        doc_blocks = create_doc_blocks(
            [
                "BALANCE SHEET",
                "ASSETS\nCash 100\nTOTAL ASSETS 100",
                "LIABILITIES AND EQUITY\nDebt 50\nTOTAL LIABILITIES AND EQUITY 100",
                "Nature of operations and going concern (Note 1)",
                "Subsequent events (Note 26)",
                "Approved and authorized for the issue on behalf of the Board",
                "1. SIGNIFICANT ACCOUNTING POLICIES"
            ],
            block_types=['paragraph', 'table', 'table', 'paragraph', 'paragraph', 'paragraph', 'paragraph']
        )
        result = self.classifier.classify(doc_blocks, confidence_threshold=0.4)
        self.assertIsNotNone(result, "SoFP should be identified")
        if result:
            self.assertEqual(result['start_block_index'], 0)
            self.assertEqual(result['end_block_index'], 5, "SoFP should include trailing non-note-heading paragraphs")

    def test_calculate_score_strong_sofp_table_content(self):
        text = """STATEMENT OF FINANCIAL POSITION
ASSETS
CASH $100
RECEIVABLES $200
TOTAL ASSETS $300
LIABILITIES
PAYABLES $50
TOTAL LIABILITIES $50
EQUITY
SHARE CAPITAL $200
RETAINED EARNINGS $50
TOTAL EQUITY $250
TOTAL LIABILITIES AND EQUITY $300
2023 2022 NOTE $"""
        expected_total_score = 39
        result = self.classifier._calculate_score(text, 0, is_title_paragraph_present=True, num_table_blocks=1)
        self.assertEqual(result["total"], expected_total_score, f"Breakdown: {result.get('breakdown')}")

    def test_calculate_score_no_match(self):
        text = "This is an unrelated annual report introduction without any keywords from test rules."
        result = self.classifier._calculate_score(text, 20, is_title_paragraph_present=False, num_table_blocks=0)
        self.assertEqual(result["total"], 0, f"Score should be 0 for non-matching text. Breakdown: {result.get('breakdown')}")

    def test_check_balancing_equation_does_not_balance(self):
        text_not_balance = """
        TOTAL ASSETS              1000
        TOTAL LIABILITIES         400
        TOTAL EQUITY              598
        """
        self.assertEqual(self.classifier._check_balancing_equation(text_not_balance), 0)

    def test_classify_identifies_sofp(self):
        doc_blocks = create_doc_blocks(
            block_contents=[
                "PREFACE",
                "STATEMENT OF FINANCIAL POSITION",
                """
                2023 $ | 2022 $ | NOTE
                ASSETS
                CASH | 100 | 90
                RECEIVABLES | 200 | 180
                TOTAL ASSETS | 300 | 270
                LIABILITIES
                PAYABLES | 50 | 45
                DEBT | 0 | 0
                TOTAL LIABILITIES | 50 | 45
                EQUITY
                SHARE CAPITAL | 200 | 195
                RETAINED EARNINGS | 50 | 30
                TOTAL EQUITY | 250 | 225
                TOTAL LIABILITIES AND EQUITY | 300 | 270
                """
            ],
            start_index=0,
            block_types=["paragraph", "paragraph", "table"]
        )

        result = self.classifier.classify(doc_blocks, confidence_threshold=0.7, start_block_index_in_list=0)
        self.assertIsNotNone(result)
        if result:
            self.assertEqual(result["section_name"], "Test Statement of Financial Position")
            self.assertEqual(result["start_block_index"], 1)
            self.assertEqual(result["end_block_index"], 2)
            self.assertAlmostEqual(result["confidence"], 1.0, places=5, msg=f"Confidence incorrect. Breakdown: {result.get('breakdown')}")


if __name__ == '__main__':
    if not os.path.exists(TEST_RULES_PATH):
        try:
            with open(TEST_RULES_PATH, "w") as f:
                f.write(TestSoFPClassifier.test_rules_content_as_string)
            print(f"Created dummy {TEST_RULES_PATH} for testing as it was missing.")
        except IOError as e:
            print(f"Could not create dummy test rules file before running main: {e}.")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)

