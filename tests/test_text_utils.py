import pytest
import re

from utils.text_utils import normalize_text, find_exact_phrases, find_regex_patterns

# --- Tests for normalize_text ---

def test_normalize_text_empty_string():
    """Test with an empty string."""
    assert normalize_text("") == ""

def test_normalize_text_none_input():
    """Test with None input."""
    assert normalize_text(None) == ""

def test_normalize_text_simple_lowercase():
    """Test with a simple lowercase string."""
    assert normalize_text("hello world") == "HELLO WORLD"

def test_normalize_text_mixed_case():
    """Test with mixed case string."""
    assert normalize_text("Hello World") == "HELLO WORLD"

def test_normalize_text_extra_spaces_leading_trailing():
    """Test with leading/trailing spaces and multiple spaces between words."""
    assert normalize_text("  hello   world  ") == "HELLO WORLD"

def test_normalize_text_newlines_and_tabs():
    """Test with newlines and tabs."""
    assert normalize_text("hello\tworld\nnext line") == "HELLO WORLD NEXT LINE"

def test_normalize_text_already_normalized():
    """Test with an already normalized string."""
    assert normalize_text("ALREADY NORMALIZED") == "ALREADY NORMALIZED"

def test_normalize_text_with_punctuation():
    """Test with punctuation (should be preserved)."""
    assert normalize_text("Hello, World! Test.") == "HELLO, WORLD! TEST."

# --- Tests for find_exact_phrases ---

def test_find_exact_phrases_no_match():
    """Test when no phrases are found."""
    text = "THIS IS A SAMPLE TEXT"
    phrases = ["HELLO", "WORLD"]
    assert find_exact_phrases(text, phrases) == []

def test_find_exact_phrases_single_match():
    """Test when one phrase is found."""
    text = "STATEMENT OF FINANCIAL POSITION"
    phrases = ["FINANCIAL POSITION", "INCOME STATEMENT"]
    assert find_exact_phrases(text, phrases) == ["FINANCIAL POSITION"]

def test_find_exact_phrases_multiple_matches():
    """Test when multiple phrases are found."""
    text = "ASSETS AND LIABILITIES AND EQUITY"
    phrases = ["ASSETS", "LIABILITIES", "EQUITY", "CASH"]
    expected = ["ASSETS", "LIABILITIES", "EQUITY"]
    assert find_exact_phrases(text, phrases) == expected


def test_find_exact_phrases_partial_word_no_match_due_to_boundary_logic():
    """Test that partial word matches are not considered if boundaries apply."""
    text = "OPERATION"
    phrases = ["OPERATE"]

    assert find_exact_phrases(text, phrases) == []

def test_find_exact_phrases_case_insensitivity_of_text():
    """Test with mixed case text (it's normalized by caller) but phrases are matched as given (then uppercased internally)."""
    text_normalized = normalize_text("statement of financial position")
    phrases = ["FINANCIAL POSITION", "INCOME STATEMENT"]
    assert find_exact_phrases(text_normalized, phrases) == ["FINANCIAL POSITION"]

def test_find_exact_phrases_empty_text():
    """Test with empty text."""
    assert find_exact_phrases("", ["HELLO"]) == []

def test_find_exact_phrases_empty_phrases_list():
    """Test with empty phrases list."""
    assert find_exact_phrases("SOME TEXT", []) == []

def test_find_exact_phrases_phrase_with_punctuation():
    """Test matching phrases that include punctuation."""
    text = "STATEMENT OF (LOSS) AND OTHER ITEMS"
    phrases = ["STATEMENT OF (LOSS)", "(LOSS)", "OTHER ITEMS"]
    expected = ["STATEMENT OF (LOSS)", "(LOSS)", "OTHER ITEMS"]
    assert find_exact_phrases(text, phrases) == expected

def test_find_exact_phrases_phrase_is_only_punctuation():
    text = "AMOUNT IS $"
    phrases = ["$"]
    assert find_exact_phrases(text, phrases) == ["$"]

def test_find_exact_phrases_phrase_starts_with_punctuation():
    text = "ACCOUNT (NET) OF DEDUCTIONS"
    phrases = ["(NET)"]
    assert find_exact_phrases(text, phrases) == ["(NET)"]


# --- Tests for find_regex_patterns ---

def test_find_regex_patterns_no_match():
    """Test when no regex patterns match."""
    text = "SIMPLE TEXT"
    patterns = [r"\d+", r"HELLO"]
    assert find_regex_patterns(text, patterns) == []

def test_find_regex_patterns_single_match():
    """Test when one regex pattern matches."""
    text = "YEAR 2023 REPORT"
    patterns = [r"\b(19|20)\d{2}\b", r"ABC"]
    expected = [r"\b(19|20)\d{2}\b"]
    assert find_regex_patterns(text, patterns) == expected

def test_find_regex_patterns_multiple_matches():
    """Test when multiple regex patterns match."""
    text = "NOTE 1.2 AND YEAR 2024"
    patterns = [r"NOTE\s+\d", r"\b(19|20)\d{2}\b", r"XYZ"]
    expected = [r"NOTE\s+\d", r"\b(19|20)\d{2}\b"]
    assert find_regex_patterns(text, patterns) == expected

def test_find_regex_patterns_case_insensitivity_in_text():
    """Test regex matching on normalized (uppercase) text."""
    text_normalized = normalize_text("year 2023 report")
    patterns = [r"\b(19|20)\d{2}\b", r"REPORT$"]
    expected = [r"\b(19|20)\d{2}\b", r"REPORT$"]
    assert find_regex_patterns(text_normalized, patterns) == expected

def test_find_regex_patterns_empty_text():
    """Test with empty text."""
    patterns = [r"\d+"]
    assert find_regex_patterns("", patterns) == []

def test_find_regex_patterns_empty_patterns_list():
    """Test with empty patterns list."""
    text = "SOME DATA 123"
    assert find_regex_patterns(text, []) == []

