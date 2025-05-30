import re
from typing import List, Tuple, Dict, Any

def normalize_text(text: str) -> str:
    """
    Converts text to uppercase, replaces multiple whitespace chars with a single space,
    and strips leading/trailing whitespace.
    """
    if not text:
        return ""
    text_upper = text.upper()
    # Replace various whitespace characters with a single space
    normalized_text = re.sub(r'\s+', ' ', text_upper)
    return normalized_text.strip()

def find_exact_phrases(text_upper_normalized: str, phrases: List[str]) -> List[str]:
    """
    Finds which of the exact phrases are present in the normalized uppercase text.
    Returns a list of found phrases.
    """
    found = []
    for phrase in phrases:
        # Ensure phrase normalized for matching if it comes from an external source
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text_upper_normalized):
            found.append(phrase)
    return found

def find_regex_patterns(text_upper_normalized: str, patterns: List[str]) -> List[str]:
    """
    Finds which of the regex patterns match in the normalized uppercase text.
    Returns a list of found patterns (or a representative name if patterns are complex).
    """
    found = []
    for pattern_str in patterns:
        if re.search(pattern_str, text_upper_normalized): # Assumes patterns are designed for uppercase text
            found.append(pattern_str)
    return found
