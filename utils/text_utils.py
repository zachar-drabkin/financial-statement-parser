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

    normalized_text = re.sub(r'\s+', ' ', text_upper)
    return normalized_text.strip()

def find_exact_phrases(text_upper_normalized: str, phrases: List[str]) -> List[str]:
    """
    Finds which of the exact phrases are present in the normalized uppercase text.
    Returns a list of found phrases (maintaining original casing from the input 'phrases' list).
    """
    found = []
    if not text_upper_normalized or not phrases:
        return found

    for phrase_original_case in phrases:
        if not phrase_original_case:
            continue

        phrase_to_match_upper = phrase_original_case.upper()
        escaped_phrase = re.escape(phrase_to_match_upper)


        prefix = r'\b' if phrase_to_match_upper[0].isalnum() else ''

        suffix = r'\b' if phrase_to_match_upper[-1].isalnum() else ''

        pattern_str = prefix + escaped_phrase + suffix


        if not pattern_str.strip():
            continue
        if pattern_str == r'\b\b' and not phrase_to_match_upper[0].isalnum() and not phrase_to_match_upper[-1].isalnum():

             pattern_str = escaped_phrase


        try:
            if re.search(pattern_str, text_upper_normalized):
                found.append(phrase_original_case)
        except re.error as e:

            pass
    return found

def find_regex_patterns(text_upper_normalized: str, patterns: List[str]) -> List[str]:
    """
    Finds which of the regex patterns match in the normalized uppercase text.
    Returns a list of found patterns (or a representative name if patterns are complex).
    """
    found = []
    if not text_upper_normalized or not patterns:
        return found

    for pattern_str in patterns:
        if not pattern_str:
            continue
        try:
            if re.search(pattern_str, text_upper_normalized):
                found.append(pattern_str)
        except re.error as e:

            pass
    return found

