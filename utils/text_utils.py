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

def normalize_multiline_text(text: str) -> str:
    if not text:
        return ""
    lines = text.split('\n')
    normalized_lines = []
    for line in lines:
        processed_line = line.upper()
        processed_line = re.sub(r'[ \t\f\v]+', ' ', processed_line) # Normalize horizontal spaces
        normalized_lines.append(processed_line.strip())
    return "\n".join(normalized_lines)

def find_exact_phrases(text_upper_normalized: str, phrases: List[str]) -> List[str]:
    """
    Finds which of the exact phrases are present in the normalized uppercase text.
    Returns a list of found phrases (maintaining original casing from the input 'phrases' list).
    """
    found = []
    if not text_upper_normalized or not phrases:
        return found

    for phrase_original_case in phrases:
        if not phrase_original_case:  # Skip empty or None phrases
            continue

        # 1. Normalize the phrase itself to match how text_upper_normalized is prepared.
        # This typically involves converting to uppercase, stripping leading/trailing whitespace,
        # and normalizing internal whitespace (e.g., multiple spaces to one).
        temp_phrase_upper = phrase_original_case.upper()
        normalized_phrase_upper = " ".join(temp_phrase_upper.split())

        if not normalized_phrase_upper:  # Handles phrases that were only whitespace
            continue

        # 2. Escape the normalized phrase for use in a regex pattern.
        escaped_phrase = re.escape(normalized_phrase_upper)

        # 3. Determine word boundaries based on the *normalized and stripped* phrase.
        # Add \b (word boundary) if the effective start/end of the phrase is alphanumeric.
        # This ensures the phrase is matched as a whole unit.

        prefix = r'\b' if normalized_phrase_upper[0].isalnum() else ''

        suffix = r'\b' if normalized_phrase_upper[-1].isalnum() else ''

        pattern_str = prefix + escaped_phrase + suffix

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

