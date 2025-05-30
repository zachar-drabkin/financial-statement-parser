import re
from typing import Dict, List, Optional, Any, Tuple

class FinancialNotesClassifier:
    """
    Rule-based Classifier for identifying Sections in Financial Statement Notes.
    It processes sequential document blocks to identify and classify financial notes.
    Includes Keyword Cohesion Check for more robust soft breaks.
    """

    COHESION_THRESHOLD_TO_PREVENT_SOFT_BREAK: int = 1
    # if a block (not a new note header) has a forward ref to a different note,
    # it needs more primary keywords of the current note's theme
    # to be considered cohesive enough to prevent a soft break.
    # if score is 0 or 1, and there's a forward ref -> it will break.
    # if score is 2 or more -> it will not soft break due to cohesion.

    def __init__(self):
        self.note_definitions = self._initialize_note_definitions()
        self.note_start_pattern = re.compile(
            r"^\s*(?:(?:Note|NOTE|ITEM|Item)\s*)?([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*)\s*[.:\-–—]?\s*(.*)",
            re.IGNORECASE
        )
        self.numeric_note_start_pattern = re.compile(
            r"^\s*(\d+[\.]?[A-Za-z]?)\s*[.:\-–—]+\s*(.*)",
            re.IGNORECASE
        )
        self.section_header_default = "Notes to Financial Statements"

    def _normalize_text(self, text: str) -> str:
        """Helper to convert text to uppercase and replace multiple whitespace chars with a single space."""
        if not text:
            return ""
        text_upper = text.upper()
        normalized_text = ' '.join(text_upper.split())
        return normalized_text.strip()

    def _initialize_note_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize a comprehensive dictionary of financial note definitions."""
        definitions = {
            "Nature of Operations and Going Concern": {
                "canonical_title": "Nature of Operations and Going Concern",
                "title_patterns": [
                    r"Nature of Operations and Going Concern",
                    r"Nature of Business and Going Concern",
                    r"Organization and Going Concern"
                ],
                "primary_keywords": [ # Made more granular
                    "nature of operations", "operations", "going concern", "business", "company",
                    "incorporated", "properties", "mineral properties", "project", "producing project",
                    "exploration", "development", "mining", "concession rights",
                    "working capital", "deficit", "material uncertainty", "continue its operations",
                    "adequate funding", "realize assets", "discharge liabilities"
                ],
                "secondary_keywords": [
                    "liquidit", "financing", "debt", "equity", "obligations", "next twelve months", "ltd", "inc",
                    "registered office", "tsx venture exchange"
                ],
                "structural_cues_rules": [
                     {"type": "phrase_exists", "phrase": "existence of a material uncertainty", "score": 2},
                     {"type": "phrase_exists", "phrase": "may cast significant doubt", "score": 2},
                ],
                "positional_heuristics": {"expected_note_numbers": ["1", "A"], "score_bonus": 3},
                "max_score_components": {"title": 5, "primary_keywords": 10, "secondary_keywords": 5, "structure": 4, "position": 3}
            },
            "Basis of Preparation": {
                "canonical_title": "Basis of Preparation",
                "title_patterns": [
                    r"Basis of Preparation",
                    r"Basis of Presentation",
                    r"General Information (and|&) Basis of Preparation"
                ],
                "primary_keywords": [
                    "basis of preparation", "basis of presentation", "statement of compliance",
                    "accounting framework", "IFRS", "US GAAP", "Generally Accepted Accounting Principles",
                    "fiscal year", "reporting period", "functional currency", "presentation currency",
                    "historical cost basis", "basis of consolidation", "subsidiaries", "accounting standards"
                ],
                "secondary_keywords": [
                    "compliance with IFRS", "compliance with US GAAP",
                    "approved by the board", "authorized for issue", "intercompany transactions",
                    "interpretations committee", "iasb"
                ],
                "structural_cues_rules": [
                    {"type": "phrase_exists", "phrase": "These financial statements have been prepared in accordance with", "score": 2},
                    {"type": "phrase_exists", "phrase": "statement of compliance", "score": 1},
                    {"type": "section_exists", "keywords": ["basis of consolidation", "subsidiaries"], "score": 2}
                ],
                "positional_heuristics": {"expected_note_numbers": ["1", "2", "A", "B"], "score_bonus": 3},
                "max_score_components": {"title": 5, "primary_keywords": 8, "secondary_keywords": 4, "structure": 5, "position": 3}
            },
            "Significant Accounting Policies": {
                "canonical_title": "Summary of Significant Accounting Policies",
                "title_patterns": [
                    r"Summary of Significant Accounting Policies", r"Significant Accounting Policies",
                    r"Accounting Policies", r"Material Accounting Policies"
                ],
                "primary_keywords": [
                    "significant accounting policies", "material accounting policies", "use of estimates",
                    "critical accounting judgements", "principles of consolidation", "revenue recognition",
                    "property, plant and equipment", "intangible assets", "leases", "income taxes",
                    "financial instruments", "cash and cash equivalents", "inventory", "impairment"
                ],
                "secondary_keywords": [
                    "new accounting standards", "recently adopted pronouncements", "recently issued pronouncements",
                    "amendments to standards", "ASC", "IFRS", "IAS"
                ],
                "structural_cues_rules": [
                    {"type": "section_exists", "keywords": ["new accounting standards", "recently issued"], "score": 3},
                    {"type": "list_of_policies_implied", "min_keywords": 5, "keyword_list_ref": "primary_keywords", "score": 2}
                ],
                "positional_heuristics": {"expected_note_numbers": ["1", "2", "3", "A", "B", "C"], "score_bonus": 3},
                "max_score_components": {"title": 5, "primary_keywords": 10, "secondary_keywords": 5, "structure": 5, "position": 3}
            },
            "Property, Plant, and Equipment": {
                "canonical_title": "Property, Plant, and Equipment",
                "title_patterns": [
                    r"Property, Plant(?:,| and) Equipment", r"Property and Equipment", r"Fixed Assets(?:, Net)?"
                ],
                "primary_keywords": [
                    "property, plant and equipment", "ppe", "fixed assets", "land", "buildings",
                    "machinery and equipment", "accumulated depreciation", "depreciation expense",
                    "useful lives", "carrying amount", "additions", "disposals", "impairment loss"
                ],
                "secondary_keywords": [
                    "straight-line method", "declining balance method", "cost model", "revaluation model",
                    "construction in progress", "capital commitments"
                ],
                "structural_cues_rules": [
                    {"type": "phrase_exists", "phrase": "reconciliation of carrying amounts", "score": 2},
                    {"type": "phrase_exists", "phrase": "movements in property, plant and equipment", "score": 2},
                    {"type": "table_headers_hint", "headers": ["Opening Balance", "Additions", "Disposals", "Depreciation"], "score": 4}
                ],
                "positional_heuristics": {},
                "max_score_components": {"title": 5, "primary_keywords": 10, "secondary_keywords": 5, "structure": 6, "position": 0}
            }
        }
        for note_name, definition in definitions.items():
            definition["max_score_exemplar"] = sum(definition["max_score_components"].values())
        return definitions

    def _parse_note_start(self, text: str) -> Optional[Tuple[str, str]]:
        """Tries to parse note number and title from a line of text."""
        if not text.strip():
            return None

        # print(f"DEBUG_PARSE_NOTE_START: Processing text: '{text[:100].strip()}'")

        pattern1 = re.compile(
            r"^\s*(?:Note|NOTE|ITEM|Item)\s+([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*)\s*[.:\-–—]?\s*(.{6,})",
            re.IGNORECASE
        )
        match = pattern1.match(text)
        if match:
            note_number, raw_title = match.groups()
            # print(f"DEBUG_PARSE_NOTE_START: Pattern 1 matched. Num='{note_number}', RawTitle='{raw_title[:50]}'")
            if not (len(raw_title) > 40 and raw_title.isupper() and ' ' in raw_title and raw_title.endswith('.')):
                # print(f"DEBUG_PARSE_NOTE_START: Pattern 1 PASSED heuristic. Returning: ('{note_number.strip('.')}', '{raw_title.strip()}')")
                return note_number.strip("."), raw_title.strip()
            # print(f"DEBUG_PARSE_NOTE_START: Pattern 1 FAILED heuristic (long uppercase sentence-like title).")

        pattern2 = re.compile(
            r"^\s*([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*\.?)\s+(?![\d\W]*$)([A-Z][A-Za-z0-9\s\(\)\-\,\&\/\.'“”‘’]{5,}.*)"
        )
        match = pattern2.match(text)
        if match:
            note_number_p2, raw_title_p2 = match.groups()
            # print(f"DEBUG_PARSE_NOTE_START: Pattern 2 matched. Num='{note_number_p2}', RawTitle='{raw_title_p2[:50]}'")

            is_likely_sentence_start_word = False
            common_sentence_starters = [
                "THE", "A", "AN", "IT", "THIS", "THAT", "THESE", "THOSE", "HE", "SHE", "WE", "THEY",
                "IN", "ON", "AT", "FOR", "AS", "OF", "IF", "ALL", "ANY", "SOME", "MANY", "MORE", "MOST",
                "UNDER", "ABOUT", "WITH", "FROM", "OTHER", "SUCH", "WHEN", "WHERE", "WHICH", "WHO",
                "TOTAL", "NET", "GROSS", "LESS", "PLUS", "EQUALS", "SHOULD", "WOULD", "COULD", "MAY", "MIGHT"
            ]
            allowed_short_alpha_numbers = [chr(i) for i in range(ord('A'), ord('Z')+1)] + \
                                         ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

            if note_number_p2.upper() in common_sentence_starters and note_number_p2.upper() not in allowed_short_alpha_numbers:
                is_likely_sentence_start_word = True
            elif len(note_number_p2) > 2 and note_number_p2.isalpha() and \
                 note_number_p2.upper() not in ["APPENDIX", "SCHEDULE", "EXHIBIT"] and \
                 note_number_p2.upper() not in allowed_short_alpha_numbers :
                is_likely_sentence_start_word = True
            elif note_number_p2.isdigit() and (len(note_number_p2) >= 4 or (len(note_number_p2) >=1 and int(note_number_p2) > 200) ):
                is_likely_sentence_start_word = True

            if is_likely_sentence_start_word:
                # print(f"DEBUG_PARSE_NOTE_START: Pattern 2 REJECTED: Num part '{note_number_p2}' looks like a sentence start word/year.")
                return None

            if not (len(raw_title_p2) > 40 and raw_title_p2.isupper() and ' ' in raw_title_p2 and raw_title_p2.endswith('.')):
                # print(f"DEBUG_PARSE_NOTE_START: Pattern 2 PASSED all heuristics. Returning: ('{note_number_p2.strip('.')}', '{raw_title_p2.strip()}')")
                return note_number_p2.strip("."), raw_title_p2.strip()
            # print(f"DEBUG_PARSE_NOTE_START: Pattern 2 FAILED heuristic (long uppercase sentence-like title).")

        # print(f"DEBUG_PARSE_NOTE_START: NO pattern matched for text: '{text[:100].strip()}'")
        return None

    def _get_tentative_note_definition_for_cohesion(self, raw_title: str, note_number: str, debug: bool = False) -> Optional[Dict[str, Any]]:
        """
        Tries to find the best matching note definition based on the raw_title
        and note_number of the note currently being built. Used for cohesion check.
        """
        raw_title_upper = raw_title.upper()
        best_match_def = None
        highest_heuristic_score = -1

        for def_name, note_def in self.note_definitions.items():
            current_heuristic_score = 0
            for pattern_str in note_def["title_patterns"]:
                if re.search(pattern_str, raw_title, re.IGNORECASE):
                    current_heuristic_score += 2
                    break
            if current_heuristic_score == 0 and (note_def["canonical_title"].upper() in raw_title_upper or raw_title_upper in note_def["canonical_title"].upper()):
                 current_heuristic_score +=1

            if current_heuristic_score > 0 and "positional_heuristics" in note_def and note_number:
                expected_nums = note_def["positional_heuristics"].get("expected_note_numbers", [])
                if note_number.strip('.').upper() in expected_nums:
                    current_heuristic_score += 1

            if current_heuristic_score > highest_heuristic_score:
                highest_heuristic_score = current_heuristic_score
                best_match_def = note_def
            elif current_heuristic_score == highest_heuristic_score and best_match_def and len(note_def["canonical_title"]) < len(best_match_def["canonical_title"]):
                 best_match_def = note_def

        if debug and best_match_def and highest_heuristic_score >=1 :
            print(f"DEBUG_COHESION_HELPER: Tentative def for title '{raw_title[:30]}' (num {note_number}) is '{best_match_def['canonical_title']}' with heuristic score {highest_heuristic_score}")
        elif debug and highest_heuristic_score < 1:
            print(f"DEBUG_COHESION_HELPER: No strong tentative def found for title '{raw_title[:30]}' (num {note_number})")
        return best_match_def if highest_heuristic_score >=1 else None

    def _calculate_block_cohesion_score(self, block_content_normalized_upper: str, note_definition: Dict[str, Any], debug: bool = False) -> int:
        """
        Calculates a simple cohesion score for a block against a note definition
        by counting matching primary keywords.
        """
        if not note_definition:
            if debug: print(f"DEBUG_COHESION_SCORE: No note definition provided for cohesion check.")
            return 0

        cohesion_score = 0
        primary_keywords = note_definition.get("primary_keywords", [])
        found_in_block_for_debug = []

        if debug:
            print(f"DEBUG_COHESION_SCORE: Checking cohesion for def '{note_definition.get('canonical_title', 'Unknown Def')}'")
            print(f"DEBUG_COHESION_SCORE: Block content (normalized upper, first 100): '{block_content_normalized_upper[:100]}'")
            print(f"DEBUG_COHESION_SCORE: Primary keywords for def: {primary_keywords}")

        for kw_original_case in primary_keywords:
            kw_upper = kw_original_case.upper()
            # using word boundaries for more precise matching of keywords
            pattern = r'\b' + re.escape(kw_upper) + r'\b'

            if debug: print(f"DEBUG_COHESION_SCORE:  Trying kw '{kw_upper}' with pattern '{pattern}'")
            match_found = re.search(pattern, block_content_normalized_upper)

            if match_found:
                cohesion_score += 1
                if debug:
                    found_in_block_for_debug.append(kw_original_case)
                    print(f"DEBUG_COHESION_SCORE:    FOUND '{kw_upper}' at pos {match_found.span()}. Score now: {cohesion_score}")
            elif debug:
                print(f"DEBUG_COHESION_SCORE:    NOT FOUND '{kw_upper}'")

        if debug:
            print(f"DEBUG_COHESION_SCORE: Final for def '{note_definition.get('canonical_title', 'Unknown Def')}'. Score = {cohesion_score}. Found Primary KWs: {found_in_block_for_debug}")
        return cohesion_score

    def _calculate_note_score(self,
                              extracted_raw_title: str,
                              note_content_upper_normalized: str,
                              note_def: Dict[str, Any],
                              extracted_note_number: Optional[str],
                              num_content_blocks: int) -> Dict[str, Any]:
        """Calculates raw score for a potential note against a note definition."""
        breakdown = {}
        total_score = 0

        title_score = 0
        max_title_score = note_def["max_score_components"]["title"]
        for pattern_str in note_def["title_patterns"]:
            if re.search(pattern_str, extracted_raw_title, re.IGNORECASE):
                title_score = max_title_score
                break
        if not title_score and note_def["canonical_title"].lower() in extracted_raw_title.lower() :
             title_score = max_title_score / 2
        breakdown["title"] = title_score
        total_score += title_score

        primary_keywords_found_count = 0
        for kw in note_def["primary_keywords"]:
            if re.search(r'\b' + re.escape(kw.upper()) + r'\b', note_content_upper_normalized):
                primary_keywords_found_count +=1

        secondary_keywords_found_count = 0
        for kw in note_def["secondary_keywords"]:
            if re.search(r'\b' + re.escape(kw.upper()) + r'\b', note_content_upper_normalized):
                secondary_keywords_found_count +=1

        pk_score = (primary_keywords_found_count / max(1, len(note_def["primary_keywords"]))) * note_def["max_score_components"]["primary_keywords"]
        sk_score = (secondary_keywords_found_count / max(1, len(note_def["secondary_keywords"]))) * note_def["max_score_components"]["secondary_keywords"]

        breakdown["primary_keywords"] = round(pk_score)
        breakdown["secondary_keywords"] = round(sk_score)
        total_score += pk_score + sk_score

        structure_score = 0
        max_structure_score = note_def["max_score_components"]["structure"]
        if "structural_cues_rules" in note_def:
            for cue in note_def["structural_cues_rules"]:
                if cue["type"] == "phrase_exists" and cue["phrase"].upper() in note_content_upper_normalized:
                    structure_score += cue["score"]
                elif cue["type"] == "section_exists":
                    if any(kw.upper() in note_content_upper_normalized for kw in cue["keywords"]):
                         structure_score += cue["score"]
                elif cue["type"] == "table_headers_hint":
                    if all(h.upper() in note_content_upper_normalized for h in cue["headers"]):
                        structure_score += cue["score"]
        structure_score = min(structure_score, max_structure_score)
        breakdown["structure"] = structure_score
        total_score += structure_score

        position_score = 0
        max_position_score = note_def["max_score_components"]["position"]
        if "positional_heuristics" in note_def and extracted_note_number:
            expected_nums = note_def["positional_heuristics"].get("expected_note_numbers", [])
            normalized_extracted_num = extracted_note_number.strip('.').upper()
            if normalized_extracted_num in expected_nums:
                position_score = note_def["positional_heuristics"].get("score_bonus", max_position_score)
        breakdown["position"] = position_score
        total_score += position_score

        content_bonus = min(1.5, num_content_blocks / 5.0) if num_content_blocks > 2 else 0
        breakdown["content_volume_bonus"] = round(content_bonus,1)
        total_score += content_bonus

        return {"total": total_score, "breakdown": breakdown}

    def classify_financial_notes(self,
                                 doc_blocks: List[Dict[str, Any]],
                                 start_block_index: int = 0,
                                 confidence_threshold: float = 0.50,
                                 max_start_block_index_to_check: Optional[int] = None,
                                 debug: bool = False) -> List[Dict[str, Any]]:
        """
        Identifies and classifies financial notes from a list of document blocks.
        Starts processing from `start_block_index` (list index in doc_blocks).
        Includes Keyword Cohesion Check for soft breaks.
        """
        identified_notes = []
        current_list_idx = start_block_index
        num_total_blocks_in_list = len(doc_blocks)

        if debug:
            start_doc_idx_field = doc_blocks[current_list_idx]['index'] if current_list_idx < num_total_blocks_in_list else 'N/A'
            print(f"\nDEBUG_CLASSIFY: Starting financial notes classification. Processing from list_idx {current_list_idx} / doc_block index field {start_doc_idx_field}.")
            print(f"DEBUG_CLASSIFY: Total blocks in provided list: {num_total_blocks_in_list}, Confidence Threshold: {confidence_threshold}")
            if max_start_block_index_to_check is not None:
                 print(f"DEBUG_CLASSIFY: Will stop initiating new notes if block's 'index' field >= {max_start_block_index_to_check}")

        while current_list_idx < num_total_blocks_in_list:
            start_block_candidate = doc_blocks[current_list_idx]
            current_block_content = start_block_candidate['content']
            current_block_index_field = start_block_candidate['index']

            if max_start_block_index_to_check is not None and current_block_index_field >= max_start_block_index_to_check:
                if debug:
                    print(f"DEBUG_CLASSIFY: Stopping search for new notes. Current block index field {current_block_index_field} >= max_start_block_index_to_check {max_start_block_index_to_check}.")
                break

            if debug:
                 print(f"\nDEBUG_CLASSIFY: Main Loop Iteration. current_list_idx={current_list_idx}, current_block_index_field={current_block_index_field}. Checking content: '{current_block_content[:80].strip()}'")

            parsed_start = self._parse_note_start(current_block_content)

            if parsed_start:
                extracted_note_number, extracted_raw_title = parsed_start
                start_block_index_val = current_block_index_field

                if debug:
                    print(f"DEBUG_CLASSIFY: SUCCESSFUL PARSE_NOTE_START at list_idx {current_list_idx} (block_index_field {start_block_index_val}): Num='{extracted_note_number}', Title='{extracted_raw_title[:50]}...'")

                note_content_list = [current_block_content]
                current_note_last_block_list_idx = current_list_idx

                current_num_normalized_for_loop = extracted_note_number.strip(".").upper()
                tentative_current_note_def = self._get_tentative_note_definition_for_cohesion(extracted_raw_title, extracted_note_number, debug=debug)

                if debug and tentative_current_note_def:
                    print(f"DEBUG_CLASSIFY:  Tentative definition for current note '{extracted_note_number} - {extracted_raw_title[:30]}' is '{tentative_current_note_def['canonical_title']}' for cohesion checks.")
                elif debug:
                    print(f"DEBUG_CLASSIFY:  Could not find a strong tentative definition for current note '{extracted_note_number} - {extracted_raw_title[:30]}' for cohesion checks.")

                if debug:
                    print(f"DEBUG_CLASSIFY:  Starting search for end of note '{extracted_note_number}' (Normalized: '{current_num_normalized_for_loop}').")

                for j_list_idx in range(current_list_idx + 1, num_total_blocks_in_list):
                    next_block_item = doc_blocks[j_list_idx]
                    next_block_content = next_block_item['content']
                    next_block_index_field = next_block_item['index']

                    if debug:
                        print(f"DEBUG_CLASSIFY:   Checking next block (list_idx {j_list_idx}, block_index_field {next_block_index_field}): '{next_block_content[:80].strip()}'")

                    potential_next_note_details = self._parse_note_start(next_block_content)
                    is_actually_a_new_different_note_header = False
                    should_soft_break = False

                    if potential_next_note_details:
                        next_note_number_parsed, next_title_parsed = potential_next_note_details
                        if debug:
                             print(f"DEBUG_CLASSIFY:    Potential next note parsed by _parse_note_start: Num='{next_note_number_parsed}', Title='{next_title_parsed[:30]}...'")

                        next_num_normalized = next_note_number_parsed.strip(".").upper()

                        if next_num_normalized != current_num_normalized_for_loop:
                            is_actually_a_new_different_note_header = True
                            if debug:
                                print(f"DEBUG_CLASSIFY:    HARD Break Condition Met: DIFFERENT Note Number found. Current='{current_num_normalized_for_loop}', Next='{next_num_normalized}'. This marks end of current note.")
                        # Same note number parsed (e.g., "(continued)")
                        elif debug:
                                print(f"DEBUG_CLASSIFY:    SAME Note Number parsed by _parse_note_start. Current='{current_num_normalized_for_loop}', Next='{next_num_normalized}'. Continuing current note based on this check.")

                    if not is_actually_a_new_different_note_header:
                        # check only for soft breaks if it's not alredy identified as a new different note header
                        # and _parse_note_start returned None for this next_block_content - it's not a header itself,
                        if potential_next_note_details is None:
                            fwd_note_ref_match = re.search(r"\(Notes?\s+([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9,\s(?:and)]+)*)\)", next_block_content, re.IGNORECASE)
                            if not fwd_note_ref_match:
                                fwd_note_ref_match = re.search(r"see\s+Note\s+([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*)", next_block_content, re.IGNORECASE)

                            if fwd_note_ref_match:
                                referenced_notes_str = fwd_note_ref_match.group(1)
                                first_referenced_note_num_str = re.split(r"[\s,and]+", referenced_notes_str)[0]
                                referenced_num_normalized = first_referenced_note_num_str.strip(".").upper()

                                if referenced_num_normalized != current_num_normalized_for_loop:
                                    # check if the referenced number is a plausible main note ID
                                    if len(referenced_num_normalized) <= 3 and (referenced_num_normalized.isalnum() or '.' in referenced_num_normalized):
                                        block_cohesion_score = 0
                                        # check cohesion only if we have a theme for current note
                                        if tentative_current_note_def:
                                            normalized_next_block_content_upper = self._normalize_text(next_block_content)
                                            block_cohesion_score = self._calculate_block_cohesion_score(normalized_next_block_content_upper, tentative_current_note_def, debug=debug)

                                        if block_cohesion_score <= self.COHESION_THRESHOLD_TO_PREVENT_SOFT_BREAK:
                                            should_soft_break = True
                                            if debug:
                                                current_def_title_for_debug = tentative_current_note_def['canonical_title'] if tentative_current_note_def else 'N/A (No Tentative Def)'
                                                print(f"DEBUG_CLASSIFY:    Soft Break Met: Forward Ref to Note '{first_referenced_note_num_str}' AND Low Cohesion ({block_cohesion_score}) with current note theme ('{current_def_title_for_debug}'). Block: {next_block_index_field}.")
                                        elif debug:
                                            current_def_title_for_debug = tentative_current_note_def['canonical_title'] if tentative_current_note_def else 'N/A (No Tentative Def)'
                                            print(f"DEBUG_CLASSIFY:    Soft Break Postponed: Forward Ref to Note '{first_referenced_note_num_str}' but Cohesion ({block_cohesion_score}) with current note theme ('{current_def_title_for_debug}') is sufficient. Block: {next_block_index_field}.")

                    if is_actually_a_new_different_note_header or should_soft_break:
                        break

                    note_content_list.append(next_block_content)
                    current_note_last_block_list_idx = j_list_idx

                end_block_index_val = doc_blocks[current_note_last_block_list_idx]['index']
                note_content_full = "\n".join(note_content_list).strip()
                note_content_upper_normalized = self._normalize_text(note_content_full)
                num_content_blocks_in_note = (current_note_last_block_list_idx - current_list_idx) + 1

                if debug:
                    print(f"DEBUG_CLASSIFY:  Note '{extracted_note_number}' (Raw Title: '{extracted_raw_title[:30]}...') determined to span block indices {start_block_index_val}-{end_block_index_val} "
                          f"({num_content_blocks_in_note} blocks from list_idx {current_list_idx} to {current_note_last_block_list_idx}).")
                    if num_content_blocks_in_note < 3 and len(note_content_full) < 200:
                         print(f"DEBUG_CLASSIFY:  Full content for short note ({len(note_content_full)} chars): '{note_content_full.replace(chr(10), ' ')}'")
                    else:
                         print(f"DEBUG_CLASSIFY:  Content preview: '{note_content_full[:150].replace(chr(10), ' ')}...'")

                best_classified_title = extracted_raw_title
                highest_confidence = 0.0
                best_score_result = {}

                for def_name, note_def_obj in self.note_definitions.items():
                    score_result = self._calculate_note_score(
                        extracted_raw_title, note_content_upper_normalized, note_def_obj,
                        extracted_note_number, num_content_blocks_in_note
                    )
                    raw_score = score_result["total"]
                    max_possible_score_for_def = note_def_obj.get("max_score_exemplar", 1)
                    current_confidence = (min(raw_score, max_possible_score_for_def) / (max_possible_score_for_def + 1.0)) if max_possible_score_for_def > 0 else 0.0
                    current_confidence = min(current_confidence, 0.99)

                    if debug:
                        print(f"DEBUG_CLASSIFY:    Trying Def: '{note_def_obj['canonical_title']}' -> Raw Score: {raw_score:.2f}/{max_possible_score_for_def}, Conf: {current_confidence:.3f}, Breakdown: {score_result.get('breakdown')}")

                    if current_confidence > highest_confidence:
                        highest_confidence = current_confidence
                        best_classified_title = note_def_obj['canonical_title']
                        best_score_result = score_result

                if highest_confidence >= confidence_threshold:
                    note_output = {
                        "section_header": self.section_header_default,
                        "note_number": extracted_note_number,
                        "note_title": best_classified_title,
                        "start_block": start_block_index_val,
                        "end_block": end_block_index_val,
                        "confidence_rate": round(highest_confidence, 3),
                        "raw_score_breakdown": best_score_result.get("breakdown", {})
                    }
                    identified_notes.append(note_output)
                    if debug:
                        print(f"DEBUG_CLASSIFY:  >>> IDENTIFIED: '{best_classified_title}' (Num: {extracted_note_number}), Conf: {highest_confidence:.3f} "
                              f"(Blocks {start_block_index_val}-{end_block_index_val})")
                elif debug:
                    print(f"DEBUG_CLASSIFY:  --- NOT CLASSIFIED (Low Confidence): RawTitle='{extracted_raw_title[:50]}...', MaxConf={highest_confidence:.3f} "
                          f"(Blocks {start_block_index_val}-{end_block_index_val})")

                current_list_idx = current_note_last_block_list_idx + 1

            else:
                if debug and current_block_content.strip():
                     print(f"DEBUG_CLASSIFY: Block index {current_block_index_field} (list_idx {current_list_idx}) - _parse_note_start returned None. Content: '{current_block_content[:80].strip()}'")
                current_list_idx += 1

        if debug:
            print(f"DEBUG_CLASSIFY: Finished classification. Total notes identified: {len(identified_notes)}")
        return identified_notes

    def display_financial_notes_results(self,
                                       identified_notes: List[Dict[str, Any]],
                                       all_doc_blocks: Optional[List[Dict[str, Any]]] = None):
        """Displays the financial note classification results in a clear format."""
        print("\nFINANCIAL NOTES CLASSIFICATION RESULTS:")
        print("-" * 40)
        if not identified_notes:
            print("\n No financial note sections identified with sufficient confidence.")
            return
        print(f"\nSuccessfully identified {len(identified_notes)} financial note section(s):")
        block_content_map = {block['index']: block['content'] for block in all_doc_blocks} if all_doc_blocks else {}
        for i, note_result in enumerate(identified_notes):
            print(f"\n--- Note Section #{i+1} ---")
            print(f"  Section Header: {note_result['section_header']}")
            print(f"  Identified Title: {note_result['note_title']}")
            print(f"  Extracted Note Number: {note_result['note_number']}")
            print(f"  Confidence: {note_result['confidence_rate']:.3f}")
            print(f"  Block Range (by 'index' field): {note_result['start_block']} to {note_result['end_block']}")
            if 'raw_score_breakdown' in note_result and note_result['raw_score_breakdown']:
                print(f"  Score Breakdown:")
                for category, score in note_result['raw_score_breakdown'].items():
                    if isinstance(score, (int, float)) and (score > 0.01 or score < -0.01) :
                        print(f"    • {category.replace('_', ' ').title()}: {score:+.2f}")
            if all_doc_blocks and block_content_map:
                first_block_content = block_content_map.get(note_result['start_block'])
                if first_block_content:
                    display_content = first_block_content.replace('\n', ' ').strip()[:150]
                    print(f"  Content Preview (Block {note_result['start_block']}): '{display_content}{'...' if len(first_block_content.replace(chr(10),' ')) > 150 else ''}'")
        print("-" * 40)

if __name__ == '__main__':
    classifier = FinancialNotesClassifier()
    sample_doc_blocks = [ # Based on user's debug log for consistency
        {"index": 0, "content": "Cover page info 1", "type": "paragraph"},
        # ... (add blocks 1-10 if needed for full context, or assume notes start later)
        {"index": 10, "content": "Some text before notes", "type": "paragraph"},
        {"index": 11, "content": "1.\tNATURE OF OPERATIONS AND GOING CONCERN", "type": "paragraph"},
        {"index": 12, "content": "BestCo Ltd. (formerly GoodCo Ltd.) (the “Company”) was incorporated pursuant to the Business Corporations Act of British Columbia on January 24, 2011. The Company’s registered office is located at 13th Floor, 1313 Lucky Street, Vancouver, British Columbia, Canada, V1C 2D3.", "type": "paragraph"},
        {"index": 13, "content": "The Company is engaged in the operation, acquisition, exploration and development of mineral properties in Latin America, with a primary focus on silver and zinc, including lead and copper. As at December 31, 2023, the Company had one producing project, the Great Mine. The Company has acquired, or has options to acquire, the mining concession rights to the following properties:", "type": "paragraph"},
        {"index": 14, "content": "The producing Great Mine located in Mexico", "type": "paragraph"}, # This was user's block 14
        {"index": 15, "content": "Various other properties in San Luis Potosi, Mexico, noting that the Rocket Project had been placed on care and maintenance in August 2023 (Note 6). This is still about our operations.", "type": "paragraph"}, # User's block 15 that caused issue
        {"index": 16, "content": "These consolidated financial statements for the years ended December 31, 2023 and 2022 (“financial statements”) have been prepared on a going concern basis. The Company has a working capital deficiency and an accumulated deficit.", "type": "paragraph"}, # User's block 16
        {"index": 17, "content": "Should the Company be unable to continue as a going concern, asset and liability realization values may be substantially different from their carrying values. These financial statements do not give effect to adjustments that would be necessary to the carrying values and classification of assets and liabilities should the Company be unable to continue as a going concern. Such adjustments could be material.", "type": "paragraph"}, # User's block 17 (end of Note 1)
        {"index": 18, "content": "2.\tBASIS OF PREPARATION", "type": "paragraph"},
        {"index": 19, "content": "Statement of compliance", "type": "paragraph"},
        {"index": 20, "content": "These financial statements were approved by the Board of Directors and authorized for issue on February 20, 2024.", "type": "paragraph"},
        {"index": 21, "content": "These financial statements have been prepared in accordance with International Financial Reporting Standards (“IFRS Accounting Standards”) as issued by the International Accounting Standards Board and interpretations of the International Financial Reporting Interpretations Committee.", "type": "paragraph"},
        {"index": 22, "content": "Basis of presentation", "type": "paragraph"},
        {"index": 23, "content": "These financial statements have been prepared using the historical cost basis, except for certain financial assets and liabilities which are measured at fair value, as specified by IFRS Accounting Standards for each type of asset, liability, income, and expense as set out in the accounting policies below.", "type": "paragraph"},
        {"index": 24, "content": "Functional and presentation currency", "type": "paragraph"},
        {"index": 25, "content": "These financial statements are presented in United States dollars (“US dollar” or “USD”). The functional currency is the currency of the primary economic environment in which an entity operates. References to “C$” or “CAD” are to Canadian dollars and references to “MXN” are to Mexican pesos.", "type": "paragraph"},
        {"index": 26, "content": "2.\tBASIS OF PREPARATION (continued)", "type": "paragraph"},
        {"index": 27, "content": "Basis of consolidation", "type": "paragraph"},
        {"index": 28, "content": "These financial statements include the accounts of the Company and its subsidiaries. All intercompany transactions and balances are eliminated on consolidation. Control exists where the parent entity has power over the investee and is exposed, or has rights, to variable returns from its involvement with the investee and has the ability to affect those returns through its power over the investee.", "type": "paragraph"},
        {"index": 29, "content": "A summary of the Company’s subsidiaries included in these financial statements as at December 31, 2023 are as follows:", "type": "paragraph"},
        {"index": 30, "content": "Name of subsidiary | Country of incorporation | Percentage ownership | Functional currency | Principal activity\nBestCo Ltd. | Canada | 100% | CAD | Holding company and head office function", "type": "table"}, # Simplified table
        {"index": 31, "content": "On April 23, 2023, the Company acquired a 100% interest in Great Mining (Note 5).", "type": "paragraph"},
        {"index": 32, "content": "3.\tSIGNIFICANT ACCOUNTING POLICIES", "type": "paragraph"},
        {"index": 33, "content": "Content for note 3 - significant accounting policies.", "type": "paragraph"},
    ]

    notes_start_doc_index = 11
    actual_start_list_idx = 0
    found_start = False
    for idx, block in enumerate(sample_doc_blocks):
        if block['index'] == notes_start_doc_index:
            actual_start_list_idx = idx
            found_start = True
            break
    if not found_start:
        print(f"Error: Could not find block with document index {notes_start_doc_index} in sample_doc_blocks to determine start_list_idx.")
        actual_start_list_idx = 0

    print(f"Starting Notes Classification from list index: {actual_start_list_idx} (corresponds to document block index {notes_start_doc_index})")

    identified_notes_sections = classifier.classify_financial_notes(
        sample_doc_blocks,
        start_block_index=actual_start_list_idx,
        confidence_threshold=0.35,
        debug=True
    )

    classifier.display_financial_notes_results(identified_notes_sections, sample_doc_blocks)

