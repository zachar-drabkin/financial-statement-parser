import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from classifiers.base_classifier import BaseClassifier
from utils.text_utils import normalize_text

logger = logging.getLogger(__name__)

class FinancialNotesClassifier(BaseClassifier):
    """
    Rule-based Classifier for identifying Sections in Financial Statement Notes.
    """

    def __init__(self, rules_file_path: str = "rules/financial_notes_rules.json"):
        super().__init__(rules_file_path=rules_file_path)

        # Compile frequently used regex patterns from rules for performance improvement
        patterns_cfg = self.rules.get("note_start_patterns", {})
        heuristic_cfg = patterns_cfg.get("heuristic_parse_note_start", {})

        self.note_start_pattern_general = re.compile(
            heuristic_cfg.get("pattern1", r"^\s*(?:Note|NOTE|ITEM|Item)\s+([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*)\s*[\.:\-–—]?\s*(.{6,})"),
            re.IGNORECASE
        )
        self.note_start_pattern_heuristic_p2 = re.compile(
             heuristic_cfg.get("pattern2", r"^\s*([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*\.?)\s+(?![\d\W]*$)([A-Z][A-Za-z0-9\s\(\)\-\,\&\/\.'“”‘’]{5,}.*)")
        )

        self.simple_note_start_pattern = re.compile(
            patterns_cfg.get("general_note_pattern", r"^\s*(?:(?:Note|NOTE|ITEM|Item)\s*)?([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*)\s*[\.:\-–—]?\s*(.*)"),
            re.IGNORECASE
        )
        self.cohesion_threshold = self.rules.get("cohesion_threshold_to_prevent_soft_break", 1)
        self.section_header_default = self.rules.get("section_header_default", "Notes to Financial Statements")

    def _parse_note_start(self, text: str) -> Optional[Tuple[str, str]]:
        """Tries to parse note number and title from a line of text using heuristic patterns."""
        if not text.strip():
            return None

        cfg = self.rules.get("note_start_patterns", {}).get("heuristic_parse_note_start", {})
        logger.debug(f"Notes Parse Start: Processing text: '{text[:100].strip()}'")

        match = self.note_start_pattern_general.match(text)
        if match:
            note_number, raw_title = match.groups()
            logger.debug(f"Notes Parse Start: Pattern 1 matched. Num='{note_number}', RawTitle='{raw_title[:50]}'")
            # avoid matching long uppercasse sentences that are not titles
            if not (len(raw_title) > cfg.get("min_long_uppercase_title_len", 40) and
                    raw_title.isupper() and ' ' in raw_title and raw_title.endswith('.')):
                logger.debug(f"Notes Parse Start: Pattern 1 PASSED heuristic. Returning: ('{note_number.strip('.')}', '{raw_title.strip()}')")
                return note_number.strip("."), raw_title.strip()
            logger.debug(f"Notes Parse Start: Pattern 1 FAILED heuristic (long uppercase sentence-like title).")

        match_p2 = self.note_start_pattern_heuristic_p2.match(text)
        if match_p2:
            note_number_p2, raw_title_p2 = match_p2.groups()
            logger.debug(f"Notes Parse Start: Pattern 2 matched. Num='{note_number_p2}', RawTitle='{raw_title_p2[:50]}'")

            is_likely_sentence_start_word = False
            common_starters = cfg.get("common_sentence_starters", [])
            allowed_short_alpha = cfg.get("allowed_short_alpha_numbers", [])
            context_words = cfg.get("short_alpha_num_context_words", [])
            max_unlikely_num_len = cfg.get("max_unlikely_note_number_length", 3)
            min_unlikely_num_val = cfg.get("min_unlikely_note_number_value", 200)

            if note_number_p2.upper() in common_starters and note_number_p2.upper() not in allowed_short_alpha:
                is_likely_sentence_start_word = True
            elif len(note_number_p2) > 2 and note_number_p2.isalpha() and \
                 note_number_p2.upper() not in context_words and \
                 note_number_p2.upper() not in allowed_short_alpha :
                is_likely_sentence_start_word = True
            # check if it's a number that looks more like a year or large number than a note number
            elif note_number_p2.isdigit() and (len(note_number_p2) > max_unlikely_num_len or int(note_number_p2) > min_unlikely_num_val):
                is_likely_sentence_start_word = True

            if is_likely_sentence_start_word:
                logger.debug(f"Notes Parse Start: Pattern 2 REJECTED: Num part '{note_number_p2}' looks like a sentence start word/year.")
                return None # skip...

            if not (len(raw_title_p2) > cfg.get("min_long_uppercase_title_len", 40) and
                    raw_title_p2.isupper() and ' ' in raw_title_p2 and raw_title_p2.endswith('.')):
                logger.debug(f"Notes Parse Start: Pattern 2 PASSED all heuristics. Returning: ('{note_number_p2.strip('.')}', '{raw_title_p2.strip()}')")
                return note_number_p2.strip("."), raw_title_p2.strip()
            logger.debug(f"Notes Parse Start: Pattern 2 FAILED heuristic (long uppercase sentence-like title).")

        logger.debug(f"Notes Parse Start: NO pattern matched for text: '{text[:100].strip()}'")
        return None

    def _get_tentative_note_definition_for_cohesion(self, raw_title: str, note_number: str) -> Optional[Dict[str, Any]]:
        raw_title_upper = raw_title.upper()
        best_match_def = None
        highest_heuristic_score = -1
        note_definitions = self.rules.get("note_definitions", {})

        for def_name, note_def in note_definitions.items():
            current_heuristic_score = 0
            for pattern_str in note_def.get("title_patterns", []):
                if re.search(pattern_str, raw_title, re.IGNORECASE):
                    current_heuristic_score += 2
                    break
            # check canonical title match if pattern didn't hit hard
            if current_heuristic_score == 0 and note_def.get("canonical_title","").upper() in raw_title_upper or \
               (current_heuristic_score == 0 and raw_title_upper in note_def.get("canonical_title","").upper()):
                 current_heuristic_score +=1

            if current_heuristic_score > 0 and "positional_heuristics" in note_def and note_number:
                pos_heuristics = note_def.get("positional_heuristics", {})
                expected_nums = pos_heuristics.get("expected_note_numbers", [])
                if note_number.strip('.').upper() in expected_nums:
                    current_heuristic_score += 1 # bonus increment

            if current_heuristic_score > highest_heuristic_score:
                highest_heuristic_score = current_heuristic_score
                best_match_def = note_def
            elif current_heuristic_score == highest_heuristic_score and best_match_def and \
                 len(note_def.get("canonical_title","")) < len(best_match_def.get("canonical_title","")):
                 best_match_def = note_def

        min_heuristic_score_for_match = 1
        if best_match_def and highest_heuristic_score >= min_heuristic_score_for_match:
            logger.debug(f"Notes Cohesion Helper: Tentative def for title '{raw_title[:30]}' (num {note_number}) is '{best_match_def.get('canonical_title')}' score {highest_heuristic_score}")
            return best_match_def

        logger.debug(f"Notes Cohesion Helper: No strong tentative def found for title '{raw_title[:30]}' (num {note_number})")
        return None

    def _calculate_block_cohesion_score(self, block_content_normalized_upper: str, note_definition: Dict[str, Any]) -> int:
        if not note_definition:
            logger.debug(f"Notes Cohesion Score: No note definition provided.")
            return 0

        cohesion_score = 0
        primary_keywords = note_definition.get("primary_keywords", [])
        found_kws_debug = []
        logger.debug(f"Notes Cohesion Score: Checking for def '{note_definition.get('canonical_title', 'Unknown Def')}' in '{block_content_normalized_upper[:100]}...'")

        for kw_original_case in primary_keywords:
            kw_upper = kw_original_case.upper()
            pattern = r'\b' + re.escape(kw_upper) + r'\b'
            if re.search(pattern, block_content_normalized_upper):
                cohesion_score += 1
                found_kws_debug.append(kw_original_case)
        logger.debug(f"Notes Cohesion Score: Final for def '{note_definition.get('canonical_title', 'Unknown Def')}'. Score = {cohesion_score}. Found: {found_kws_debug}")
        return cohesion_score

    def _calculate_note_score(self,
                              extracted_raw_title: str,
                              note_content_upper_normalized: str,
                              note_def_name: str,
                              extracted_note_number: Optional[str],
                              num_content_blocks: int) -> Dict[str, Any]:

        note_def = self.rules.get("note_definitions", {}).get(note_def_name, {})
        if not note_def:
            logger.warning(f"Note definition '{note_def_name}' not found in rules.")
            return {"total": 0, "breakdown": {}}

        breakdown: Dict[str, Any] = {}
        total_score = 0.0
        max_scores = note_def.get("max_score_components", {})

        # title score
        title_score = 0.0
        max_title_score = float(max_scores.get("title", 0))
        for pattern_str in note_def.get("title_patterns", []): # Patterns are regex
            if re.search(pattern_str, extracted_raw_title, re.IGNORECASE):
                title_score = max_title_score
                break
        if not title_score and note_def.get("canonical_title","").lower() in extracted_raw_title.lower():
             title_score = max_title_score / 2.0
        breakdown["title"] = title_score
        total_score += title_score

        # keyword scores
        pk_found_count = sum(1 for kw in note_def.get("primary_keywords", []) if re.search(r'\b' + re.escape(kw.upper()) + r'\b', note_content_upper_normalized))
        sk_found_count = sum(1 for kw in note_def.get("secondary_keywords", []) if re.search(r'\b' + re.escape(kw.upper()) + r'\b', note_content_upper_normalized))

        num_pk = len(note_def.get("primary_keywords", []))
        num_sk = len(note_def.get("secondary_keywords", []))
        max_pk_score = float(max_scores.get("primary_keywords",0))
        max_sk_score = float(max_scores.get("secondary_keywords",0))

        pk_score = (pk_found_count / max(1, num_pk)) * max_pk_score if num_pk > 0 else 0
        sk_score = (sk_found_count / max(1, num_sk)) * max_sk_score if num_sk > 0 else 0
        breakdown["primary_keywords"] = round(pk_score)
        breakdown["secondary_keywords"] = round(sk_score)
        total_score += pk_score + sk_score

        # structural cues score
        structure_score = 0.0
        max_structure_score = float(max_scores.get("structure",0))
        for cue in note_def.get("structural_cues_rules", []):
            cue_type = cue.get("type")
            cue_score_val = float(cue.get("score",0))
            if cue_type == "phrase_exists" and cue.get("phrase","").upper() in note_content_upper_normalized:
                structure_score += cue_score_val
            elif cue_type == "section_exists" and any(kw.upper() in note_content_upper_normalized for kw in cue.get("keywords",[])):
                structure_score += cue_score_val
            elif cue_type == "table_headers_hint" and all(h.upper() in note_content_upper_normalized for h in cue.get("headers",[])):
                structure_score += cue_score_val

        structure_score = min(structure_score, max_structure_score)
        breakdown["structure"] = structure_score
        total_score += structure_score

        # positional scor
        position_score = 0.0
        max_position_score = float(max_scores.get("position",0))
        pos_heuristics = note_def.get("positional_heuristics", {})
        if pos_heuristics and extracted_note_number:
            expected_nums = pos_heuristics.get("expected_note_numbers", [])
            normalized_extracted_num = extracted_note_number.strip('.').upper()
            if normalized_extracted_num in expected_nums:
                position_score = float(pos_heuristics.get("score_bonus", max_position_score))
        breakdown["position"] = position_score
        total_score += position_score

        # content volume bonus
        content_bonus_max = float(max_scores.get("content_volume_bonus", 1.5))
        content_bonus = min(content_bonus_max, num_content_blocks / 5.0) if num_content_blocks > 2 else 0.0
        breakdown["content_volume_bonus"] = round(content_bonus,1)
        total_score += content_bonus

        return {"total": total_score, "breakdown": breakdown}

    def _find_note_span_and_type(self, doc_blocks: List[Dict[str, Any]],
                                 start_list_idx: int,
                                 current_note_number_normalized: str,
                                 current_note_raw_title: str,
                                 confidence_threshold: float
                                ) -> Tuple[Optional[Dict[str,Any]], int]:
        """
        Finds the end of the current note, gathers its content, scores it against definitions,
        and returns the classified note object and the list index of the block after this note.
        """
        note_content_list = [doc_blocks[start_list_idx]['content']]
        current_note_last_block_list_idx = start_list_idx
        num_total_blocks_in_list = len(doc_blocks)

        tentative_current_note_def = self._get_tentative_note_definition_for_cohesion(current_note_raw_title, current_note_number_normalized)
        if tentative_current_note_def:
            logger.debug(f"Notes Span: Tentative def for current note '{current_note_number_normalized} - {current_note_raw_title[:30]}' is '{tentative_current_note_def.get('canonical_title')}'")
        else:
            logger.debug(f"Notes Span: No strong tentative def for current note '{current_note_number_normalized} - {current_note_raw_title[:30]}'")

        # Expand window to find end of current note
        for j_list_idx in range(start_list_idx + 1, num_total_blocks_in_list):
            next_block_item = doc_blocks[j_list_idx]
            next_block_content = next_block_item['content']
            is_hard_break = False
            is_soft_break = False

            potential_next_note_details = self._parse_note_start(next_block_content)
            if potential_next_note_details:
                next_note_number_parsed, _ = potential_next_note_details
                if next_note_number_parsed.strip(".").upper() != current_note_number_normalized:
                    is_hard_break = True
                    logger.debug(f"Notes Span: HARD Break. Next block is new note '{next_note_number_parsed}'.")

            if not is_hard_break and not potential_next_note_details: # Only check soft break if not a new note header
                # Forward reference check (simplified from original)
                fwd_ref_match = re.search(r"(?:see\s+Note|\(Notes?)\s+([A-Za-z0-9]+)", next_block_content, re.IGNORECASE)
                if fwd_ref_match:
                    referenced_num_normalized = fwd_ref_match.group(1).strip(".").upper()
                    if referenced_num_normalized != current_note_number_normalized and \
                       (len(referenced_num_normalized) <= 3 and (referenced_num_normalized.isalnum() or '.' in referenced_num_normalized)): # Plausible note ID
                        block_cohesion_score = 0
                        if tentative_current_note_def:
                            normalized_next_block_upper = normalize_text(next_block_content)
                            block_cohesion_score = self._calculate_block_cohesion_score(normalized_next_block_upper, tentative_current_note_def)

                        if block_cohesion_score <= self.cohesion_threshold:
                            is_soft_break = True
                            current_def_title_debug = tentative_current_note_def.get('canonical_title', 'N/A') if tentative_current_note_def else 'N/A'
                            logger.debug(f"Notes Span: SOFT Break. Fwd Ref to Note '{referenced_num_normalized}' & Low Cohesion ({block_cohesion_score}) with '{current_def_title_debug}'. Block: {next_block_item['index']}.")

            if is_hard_break or is_soft_break:
                break

            note_content_list.append(next_block_content)
            current_note_last_block_list_idx = j_list_idx

        # Score the identified note span
        start_block_doc_idx = doc_blocks[start_list_idx]['index']
        end_block_doc_idx = doc_blocks[current_note_last_block_list_idx]['index']
        note_content_full = "\n".join(note_content_list).strip()
        note_content_upper_normalized = normalize_text(note_content_full)
        num_content_blocks_in_note = (current_note_last_block_list_idx - start_list_idx) + 1

        logger.debug(f"Notes Span: Note '{current_note_number_normalized}' ({current_note_raw_title[:30]}) spans blocks {start_block_doc_idx}-{end_block_doc_idx} ({num_content_blocks_in_note} blocks).")

        best_classified_title = current_note_raw_title # Default to raw title
        highest_confidence = 0.0
        best_score_result = {}
        note_definitions = self.rules.get("note_definitions", {})

        for def_name, note_def_obj in note_definitions.items():
            score_result = self._calculate_note_score(
                current_note_raw_title, note_content_upper_normalized, def_name,
                current_note_number_normalized, num_content_blocks_in_note
            )
            raw_score = score_result.get("total", 0.0)
            max_possible_score_for_def = float(note_def_obj.get("max_score_exemplar", 1.0))
            current_confidence = (min(raw_score, max_possible_score_for_def) / (max_possible_score_for_def + 1e-9)) if max_possible_score_for_def > 0 else 0.0
            current_confidence = min(current_confidence, 0.99) # Cap confidence

            logger.debug(f"Notes Span Score: Trying Def '{note_def_obj.get('canonical_title')}': Raw Score {raw_score:.2f}/{max_possible_score_for_def}, Conf {current_confidence:.3f}")

            if current_confidence > highest_confidence:
                highest_confidence = current_confidence
                best_classified_title = note_def_obj.get('canonical_title', current_note_raw_title)
                best_score_result = score_result

        identified_note_object = None
        if highest_confidence >= confidence_threshold:
            identified_note_object = {
                "section_header": self.section_header_default,
                "note_number": current_note_number_normalized, # Store normalized
                "note_title": best_classified_title,
                "start_block": start_block_doc_idx,
                "end_block": end_block_doc_idx,
                "confidence_rate": round(highest_confidence, 3),
                "raw_score_breakdown": best_score_result.get("breakdown", {})
            }
            logger.info(f"Notes Span: IDENTIFIED Note '{best_classified_title}' (Num: {current_note_number_normalized}), Conf: {highest_confidence:.3f}")
        else:
            logger.debug(f"Notes Span: NOT CLASSIFIED (Low Confidence). RawTitle='{current_note_raw_title[:50]}...', MaxConf={highest_confidence:.3f}")

        return identified_note_object, current_note_last_block_list_idx + 1


    def classify(self, doc_blocks: List[Dict[str, Any]],
                 confidence_threshold: float = 0.50,
                 max_start_block_index_to_check: Optional[int] = None,
                 **kwargs) -> List[Dict[str, Any]]: # Changed return to List

        identified_notes_list: List[Dict[str, Any]] = []
        current_list_idx = kwargs.get("start_block_index_in_list", 0)
        num_total_blocks_in_list = len(doc_blocks)

        logger.info(f"Starting financial notes classification. Processing from list_idx {current_list_idx}. Total blocks: {num_total_blocks_in_list}")

        while current_list_idx < num_total_blocks_in_list:
            start_block_candidate = doc_blocks[current_list_idx]
            current_block_doc_idx = start_block_candidate['index']

            if max_start_block_index_to_check is not None and current_block_doc_idx >= max_start_block_index_to_check:
                logger.debug(f"Notes Classify: Stopping search. Current block doc index {current_block_doc_idx} >= {max_start_block_index_to_check}.")
                break

            logger.debug(f"\nNotes Classify: Main Loop. current_list_idx={current_list_idx}, current_block_doc_idx={current_block_doc_idx}. Checking content: '{start_block_candidate['content'][:80].strip()}'")

            # use more detailed _parse_note_start for identifying note headers
            parsed_start_details = self._parse_note_start(start_block_candidate['content'])

            if parsed_start_details:
                extracted_note_number, extracted_raw_title = parsed_start_details
                logger.debug(f"Notes Classify: Successful _parse_note_start at list_idx {current_list_idx}: Num='{extracted_note_number}', Title='{extracted_raw_title[:50]}...'")

                note_object, next_list_idx_to_process = self._find_note_span_and_type(
                    doc_blocks,
                    current_list_idx,
                    extracted_note_number.strip(".").upper(),
                    extracted_raw_title,
                    confidence_threshold
                )
                if note_object:
                    identified_notes_list.append(note_object)
                current_list_idx = next_list_idx_to_process
            else:
                if start_block_candidate['content'].strip():
                     logger.debug(f"Notes Classify: Block doc_idx {current_block_doc_idx} (list_idx {current_list_idx}) - _parse_note_start returned None.")
                current_list_idx += 1

        logger.info(f"Financial notes classification finished. Total notes identified: {len(identified_notes_list)}")
        return identified_notes_list

    def display_results(self, identified_notes: Optional[List[Dict[str, Any]]], # Changed type hint
                        all_doc_blocks: Optional[List[Dict[str, Any]]] = None) -> None:

        logger.info("\nFINANCIAL NOTES CLASSIFICATION RESULTS:")
        logger.info("-" * 40)
        if not identified_notes: # Check if the list is empty or None
            logger.info("No financial note sections identified with sufficient confidence.")
            return

        logger.info(f"Successfully identified {len(identified_notes)} financial note section(s):")
        block_content_map = {block['index']: block['content'] for block in all_doc_blocks} if all_doc_blocks else {}

        for i, note_result in enumerate(identified_notes):
            logger.info(f"\n--- Note Section #{i+1} ---")
            logger.info(f"  Section Header: {note_result.get('section_header')}")
            logger.info(f"  Identified Title: {note_result.get('note_title')}")
            logger.info(f"  Extracted Note Number: {note_result.get('note_number')}")
            logger.info(f"  Confidence: {note_result.get('confidence_rate'):.3f}")
            logger.info(f"  Block Range (doc indices): {note_result.get('start_block')} to {note_result.get('end_block')}")

            if logger.isEnabledFor(logging.DEBUG):
                breakdown = note_result.get('raw_score_breakdown', {})
                if breakdown:
                    logger.debug(f"  Score Breakdown:")
                    for category, score in breakdown.items():
                        if isinstance(score, (int, float)) and (score > 0.001 or score < -0.001) : # Show non-zero scores
                            logger.debug(f"    • {category.replace('_', ' ').title()}: {score:+.2f}")

                if all_doc_blocks and block_content_map:
                    first_block_content = block_content_map.get(note_result.get('start_block'))
                    if first_block_content:
                        display_content = normalize_text(first_block_content)[:150] # Use normalize_text for preview
                        logger.debug(f"  Content Preview (Block {note_result.get('start_block')}): '{display_content}{'...' if len(first_block_content) > 150 else ''}'")
        logger.info("-" * 40)

    def _calculate_score(self, combined_text: str, first_block_index: int, **kwargs) -> Dict[str, Any]:
        logger.warning("FinancialNotesClassifier._calculate_score called directly; this method is not typically used for primary note classification.")
        return {"total": 0, "breakdown": {"message": "Use classify method for notes."}}
