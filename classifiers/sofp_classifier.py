import re
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from classifiers.base_classifier import BaseClassifier
from utils.text_utils import normalize_text, find_exact_phrases, find_regex_patterns

logger = logging.getLogger(__name__)

# --- Constants ---
SECTION_NAME_SOFP = "Statement of Financial Position"
BLOCK_TYPE_PARAGRAPH = "paragraph"
BLOCK_TYPE_TABLE = "table"


class SoFPClassifier(BaseClassifier):
    """
    Rule-based Classifier for identifying Statement of Financial Position (SoFP)
    sections in Financial Statements.
    """

    def __init__(self, rules_file_path: str = "rules/sofp_rules.json"): # Adjust default path
        super().__init__(rules_file_path=rules_file_path)

    def _is_hard_termination_block(self, text_content: str, block_type: Optional[str] = None) -> bool:
        text_upper_normalized = normalize_text(text_content)
        if not text_upper_normalized:
            return False

        term_rules = self.rules.get("hard_termination_section_starters", {})
        keywords = term_rules.get("keywords", [])
        regex_patterns = term_rules.get("regex_patterns", [])
        period_indicators = term_rules.get("content_indicator_regex_for_period_statements", [])

        if find_exact_phrases(text_upper_normalized, keywords):
            logger.debug(f"SoFP Term: Matched keyword in '{text_content[:60]}...'")
            return True
        if find_regex_patterns(text_upper_normalized, regex_patterns): # Assuming patterns are case-insensitive or text is already upper
            logger.debug(f"SoFP Term: Matched regex_pattern in '{text_content[:60]}...'")
            return True

        if block_type == BLOCK_TYPE_TABLE:
            year_pattern_hdr = self.rules.get("structural_cues", {}).get("comparative_year_pattern", r'\b(19|20)\d{2}\b')
            for pattern_str in period_indicators:
                if re.search(pattern_str, text_upper_normalized, re.IGNORECASE): # some period indicators might have mixed case
                    years_found_hdr = re.findall(year_pattern_hdr, text_upper_normalized)
                    if len(years_found_hdr) >= 1:
                        logger.debug(f"SoFP Term: Matched period content indicator '{pattern_str}' and >=1 year(s) in a table.")
                        return True
        return False

    def _check_sofp_title_phrases(self, text_upper_normalized: str) -> int:
        title_rules = self.rules.get("title_phrases", {})
        phrases = title_rules.get("keywords", [])
        buffer = title_rules.get("length_check_buffer", 20)
        score_value = title_rules.get("score", 0)

        for phrase in phrases:
            pattern = r'\b' + re.escape(phrase) + r'\b' # keywords in JSON are upperrcase
            match = re.search(pattern, text_upper_normalized)
            if match:
                if len(text_upper_normalized) < (len(phrase) + buffer):
                    return score_value
        return 0

    def _check_major_sections(self, text_upper_normalized: str) -> int:
        major_section_rules = self.rules.get("major_section_keywords", {})
        score = 0
        found_sections = set()

        if find_exact_phrases(text_upper_normalized, major_section_rules.get("assets", [])):
            found_sections.add("assets")
        if find_exact_phrases(text_upper_normalized, major_section_rules.get("liabilities", [])):
            found_sections.add("liabilities")
        if find_exact_phrases(text_upper_normalized, major_section_rules.get("equity", [])):
            found_sections.add("equity")
        if find_exact_phrases(text_upper_normalized, major_section_rules.get("combined_liabilities_equity", [])):
            found_sections.add("liabilities")
            found_sections.add("equity")

        score = len(found_sections) * major_section_rules.get("score_per_section", 1)
        return min(score, major_section_rules.get("max_score", 3))

    def _count_and_score_items(self, text_upper_normalized: str, item_category_key: str) -> int:
        item_rules = self.rules.get(item_category_key, {})
        keywords = item_rules.get("keywords", [])
        max_score = item_rules.get("max_score", 0)
        count_for_max_score = item_rules.get("count_for_max_score", 1)
        if count_for_max_score == 0: return 0 # avoid dividing by zero

        found_items_texts = find_exact_phrases(text_upper_normalized, keywords)
        unique_found_count = len(set(found_items_texts))

        if unique_found_count == 0:
            return 0
        if unique_found_count >= count_for_max_score:
            return max_score
        return max(1, int(round((unique_found_count / count_for_max_score) * max_score)))

    def _check_structural_cues(self, text_upper_normalized: str) -> int:
        cue_rules = self.rules.get("structural_cues", {})
        score = 0

        year_pattern = cue_rules.get("comparative_year_pattern", r'\b(19|20)\d{2}\b')
        years_found = re.findall(year_pattern, text_upper_normalized)
        if len(set(years_found)) >= 2:
            score += cue_rules.get("score_comparative_years", 1)

        note_keywords_found = find_regex_patterns(text_upper_normalized, cue_rules.get("note_column_keywords", []))
        currency_indicators_found = find_regex_patterns(text_upper_normalized, cue_rules.get("currency_indicators", []))

        if note_keywords_found or currency_indicators_found:
            score += cue_rules.get("score_notes_or_currency", 1)

        return min(score, cue_rules.get("score", 2))

    def _parse_financial_number(self, value_str: str) -> Optional[float]:
        logger.debug(f"SoFP BalEq: Parsing value string: '{value_str}'")
        if not value_str: return None
        cleaned_value = str(value_str).strip().replace(',', '')
        is_negative = False
        if cleaned_value.startswith('(') and cleaned_value.endswith(')'):
            is_negative = True
            cleaned_value = cleaned_value[1:-1]
        if cleaned_value == '-' or not cleaned_value or not re.search(r'\d', cleaned_value):
            return None
        try:
            number = float(cleaned_value)
            return -number if is_negative else number
        except ValueError:
            logger.debug(f"SoFP BalEq: Could not parse '{cleaned_value}' to float.")
            return None

    def _find_value_for_label(self, text_content_upper_with_newlines: str, label_variants: List[str], year_column_idx: int = 0) -> Optional[float]:
        for label in label_variants:
            line_pattern = re.compile(r"^\s*" + re.escape(label) + r"\b(.*)$", re.MULTILINE | re.IGNORECASE) # label might not be upper in raw text
            match = line_pattern.search(text_content_upper_with_newlines) # Search in text that preserves newlines
            if match:
                rest_of_line = match.group(1)
                logger.debug(f"SoFP BalEq: Found line for '{label}': '{rest_of_line[:100]}'")
                potential_value_strings = re.findall(r"[\(\)\d,\.\-]+", rest_of_line)
                parsed_numbers_on_line = []
                for s_val in potential_value_strings:
                    if not (re.search(r"\d.*\d", s_val) or (s_val.count('(') + s_val.count(')') + s_val.count(',')) > 0 or len(s_val) > 2) and s_val.strip('-').isdigit() and len(s_val.strip('-')) <= 2 :
                        logger.debug(f"SoFP BalEq: Skipping potential note ref/small num: '{s_val}'")
                        continue
                    num = self._parse_financial_number(s_val)
                    if num is not None:
                        parsed_numbers_on_line.append(num)

                logger.debug(f"SoFP BalEq: Parsed numbers from line for '{label}': {parsed_numbers_on_line}")
                if year_column_idx < len(parsed_numbers_on_line):
                    return parsed_numbers_on_line[year_column_idx]
        logger.debug(f"SoFP BalEq: Value not found for labels '{label_variants}' (year_column_idx {year_column_idx}).")
        return None

    def _check_balancing_equation(self, raw_text_content: str) -> int:
        if not raw_text_content: return 0

        bal_eq_rules = self.rules.get("balancing_equation", {})
        score_value = bal_eq_rules.get("score", 0)
        rel_tol = bal_eq_rules.get("relative_tolerance", 1e-5)
        abs_tol = bal_eq_rules.get("absolute_tolerance", 0.51)

        text_for_search = raw_text_content.upper()

        assets = self._find_value_for_label(text_for_search, bal_eq_rules.get("total_assets_labels", []))
        total_liab_equity = self._find_value_for_label(text_for_search, bal_eq_rules.get("total_liabilities_and_equity_labels", []))

        if assets is not None and total_liab_equity is not None:
            logger.debug(f"SoFP BalEq: Check 1 -> Assets: {assets}, Total L+E (combined): {total_liab_equity}")
            if math.isclose(assets, total_liab_equity, rel_tol=rel_tol, abs_tol=abs_tol):
                logger.debug(f"SoFP BalEq: Balances! (Assets vs Total L+E combined). Score: {score_value}")
                return score_value

        liabilities = self._find_value_for_label(text_for_search, bal_eq_rules.get("total_liabilities_labels", []))
        equity = self._find_value_for_label(text_for_search, bal_eq_rules.get("total_equity_labels", []))

        if assets is not None and liabilities is not None and equity is not None:
            sum_liab_equity = liabilities + equity
            logger.debug(f"SoFP BalEq: Check 2 -> Assets: {assets}, Liab: {liabilities}, Eq: {equity}, Sum L+E: {sum_liab_equity}")
            if math.isclose(assets, sum_liab_equity, rel_tol=rel_tol, abs_tol=abs_tol):
                logger.debug(f"SoFP BalEq: Balances! (Assets vs L + E separate). Score: {score_value}")
                return score_value

        logger.debug(f"SoFP BalEq: Equation does not balance or values not found. Score: 0")
        return 0

    def _calculate_score(self, combined_text_original_case: str, first_block_index: int,
                         is_title_paragraph_present: bool = False, num_table_blocks: int = 0) -> Dict[str, Any]:
        if not combined_text_original_case.strip():
            return {"total": 0, "breakdown": {}}

        text_upper_normalized = normalize_text(combined_text_original_case)

        title_score = 0
        # if windo started with a paragraph assumed to be a title
        if is_title_paragraph_present:
             # use the first part of the combined text for a more focused title check,
             # assuming the title paragraph is at the beginning.
             # heuristic, ideally, the title paragraph's content would be passed separately.
             title_check_text = normalize_text(combined_text_original_case.split('\n', 1)[0])
             title_score = self._check_sofp_title_phrases(title_check_text)
        # ff it's table-only - still check for titles within the table content
        elif num_table_blocks > 0:
            title_score = self._check_sofp_title_phrases(text_upper_normalized)


        major_sections_score = self._check_major_sections(text_upper_normalized)
        asset_items_score = self._count_and_score_items(text_upper_normalized, "asset_keywords")
        liability_items_score = self._count_and_score_items(text_upper_normalized, "liability_keywords")
        equity_items_score = self._count_and_score_items(text_upper_normalized, "equity_keywords")
        total_indicators_score = self._count_and_score_items(text_upper_normalized, "total_indicator_keywords")
        structural_cues_score = self._check_structural_cues(text_upper_normalized)

        # pas original case text with newlines for balancing equation checks
        balancing_equation_score = self._check_balancing_equation(combined_text_original_case)

        block_bonus = self.rules.get("block_bonus_score", 0) if first_block_index < self.rules.get("block_bonus_max_index", 60) else 0

        combination_bonus = 0
        combo_score_val = self.rules.get("combination_bonus_score", 1)
        if is_title_paragraph_present and num_table_blocks > 0 and title_score > 0:
            combination_bonus += combo_score_val
        if (major_sections_score >= self.rules.get("strong_table_min_major_sections", 2) and
            total_indicators_score >= self.rules.get("strong_table_min_total_indicators", 1) and
            structural_cues_score >= self.rules.get("strong_table_min_structural_cues", 1) and
            num_table_blocks > 0):
            combination_bonus += combo_score_val
        combination_bonus = min(combination_bonus, 2 * combo_score_val) # Cap bonus

        total_score = sum([
            title_score, major_sections_score, asset_items_score, liability_items_score,
            equity_items_score, total_indicators_score, structural_cues_score,
            balancing_equation_score, block_bonus, combination_bonus
        ])

        breakdown = {
            "title": title_score, "major_sections": major_sections_score,
            "asset_items": asset_items_score, "liability_items": liability_items_score,
            "equity_items": equity_items_score, "total_indicators": total_indicators_score,
            "structural_cues": structural_cues_score, "balancing_equation": balancing_equation_score,
            "block_bonus": block_bonus, "combination_bonus": combination_bonus
        }
        return {"total": total_score, "breakdown": breakdown}

    def _identify_and_evaluate_core_window(self,
                                         doc_blocks: List[Dict[str, Any]],
                                         current_doc_block_list_idx: int,
                                         confidence_threshold: float
                                         ) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Identifies a potential CORE SoFP window (title + tables, or tables only),
        evaluates it, and returns the result and the next list index to check from.
        """
        start_block_candidate = doc_blocks[current_doc_block_list_idx]
        current_core_sofp_blocks = []
        current_core_sofp_content = ""
        is_potential_title_paragraph_for_core = False
        num_table_blocks_in_core = 0
        core_window_last_list_idx = current_doc_block_list_idx - 1

        block_type_of_start_candidate = start_block_candidate.get('type')

        if block_type_of_start_candidate == BLOCK_TYPE_PARAGRAPH:
            # check title on normalized, uppercase
            title_check_text = normalize_text(start_block_candidate['content'])
            temp_title_score = self._check_sofp_title_phrases(title_check_text)

            min_title_score_threshold = self.rules.get("title_phrases",{}).get("score", 4) * 0.5 # e.g. 50% of max title score
            if temp_title_score >= min_title_score_threshold:
                logger.debug(f"SoFP Core: Potential title paragraph {start_block_candidate['index']} (Score: {temp_title_score})")
                current_core_sofp_blocks.append(start_block_candidate)
                current_core_sofp_content += start_block_candidate['content'] + "\n" # Keep original case for balancing eq.
                is_potential_title_paragraph_for_core = True
                core_window_last_list_idx = current_doc_block_list_idx

                # expand with subsequent tables
                for k_core in range(current_doc_block_list_idx + 1, len(doc_blocks)):
                    core_block_to_add = doc_blocks[k_core]
                    if self._is_hard_termination_block(core_block_to_add['content'], core_block_to_add.get('type')):
                        logger.debug(f"SoFP Core (after para) table sequence stopped by HARD termination: Block {core_block_to_add['index']}")
                        break
                    if core_block_to_add.get('type') == BLOCK_TYPE_TABLE:
                        current_core_sofp_blocks.append(core_block_to_add)
                        current_core_sofp_content += core_block_to_add['content'] + "\n"
                        num_table_blocks_in_core += 1
                        core_window_last_list_idx = k_core
                    else:
                        logger.debug(f"SoFP Core (after para) table sequence stopped by non-table: Block {core_block_to_add['index']}")
                        break
                if is_potential_title_paragraph_for_core and num_table_blocks_in_core == 0:
                    logger.debug(f"SoFP Core: Discarding title para {start_block_candidate['index']} as it has no subsequent tables.")
                    current_core_sofp_blocks = [] # reset if title para has no tables
                    is_potential_title_paragraph_for_core = False # reset flag
                    core_window_last_list_idx = current_doc_block_list_idx -1 # effectively no window formed

        # this 'elif' allows a table-only start if the paragraph start didn't yield a valid core
        if not current_core_sofp_blocks and block_type_of_start_candidate == BLOCK_TYPE_TABLE:
            logger.debug(f"SoFP Core: Potential table-only start at block {start_block_candidate['index']}")
            for k_core in range(current_doc_block_list_idx, len(doc_blocks)):
                core_block_to_add = doc_blocks[k_core]
                if self._is_hard_termination_block(core_block_to_add['content'], core_block_to_add.get('type')):
                    logger.debug(f"SoFP Core (table-only) sequence stopped by HARD termination: Block {core_block_to_add['index']}")
                    break
                if core_block_to_add.get('type') == BLOCK_TYPE_TABLE:
                    current_core_sofp_blocks.append(core_block_to_add)
                    current_core_sofp_content += core_block_to_add['content'] + "\n"
                    num_table_blocks_in_core += 1
                    core_window_last_list_idx = k_core
                else:
                    logger.debug(f"SoFP Core (table-only) sequence stopped by non-table: Block {core_block_to_add['index']}")
                    break

        next_i_to_check = core_window_last_list_idx + 1 if core_window_last_list_idx >= current_doc_block_list_idx else current_doc_block_list_idx + 1

        if not current_core_sofp_blocks:
            return None, next_i_to_check

        # evaluate the formed core window
        final_core_sofp_content = current_core_sofp_content.strip()
        first_block_in_core_doc_idx = current_core_sofp_blocks[0]['index']

        score_result = self._calculate_score(
            final_core_sofp_content,
            first_block_in_core_doc_idx,
            is_potential_title_paragraph_for_core,
            num_table_blocks_in_core
        )
        raw_core_score = score_result["total"]
        capped_core_score = min(raw_core_score, self.max_score_exemplar)
        final_core_confidence = (capped_core_score / (self.max_score_exemplar + 1e-9)) # Epsilon for safety

        core_indices_str = str([b['index'] for b in current_core_sofp_blocks])
        logger.debug(f"EVALUATING CORE SoFP: Blocks {core_indices_str}, Content: '{final_core_sofp_content[:150].replace(chr(10), ' ')}...'")
        logger.debug(f"  Core Raw Score: {raw_core_score}, Core Confidence: {final_core_confidence:.3f}, Breakdown: {score_result['breakdown']}")

        if final_core_confidence >= confidence_threshold:
            logger.info(f"  SoFP Core QUALIFIES (Blocks {core_indices_str}, Confidence {final_core_confidence:.3f}). Will attempt expansion.")
            # prepare result based on core, may be expanded later
            core_result = {
                "section_name": SECTION_NAME_SOFP,
                "start_block_index": current_core_sofp_blocks[0]['index'],
                "end_block_index": current_core_sofp_blocks[-1]['index'],
                "block_indices": [b['index'] for b in current_core_sofp_blocks],
                "num_blocks": len(current_core_sofp_blocks),
                "confidence": final_core_confidence,
                "raw_score": raw_core_score,
                "breakdown": score_result["breakdown"],
                "content_preview": final_core_sofp_content[:300].strip(),
                "full_content": final_core_sofp_content
            }
            return core_result, next_i_to_check
        else:
            logger.debug(f"  SoFP Core (Blocks {core_indices_str}) did NOT qualify (Confidence {final_core_confidence:.3f}).")
            return None, next_i_to_check


    def classify(self, doc_blocks: List[Dict[str, Any]],
                 confidence_threshold: float = 0.6,
                 max_start_block_index_to_check: int = 600,
                 **kwargs) -> Optional[Dict[str, Any]]:

        # 'start_block_index_in_list' is the Python list index in doc_blocks to start searching from
        current_list_idx = kwargs.get("start_block_index_in_list", 0)

        while current_list_idx < len(doc_blocks):
            start_block_candidate_for_iteration = doc_blocks[current_list_idx]

            # check against the document's actual block index for max_start_block_index_to_check
            if start_block_candidate_for_iteration['index'] >= max_start_block_index_to_check:
                logger.debug(f"SoFP Classify: Stopping search. Current block doc index {start_block_candidate_for_iteration['index']} "
                             f">= max_start_block_index_to_check ({max_start_block_index_to_check}).")
                break

            if self._is_hard_termination_block(start_block_candidate_for_iteration['content'], start_block_candidate_for_iteration.get('type')):
                logger.debug(f"SoFP Classify: Skipping block {start_block_candidate_for_iteration['index']} (list idx {current_list_idx}) as start: It's a hard termination boundary.")
                current_list_idx += 1
                continue

            core_sofp_result_dict, next_list_idx_after_core = self._identify_and_evaluate_core_window(
                doc_blocks,
                current_list_idx,
                confidence_threshold
            )

            if core_sofp_result_dict:
                logger.debug(f"SoFP Core identified ending at list index {next_list_idx_after_core - 1}. Attempting expansion.")
                # --- Stage 2: Expand the QUALIFIED CORE SoFP ---
                # get the actual block objects for the core window
                # this requires knowiing the list indices of the core blocks.
                # _identify_and_evaluate_core_window gives us the next list index after the core.
                # The core blocks list would be doc_blocks[current_list_idx : next_list_idx_after_core]

                # reconstruct current_window_blocks and current_window_content from the core result for expansion
                # This is a bit simplified.ideally _identify_and_evaluate_core_window would return the blocks list.
                # Currently we assume core_sofp_result_dict has 'block_indices' (doc indices).
                # We need to map these back to actual block dicts for expansion.

                # Find the list indices for the core blocks
                core_start_list_idx = -1
                core_end_list_idx = -1
                core_block_doc_indices = core_sofp_result_dict['block_indices']

                temp_core_blocks = []
                for list_idx_map, block_in_list in enumerate(doc_blocks):
                    if block_in_list['index'] == core_block_doc_indices[0] and core_start_list_idx == -1:
                        core_start_list_idx = list_idx_map
                    if block_in_list['index'] == core_block_doc_indices[-1]:
                        core_end_list_idx = list_idx_map
                        # Add blocks from core_start_list_idx up to and including core_end_list_idx
                        temp_core_blocks = doc_blocks[core_start_list_idx : core_end_list_idx + 1]
                        break

                expanded_window_blocks = list(temp_core_blocks) # start with blocks from the core
                expanded_window_content = str(core_sofp_result_dict["full_content"]) # start with content from the core

                last_expanded_list_idx = core_end_list_idx # list index of the last block in the core

                # start expanding from the block after the identified core wind
                list_idx_for_expansion_start = core_end_list_idx + 1

                if list_idx_for_expansion_start < len(doc_blocks):
                    logger.debug(f"SoFP Expansion: Starting expansion from list index {list_idx_for_expansion_start} (doc index {doc_blocks[list_idx_for_expansion_start]['index']}).")
                    for j_expand in range(list_idx_for_expansion_start, len(doc_blocks)):
                        block_to_add_for_expansion = doc_blocks[j_expand]
                        if self._is_hard_termination_block(block_to_add_for_expansion['content'], block_to_add_for_expansion.get('type')):
                            logger.debug(f"SoFP Expansion: Stopped by HARD termination boundary: Block {block_to_add_for_expansion['index']} ('{block_to_add_for_expansion['content'][:60].strip().replace(chr(10), ' ')}...')")
                            break
                        else:
                            # SoFP expansion is typically conservative: mostly tables or directly associated short paragraphs.
                            # for now, let's include any non-terminating block.
                            logger.debug(f"SoFP Expansion: Expanding with block {block_to_add_for_expansion['index']} ('{block_to_add_for_expansion['content'][:60].strip().replace(chr(10), ' ')}...')")
                            expanded_window_blocks.append(block_to_add_for_expansion)
                            expanded_window_content += "\n" + block_to_add_for_expansion['content']
                            last_expanded_list_idx = j_expand
                else:
                    logger.debug("SoFP Expansion: No further blocks to check for expansion after core.")

                # update the result with the fully expanded window details
                # the confidence and raw_score are still bassed on the initial core identification.
                # this matches the original SoFP classifier's behavior where expansion didn't re-score.
                final_result = {
                    "section_name": SECTION_NAME_SOFP,
                    "start_block_index": expanded_window_blocks[0]['index'],
                    "end_block_index": expanded_window_blocks[-1]['index'],
                    "block_indices": [b['index'] for b in expanded_window_blocks],
                    "num_blocks": len(expanded_window_blocks),
                    "confidence": core_sofp_result_dict['confidence'], # Confidence from core
                    "raw_score": core_sofp_result_dict['raw_score'],     # Raw score from core
                    "breakdown": core_sofp_result_dict['breakdown'],   # Breakdown from core
                    "content_preview": expanded_window_content.strip()[:300],
                    "full_content": expanded_window_content.strip()
                }
                logger.info(f"SoFP Classified & Expanded: Section from block {final_result['start_block_index']} "
                            f"to {final_result['end_block_index']} (confidence {final_result['confidence']:.3f})")
                return final_result # Return the first qualifying and expanded section

            # advance main loop iterator based on what _identify_and_evaluate_core_window processed
            current_list_idx = next_list_idx_after_core

        logger.info("No SoFP section identified with sufficient confidence after checking all potential windows.")
        return None

    def display_results(self, sofp_result: Optional[Dict[str, Any]], all_doc_blocks: Optional[List[Dict[str, Any]]] = None) -> None:
        super().display_results(classification_result=sofp_result, all_doc_blocks=all_doc_blocks)

        # add SoFP-specific logging details if needed, especially for debug
        if sofp_result and logger.isEnabledFor(logging.DEBUG):
            # Example: logger.debug(f"  Balancing Equation Score: {sofp_result.get('breakdown', {}).get('balancing_equation', 'N/A')}")
            if all_doc_blocks: # Log block by block content if available and in debug
                logger.debug(f"\n  Identified SoFP content (block by block):")
                block_content_map = {block['index']: block['content'] for block in all_doc_blocks}
                for block_idx in sofp_result.get('block_indices', []):
                    content = block_content_map.get(block_idx, "Content not found.")
                    display_content = content.replace('\n', ' ').replace('\r', ' ').strip()[:100] + "..."
                    logger.debug(f"    Block {block_idx} (type: {next((b['type'] for b in all_doc_blocks if b['index']==block_idx),'N/A')}): {display_content}")
        elif not sofp_result:
             logger.info("  Suggestions for SoFP detection:")
             logger.info("   - Adjust `confidence_threshold`.")
             logger.info("   - Verify `max_start_block_index_to_check`.")
             logger.info("   - Check `sofp_rules.json` for keyword accuracy and completeness.")
