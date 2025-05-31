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
    Enhanced for robustness in termination logic, financial number parsing,
    balancing equation checks, and window evaluation.
    """

    def __init__(self, rules_file_path: str = "rules/sofp_rules.json"):
        super().__init__(rules_file_path=rules_file_path)

    def _is_hard_termination_block(self, text_content: str, block_type: Optional[str] = None) -> bool:
        text_upper_normalized = normalize_text(text_content)
        if not text_upper_normalized:
            return False

        term_rules = self.rules.get("hard_termination_section_starters", {})
        keywords = term_rules.get("keywords", [])
        regex_patterns = term_rules.get("regex_patterns", [])

        # Check 1: Standard full titles or explicit note section patterns
        if find_exact_phrases(text_upper_normalized, keywords):
            logger.debug(f"SoFP Term: Matched keyword in '{text_content[:80].strip()}...'")
            return True
        if find_regex_patterns(text_upper_normalized, regex_patterns):
            logger.debug(f"SoFP Term: Matched regex_pattern in '{text_content[:80].strip()}...'")
            return True

        # Check 2: Special logic for table blocks that might be headers of other statements
        if block_type == BLOCK_TYPE_TABLE:
            is_potentially_other_statement_header = False
            year_pattern_hdr = self.rules.get("structural_cues", {}).get("comparative_year_pattern", r'\b(19|20)\d{2}\b')

            # Rule group 1: General period indicators (e.g., "FOR THE YEAR ENDED", "YEARS ENDED")
            general_period_indicators_regex = term_rules.get("content_indicator_regex_for_period_statements", [])
            # Rule group 2: Strong period phrases that imply a date by themselves (e.g., "Years ended July 31")
            # These are regexes. Example: r"YEARS ENDED\s+(?:JANUARY|FEBRUARY|...)\s+\d{1,2}"
            strong_period_phrases_standalone_regex = term_rules.get("strong_period_header_phrases_standalone_regex", [])

            # Test for strong standalone period phrases first
            for strong_pattern_str in strong_period_phrases_standalone_regex:
                if re.search(strong_pattern_str, text_upper_normalized, re.IGNORECASE):
                    is_potentially_other_statement_header = True
                    logger.debug(f"SoFP Term Table: Matched strong_period_header_phrases_standalone_regex '{strong_pattern_str}'.")
                    break

            # If not caught by a strong phrase, check for general period indicators IF a YYYY year is also present
            if not is_potentially_other_statement_header:
                for gen_pattern_str in general_period_indicators_regex:
                    if re.search(gen_pattern_str, text_upper_normalized, re.IGNORECASE):
                        years_found_hdr = re.findall(year_pattern_hdr, text_upper_normalized)
                        if len(years_found_hdr) >= 1: # Requires YYYY year for these more general phrases
                            is_potentially_other_statement_header = True
                            logger.debug(f"SoFP Term Table: Matched general_period_indicator '{gen_pattern_str}' AND YYYY year.")
                            break

            if is_potentially_other_statement_header:
                major_sofp_sections_present_in_header = self._check_major_sections(text_upper_normalized, for_termination_check=True) > 0
                min_other_stmt_kws_for_term = term_rules.get("min_other_statement_keywords_for_table_termination", 1)
                other_stmt_kws_rules = term_rules.get("other_statement_termination_keywords", {}).get("line_items", [])

                num_other_stmt_kws = 0
                if other_stmt_kws_rules:
                    found_other_kws = find_exact_phrases(text_upper_normalized, other_stmt_kws_rules)
                    num_other_stmt_kws = len(found_other_kws)

                if not major_sofp_sections_present_in_header or num_other_stmt_kws >= min_other_stmt_kws_for_term:
                    logger.debug(f"SoFP Term Table: Is potential other statement header AND (lacks SoFP major sections OR has {num_other_stmt_kws} other statement keywords). Terminating. Text: '{text_content[:80].strip()}...'")
                    return True
                else:
                    logger.debug(f"SoFP Term Table: Is potential other statement header BUT still has SoFP major sections ({major_sofp_sections_present_in_header}) AND few other statement keywords ({num_other_stmt_kws}). Not terminating on this basis alone.")

        return False

    def _check_sofp_title_phrases(self, text_upper_normalized: str) -> int:
        title_rules = self.rules.get("title_phrases", {})
        phrases = title_rules.get("keywords", [])
        # Buffer to ensure the block is not much longer than the title itself.
        buffer = title_rules.get("length_check_buffer", 30)
        score_value = title_rules.get("score", 0)

        for phrase in phrases:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            match = re.search(pattern, text_upper_normalized)
            if match:
                if len(text_upper_normalized) < (len(phrase) + buffer):
                    return score_value
        return 0

    def _check_major_sections(self, text_upper_normalized: str, for_termination_check: bool = False) -> int:
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

        if for_termination_check:
            return len(found_sections)

        score = len(found_sections) * major_section_rules.get("score_per_section", 1)
        return min(score, major_section_rules.get("max_score", 3))

    def _count_and_score_items(self, text_upper_normalized: str, item_category_key: str) -> int:
        item_rules = self.rules.get(item_category_key, {})
        keywords = item_rules.get("keywords", [])
        max_score = item_rules.get("max_score", 0)
        count_for_max_score = item_rules.get("count_for_max_score", 1)
        score_per_item = item_rules.get("score_per_item", max_score / (count_for_max_score + 1e-9) if count_for_max_score > 0 else 0)


        if count_for_max_score == 0: return 0

        found_items_texts = find_exact_phrases(text_upper_normalized, keywords)
        unique_found_count = len(set(found_items_texts))

        if unique_found_count == 0:
            return 0

        calculated_score = unique_found_count * score_per_item
        return min(int(round(calculated_score)), max_score)


    def _check_structural_cues(self, text_upper_normalized: str) -> int:
        cue_rules = self.rules.get("structural_cues", {})
        score = 0

        year_pattern = cue_rules.get("comparative_year_pattern", r'\b(19|20)\d{2}\b')
        years_found = re.findall(year_pattern, text_upper_normalized)
        if len(set(years_found)) >= 2:
            score += cue_rules.get("score_comparative_years", 1)
        elif years_found:
            score += cue_rules.get("score_single_year", 0.5)

        note_keywords_found = find_regex_patterns(text_upper_normalized, cue_rules.get("note_column_keywords_regex", [])) # Expects regex list
        currency_indicators_found = find_regex_patterns(text_upper_normalized, cue_rules.get("currency_indicators_regex", [])) # Expects regex list

        if note_keywords_found or currency_indicators_found:
            score += cue_rules.get("score_notes_or_currency", 1)

        return min(int(round(score)), cue_rules.get("max_score", 2))


    def _parse_financial_number(self, value_str: str) -> Optional[float]:
        if not value_str: return None
        cleaned_value = str(value_str).strip()

        is_negative = False
        if cleaned_value.startswith('(') and cleaned_value.endswith(')'):
            is_negative = True
            cleaned_value = cleaned_value[1:-1].strip()

        cleaned_value = cleaned_value.replace(',', '')

        enable_note_ref_heuristic = self.rules.get("balancing_equation", {}).get("enable_note_ref_heuristic", True)
        if enable_note_ref_heuristic and \
           cleaned_value.isdigit() and \
           not is_negative and \
           len(cleaned_value) <= self.rules.get("balancing_equation", {}).get("max_len_for_note_ref_heuristic", 2):
            try:
                small_val = int(cleaned_value)
                if small_val < self.rules.get("balancing_equation", {}).get("max_value_for_note_ref_heuristic", 10):
                    logger.debug(f"SoFP BalEq: Potential note ref '{value_str}' (parsed as small int {small_val}), skipping.")
                    return None
            except ValueError:
                pass

        if not cleaned_value or cleaned_value == '-' or not re.search(r'\d', cleaned_value):
            return None

        try:
            number = float(cleaned_value)
            return -number if is_negative else number
        except ValueError:
            logger.debug(f"SoFP BalEq: Could not parse '{cleaned_value}' (from original '{value_str}') to float.")
            return None

    def _find_values_on_line(self, line_text: str) -> List[float]:
        """Helper to extract all parseable financial numbers from a line of text."""
        potential_values_str = re.findall(r"\(?\s*-?[\d,]+\.?\d*\s*\)?|\(?\s*[\d,]*\.?\d+\s*\)?", line_text)
        parsed_numbers = []
        for val_str in potential_values_str:
            num = self._parse_financial_number(val_str)
            if num is not None:
                parsed_numbers.append(num)
        return parsed_numbers

    def _find_value_for_label(self, text_content_normalized_upper_with_newlines: str,
                               label_variants: List[str], column_index_to_try: int) -> Optional[float]:
        """
        Finds a financial value associated with a label for a specific column index.
        Searches for label variants, then extracts numbers from that line, returning the one at column_index_to_try.
        """
        for label_raw in label_variants:
            label_normalized = " ".join(label_raw.upper().split())
            pattern_text = r'(?:^|\W)' + r'\b' + re.escape(label_normalized) + r'\b' + r'(.*)$'

            for line in text_content_normalized_upper_with_newlines.splitlines():
                line_normalized_upper = normalize_text(line)
                match = re.search(pattern_text, line_normalized_upper)
                if match:
                    rest_of_line = match.group(1)
                    logger.debug(f"SoFP BalEq: Found line for '{label_normalized}': '{line_normalized_upper[:100]}'. Remainder: '{rest_of_line[:50]}'")

                    numbers_on_line = self._find_values_on_line(line_normalized_upper)

                    logger.debug(f"SoFP BalEq: Parsed numbers from line for '{label_normalized}': {numbers_on_line}")
                    if 0 <= column_index_to_try < len(numbers_on_line):
                        selected_value = numbers_on_line[column_index_to_try]
                        logger.debug(f"SoFP BalEq: Selected value '{selected_value}' from column index {column_index_to_try} for label '{label_normalized}'.")
                        return selected_value
                    else:
                        logger.debug(f"SoFP BalEq: Label '{label_normalized}' found, but not enough numbers for column index {column_index_to_try} on line. Numbers found: {numbers_on_line}")
                        return None

        logger.debug(f"SoFP BalEq: Value not found for labels '{label_variants}' for column index {column_index_to_try}.")
        return None


    def _check_balancing_equation(self, raw_text_content: str) -> int:
        if not raw_text_content.strip(): return 0

        bal_eq_rules = self.rules.get("balancing_equation", {})
        score_value = bal_eq_rules.get("score", 0)
        rel_tol = bal_eq_rules.get("relative_tolerance", 1e-4)
        abs_tol = bal_eq_rules.get("absolute_tolerance", 1.01)
        num_columns_to_check = bal_eq_rules.get("num_columns_to_check_for_balance", 2)

        lines = raw_text_content.splitlines()
        processed_lines = [normalize_text(line) for line in lines]
        text_for_search = "\n".join(processed_lines)

        logger.debug(f"SoFP BalEq: Checking balance for up to {num_columns_to_check} columns.")

        for col_idx in range(num_columns_to_check):
            logger.debug(f"SoFP BalEq: Attempting balance check for column index {col_idx}.")
            assets = self._find_value_for_label(text_for_search, bal_eq_rules.get("total_assets_labels", []), col_idx)
            total_liab_equity = self._find_value_for_label(text_for_search, bal_eq_rules.get("total_liabilities_and_equity_labels", []), col_idx)

            if assets is not None and total_liab_equity is not None:
                logger.debug(f"SoFP BalEq (Col {col_idx}): Assets: {assets}, Total L+E (combined): {total_liab_equity}")
                if math.isclose(assets, total_liab_equity, rel_tol=rel_tol, abs_tol=abs_tol):
                    logger.info(f"SoFP BalEq: Balances! (Assets vs Total L+E combined) for column {col_idx}. Score: {score_value}")
                    return score_value

            liabilities = self._find_value_for_label(text_for_search, bal_eq_rules.get("total_liabilities_labels", []), col_idx)
            equity = self._find_value_for_label(text_for_search, bal_eq_rules.get("total_equity_labels", []), col_idx)

            if assets is not None and liabilities is not None and equity is not None:
                sum_liab_equity = liabilities + equity
                logger.debug(f"SoFP BalEq (Col {col_idx}): Assets: {assets}, Liab: {liabilities}, Eq: {equity}, Sum L+E: {sum_liab_equity}")
                if math.isclose(assets, sum_liab_equity, rel_tol=rel_tol, abs_tol=abs_tol):
                    logger.info(f"SoFP BalEq: Balances! (Assets vs L + E separate) for column {col_idx}. Score: {score_value}")
                    return score_value

            if assets is None and liabilities is None and equity is None and total_liab_equity is None and col_idx == 0:
                logger.debug(f"SoFP BalEq: No primary balance sheet items found in column {col_idx}. Unlikely SoFP data for this column.")

        logger.debug(f"SoFP BalEq: Equation did not balance for any checked column or values not found.")
        return 0


    def _calculate_score(self, combined_text_original_case: str, first_block_index: int,
                         is_title_paragraph_present: bool = False, num_table_blocks: int = 0) -> Dict[str, Any]:
        if not combined_text_original_case.strip():
            return {"total": 0, "breakdown": {}}

        text_upper_normalized_for_keywords = normalize_text(combined_text_original_case)

        title_score = 0
        if is_title_paragraph_present:
             title_check_text = normalize_text(combined_text_original_case.split('\n', 1)[0])
             title_score = self._check_sofp_title_phrases(title_check_text)
        elif num_table_blocks > 0:
            title_score = self._check_sofp_title_phrases(text_upper_normalized_for_keywords)


        major_sections_score = self._check_major_sections(text_upper_normalized_for_keywords)
        asset_items_score = self._count_and_score_items(text_upper_normalized_for_keywords, "asset_keywords")
        liability_items_score = self._count_and_score_items(text_upper_normalized_for_keywords, "liability_keywords")
        equity_items_score = self._count_and_score_items(text_upper_normalized_for_keywords, "equity_keywords")
        total_indicators_score = self._count_and_score_items(text_upper_normalized_for_keywords, "total_indicator_keywords")
        structural_cues_score = self._check_structural_cues(text_upper_normalized_for_keywords)

        balancing_equation_score = self._check_balancing_equation(combined_text_original_case)

        block_bonus = 0
        block_bonus_val = self.rules.get("block_bonus_score", 1)
        full_bonus_max_idx = self.rules.get("block_bonus_full_score_max_index", 30) # Increased default for SoFP
        partial_bonus_max_idx = self.rules.get("block_bonus_max_index", 60)
        partial_bonus_factor = self.rules.get("block_bonus_partial_factor", 0.5)
        if first_block_index < full_bonus_max_idx:
            block_bonus = block_bonus_val
        elif first_block_index < partial_bonus_max_idx:
            block_bonus = block_bonus_val * partial_bonus_factor

        combination_bonus = 0
        combo_score_val = self.rules.get("combination_bonus_score", 2)
        min_score_factor_for_strong = self.rules.get("min_score_factor_for_strong_indicator", 0.5)
        min_strong_indicators_for_combo = self.rules.get("min_strong_indicators_for_combo", 3)


        is_strong_title = title_score >= (self.rules.get("title_phrases",{}).get("score",1) * min_score_factor_for_strong)
        is_strong_major_sections = major_sections_score >= (self.rules.get("major_section_keywords",{}).get("max_score",1) * min_score_factor_for_strong)
        avg_items_score = (asset_items_score + liability_items_score + equity_items_score) / 3
        min_avg_items_score_for_strong = self.rules.get("min_avg_items_score_for_strong_combo", 1.5)
        is_strong_items = avg_items_score >= min_avg_items_score_for_strong
        is_strong_totals = total_indicators_score > 0
        is_strong_structure = structural_cues_score > 0

        num_strong_indicators = sum([
            1 if is_strong_title and is_title_paragraph_present else 0,
            1 if is_strong_major_sections else 0,
            1 if is_strong_items and num_table_blocks > 0 else 0,
            1 if is_strong_totals else 0,
            1 if balancing_equation_score > 0 else 0
        ])

        if num_strong_indicators >= min_strong_indicators_for_combo:
            combination_bonus = combo_score_val

        if num_table_blocks > 0 and is_strong_major_sections and is_strong_items and is_strong_totals:
            combination_bonus = max(combination_bonus, self.rules.get("strong_table_combination_bonus", combo_score_val + 1))

        total_score = sum(filter(None, [
            title_score, major_sections_score, asset_items_score, liability_items_score,
            equity_items_score, total_indicators_score, structural_cues_score,
            balancing_equation_score, block_bonus, combination_bonus
        ]))

        breakdown = {
            "title": title_score, "major_sections": major_sections_score,
            "asset_items": asset_items_score, "liability_items": liability_items_score,
            "equity_items": equity_items_score, "total_indicators": total_indicators_score,
            "structural_cues": structural_cues_score, "balancing_equation": balancing_equation_score,
            "block_bonus": block_bonus, "combination_bonus": combination_bonus,
            "num_strong_indicators_for_combo": num_strong_indicators
        }
        return {"total": total_score, "breakdown": breakdown}

    def _identify_and_evaluate_core_window(self,
                                         doc_blocks: List[Dict[str, Any]],
                                         current_doc_block_list_idx: int
                                         ) -> Tuple[Optional[List[Dict[str,Any]]], str, bool, int, int]:
        """
        Identifies a potential CORE SoFP window (title + tables/paras, or tables/paras only).
        Returns: (core_window_blocks, core_window_content, is_title_para_present, num_tables_in_core, next_list_idx_to_check)
        """
        start_block_candidate = doc_blocks[current_doc_block_list_idx]
        current_core_sofp_blocks: List[Dict[str, Any]] = []
        current_core_sofp_content = ""
        is_potential_title_paragraph_for_core = False
        num_table_blocks_in_core = 0
        core_window_last_list_idx = current_doc_block_list_idx -1

        block_type_of_start_candidate = start_block_candidate.get('type')

        if block_type_of_start_candidate == BLOCK_TYPE_PARAGRAPH:
            title_check_text = normalize_text(start_block_candidate['content'])
            min_raw_title_score_for_start = (self.rules.get("title_phrases",{}).get("score", 4) * 0.25) # e.g., 1 if score is 4
            if self._check_sofp_title_phrases(title_check_text) >= min_raw_title_score_for_start :
                logger.debug(f"SoFP Core: Potential title paragraph {start_block_candidate['index']} to start window.")
                current_core_sofp_blocks.append(start_block_candidate)
                current_core_sofp_content += start_block_candidate['content'] + "\n"
                is_potential_title_paragraph_for_core = True
                core_window_last_list_idx = current_doc_block_list_idx

        accumulation_start_list_idx = core_window_last_list_idx + 1

        if not is_potential_title_paragraph_for_core and block_type_of_start_candidate == BLOCK_TYPE_TABLE:
            current_core_sofp_blocks.append(start_block_candidate)
            current_core_sofp_content += start_block_candidate['content'] + "\n"
            num_table_blocks_in_core += 1
            core_window_last_list_idx = current_doc_block_list_idx
            accumulation_start_list_idx = current_doc_block_list_idx + 1
        elif not is_potential_title_paragraph_for_core and block_type_of_start_candidate != BLOCK_TYPE_TABLE:
            return None, "", False, 0, current_doc_block_list_idx + 1

        max_core_blocks = self.rules.get("max_blocks_in_core_sofp_window", 20) # New rule
        for k_core in range(accumulation_start_list_idx, min(len(doc_blocks), accumulation_start_list_idx + max_core_blocks)):
            core_block_to_add = doc_blocks[k_core]
            if self._is_hard_termination_block(core_block_to_add['content'], core_block_to_add.get('type')):
                logger.debug(f"SoFP Core: Sequence stopped by HARD termination: Block {core_block_to_add['index']}")
                break

            block_type_to_add = core_block_to_add.get('type')
            if block_type_to_add == BLOCK_TYPE_TABLE or block_type_to_add == BLOCK_TYPE_PARAGRAPH:
                current_core_sofp_blocks.append(core_block_to_add)
                current_core_sofp_content += core_block_to_add['content'] + "\n"
                if block_type_to_add == BLOCK_TYPE_TABLE:
                    num_table_blocks_in_core += 1
                core_window_last_list_idx = k_core
            else:
                logger.debug(f"SoFP Core: Sequence stopped by non-table/non-paragraph block: {core_block_to_add['index']} (type: {block_type_to_add})")
                break

        min_blocks_after_title_for_core = self.rules.get("min_content_blocks_after_title_for_core", 1) # New rule
        if is_potential_title_paragraph_for_core and (len(current_core_sofp_blocks) -1) < min_blocks_after_title_for_core :
            if num_table_blocks_in_core == 0 and (len(current_core_sofp_blocks) -1) < (min_blocks_after_title_for_core + 1): # Stricter if no tables
                logger.debug(f"SoFP Core: Discarding title-led window for block {start_block_candidate['index']} due to insufficient subsequent content ({len(current_core_sofp_blocks)-1} blocks, {num_table_blocks_in_core} tables).")
                current_core_sofp_blocks = []

        if not current_core_sofp_blocks:
             next_i_to_check = current_doc_block_list_idx + 1
             return None, "", False, 0, next_i_to_check

        next_i_to_check = core_window_last_list_idx + 1
        return current_core_sofp_blocks, current_core_sofp_content.strip(), is_potential_title_paragraph_for_core, num_table_blocks_in_core, next_i_to_check


    def classify(self, doc_blocks: List[Dict[str, Any]],
                 confidence_threshold: float = 0.55,
                 max_start_block_index_to_check: int = 700,
                 **kwargs) -> Optional[Dict[str, Any]]:

        current_list_idx = kwargs.get("start_block_index_in_list", 0)
        best_sofp_result = None

        while current_list_idx < len(doc_blocks):
            start_block_candidate_for_iteration = doc_blocks[current_list_idx]

            if start_block_candidate_for_iteration['index'] >= max_start_block_index_to_check:
                logger.debug(f"SoFP Classify: Stopping search. Current block doc index {start_block_candidate_for_iteration['index']} >= {max_start_block_index_to_check}.")
                break

            if self._is_hard_termination_block(start_block_candidate_for_iteration['content'], start_block_candidate_for_iteration.get('type')):
                logger.debug(f"SoFP Classify: Skipping block {start_block_candidate_for_iteration['index']} as start: hard termination.")
                current_list_idx += 1
                continue

            core_blocks, core_content, is_title_para_start, num_tables_in_core, next_list_idx_after_core_attempt = \
                self._identify_and_evaluate_core_window(doc_blocks, current_list_idx)

            if not core_blocks:
                current_list_idx = next_list_idx_after_core_attempt
                continue

            # --- Evaluate the Core Window ---
            first_block_in_core_doc_idx = core_blocks[0]['index']
            core_score_result = self._calculate_score(
                core_content,
                first_block_in_core_doc_idx,
                is_title_para_start,
                num_tables_in_core
            )
            core_raw_score = core_score_result["total"]
            core_capped_score = min(core_raw_score, self.max_score_exemplar)
            core_confidence = (core_capped_score / (self.max_score_exemplar + 1e-9))

            logger.debug(f"SoFP Core EVALUATED: Blocks {[b['index'] for b in core_blocks]}. Score: {core_raw_score}, Conf: {core_confidence:.3f}")

            current_expanded_result = None
            if core_confidence >= confidence_threshold * self.rules.get("core_confidence_factor_for_expansion", 0.9): # Core must be reasonably good
                logger.debug(f"SoFP Core (Blocks {[b['index'] for b in core_blocks]}) qualified for expansion attempt.")
                # --- Expansion Phase ---
                expanded_window_blocks = list(core_blocks)
                expanded_window_content_original_case = str(core_content)

                max_expansion_blocks = self.rules.get("max_blocks_for_sofp_expansion", 15) #
                expansion_start_list_idx = next_list_idx_after_core_attempt

                for j_expand in range(expansion_start_list_idx, min(len(doc_blocks), expansion_start_list_idx + max_expansion_blocks)):
                    block_to_add = doc_blocks[j_expand]
                    if self._is_hard_termination_block(block_to_add['content'], block_to_add.get('type')):
                        logger.debug(f"SoFP Expansion: Stopped by hard termination at block {block_to_add['index']}.")
                        break

                    expanded_window_blocks.append(block_to_add)
                    expanded_window_content_original_case += "\n" + block_to_add['content']
                    logger.debug(f"SoFP Expansion: Added block {block_to_add['index']}.")

                # --- Re-score the Expanded Window ---
                num_tables_in_expanded = sum(1 for b in expanded_window_blocks if b.get('type') == BLOCK_TYPE_TABLE)
                expanded_score_result = self._calculate_score(
                    expanded_window_content_original_case,
                    expanded_window_blocks[0]['index'],
                    is_title_para_start,
                    num_tables_in_expanded
                )
                expanded_raw_score = expanded_score_result["total"]
                expanded_capped_score = min(expanded_raw_score, self.max_score_exemplar)
                expanded_confidence = (expanded_capped_score / (self.max_score_exemplar + 1e-9))

                logger.info(f"SoFP Expanded Window (Blocks {[b['index'] for b in expanded_window_blocks]}) RESCORED. Score: {expanded_raw_score}, Conf: {expanded_confidence:.3f}")

                if expanded_confidence >= confidence_threshold:
                    current_expanded_result = {
                        "section_name": self.rules.get("section_name", SECTION_NAME_SOFP),
                        "start_block_index": expanded_window_blocks[0]['index'],
                        "end_block_index": expanded_window_blocks[-1]['index'],
                        "block_indices": [b['index'] for b in expanded_window_blocks],
                        "num_blocks": len(expanded_window_blocks),
                        "confidence": expanded_confidence,
                        "raw_score": expanded_raw_score,
                        "breakdown": expanded_score_result["breakdown"],
                        "content_preview": normalize_text(expanded_window_content_original_case)[:300],
                    }

            if current_expanded_result:
                if best_sofp_result is None or current_expanded_result['confidence'] > best_sofp_result['confidence']:
                    best_sofp_result = current_expanded_result
                    logger.info(f"SoFP New Best Candidate: Blocks {best_sofp_result['block_indices']} (Conf: {best_sofp_result['confidence']:.3f})")

                current_list_idx = doc_blocks.index(expanded_window_blocks[-1]) + 1 if expanded_window_blocks[-1] in doc_blocks else next_list_idx_after_core_attempt
            else:
                current_list_idx = next_list_idx_after_core_attempt

        if best_sofp_result:
            logger.info(f"SoFP Final Best Classification: Blocks {best_sofp_result['block_indices']} (Confidence {best_sofp_result['confidence']:.3f})")
            return best_sofp_result

        logger.info("No SoFP section identified with sufficient confidence after checking all potential windows.")
        return None

    def display_results(self, sofp_result: Optional[Dict[str, Any]], all_doc_blocks: Optional[List[Dict[str, Any]]] = None) -> None:
        super().display_results(classification_result=sofp_result, all_doc_blocks=all_doc_blocks)
        if sofp_result and logger.isEnabledFor(logging.DEBUG):
            if all_doc_blocks:
                logger.debug(f"\n  Identified SoFP content (block by block preview):")
                block_content_map = {block['index']: block['content'] for block in all_doc_blocks}
                for i, block_idx in enumerate(sofp_result.get('block_indices', [])):
                    if i >= 5 and len(sofp_result['block_indices']) > 5 : # Preview first 5
                        logger.debug(f"    ... and {len(sofp_result['block_indices']) - 5} more blocks.")
                        break
                    content = block_content_map.get(block_idx, "Content not found.")
                    block_type = next((b.get('type', 'N/A') for b in all_doc_blocks if b['index'] == block_idx), 'N/A')
                    display_content = content.replace('\n', ' ').replace('\r', ' ').strip()[:80] + "..."
                    logger.debug(f"    Block {block_idx} (type: {block_type}): {display_content}")
        elif not sofp_result:
             logger.info("  Suggestions for SoFP detection:")
             logger.info("   - Adjust `confidence_threshold` in main script.")
             logger.info("   - Verify `max_start_block_index_to_check`.")
             logger.info("   - Review `sofp_rules.json` for keyword accuracy, scores, and new parameters like `num_columns_to_check_for_balance`.")
             logger.info("   - Ensure `max_score_exemplar` in rules reflects potential scores from updated logic.")
