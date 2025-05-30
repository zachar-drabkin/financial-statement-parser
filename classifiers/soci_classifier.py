import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from classifiers.base_classifier import BaseClassifier
from utils.text_utils import normalize_text

logger = logging.getLogger(__name__)


class SoCIClassifier(BaseClassifier):
    """
    Rule-based Classifier for identifying Statement of Comprehensive Income (SoCI)
    sections in Financial Statements.
    """

    def __init__(self, rules_file_path: str = "rules/soci_rules.json"):
        super().__init__(rules_file_path=rules_file_path)
        self.section_name = self.rules.get("section_name", "Statement of Comprehensive Income (Fallback Name)")
        self._normalized_termination_titles = self._get_all_known_statement_titles_normalized()

    def _get_all_known_statement_titles_normalized(self) -> List[str]:
        """Helper to get all known statement titles, normalized for termination checks."""
        titles = []
        soci_title_rules = self.rules.get("soci_titles", {})
        for kw_list_name, kw_list_data in soci_title_rules.items():
            if isinstance(kw_list_data, list): # Ensure it's a list of title dicts
                for kw_item in kw_list_data:
                    if isinstance(kw_item, dict) and "text" in kw_item:
                        titles.append(normalize_text(kw_item['text']))

        term_kw_rules = self.rules.get("termination_keywords", {})
        for kw_item_text in term_kw_rules.get("statement_titles", []):
            titles.append(normalize_text(kw_item_text))
        return list(set(titles))

    def _find_keywords(self, normalized_text_upper: str, keyword_definitions: List[Dict[str, Any]], unique: bool = True) -> Tuple[int, List[str], int]:
        """
        Finds keywords in text and calculates a score based on definitions.
        Each item in keyword_definitions should be a dict with 'text' and 'score'.
        """
        found_score = 0
        found_items_texts = [] # store for debug
        count = 0
        seen_keywords_upper = set()

        for item_def in keyword_definitions:
            keyword_original_case = item_def.get("text", "")
            item_score = item_def.get("score", 0)
            keyword_upper = keyword_original_case.upper()

            pattern = r'\b' + re.escape(keyword_upper) + r'\b'
            if re.search(pattern, normalized_text_upper):
                if unique and keyword_upper in seen_keywords_upper:
                    continue
                seen_keywords_upper.add(keyword_upper)
                found_score += item_score
                found_items_texts.append(keyword_original_case)
                count += 1
        return found_score, found_items_texts, count

    def _check_soci_titles(self, normalized_text_upper_block: str) -> Tuple[int, Optional[str], Optional[str]]:
        max_score = 0
        best_title_text = None
        best_title_type = None

        title_rules = self.rules.get("soci_titles", {})
        buffer = title_rules.get("length_check_buffer", 30)

        all_title_keywords_with_details = []
        # list the keys whose values are lists of titl definition dictionaries
        title_definition_list_keys = ["ifrs_primary", "gaap_primary", "shared_ambiguous"]

        for key in title_definition_list_keys:
            category_list = title_rules.get(key) # get the list associated with this key
            if isinstance(category_list, list):
                for item in category_list:
                    if isinstance(item, dict) and "text" in item:
                        all_title_keywords_with_details.append(item)
                    elif isinstance(item, dict):
                        logger.warning(f"SoCI Title Rule: Item in '{key}' is a dict but missing 'text' key: {item}")
                    else:
                        logger.warning(f"SoCI Title Rule: Unexpected item type in '{key}'. Expected dict, got {type(item)}: {item}")
            elif category_list is not None: # key exists but is not a list
                 logger.warning(f"SoCI Title Rule: Expected list for '{key}', but got {type(category_list)}.")

        if not all_title_keywords_with_details:
            logger.debug(f"No valid title definition dictionaries found in soci_titles rules for checking '{normalized_text_upper_block[:50]}...'")
            return 0, None, None

        # sort by length of text to match longer phrases first
        sorted_title_keywords = sorted(all_title_keywords_with_details, key=lambda x: len(x.get("text","")), reverse=True)

        for title_item in sorted_title_keywords:
            title_keyword_original = title_item.get("text", "")
            if not title_keyword_original: # skip if text is empty for some reason
                continue
            title_keyword_upper = title_keyword_original.upper()
            item_score = title_item.get("score", 0)
            item_type = title_item.get("type")

            pattern = r'\b' + re.escape(title_keyword_upper) + r'\b'
            match = re.search(pattern, normalized_text_upper_block)
            if match:
                if len(normalized_text_upper_block) < (len(title_keyword_upper) + buffer):
                    if item_score > max_score:
                        max_score = item_score
                        best_title_text = title_keyword_original
                        best_title_type = item_type
        return max_score, best_title_text, best_title_type

    def _check_pl_keywords(self, normalized_text_upper: str) -> Tuple[int, List[str], bool, bool]:
        total_score = 0
        all_found_keywords_texts = set()
        has_revenue_flag = False
        has_net_income_flag = False
        pl_rules = self.rules.get("pl_keywords", {})

        revenue_score, revenue_kw_texts, _ = self._find_keywords(normalized_text_upper, pl_rules.get("revenue", []))
        if revenue_kw_texts:
            total_score += revenue_score
            all_found_keywords_texts.update(revenue_kw_texts)
            has_revenue_flag = True

        net_income_score, net_income_kw_texts, _ = self._find_keywords(normalized_text_upper, pl_rules.get("net_income", []))
        if net_income_kw_texts:
            total_score += net_income_score
            all_found_keywords_texts.update(net_income_kw_texts)
            has_net_income_flag = True

        for category_name, keywords_list_defs in pl_rules.items():
            if category_name in ["revenue", "net_income"]: continue # -> already processed
            if isinstance(keywords_list_defs, list):
                cat_score, cat_kw_texts, _ = self._find_keywords(normalized_text_upper, keywords_list_defs)
                if cat_kw_texts:
                    total_score += cat_score
                    all_found_keywords_texts.update(cat_kw_texts)
        return total_score, list(all_found_keywords_texts), has_revenue_flag, has_net_income_flag

    def _check_oci_keywords(self, normalized_text_upper: str) -> Tuple[int, List[str]]:
        total_score = 0
        all_found_keywords_texts = set()
        oci_rules = self.rules.get("oci_keywords", {})
        for category_name, keywords_list_defs in oci_rules.items():
            if isinstance(keywords_list_defs, list):
                cat_score, cat_kw_texts, _ = self._find_keywords(normalized_text_upper, keywords_list_defs)
                if cat_kw_texts:
                    total_score += cat_score
                    all_found_keywords_texts.update(cat_kw_texts)
        return total_score, list(all_found_keywords_texts)

    def _check_structural_cues(self, normalized_text_upper: str) -> Tuple[int, List[str]]:
        score = 0
        found_cues_texts = []
        cue_rules = self.rules.get("structural_cues", {})

        period_score, period_kws_texts, _ = self._find_keywords(normalized_text_upper, cue_rules.get("period_indicators_keywords", []))
        if period_kws_texts: score += period_score; found_cues_texts.extend(period_kws_texts)

        for pattern_str in cue_rules.get("period_indicators_patterns", []):
            if re.search(pattern_str, normalized_text_upper, re.IGNORECASE):
                score += cue_rules.get("period_pattern_score", 5)
                found_cues_texts.append(f"Date Pattern: {pattern_str.split(':', 1)[0]}") # Cleaner display
                break

        year_matches = re.findall(cue_rules.get("year_pattern", r'\b(?:19|20)\d{2}\b'), normalized_text_upper)
        if len(set(year_matches)) >= 2: score += cue_rules.get("comparative_year_score", 5); found_cues_texts.append(f"Comparative Years: {list(set(year_matches))}")
        elif year_matches: score += cue_rules.get("single_year_score", 2); found_cues_texts.append(f"Year Found: {list(set(year_matches))}")

        currency_score_gained_this_check = False
        curr_kw_score, curr_kws_texts, _ = self._find_keywords(normalized_text_upper, cue_rules.get("currency_keywords",[]))
        if curr_kws_texts: score += curr_kw_score; found_cues_texts.extend(curr_kws_texts); currency_score_gained_this_check = True

        if not currency_score_gained_this_check:
            for iso_code in cue_rules.get("currency_iso_codes", []):
                if re.search(r'\b' + re.escape(iso_code) + r'\b', normalized_text_upper): # ISO codes are case sensitive often
                    score += cue_rules.get("currency_iso_code_score", 2); found_cues_texts.append(f"ISO Code: {iso_code}"); currency_score_gained_this_check = True; break
        if not currency_score_gained_this_check:
            for symbol in cue_rules.get("currency_symbols", []):
                if symbol in normalized_text_upper:
                    score += cue_rules.get("currency_symbol_score", 1); found_cues_texts.append(f"Symbol: {symbol}"); break
        return score, found_cues_texts

    def _check_eps_discontinued_ops(self, normalized_text_upper: str) -> Tuple[int, List[str]]:
        score, kws_texts, _ = self._find_keywords(normalized_text_upper, self.rules.get("eps_discontinued_ops", []))
        return score, kws_texts

    def _is_termination_block(self, block_content_original_case: str, block_type: Optional[str]) -> bool:
        normalized_content = normalize_text(block_content_original_case)
        if not normalized_content: return False

        term_rules = self.rules.get("termination_keywords", {})
        title_buffer = term_rules.get("title_buffer_for_termination_check", 60)
        section_header_buffer = term_rules.get("section_header_buffer_for_termination_check", 80)

        # check in pre-normalized known SOCI titles
        # prevent self-termination if an SOCI title variation appears in termination list.
        current_block_is_soci_title_itself = False
        soci_title_score, _, _ = self._check_soci_titles(normalized_content)
        if soci_title_score > 0:
            current_block_is_soci_title_itself = True

        for title_text_normalized in self._normalized_termination_titles: # Uses cached list
            # avoid self-terminating if this block is an sOCI title
            if current_block_is_soci_title_itself and title_text_normalized in [normalize_text(k.get('text','')) for k_list in self.rules.get("soci_titles",{}).values() if isinstance(k_list, list) for k in k_list]:
                continue

            pattern = r'\b' + re.escape(title_text_normalized) + r'\b'
            match = re.search(pattern, normalized_content)
            if match:
                if len(normalized_content) < (len(title_text_normalized) + title_buffer):
                    logger.debug(f"SoCI Term: Statement title '{title_text_normalized}' in block '{normalized_content[:100]}...'")
                    return True

        for section_header_text in term_rules.get("other_sections", []):
            normalized_header = normalize_text(section_header_text)
            pattern = r'\b' + re.escape(normalized_header) + r'\b'
            if re.search(pattern, normalized_content):
                if len(normalized_content) < (len(normalized_header) + section_header_buffer):
                    logger.debug(f"SoCI Term: Other section header '{normalized_header}' in block '{normalized_content[:100]}...'")
                    return True

        for note_pattern_regex in term_rules.get("note_section_patterns", []):
            # use original case for regex that might depend on it, but strip for general patterns
            if re.search(note_pattern_regex, block_content_original_case.strip(), re.IGNORECASE | re.MULTILINE):
                logger.debug(f"SoCI Term: Note section pattern '{note_pattern_regex}' in block '{normalized_content[:100]}...'")
                return True
        return False

    def _calculate_score(self, combined_text_original_case: str, first_block_index: int,
                         title_info: Optional[Tuple[int, str, str]] = None,
                         is_table_only_core: bool = False, **kwargs) -> Dict[str, Any]:
        if not combined_text_original_case.strip():
            return {"total": 0, "breakdown": {}, "best_title_text": None, "best_title_type": None,
                    "is_potential_ifrs_pl": False, "is_missing_critical_pl": True}

        normalized_combined_text = normalize_text(combined_text_original_case)
        breakdown: Dict[str, Any] = {}
        bonus_rules = self.rules.get("bonuses_penalties", {})
        pl_rules = self.rules.get("pl_keywords", {})

        title_score, best_title_text, best_title_type = 0, None, None
        if title_info:
            title_score, best_title_text, best_title_type = title_info
        elif not is_table_only_core:
            temp_title_score, temp_bt, temp_btt = self._check_soci_titles(normalized_combined_text)
            if temp_title_score > 0:
                title_score, best_title_text, best_title_type = temp_title_score, temp_bt, temp_btt
        breakdown["title"] = title_score

        pl_score, found_pl_kws, has_revenue, has_net_income = self._check_pl_keywords(normalized_combined_text)
        is_missing_critical_pl = not (has_revenue and has_net_income)
        if is_missing_critical_pl and pl_score > pl_rules.get("min_pl_score_for_critical_item_penalty", 10) :
            original_pl_score = pl_score
            penalty_divisor = pl_rules.get("critical_item_missing_penalty_divisor", 3)
            pl_score = max(0, pl_score // penalty_divisor if penalty_divisor != 0 else 0)
            breakdown["pl_penalty_critical_items_missing"] = pl_score - original_pl_score
        breakdown["p&l_content"] = pl_score
        breakdown["found_pl_keywords"] = found_pl_kws

        oci_score, found_oci_kws = self._check_oci_keywords(normalized_combined_text)
        breakdown["oci_content"] = oci_score
        breakdown["found_oci_keywords"] = found_oci_kws

        structural_score, found_structural_cues = self._check_structural_cues(normalized_combined_text)
        breakdown["structural_cues"] = structural_score
        breakdown["found_structural_cues"] = found_structural_cues

        eps_dis_score, found_eps_dis_kws = self._check_eps_discontinued_ops(normalized_combined_text)
        breakdown["eps_discontinued_ops"] = eps_dis_score
        breakdown["found_eps_discontinued_ops_keywords"] = found_eps_dis_kws

        ifrs_two_statement_bonus = breakdown.get("ifrs_two_statement_bonus", 0) # Applied later if IFRS 2-stmt confirmed

        block_b_score = bonus_rules.get("block_bonus_score", 5)
        block_bonus = block_b_score if first_block_index < bonus_rules.get("block_bonus_max_index", 30) else 0
        breakdown["block_bonus"] = block_bonus

        combination_bonus = 0
        title_is_strong = title_score >= bonus_rules.get("min_title_score_for_combo", 5)
        pl_is_strong = pl_score >= bonus_rules.get("min_pl_score_for_combo", 40) and not is_missing_critical_pl
        oci_expected = best_title_type and ("soci" in best_title_type or "oci" in best_title_type or "comprehensive" in best_title_type.lower())
        oci_is_present_or_not_needed = (oci_expected and oci_score >= bonus_rules.get("min_oci_score_if_expected_for_combo", 10)) or (not oci_expected)

        if (title_is_strong or is_table_only_core) and pl_is_strong and oci_is_present_or_not_needed:
            combination_bonus = bonus_rules.get("combination_bonus_score", 10)
        breakdown["combination_bonus"] = combination_bonus

        total_score = sum(val for val in [title_score, pl_score, oci_score, structural_score, eps_dis_score,
                                          ifrs_two_statement_bonus, block_bonus, combination_bonus] if isinstance(val, (int, float)))

        is_potential_ifrs_pl = False
        if best_title_type == "ifrs_pl" and pl_is_strong and oci_score < bonus_rules.get("min_oci_score_if_expected_for_combo", 10) and not is_missing_critical_pl:
            is_potential_ifrs_pl = True

        return {
            "total": total_score, "breakdown": breakdown, "best_title_text": best_title_text,
            "best_title_type": best_title_type, "is_potential_ifrs_pl": is_potential_ifrs_pl,
            "is_missing_critical_pl": is_missing_critical_pl
        }

    def _attempt_window_formation(self, doc_blocks: List[Dict[str, Any]],
                                  start_list_idx: int, max_blocks_in_window: int
                                 ) -> Tuple[Optional[List[Dict[str,Any]]], str, Optional[Tuple[int,str,str]], bool, int]:
        """
        Attempts to form a candidate SOCI window starting from start_list_idx.
        Returns: (window_blocks, window_content, initial_title_info, is_table_only_start, window_last_list_idx)
        """
        start_block_candidate = doc_blocks[start_list_idx]
        current_window_blocks = []
        current_window_content_original = ""
        initial_title_info = None
        is_table_only_start = False
        window_last_list_idx = start_list_idx -1
        block_type_of_start = start_block_candidate.get('type')
        soci_title_rules = self.rules.get("soci_titles", {})

        if block_type_of_start == 'paragraph':
            normalized_start_content = normalize_text(start_block_candidate['content'])
            prelim_title_score, prelim_title_text, prelim_title_type = self._check_soci_titles(normalized_start_content)
            min_title_score_start = soci_title_rules.get("min_title_score_paragraph_start", 4)
            indicative_terms = soci_title_rules.get("indicative_terms_in_title_paragraph", [])
            has_indicative = any(re.search(term, normalized_start_content, re.IGNORECASE) for term in indicative_terms)

            if prelim_title_score >= min_title_score_start or (prelim_title_score > 0 and has_indicative):
                logger.debug(f"SoCI Window: Potential title paragraph {start_block_candidate['index']} (Score: {prelim_title_score})")
                current_window_blocks.append(start_block_candidate)
                current_window_content_original += start_block_candidate['content'] + "\n"
                initial_title_info = (prelim_title_score, prelim_title_text, prelim_title_type)
                window_last_list_idx = start_list_idx

                # append subsequent tables
                for k_follow in range(start_list_idx + 1, min(start_list_idx + 1 + max_blocks_in_window - 1, len(doc_blocks))):
                    block_to_add = doc_blocks[k_follow]
                    if self._is_termination_block(block_to_add['content'], block_to_add.get('type')):
                        logger.debug(f"SoCI Window (title-led): Table sequence stopped by termination: Block {block_to_add['index']}")
                        break
                    if block_to_add.get('type') == 'table':
                        current_window_blocks.append(block_to_add)
                        current_window_content_original += block_to_add['content'] + "\n"
                        window_last_list_idx = k_follow
                    else:
                        logger.debug(f"SoCI Window (title-led): Table sequence stopped by non-table: Block {block_to_add['index']}")
                        break
            else:
                 logger.debug(f"SoCI Window: Skipping paragraph {start_block_candidate['index']} as start: low title score or no indicative terms.")

        if not current_window_blocks and block_type_of_start == 'table': # Table-only start
            logger.debug(f"SoCI Window: Potential table-only start at block {start_block_candidate['index']}")
            is_table_only_start = True
            for k_follow in range(start_list_idx, min(start_list_idx + max_blocks_in_window, len(doc_blocks))):
                block_to_add = doc_blocks[k_follow]
                if self._is_termination_block(block_to_add['content'], block_to_add.get('type')):
                    logger.debug(f"SoCI Window (table-only): Sequence stopped by termination: Block {block_to_add['index']}")
                    break
                if block_to_add.get('type') == 'table':
                    current_window_blocks.append(block_to_add)
                    current_window_content_original += block_to_add['content'] + "\n"
                    window_last_list_idx = k_follow
                else:
                    logger.debug(f"SoCI Window (table-only): Sequence stopped by non-table: Block {block_to_add['index']}")
                    break

        return current_window_blocks if current_window_blocks else None, current_window_content_original, initial_title_info, is_table_only_start, window_last_list_idx

    def _handle_ifrs_two_statement(self, current_window_blocks: List[Dict[str,Any]],
                                   current_window_content_original: str,
                                   score_result_so_far: Dict[str,Any],
                                   window_last_list_idx: int,
                                   doc_blocks: List[Dict[str,Any]]
                                  ) -> Tuple[List[Dict[str,Any]], str, Dict[str,Any], int]:
        """Handles IFRS two-statement logic by looking for a subsequent OCI part."""
        bonus_rules = self.rules.get("bonuses_penalties", {})
        oci_rules = self.rules.get("oci_keywords", {})
        general_cfg = self.rules.get("general_config", {})

        logger.debug(f"SoCI IFRS: Potential IFRS P&L statement found (ends list_idx {window_last_list_idx}). Looking for OCI part...")
        oci_window_blocks_extension = []
        oci_window_content_extension_original = ""
        idx_after_pl_part_in_list = window_last_list_idx + 1
        max_oci_blocks_to_check = general_cfg.get("max_blocks_in_oci_extension", 5)

        for k_oci in range(idx_after_pl_part_in_list, min(idx_after_pl_part_in_list + max_oci_blocks_to_check, len(doc_blocks))):
            oci_candidate_block = doc_blocks[k_oci]
            if self._is_termination_block(oci_candidate_block['content'], oci_candidate_block.get('type')):
                logger.debug(f"SoCI IFRS: OCI search stopped by termination: Block {oci_candidate_block['index']}")
                break

            normalized_oci_candidate_content = normalize_text(oci_candidate_block['content'])
            oci_header_score, _ = self._check_oci_keywords(normalized_oci_candidate_content)

            min_header_score = oci_rules.get("min_oci_header_score_for_extension", 5)
            if oci_header_score >= min_header_score or (len(oci_window_blocks_extension) > 0 and oci_candidate_block.get('type') == 'table'):
                oci_window_blocks_extension.append(oci_candidate_block)
                oci_window_content_extension_original += oci_candidate_block['content'] + "\n"
                logger.debug(f"SoCI IFRS: Added block {oci_candidate_block['index']} to OCI extension.")

                temp_oci_score_ext, _ = self._check_oci_keywords(normalize_text(oci_window_content_extension_original))
                min_total_score_stop = oci_rules.get("min_total_oci_score_for_extension_stop", 15)
                if temp_oci_score_ext >= min_total_score_stop:
                    logger.debug(f"SoCI IFRS: Sufficient OCI content found in extension (score {temp_oci_score_ext}).")
                    break
            elif oci_window_blocks_extension: # current block doesn't fit
                logger.debug(f"SoCI IFRS: Block {oci_candidate_block['index']} does not fit OCI pattern well (OCI score: {oci_header_score}). Stopping OCI extension.")
                break
            else: # first block for OCI extension not fitting
                logger.debug(f"SoCI IFRS: Block {oci_candidate_block['index']} not a strong start for OCI part (OCI score: {oci_header_score}).")
                break

        if oci_window_blocks_extension:
            logger.info(f"SoCI IFRS: OCI part identified. Merging with P&L part.")
            updated_window_blocks = current_window_blocks + oci_window_blocks_extension
            updated_window_content = current_window_content_original.strip() + "\n" + oci_window_content_extension_original.strip()

            # rescore the merged window
            updated_score_result = self._calculate_score(
                updated_window_content,
                updated_window_blocks[0]['index'], # first_block_index from merged
                title_info=(score_result_so_far["best_title_text"], score_result_so_far["best_title_type"], score_result_so_far["breakdown"]["title"]), # Pass existing title info
                is_table_only_core=score_result_so_far.get("is_table_only_core_flag_for_rescore", False) # need this if relevant
            )
            updated_score_result["breakdown"]["ifrs_two_statement_bonus"] = bonus_rules.get("ifrs_two_statement_bonus", 15)
            updated_score_result["total"] += bonus_rules.get("ifrs_two_statement_bonus", 15)
            updated_score_result["is_ifrs_two_statement_confirmed"] = True

            updated_window_last_list_idx = window_last_list_idx + len(oci_window_blocks_extension)
            return updated_window_blocks, updated_window_content, updated_score_result, updated_window_last_list_idx

        # no OCI extension found, return original
        return current_window_blocks, current_window_content_original, score_result_so_far, window_last_list_idx


    def classify(self, doc_blocks: List[Dict[str, Any]],
                 confidence_threshold: float = 0.50,
                 max_start_block_index_to_check: int = 500,
                 max_blocks_in_soci_window: int = 30, # max blocks for a single statement attempt
                 **kwargs) -> Optional[Dict[str, Any]]:

        best_soci_candidate = None
        current_list_idx = kwargs.get("start_block_index_in_list", 0)
        bonus_rules = self.rules.get("bonuses_penalties", {})
        pl_rules = self.rules.get("pl_keywords", {})

        while current_list_idx < len(doc_blocks):
            start_block_candidate_for_iteration = doc_blocks[current_list_idx]

            if start_block_candidate_for_iteration['index'] >= max_start_block_index_to_check:
                logger.debug(f"SoCI Classify: Stopping search. Block doc index {start_block_candidate_for_iteration['index']} >= {max_start_block_index_to_check}.")
                break

            if self._is_termination_block(start_block_candidate_for_iteration['content'], start_block_candidate_for_iteration.get('type')) and \
               self._check_soci_titles(normalize_text(start_block_candidate_for_iteration['content']))[0] == 0:
                logger.debug(f"SoCI Classify: Skipping block {start_block_candidate_for_iteration['index']} as start: hard termination and not SoCI title.")
                current_list_idx += 1
                continue

            window_blocks, window_content, initial_title_info, is_table_only_start, window_last_list_idx = \
                self._attempt_window_formation(doc_blocks, current_list_idx, max_blocks_in_soci_window)

            if window_blocks:
                final_window_content_original = window_content.strip()
                first_block_in_window_doc_idx = window_blocks[0]['index']

                score_result = self._calculate_score(
                    final_window_content_original,
                    first_block_in_window_doc_idx,
                    title_info=initial_title_info,
                    is_table_only_core=is_table_only_start
                )
                # store is_table_only_start flag in score_result if needed by IFRS rescore logic
                score_result["is_table_only_core_flag_for_rescore"] = is_table_only_start

                # --- IFRS Two-Statement Logic ---
                min_pl_score_ifrs_check = bonus_rules.get("min_initial_pl_score_for_ifrs_two_statement_check", 40)
                if score_result.get("is_potential_ifrs_pl", False) and score_result.get("total",0) > min_pl_score_ifrs_check :
                    window_blocks, final_window_content_original, score_result, window_last_list_idx = \
                        self._handle_ifrs_two_statement(window_blocks, final_window_content_original, score_result, window_last_list_idx, doc_blocks)

                raw_score = score_result["total"]
                breakdown_details = score_result["breakdown"]
                normalized_display_title = score_result.get("best_title_text") or f"{self.section_name} (Inferred)"

                capped_score = min(raw_score, self.max_score_exemplar)
                final_confidence = (capped_score / (self.max_score_exemplar + 1e-9)) # Epsilon for safety

                if score_result.get("is_missing_critical_pl", False) and raw_score > pl_rules.get("min_pl_score_for_critical_item_penalty",10):
                     final_confidence = min(final_confidence, bonus_rules.get("confidence_cap_if_critical_pl_missing", 0.2))
                     logger.warning(f"SoCI Window (Blocks starting {window_blocks[0]['index']}): Critical P&L items missing. Confidence capped to {final_confidence:.3f}.")

                window_indices_str = str([b['index'] for b in window_blocks])
                logger.debug(f"SoCI EVALUATING WINDOW: Blocks Doc Indices {window_indices_str} (List idx {current_list_idx}-{window_last_list_idx})")
                logger.debug(f"  Raw Score: {raw_score:.2f}, Confidence: {final_confidence:.3f}, Title: '{score_result.get('best_title_text')}' ({score_result.get('best_title_type')})")
                if logger.isEnabledFor(logging.DEBUG): # avoid formatting large dict if not needed
                    logger.debug(f"  Breakdown: {breakdown_details}")


                if final_confidence >= confidence_threshold:
                    logger.info(f"  >>> SoCI SECTION CANDIDATE MEETS THRESHOLD (Window {window_indices_str}), Conf: {final_confidence:.3f}")
                    candidate_result = {
                        "section_name": self.section_name,
                        "normalized_section_name": normalized_display_title,
                        "start_block_index": window_blocks[0]['index'],
                        "end_block_index": window_blocks[-1]['index'],
                        "block_indices": [b['index'] for b in window_blocks],
                        "num_blocks": len(window_blocks),
                        "confidence": final_confidence,
                        "raw_score": raw_score,
                        "breakdown": breakdown_details,
                        "content_preview": normalize_text(final_window_content_original)[:300],
                        "is_ifrs_two_statement": score_result.get("is_ifrs_two_statement_confirmed", False)
                    }
                    if best_soci_candidate is None or final_confidence > best_soci_candidate["confidence"] or \
                       (abs(final_confidence - best_soci_candidate["confidence"]) < 0.01 and len(window_blocks) > best_soci_candidate["num_blocks"]):
                        best_soci_candidate = candidate_result
                        logger.debug(f"    Updated best_soci_candidate (Confidence: {final_confidence:.3f})")

                current_list_idx = window_last_list_idx + 1
            else: # no windo formed
                current_list_idx += 1

        if best_soci_candidate:
            logger.info(f"Returning best SoCI candidate found with confidence {best_soci_candidate['confidence']:.3f}")
            return best_soci_candidate

        logger.info("No SoCI section identified with sufficient confidence after checking all potential windows.")
        return None

    def display_results(self, soci_result: Optional[Dict[str, Any]], all_doc_blocks: Optional[List[Dict[str, Any]]] = None) -> None:
        super().display_results(classification_result=soci_result, all_doc_blocks=all_doc_blocks)
        if soci_result and logger.isEnabledFor(logging.DEBUG): # Specific debug details for SoCI
            logger.debug(f"  SoCI Specific: IFRS Two-Statement: {soci_result.get('is_ifrs_two_statement', False)}")
            if all_doc_blocks:
                logger.debug(f"\n  Identified SoCI content (first 3 blocks preview, normalized):")
                block_content_map = {block['index']: block['content'] for block in all_doc_blocks}
                for i, block_idx in enumerate(soci_result.get('block_indices', [])):
                    if i >= 3 and len(soci_result['block_indices']) > 3:
                        logger.debug(f"    ... and {len(soci_result['block_indices']) - 3} more blocks.")
                        break
                    content = block_content_map.get(block_idx, "Content not found.")
                    display_content = normalize_text(content)[:100] + "..."
                    logger.debug(f"    Block {block_idx}: {display_content}")
        elif not soci_result:
            logger.info("  Suggestions for SoCI detection:")
            logger.info("   - Adjust `confidence_threshold` or other classification parameters.")
            logger.info("   - Verify `soci_rules.json` for accuracy and completeness.")
