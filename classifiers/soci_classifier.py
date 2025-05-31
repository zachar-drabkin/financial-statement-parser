import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
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
            if isinstance(kw_list_data, list):
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
        found_items_texts = []
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

        relative_buffer_ratio = title_rules.get("title_relative_buffer_ratio", 0.5)
        absolute_buffer_min = title_rules.get("title_absolute_buffer_min", 10)
        absolute_buffer_max = title_rules.get("title_absolute_buffer_max", 50)

        suspicious_trailing_len = title_rules.get("title_suspicious_trailing_content_len_threshold", 25)
        suspicious_punctuation = tuple(title_rules.get("title_suspicious_trailing_punctuation", [".", ";"]))
        suspicious_word_count = title_rules.get("title_suspicious_trailing_word_count", 4)

        all_title_keywords_with_details = []
        title_definition_list_keys = ["ifrs_primary", "gaap_primary", "shared_ambiguous"]

        for key in title_definition_list_keys:
            category_list = title_rules.get(key)
            if isinstance(category_list, list):
                for item in category_list:
                    if isinstance(item, dict) and "text" in item:
                        all_title_keywords_with_details.append(item)
                    elif isinstance(item, dict):
                        logger.warning(f"SoCI Title Rule: Item in '{key}' is a dict but missing 'text' key: {item}")
            elif category_list is not None:
                 logger.warning(f"SoCI Title Rule: Expected list for '{key}', but got {type(category_list)}.")

        if not all_title_keywords_with_details:
            logger.debug(f"No valid title definition dictionaries found for checking '{normalized_text_upper_block[:50]}...'")
            return 0, None, None

        sorted_title_keywords = sorted(all_title_keywords_with_details, key=lambda x: len(x.get("text","")), reverse=True)

        for title_item in sorted_title_keywords:
            title_keyword_original = title_item.get("text", "")
            if not title_keyword_original: continue
            title_keyword_upper = title_keyword_original.upper()
            item_score = title_item.get("score", 0)
            item_type = title_item.get("type")

            pattern = r'\b' + re.escape(title_keyword_upper) + r'\b'
            match = re.search(pattern, normalized_text_upper_block)

            if match:
                dynamic_buffer = int(relative_buffer_ratio * len(title_keyword_upper)) + absolute_buffer_min
                dynamic_buffer = min(dynamic_buffer, absolute_buffer_max)

                if len(normalized_text_upper_block) < (match.end() + dynamic_buffer):
                    is_likely_just_title = True
                    content_after_title = normalized_text_upper_block[match.end():].strip()

                    if len(content_after_title) > suspicious_trailing_len and \
                       content_after_title.endswith(suspicious_punctuation) and \
                       len(content_after_title.split()) > suspicious_word_count:
                        is_likely_just_title = False
                        logger.debug(f"SoCI Title: Candidate '{title_keyword_original}' matched, but block has sentence-like trailing content: '{content_after_title[:50]}...'")

                    if is_likely_just_title:
                        if item_score > max_score:
                            max_score = item_score
                            best_title_text = title_keyword_original
                            best_title_type = item_type
        return max_score, best_title_text, best_title_type

    def _check_pl_keywords(self, normalized_text_upper: str) -> Tuple[int, List[str], Set[str]]:
        total_score = 0
        all_found_keywords_texts = set()
        pl_rules = self.rules.get("pl_keywords", {})
        critical_categories_defined = set(pl_rules.get("critical_item_categories", ["revenue", "net_income"]))
        found_critical_categories = set()
        logger.debug(f"VERBOSE DEBUG: _check_pl_keywords: Input text preview: '{normalized_text_upper[:100]}...'")
        logger.debug(f"VERBOSE DEBUG: _check_pl_keywords: Defined critical categories: {critical_categories_defined}")

        for category_name, keywords_list_defs in pl_rules.items():
            if category_name in ["critical_item_categories", "min_critical_categories_to_avoid_penalty",
                                 "graduated_penalty_factors", "critical_item_missing_penalty_divisor",
                                 "min_pl_score_for_critical_item_penalty"]:
                continue

            if isinstance(keywords_list_defs, list):
                cat_score, cat_kw_texts, _ = self._find_keywords(normalized_text_upper, keywords_list_defs)
                if cat_kw_texts:
                    total_score += cat_score
                    all_found_keywords_texts.update(cat_kw_texts)
                    if category_name in critical_categories_defined:
                        logger.debug(f"VERBOSE DEBUG: _check_pl_keywords: Category '{category_name}' IS CRITICAL and keywords were found. Adding to found_critical_categories.")
                        found_critical_categories.add(category_name)
            else:
                logger.warning(f"SoCI P&L Keywords: Expected list for '{category_name}', but got {type(keywords_list_defs)}.")

        logger.debug(f"VERBOSE DEBUG: _check_pl_keywords: Returning Total P&L Score: {total_score}, All Found P&L Keywords: {list(all_found_keywords_texts)}, Found Critical Categories: {found_critical_categories}")
        return total_score, list(all_found_keywords_texts), found_critical_categories

    def _check_oci_keywords(self, normalized_text_upper: str) -> Tuple[int, List[str]]:
        total_score = 0
        all_found_keywords_texts = set()
        oci_rules = self.rules.get("oci_keywords", {})
        # logger.debug(f"VERBOSE DEBUG: _check_oci_keywords: Input text preview: '{normalized_text_upper[:100]}...'")

        for category_name, keywords_list_defs in oci_rules.items():
            if category_name in ["min_oci_header_score_for_extension_absolute", "min_oci_header_score_for_extension_pnl_factor",
                                 "min_total_oci_score_for_extension_stop_absolute", "min_total_oci_score_for_extension_stop_pnl_factor",
                                 "min_oci_header_score_for_extension", "min_total_oci_score_for_extension_stop"]:
                continue

            if isinstance(keywords_list_defs, list):
                cat_score, cat_kw_texts, _ = self._find_keywords(normalized_text_upper, keywords_list_defs)
                if cat_kw_texts:
                    # logger.debug(f"VERBOSE DEBUG: _check_oci_keywords: Category '{category_name}' - Found keywords: {cat_kw_texts}, Score: {cat_score}")
                    total_score += cat_score
                    all_found_keywords_texts.update(cat_kw_texts)
        # logger.debug(f"VERBOSE DEBUG: _check_oci_keywords: Returning Total OCI Score: {total_score}, Found OCI Keywords: {list(all_found_keywords_texts)}")
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
                found_cues_texts.append(f"Date Pattern: {pattern_str.split(':', 1)[0]}")
                break

        year_pattern_str = cue_rules.get("year_pattern", r'\b(?:19|20)\d{2}\b')
        year_matches = re.findall(year_pattern_str, normalized_text_upper)
        if len(set(year_matches)) >= 2:
            score += cue_rules.get("comparative_year_score", 5)
            found_cues_texts.append(f"Comparative Years: {list(set(year_matches))}")
        elif year_matches:
            score += cue_rules.get("single_year_score", 2)
            found_cues_texts.append(f"Year Found: {list(set(year_matches))}")

        currency_score_gained_this_check = False
        curr_kw_score, curr_kws_texts, _ = self._find_keywords(normalized_text_upper, cue_rules.get("currency_keywords",[]), unique=False)
        if curr_kws_texts:
            score += curr_kw_score;
            found_cues_texts.extend(curr_kws_texts)
            currency_score_gained_this_check = True

        if not currency_score_gained_this_check:
            for iso_code in cue_rules.get("currency_iso_codes", []):
                if re.search(r'\b' + re.escape(iso_code) + r'\b', normalized_text_upper):
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
        title_rules = self.rules.get("soci_titles", {})
        relative_buffer_ratio = title_rules.get("title_relative_buffer_ratio", 0.5)
        absolute_buffer_min = title_rules.get("title_absolute_buffer_min", 10)
        absolute_buffer_max = title_rules.get("title_absolute_buffer_max", 50)
        section_header_buffer = term_rules.get("section_header_buffer_for_termination_check", 80)

        for title_text_normalized in self._normalized_termination_titles:
            dynamic_buffer = int(relative_buffer_ratio * len(title_text_normalized)) + absolute_buffer_min
            dynamic_buffer = min(dynamic_buffer, absolute_buffer_max)

            pattern = r'\b' + re.escape(title_text_normalized) + r'\b'
            match = re.search(pattern, normalized_content)
            if match:
                if len(normalized_content) < (match.end() + dynamic_buffer):
                    logger.debug(f"SoCI Term: Statement title '{title_text_normalized}' in block '{normalized_content[:100]}...'")
                    return True

        for section_header_text in term_rules.get("other_sections", []):
            normalized_header = normalize_text(section_header_text)
            dynamic_buffer_header = int(relative_buffer_ratio * len(normalized_header)) + absolute_buffer_min
            dynamic_buffer_header = min(dynamic_buffer_header, absolute_buffer_max)

            pattern = r'\b' + re.escape(normalized_header) + r'\b'
            match_header = re.search(pattern, normalized_content)
            if match_header:
                if len(normalized_content) < (match_header.end() + dynamic_buffer_header):
                    logger.debug(f"SoCI Term: Other section header '{normalized_header}' in block '{normalized_content[:100]}...'")
                    return True

        for note_pattern_regex in term_rules.get("note_section_patterns", []):
            if re.search(note_pattern_regex, block_content_original_case.strip(), re.IGNORECASE | re.MULTILINE):
                logger.debug(f"SoCI Term: Note section pattern '{note_pattern_regex}' in block '{normalized_content[:100]}...'")
                return True
        return False

    def _calculate_score(self, combined_text_original_case: str, first_block_index: int,
                         title_info: Optional[Tuple[int, str, str]] = None,
                         is_table_only_core: bool = False, **kwargs) -> Dict[str, Any]:
        logger.debug(f"VERBOSE DEBUG: _calculate_score: Called for first_block_index: {first_block_index}, is_table_only_core: {is_table_only_core}, title_info: {title_info}")
        logger.debug(f"VERBOSE DEBUG: _calculate_score: Combined text preview: '{combined_text_original_case[:200].replace(chr(10), ' ')}...'")

        if not combined_text_original_case.strip():
            logger.debug("VERBOSE DEBUG: _calculate_score: Combined text is empty, returning zero score.")
            return {"total": 0, "breakdown": {}, "best_title_text": None, "best_title_type": None,
                    "is_potential_ifrs_pl": False, "is_missing_critical_pl": True, "found_critical_pl_categories": []}

        normalized_combined_text = normalize_text(combined_text_original_case)
        breakdown: Dict[str, Any] = {}
        bonus_rules = self.rules.get("bonuses_penalties", {})
        pl_rules = self.rules.get("pl_keywords", {})

        title_score, best_title_text, best_title_type = 0, None, None
        if title_info:
            title_score, best_title_text, best_title_type = title_info
            logger.debug(f"VERBOSE DEBUG: _calculate_score: Using provided title_info: Score={title_score}, Text='{best_title_text}', Type='{best_title_type}'")
        elif not is_table_only_core:
            title_search_text = normalized_combined_text[:500]
            temp_title_score, temp_bt, temp_btt = self._check_soci_titles(title_search_text)
            if temp_title_score > 0:
                title_score, best_title_text, best_title_type = temp_title_score, temp_bt, temp_btt
            logger.debug(f"VERBOSE DEBUG: _calculate_score: Checked for title in content (not table-only start): Score={title_score}, Text='{best_title_text}', Type='{best_title_type}'")
        breakdown["title"] = title_score

        pl_score, found_pl_kws, found_critical_pl_categories = self._check_pl_keywords(normalized_combined_text)
        logger.debug(f"VERBOSE DEBUG: _calculate_score: Initial pl_score from _check_pl_keywords: {pl_score}, Found critical categories: {found_critical_pl_categories}")
        breakdown["found_pl_keywords"] = found_pl_kws
        breakdown["found_critical_pl_categories"] = list(found_critical_pl_categories)

        critical_categories_defined = set(pl_rules.get("critical_item_categories", ["revenue", "net_income"]))
        min_critical_to_avoid_penalty = pl_rules.get("min_critical_categories_to_avoid_penalty", len(critical_categories_defined))
        num_found_critical = len(found_critical_pl_categories)
        is_missing_critical_pl = num_found_critical < min_critical_to_avoid_penalty
        logger.debug(f"VERBOSE DEBUG: _calculate_score: num_found_critical={num_found_critical}, min_critical_to_avoid_penalty={min_critical_to_avoid_penalty}, is_missing_critical_pl={is_missing_critical_pl}")

        breakdown["num_critical_pl_categories_found"] = num_found_critical
        breakdown["num_critical_pl_categories_expected_min"] = min_critical_to_avoid_penalty

        original_pl_score_for_penalty_calc = pl_score
        if is_missing_critical_pl and original_pl_score_for_penalty_calc > pl_rules.get("min_pl_score_for_critical_item_penalty", 10):
            logger.debug(f"VERBOSE DEBUG: _calculate_score: Applying P&L penalty. Original P&L score: {original_pl_score_for_penalty_calc}")
            graduated_penalty_factors = pl_rules.get("graduated_penalty_factors", {})
            factor_str_key = str(num_found_critical)
            factor = graduated_penalty_factors.get(factor_str_key)
            logger.debug(f"VERBOSE DEBUG: _calculate_score: Graduated penalty factor for '{factor_str_key}' found items: {factor}")

            if factor is not None:
                 try:
                    factor = float(factor)
                    if 0.0 <= factor <= 1.0:
                        pl_score = original_pl_score_for_penalty_calc * factor
                        logger.debug(f"SoCI P&L: Applied graduated penalty. Found {num_found_critical} critical items. Factor: {factor}. PL score {original_pl_score_for_penalty_calc:.2f} -> {pl_score:.2f}")
                    else:
                        logger.warning(f"SoCI P&L: Invalid graduated_penalty_factor {factor} for {num_found_critical} items. Not applying.")
                 except ValueError:
                    logger.warning(f"SoCI P&L: Could not parse graduated_penalty_factor '{factor}' to float for key '{factor_str_key}'.")
            else:
                penalty_divisor = pl_rules.get("critical_item_missing_penalty_divisor", 3.0)
                if penalty_divisor > 0:
                    pl_score = max(0, original_pl_score_for_penalty_calc / penalty_divisor)
                logger.debug(f"SoCI P&L: Applied divisor penalty (divisor: {penalty_divisor}). PL score {original_pl_score_for_penalty_calc:.2f} -> {pl_score:.2f}")
            breakdown["pl_penalty_critical_items_missing"] = pl_score - original_pl_score_for_penalty_calc
        breakdown["p&l_content"] = pl_score
        logger.debug(f"VERBOSE DEBUG: _calculate_score: Final pl_score (after potential penalty): {pl_score}")

        # THIS IS THE LINE FROM THE TRACEBACK - Ensure _check_oci_keywords exists and is a method of this class
        oci_score, found_oci_kws = self._check_oci_keywords(normalized_combined_text)
        breakdown["oci_content"] = oci_score
        breakdown["found_oci_keywords"] = found_oci_kws

        structural_score, found_structural_cues = self._check_structural_cues(normalized_combined_text)
        breakdown["structural_cues"] = structural_score
        breakdown["found_structural_cues"] = found_structural_cues

        eps_dis_score, found_eps_dis_kws = self._check_eps_discontinued_ops(normalized_combined_text)
        breakdown["eps_discontinued_ops"] = eps_dis_score
        breakdown["found_eps_discontinued_ops_keywords"] = found_eps_dis_kws

        ifrs_two_statement_bonus = breakdown.get("ifrs_two_statement_bonus", 0)

        block_b_score = bonus_rules.get("block_bonus_score", 5)
        block_bonus = block_b_score if first_block_index < bonus_rules.get("block_bonus_max_index", 30) else 0
        breakdown["block_bonus"] = block_bonus

        combination_bonus = 0
        title_is_strong = title_score >= bonus_rules.get("min_title_score_for_combo", 5)
        pl_is_strong_for_combo = pl_score >= bonus_rules.get("min_pl_score_for_combo", 40)
        pl_combo_condition = pl_is_strong_for_combo and not is_missing_critical_pl
        logger.debug(f"VERBOSE DEBUG: _calculate_score: Combo bonus checks: title_is_strong={title_is_strong}, pl_is_strong_for_combo (post-penalty)={pl_is_strong_for_combo}, pl_combo_condition (includes !is_missing_critical_pl)={pl_combo_condition}")

        oci_expected = best_title_type and ("soci" in best_title_type or "oci" in best_title_type or "comprehensive" in best_title_type.lower())
        oci_is_present_or_not_needed = (oci_expected and oci_score >= bonus_rules.get("min_oci_score_if_expected_for_combo", 10)) or (not oci_expected)
        logger.debug(f"VERBOSE DEBUG: _calculate_score: Combo bonus checks: oci_expected={oci_expected}, oci_is_present_or_not_needed={oci_is_present_or_not_needed}")

        if (title_is_strong or is_table_only_core) and pl_combo_condition and oci_is_present_or_not_needed:
            combination_bonus = bonus_rules.get("combination_bonus_score", 10)
            logger.debug(f"VERBOSE DEBUG: _calculate_score: Combination bonus GRANTED: {combination_bonus}")
        else:
            logger.debug(f"VERBOSE DEBUG: _calculate_score: Combination bonus NOT granted.")
        breakdown["combination_bonus"] = combination_bonus

        total_score = sum(val for val in [title_score, pl_score, oci_score, structural_score, eps_dis_score,
                                          ifrs_two_statement_bonus, block_bonus, combination_bonus] if isinstance(val, (int, float)))
        logger.debug(f"VERBOSE DEBUG: _calculate_score: Individual scores for sum: Title={title_score}, P&L={pl_score}, OCI={oci_score}, Struct={structural_score}, EPS={eps_dis_score}, IFRSBonus={ifrs_two_statement_bonus}, BlockBonus={block_bonus}, ComboBonus={combination_bonus}")

        is_potential_ifrs_pl = False
        if best_title_type == "ifrs_pl" and \
           (original_pl_score_for_penalty_calc >= bonus_rules.get("min_pl_score_for_combo", 40) and not is_missing_critical_pl) and \
           oci_score < bonus_rules.get("min_oci_score_if_expected_for_combo", 10):
            is_potential_ifrs_pl = True
        logger.debug(f"VERBOSE DEBUG: _calculate_score: is_potential_ifrs_pl: {is_potential_ifrs_pl} (based on title_type='{best_title_type}', original P&L score {original_pl_score_for_penalty_calc}, !is_missing_critical_pl={not is_missing_critical_pl}, oci_score={oci_score})")

        logger.debug(f"VERBOSE DEBUG: _calculate_score: Final calculated total_score: {total_score}")
        result_dict = {
            "total": total_score, "breakdown": breakdown, "best_title_text": best_title_text,
            "best_title_type": best_title_type, "is_potential_ifrs_pl": is_potential_ifrs_pl,
            "is_missing_critical_pl": is_missing_critical_pl,
            "found_critical_pl_categories": list(found_critical_pl_categories)
        }
        logger.debug(f"VERBOSE DEBUG: _calculate_score: Returning result_dict: {result_dict}")
        return result_dict

    def _attempt_window_formation(self, doc_blocks: List[Dict[str, Any]],
                                  start_list_idx: int, max_blocks_in_window: int
                                 ) -> Tuple[Optional[List[Dict[str,Any]]], str, Optional[Tuple[int,str,str]], bool, int]:
        logger.debug(f"VERBOSE DEBUG: _attempt_window_formation: Called with start_list_idx={start_list_idx}, max_blocks_in_window={max_blocks_in_window}")
        start_block_candidate = doc_blocks[start_list_idx]
        current_window_blocks = []
        current_window_content_original = ""
        initial_title_info = None
        is_table_only_start = False
        window_last_list_idx = start_list_idx -1
        block_type_of_start = start_block_candidate.get('type')
        logger.debug(f"VERBOSE DEBUG: _attempt_window_formation: Start block candidate (doc_idx {start_block_candidate['index']}, list_idx {start_list_idx}) type: {block_type_of_start}, content preview: '{start_block_candidate['content'][:100].replace(chr(10), ' ')}...'")

        soci_title_rules = self.rules.get("soci_titles", {})

        if block_type_of_start == 'paragraph':
            normalized_start_content = normalize_text(start_block_candidate['content'])
            prelim_title_score, prelim_title_text, prelim_title_type = self._check_soci_titles(normalized_start_content)
            min_title_score_start = soci_title_rules.get("min_title_score_paragraph_start", 4)
            indicative_terms = soci_title_rules.get("indicative_terms_in_title_paragraph", [])
            has_indicative = any(re.search(r'\b' + term + r'\b', normalized_start_content, re.IGNORECASE) for term in indicative_terms)
            logger.debug(f"VERBOSE DEBUG: _attempt_window_formation (paragraph): Prelim title score={prelim_title_score}, has_indicative={has_indicative}")

            if prelim_title_score >= min_title_score_start or (prelim_title_score > 0 and has_indicative):
                logger.debug(f"SoCI Window: Potential title paragraph {start_block_candidate['index']} (Score: {prelim_title_score}, Title: '{prelim_title_text}')")
                current_window_blocks.append(start_block_candidate)
                current_window_content_original += start_block_candidate['content'] + "\n"
                initial_title_info = (prelim_title_score, prelim_title_text, prelim_title_type)
                window_last_list_idx = start_list_idx

                for k_follow in range(start_list_idx + 1, min(start_list_idx + 1 + max_blocks_in_window -1 , len(doc_blocks))):
                    block_to_add = doc_blocks[k_follow]
                    if self._is_termination_block(block_to_add['content'], block_to_add.get('type')):
                        logger.debug(f"SoCI Window (title-led): Sequence stopped by termination: Block {block_to_add['index']}")
                        break
                    if block_to_add.get('type') == 'table':
                        current_window_blocks.append(block_to_add)
                        current_window_content_original += block_to_add['content'] + "\n"
                        window_last_list_idx = k_follow
                    else:
                        logger.debug(f"SoCI Window (title-led): Table sequence stopped by non-table: Block {block_to_add['index']}")
                        break
            else:
                 logger.debug(f"SoCI Window: Skipping paragraph {start_block_candidate['index']} as start: low title score ({prelim_title_score}) or no indicative terms.")

        if not current_window_blocks and block_type_of_start == 'table':
            logger.debug(f"VERBOSE DEBUG: _attempt_window_formation: Trying table-only start for block doc_idx {start_block_candidate['index']}")
            is_table_only_start = True
            for k_follow in range(start_list_idx, min(start_list_idx + max_blocks_in_window, len(doc_blocks))):
                block_to_add = doc_blocks[k_follow]
                logger.debug(f"VERBOSE DEBUG: _attempt_window_formation (table-only): Considering block doc_idx {block_to_add['index']} (list_idx {k_follow}), type {block_to_add.get('type')}")
                if self._is_termination_block(block_to_add['content'], block_to_add.get('type')):
                    logger.debug(f"SoCI Window (table-only): Sequence stopped by termination: Block {block_to_add['index']}")
                    break
                if block_to_add.get('type') == 'table':
                    logger.debug(f"VERBOSE DEBUG: _attempt_window_formation (table-only): Adding table block doc_idx {block_to_add['index']}")
                    current_window_blocks.append(block_to_add)
                    current_window_content_original += block_to_add['content'] + "\n"
                    window_last_list_idx = k_follow
                else:
                    logger.debug(f"SoCI Window (table-only): Sequence stopped by non-table: Block {block_to_add['index']}")
                    break

        if not current_window_blocks:
            window_last_list_idx = start_list_idx -1
            logger.debug(f"VERBOSE DEBUG: _attempt_window_formation: No window blocks formed. Returning None for blocks.")
        else:
            logger.debug(f"VERBOSE DEBUG: _attempt_window_formation: Formed window with {len(current_window_blocks)} blocks. Last list_idx in window: {window_last_list_idx}. is_table_only_start: {is_table_only_start}")

        return current_window_blocks if current_window_blocks else None, current_window_content_original.strip(), initial_title_info, is_table_only_start, window_last_list_idx

    def _handle_ifrs_two_statement(self, current_window_blocks: List[Dict[str,Any]],
                                   current_window_content_original: str,
                                   score_result_so_far: Dict[str,Any],
                                   window_last_list_idx: int,
                                   doc_blocks: List[Dict[str,Any]]
                                  ) -> Tuple[List[Dict[str,Any]], str, Dict[str,Any], int]:
        logger.debug(f"VERBOSE DEBUG: _handle_ifrs_two_statement called. Current window ends list_idx {window_last_list_idx}. Score so far: {score_result_so_far.get('total')}")
        bonus_rules = self.rules.get("bonuses_penalties", {})
        oci_rules = self.rules.get("oci_keywords", {})
        general_cfg = self.rules.get("general_config", {})

        oci_window_blocks_extension = []
        oci_window_content_extension_original = ""
        idx_after_pl_part_in_list = window_last_list_idx + 1

        pnl_core_score = score_result_so_far.get("total", 0) - \
                         score_result_so_far.get("breakdown",{}).get("block_bonus",0) - \
                         score_result_so_far.get("breakdown",{}).get("combination_bonus",0) - \
                         score_result_so_far.get("breakdown",{}).get("ifrs_two_statement_bonus",0)

        min_header_abs = oci_rules.get("min_oci_header_score_for_extension_absolute", 5)
        min_header_pnl_factor = oci_rules.get("min_oci_header_score_for_extension_pnl_factor", 0.05)
        dynamic_min_header_score = max(min_header_abs, pnl_core_score * min_header_pnl_factor)
        logger.debug(f"SoCI IFRS: Dynamic min OCI header score for extension: {dynamic_min_header_score:.2f} (P&L score influence: {pnl_core_score:.2f})")

        min_total_stop_abs = oci_rules.get("min_total_oci_score_for_extension_stop_absolute", 15)
        min_total_stop_pnl_factor = oci_rules.get("min_total_oci_score_for_extension_stop_pnl_factor", 0.10)
        dynamic_min_total_oci_score_stop = max(min_total_stop_abs, pnl_core_score * min_total_stop_pnl_factor)
        logger.debug(f"SoCI IFRS: Dynamic min total OCI score to stop extension: {dynamic_min_total_oci_score_stop:.2f}")

        max_consecutive_non_oci = general_cfg.get("oci_extension_max_consecutive_non_contributing_blocks", 1)
        max_total_scan = general_cfg.get("oci_extension_max_total_blocks_to_scan", 8)
        consecutive_non_oci_count = 0

        for k_oci in range(idx_after_pl_part_in_list, min(idx_after_pl_part_in_list + max_total_scan, len(doc_blocks))):
            oci_candidate_block = doc_blocks[k_oci]
            if self._is_termination_block(oci_candidate_block['content'], oci_candidate_block.get('type')):
                logger.debug(f"SoCI IFRS: OCI search stopped by termination: Block {oci_candidate_block['index']}")
                break

            normalized_oci_candidate_content = normalize_text(oci_candidate_block['content'])
            if not normalized_oci_candidate_content.strip():
                consecutive_non_oci_count +=1
                if consecutive_non_oci_count >= max_consecutive_non_oci and oci_window_blocks_extension:
                    logger.debug(f"SoCI IFRS: Stopping OCI extension due to {consecutive_non_oci_count} consecutive non-contributing/empty blocks.")
                    break
                continue

            oci_header_score_current_block, _ = self._check_oci_keywords(normalized_oci_candidate_content)
            added_this_block = False
            if oci_header_score_current_block >= dynamic_min_header_score or \
               (len(oci_window_blocks_extension) > 0 and oci_candidate_block.get('type') == 'table' and oci_header_score_current_block > 0):
                oci_window_blocks_extension.append(oci_candidate_block)
                oci_window_content_extension_original += oci_candidate_block['content'] + "\n"
                logger.debug(f"SoCI IFRS: Added block {oci_candidate_block['index']} to OCI extension (block OCI score: {oci_header_score_current_block:.2f}).")
                consecutive_non_oci_count = 0
                added_this_block = True

                temp_oci_score_ext_total, _ = self._check_oci_keywords(normalize_text(oci_window_content_extension_original))
                if temp_oci_score_ext_total >= dynamic_min_total_oci_score_stop:
                    logger.debug(f"SoCI IFRS: Sufficient OCI content found in extension (total OCI score {temp_oci_score_ext_total:.2f} >= {dynamic_min_total_oci_score_stop:.2f}).")
                    break

            if not added_this_block:
                consecutive_non_oci_count += 1
                logger.debug(f"SoCI IFRS: Block {oci_candidate_block['index']} (type: {oci_candidate_block.get('type')}) does not contribute to OCI (block OCI score: {oci_header_score_current_block:.2f}). Non-OCI strike {consecutive_non_oci_count}.")
                if consecutive_non_oci_count >= max_consecutive_non_oci:
                    if oci_window_blocks_extension:
                         logger.debug(f"SoCI IFRS: Stopping OCI extension due to {consecutive_non_oci_count} consecutive non-contributing blocks.")
                    else:
                         logger.debug(f"SoCI IFRS: No plausible OCI extension started after {consecutive_non_oci_count} non-contributing blocks.")
                    break

        if oci_window_blocks_extension:
            logger.info(f"SoCI IFRS: OCI part identified ({len(oci_window_blocks_extension)} blocks). Merging with P&L part.")
            updated_window_blocks = current_window_blocks + oci_window_blocks_extension
            updated_window_content = current_window_content_original.strip() + "\n" + oci_window_content_extension_original.strip()

            title_info_for_rescore = (
                score_result_so_far.get("breakdown", {}).get("title", 0),
                score_result_so_far.get("best_title_text"),
                score_result_so_far.get("best_title_type")
            )

            updated_score_result = self._calculate_score(
                updated_window_content,
                updated_window_blocks[0]['index'],
                title_info=title_info_for_rescore,
                is_table_only_core=score_result_so_far.get("is_table_only_core_flag_for_rescore", False)
            )
            updated_score_result["breakdown"]["ifrs_two_statement_bonus"] = bonus_rules.get("ifrs_two_statement_bonus", 15)
            updated_score_result["total"] += bonus_rules.get("ifrs_two_statement_bonus", 15)
            updated_score_result["is_ifrs_two_statement_confirmed"] = True
            updated_window_last_list_idx = window_last_list_idx + len(oci_window_blocks_extension)
            return updated_window_blocks, updated_window_content, updated_score_result, updated_window_last_list_idx

        return current_window_blocks, current_window_content_original, score_result_so_far, window_last_list_idx

    def classify(self, doc_blocks: List[Dict[str, Any]],
                 confidence_threshold: float = 0.50,
                 max_start_block_index_to_check: int = 500,
                 max_blocks_in_soci_window: int = 30,
                 **kwargs) -> Optional[Dict[str, Any]]:
        logger.debug("VERBOSE DEBUG: SoCIClassifier.classify START")

        best_soci_candidate = None
        overall_start_list_idx = kwargs.get("start_block_index_in_list", 0)
        current_list_idx = overall_start_list_idx

        bonus_rules = self.rules.get("bonuses_penalties", {})
        pl_rules = self.rules.get("pl_keywords", {})

        max_doc_block_idx_overall = doc_blocks[-1]['index'] if doc_blocks else 0
        effective_max_start_doc_idx = min(max_start_block_index_to_check, max_doc_block_idx_overall)
        logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: overall_start_list_idx={overall_start_list_idx}, effective_max_start_doc_idx={effective_max_start_doc_idx}")

        while current_list_idx < len(doc_blocks):
            iteration_start_list_idx = current_list_idx
            logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: Top of WHILE loop. current_list_idx={current_list_idx}, iteration_start_list_idx={iteration_start_list_idx}")

            start_block_candidate_for_iteration = doc_blocks[current_list_idx]
            logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: Processing start_block_candidate_for_iteration (doc_idx {start_block_candidate_for_iteration['index']})")

            if start_block_candidate_for_iteration['index'] > effective_max_start_doc_idx:
                logger.debug(f"SoCI Classify: Stopping search. Block doc index {start_block_candidate_for_iteration['index']} > {effective_max_start_doc_idx}.")
                break

            normalized_start_content_for_term_check = normalize_text(start_block_candidate_for_iteration['content'])
            is_potential_soci_title_here = self._check_soci_titles(normalized_start_content_for_term_check)[0] > 0
            logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: Block doc_idx {start_block_candidate_for_iteration['index']}: is_potential_soci_title_here={is_potential_soci_title_here}")

            if self._is_termination_block(start_block_candidate_for_iteration['content'], start_block_candidate_for_iteration.get('type')) and \
               not is_potential_soci_title_here:
                logger.debug(f"SoCI Classify: Skipping block {start_block_candidate_for_iteration['index']} as start: it's a termination signal and not an SoCI title itself.")
                current_list_idx += 1
                continue

            logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: Calling _attempt_window_formation for list_idx {current_list_idx}")
            window_blocks, window_content_str, initial_title_info, is_table_only_start, current_window_last_list_idx = \
                self._attempt_window_formation(doc_blocks, current_list_idx, max_blocks_in_soci_window)

            logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: _attempt_window_formation returned: window_blocks is {'None' if not window_blocks else f'{len(window_blocks)} blocks'}, initial_title_info={initial_title_info}, is_table_only_start={is_table_only_start}, current_window_last_list_idx={current_window_last_list_idx}")

            if window_blocks:
                logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: Window formed. Number of blocks: {len(window_blocks)}. First block in window doc_idx: {window_blocks[0]['index']}.")
                final_window_content_original_str = window_content_str
                first_block_in_window_doc_idx = window_blocks[0]['index']

                logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: Calling _calculate_score for window starting doc_idx {first_block_in_window_doc_idx}")
                score_result = self._calculate_score(
                    final_window_content_original_str,
                    first_block_in_window_doc_idx,
                    title_info=initial_title_info,
                    is_table_only_core=is_table_only_start
                )
                score_result["is_table_only_core_flag_for_rescore"] = is_table_only_start
                logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: _calculate_score returned: {score_result}")

                min_pl_score_ifrs_check = bonus_rules.get("min_initial_pl_score_for_ifrs_two_statement_check", 40)
                pnl_plus_title_for_ifrs_check = score_result.get("breakdown",{}).get("p&l_content",0) + score_result.get("breakdown",{}).get("title",0)
                logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: IFRS check: is_potential_ifrs_pl={score_result.get('is_potential_ifrs_pl')}, pnl_plus_title_for_ifrs_check={pnl_plus_title_for_ifrs_check}, min_pl_score_ifrs_check={min_pl_score_ifrs_check}")

                if score_result.get("is_potential_ifrs_pl", False) and pnl_plus_title_for_ifrs_check > min_pl_score_ifrs_check :
                    logger.info(f"SoCI Classify: Potential IFRS P&L. Initial Combined P&L+Title Score: {pnl_plus_title_for_ifrs_check}. Checking for OCI part.")
                    window_blocks, final_window_content_original_str, score_result, current_window_last_list_idx = \
                        self._handle_ifrs_two_statement(window_blocks, final_window_content_original_str, score_result, current_window_last_list_idx, doc_blocks)
                    logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: After _handle_ifrs_two_statement: score_result={score_result}, new current_window_last_list_idx={current_window_last_list_idx}")

                raw_score = score_result["total"]
                breakdown_details = score_result["breakdown"]
                normalized_display_title = score_result.get("best_title_text") or f"{self.section_name} (Inferred)"

                max_exemplar_score = float(self.max_score_exemplar) if self.max_score_exemplar else 1.0
                if max_exemplar_score == 0: max_exemplar_score = 100

                capped_score = min(raw_score, max_exemplar_score)
                final_confidence = (capped_score / (max_exemplar_score + 1e-9))
                logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: Raw score={raw_score}, Capped score={capped_score}, Max exemplar={max_exemplar_score}, Initial confidence={final_confidence:.4f}")

                original_pl_score_from_breakdown = breakdown_details.get("p&l_content", 0) - breakdown_details.get("pl_penalty_critical_items_missing", 0)
                min_pl_score_for_cap_check = pl_rules.get("min_pl_score_for_critical_item_penalty",10)
                logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: Confidence cap check: is_missing_critical_pl={score_result.get('is_missing_critical_pl')}, original_pl_score_from_breakdown={original_pl_score_from_breakdown}, min_pl_score_for_cap_check={min_pl_score_for_cap_check}")

                if score_result.get("is_missing_critical_pl", False) and original_pl_score_from_breakdown > min_pl_score_for_cap_check:
                     confidence_cap_value = bonus_rules.get("confidence_cap_if_critical_pl_missing", 0.25)
                     final_confidence = min(final_confidence, confidence_cap_value)
                     logger.warning(f"SoCI Window (Blocks starting {window_blocks[0]['index']}): Critical P&L items missing or below threshold. Confidence capped from {capped_score / (max_exemplar_score + 1e-9):.4f} to {final_confidence:.4f} (cap value: {confidence_cap_value}).")

                logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: Final confidence after potential cap: {final_confidence:.4f}")

                window_indices_str = str([b['index'] for b in window_blocks])
                logger.debug(f"SoCI EVALUATING WINDOW: Blocks Doc Indices {window_indices_str} (List idx {iteration_start_list_idx}-{current_window_last_list_idx})")
                logger.debug(f"  Raw Score: {raw_score:.2f}, Confidence: {final_confidence:.4f}, Title: '{score_result.get('best_title_text')}' ({score_result.get('best_title_type')})")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  Breakdown: {breakdown_details}")
                    logger.debug(f"  Is Missing Critical P&L: {score_result.get('is_missing_critical_pl')}, Found Critical: {score_result.get('found_critical_pl_categories')}")

                if final_confidence >= confidence_threshold:
                    logger.info(f"  >>> SoCI SECTION CANDIDATE MEETS THRESHOLD (Window {window_indices_str}), Conf: {final_confidence:.4f}")
                    candidate_result = {
                        "section_name": self.section_name,
                        "normalized_section_name": normalized_display_title,
                        "start_block_index": window_blocks[0]['index'],
                        "end_block_index": window_blocks[-1]['index'],
                        "block_indices": [b['index'] for b in window_blocks],
                        "num_blocks": len(window_blocks),
                        "confidence": round(final_confidence,4),
                        "raw_score": raw_score,
                        "breakdown": breakdown_details,
                        "content_preview": normalize_text(final_window_content_original_str)[:300],
                        "is_ifrs_two_statement": score_result.get("is_ifrs_two_statement_confirmed", False),
                        "is_missing_critical_pl": score_result.get("is_missing_critical_pl", True),
                        "found_critical_pl_categories": score_result.get("found_critical_pl_categories", [])
                    }
                    if best_soci_candidate is None or \
                       final_confidence > best_soci_candidate["confidence"] or \
                       (abs(final_confidence - best_soci_candidate["confidence"]) < 0.01 and len(window_blocks) > best_soci_candidate["num_blocks"]):
                        best_soci_candidate = candidate_result
                        logger.info(f"    Updated best_soci_candidate (Confidence: {final_confidence:.4f}, Blocks: {len(window_blocks)})")
                else:
                    logger.debug(f"VERBOSE DEBUG: SoCIClassifier.classify: Candidate with conf {final_confidence:.4f} DOES NOT MEET threshold {confidence_threshold}")

                current_list_idx = current_window_last_list_idx + 1
            else:
                current_list_idx += 1
                if current_list_idx == iteration_start_list_idx + 1:
                     logger.debug(f"SoCI Classify: No window formed starting at list_idx {iteration_start_list_idx}. Moving to next block ({current_list_idx}).")
                elif current_list_idx <= iteration_start_list_idx :
                     logger.error(f"SoCI Classify: list_idx did not advance as expected. Forcing increment. Before: {iteration_start_list_idx}, After attempt: {current_list_idx}")
                     current_list_idx = iteration_start_list_idx + 1

        if best_soci_candidate:
            logger.info(f"Returning best SoCI candidate found: Conf {best_soci_candidate['confidence']:.4f}, Blocks {best_soci_candidate['start_block_index']}-{best_soci_candidate['end_block_index']}")
            return best_soci_candidate

        logger.info("No SoCI section identified with sufficient confidence after checking all potential windows.")
        logger.debug("VERBOSE DEBUG: SoCIClassifier.classify END - No candidate found or best_soci_candidate is None.")
        return None

    def display_results(self, soci_result: Optional[Dict[str, Any]], all_doc_blocks: Optional[List[Dict[str, Any]]] = None) -> None:
        super().display_results(classification_result=soci_result, all_doc_blocks=all_doc_blocks)
        if soci_result and logger.isEnabledFor(logging.INFO):
            logger.info(f"  SoCI Specifics: IFRS Two-Statement: {soci_result.get('is_ifrs_two_statement', False)}")
            logger.info(f"  SoCI Specifics: Missing Critical P&L Items: {soci_result.get('is_missing_critical_pl', 'N/A')}")
            logger.info(f"  SoCI Specifics: Found Critical P&L Categories: {soci_result.get('found_critical_pl_categories', [])}")
            if logger.isEnabledFor(logging.DEBUG) and all_doc_blocks:
                logger.debug(f"\n  Identified SoCI content (first 3 blocks preview, normalized):")
                block_content_map = {block['index']: block['content'] for block in all_doc_blocks}
                for i, block_idx in enumerate(soci_result.get('block_indices', [])):
                    if i >= 3 and len(soci_result['block_indices']) > 3:
                        logger.debug(f"    ... and {len(soci_result['block_indices']) - 3} more blocks.")
                        break
                    content = block_content_map.get(block_idx, "Content not found.")
                    display_content = normalize_text(content)[:100].replace('\n', ' ') + "..."
                    logger.debug(f"    Block {block_idx} (Type: {next((b.get('type') for b in all_doc_blocks if b['index'] == block_idx),'N/A')}): {display_content}")
        elif not soci_result:
            logger.info("  Suggestions for SoCI detection:")
            logger.info("   - Review logs for window formation, scoring, and termination decisions.")
            logger.info("   - Adjust `confidence_threshold` or other classification parameters in `soci_rules.json`.")
            logger.info("   - Verify `soci_rules.json` for accuracy and completeness of keywords, critical items, and scoring logic.")
