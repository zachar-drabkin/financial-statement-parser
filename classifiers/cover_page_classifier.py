import re
from typing import Dict, List, Optional, Tuple, Any
import logging
from utils.text_utils import normalize_text, find_exact_phrases, find_regex_patterns
from classifiers.base_classifier import BaseClassifier

logger = logging.getLogger(__name__)

# --- Constants ---
SECTION_NAME_COVER_PAGE = "Cover Page"
YEAR_PATTERN_REGEX = r'\b(19|20)\d{2}\b' # Standard year pattern

class CoverPageClassifier(BaseClassifier):
    """
    Rule-based Classifier for identifying Cover Page Sections in Financial Statements.
    Enhanced with more robust heuristics for currency, bonuses, and indicator strength.
    """

    def __init__(self, rules_file_path="rules/cover_page_rules.json"):
        super().__init__(rules_file_path=rules_file_path)

    def _check_title_phrases(self, text_upper_normalized: str) -> int:
        rule_set = self.rules.get("title_phrases", {})
        phrases = rule_set.get("keywords", [])
        score_value = rule_set.get("score", 0)

        found_phrases = find_exact_phrases(text_upper_normalized, phrases)
        if found_phrases:
            return score_value
        return 0

    def _check_date_phrases(self, text_upper_normalized: str) -> int:
        rule_set = self.rules.get("date_phrases", {})
        keywords = rule_set.get("keywords", [])
        patterns = rule_set.get("patterns", [])
        score_value = rule_set.get("score", 0)

        if find_exact_phrases(text_upper_normalized, keywords):
            return score_value
        if find_regex_patterns(text_upper_normalized, patterns):
            return score_value
        return 0

    def _check_currency_phrases(self, text_upper_normalized: str) -> int:
        score = 0
        rule_set = self.rules.get("currency_phrases", {})
        keywords = rule_set.get("keywords", [])
        iso_codes = rule_set.get("iso_codes", [])
        symbols = rule_set.get("symbols", [])
        max_score_for_type = rule_set.get("score", 0)
        single_indicator_score = rule_set.get("single_indicator_score", 1)

        # Primary check: Full currency phrases like "EXPRESSED IN USD"
        if find_exact_phrases(text_upper_normalized, keywords):
            return max_score_for_type

        # Secondary check: ISO codes (e.g., "USD")
        # find_exact_phrases usually implies word boundaries, good for ISO codes
        if find_exact_phrases(text_upper_normalized, iso_codes):
            score = max(score, single_indicator_score)

        # Tertiary check: Currency symbols (e.g., "$", "€") with more context
        # We look for symbols adjacent to numbers to reduce false positives.
        found_contextual_symbol = False
        for sym_char in symbols:
            # Pattern: symbol then optional space then digit OR digit then optional space then symbol
            contextual_symbol_pattern = r'(?:{}\s*\d|\d\s*{})'.format(re.escape(sym_char), re.escape(sym_char))
            if re.search(contextual_symbol_pattern, text_upper_normalized):
                score = max(score, single_indicator_score) # Contextual symbol grants the single_indicator_score
                found_contextual_symbol = True
                break

        # Optional: Fallback for symbols without strong context, with a lower score,
        # if no other currency indicators have already met `single_indicator_score`.
        # This could be controlled by a new rule: "allow_contextless_symbols_partial_score": true/false
        if not found_contextual_symbol and score < single_indicator_score:
             allow_contextless_fallback = self.rules.get("allow_contextless_symbols_partial_score", False) # New potential rule
             if allow_contextless_fallback:
                contextless_symbol_score_factor = self.rules.get("contextless_symbol_score_factor", 0.5) # New potential rule
                for sym_char in symbols:
                    if sym_char in text_upper_normalized: # Original weaker check
                        score = max(score, int(single_indicator_score * contextless_symbol_score_factor))
                        break
        return score

    def _check_company_indicators(self, text_upper_normalized: str) -> int:
        rule_set = self.rules.get("company_indicators", {})
        suffixes = rule_set.get("suffixes", []) # E.g., "LTD", "INC."
        score_value = rule_set.get("score", 0)

        # Robustness: find_exact_phrases should handle variations like "LTD." vs "LTD"
        # If not, rules might need to list common variations, or a more advanced matching utility is needed.
        if find_exact_phrases(text_upper_normalized, suffixes):
            return score_value
        return 0

    def _check_year_pattern(self, text_upper_normalized: str) -> int:
        # This score is for the mere presence of a year, which is common on cover pages.
        score_value = self.rules.get("year_pattern_score", 0) # e.g., 1
        if re.search(YEAR_PATTERN_REGEX, text_upper_normalized):
            return score_value
        return 0

    def _calculate_score(self, combined_text: str, first_block_index: int) -> Dict[str, Any]:
        """
        Calculate raw score for a combined text of a window, with detailed breakdown.
        This method is the implementation for the abstract _calculate_score in BaseClassifier.
        Enhanced with graduated block bonus and combination bonus based on stronger indicators.
        """
        if not combined_text.strip():
            return {"total": 0, "breakdown": {}}

        text_upper_normalized = normalize_text(combined_text)

        title_score = self._check_title_phrases(text_upper_normalized)
        date_score = self._check_date_phrases(text_upper_normalized)
        currency_score = self._check_currency_phrases(text_upper_normalized)
        company_score = self._check_company_indicators(text_upper_normalized)
        year_score = self._check_year_pattern(text_upper_normalized)

        # --- Graduated Block Bonus ---
        # Rewards sections found very early in the document.
        block_bonus = 0
        block_bonus_val = self.rules.get("block_bonus_score", 1) # Max potential bonus score
        # Max index for full bonus (e.g., first 5 blocks)
        full_bonus_max_idx = self.rules.get("block_bonus_full_score_max_index", 5) # New rule, defaults to 5
        # Max index for partial bonus (e.g., up to block 10)
        partial_bonus_max_idx = self.rules.get("block_bonus_max_index", 10) # Existing rule, default 10
        partial_bonus_factor = self.rules.get("block_bonus_partial_factor", 0.5) # New rule, score is val * factor

        if first_block_index < full_bonus_max_idx:
            block_bonus = block_bonus_val
        elif first_block_index < partial_bonus_max_idx:
            block_bonus = block_bonus_val * partial_bonus_factor

        # --- Combination Bonus for Multiple Strong Indicators ---
        # Rewards the co-occurrence of several strong signals.
        combination_bonus = 0
        combo_bonus_val = self.rules.get("combination_bonus_score", 1)
        min_strong_indicators_for_combo = self.rules.get("combination_bonus_min_strong_indicators", 3)
        min_score_factor_for_strong = self.rules.get("min_score_factor_for_strong_indicator", 0.5)
        title_base_score = self.rules.get("title_phrases", {}).get("score", 0) or 1
        date_base_score = self.rules.get("date_phrases", {}).get("score", 0) or 1
        currency_main_base_score = self.rules.get("currency_phrases", {}).get("score", 0) or 1
        currency_single_base_score = self.rules.get("currency_phrases", {}).get("single_indicator_score", 0) or 1
        company_base_score = self.rules.get("company_indicators", {}).get("score", 0) or 1
        year_base_score = self.rules.get("year_pattern_score", 0) or 1

        is_strong_title = title_score >= (title_base_score * min_score_factor_for_strong)
        is_strong_date = date_score >= (date_base_score * min_score_factor_for_strong)
        is_strong_currency = currency_score >= (currency_main_base_score * min_score_factor_for_strong) or \
                             (currency_score >= currency_single_base_score and currency_single_base_score > 0)
        is_strong_company = company_score >= (company_base_score * min_score_factor_for_strong)
        is_strong_year = year_score >= (year_base_score * min_score_factor_for_strong)

        num_strong_found_indicators = sum([
            1 if is_strong_title else 0, 1 if is_strong_date else 0,
            1 if is_strong_currency else 0, 1 if is_strong_company else 0,
            1 if is_strong_year else 0
        ])

        if num_strong_found_indicators >= min_strong_indicators_for_combo:
            combination_bonus = combo_bonus_val

        total_score = (title_score + date_score + currency_score +
                       company_score + year_score + block_bonus + combination_bonus)

        breakdown = {
            "title": title_score, "date": date_score, "currency": currency_score,
            "company": company_score, "year": year_score, "block_bonus": block_bonus,
            "combination_bonus": combination_bonus,
            "num_strong_indicators": num_strong_found_indicators # For debugging/analysis
        }
        return {"total": total_score, "breakdown": breakdown}

    def _expand_window(self, doc_blocks: List[Dict[str, Any]], start_list_idx: int) -> Tuple[List[Dict[str, Any]], str]:
        """
        Expands a window of sequential paragraph blocks from a starting index.
        Returns the list of blocks in the window and their combined content.
        (No changes to this method's core logic, as it's generic windowing)
        """
        current_window_blocks = []
        current_window_content = ""
        for j in range(start_list_idx, len(doc_blocks)):
            current_block_in_expansion = doc_blocks[j]

            if current_block_in_expansion.get('type') != 'paragraph':
                if current_window_blocks:
                    logger.debug(f"Window ending at block {current_window_blocks[-1]['index']}: "
                                 f"Next block {current_block_in_expansion['index']} is not a paragraph "
                                 f"(type: {current_block_in_expansion.get('type')}).")
                break

            current_window_blocks.append(current_block_in_expansion)
            current_window_content += current_block_in_expansion['content'] + "\n"

            logger.debug(f"Window expanded to include block {current_block_in_expansion['index']} "
                         f"(type: {current_block_in_expansion.get('type')}). "
                         f"Window indices: {[b['index'] for b in current_window_blocks]}.")
        return current_window_blocks, current_window_content

    def _evaluate_window(self,
                         current_window_blocks: List[Dict[str, Any]],
                         current_window_content: str,
                         confidence_threshold: float
                         ) -> Optional[Dict[str, Any]]:
        """
        Evaluates a formed window, calculates its score and confidence,
        and returns a formatted result if it meets the threshold.
        (No changes to this method's core logic beyond using the updated _calculate_score)
        """
        if not current_window_blocks:
            return None

        final_window_content = current_window_content.strip()
        first_block_in_window_idx = current_window_blocks[0]['index']

        score_result = self._calculate_score(final_window_content, first_block_in_window_idx)
        raw_score = score_result["total"]
        breakdown = score_result["breakdown"]

        capped_score = min(raw_score, self.max_score_exemplar)
        final_confidence = (capped_score / (self.max_score_exemplar + 1e-9))

        window_indices_str = str([b['index'] for b in current_window_blocks])
        logger.debug(f"  EVALUATING WINDOW: Blocks {window_indices_str}, Start Index: {first_block_in_window_idx}\n"
                     f"  Combined Content: '{final_window_content[:150].replace(chr(10), ' ')}...'\n"
                     f"  Raw Score: {raw_score}, Confidence: {final_confidence:.3f}, Breakdown: {breakdown}")

        if final_confidence >= confidence_threshold:
            logger.info(f"  >>> COVER PAGE IDENTIFIED (Window): Blocks {window_indices_str}, "
                        f"Confidence: {final_confidence:.3f} >= Threshold: {confidence_threshold}")
            return {
                "section_name": SECTION_NAME_COVER_PAGE,
                "start_block_index": current_window_blocks[0]['index'],
                "end_block_index": current_window_blocks[-1]['index'],
                "block_indices": [b['index'] for b in current_window_blocks],
                "num_blocks": len(current_window_blocks),
                "confidence": final_confidence,
                "raw_score": raw_score,
                "breakdown": breakdown,
                "content": final_window_content,
                "details": {
                    "confidence": final_confidence,
                    "raw_score": raw_score,
                    "breakdown": breakdown,
                    "content_preview": final_window_content[:200].strip()
                }
            }
        return None

    def classify(self, doc_blocks: List[Dict[str, Any]],
                 confidence_threshold: float = 0.6, #
                 max_start_block_index_to_check: int = 500) -> Optional[Dict[str, Any]]:
        """
        Identify Cover Page Section from document blocks using a sliding window.
        Implements the abstract 'classify' method from BaseClassifier.
        (No changes to this method's core logic beyond using updated helper methods)
        """
        for i in range(len(doc_blocks)):
            start_block_candidate = doc_blocks[i]

            if start_block_candidate['index'] >= max_start_block_index_to_check:
                logger.debug(f"Stopping search: Start block index {start_block_candidate['index']} "
                             f"is >= max_start_block_index_to_check ({max_start_block_index_to_check}).")
                break


            if start_block_candidate.get('type') != 'paragraph':
                logger.debug(f"Skipping block {start_block_candidate['index']} as start of window: "
                             f"not a paragraph (type: {start_block_candidate.get('type')}).")
                continue

            current_window_blocks, current_window_content = self._expand_window(doc_blocks, i)

            result = self._evaluate_window(current_window_blocks, current_window_content, confidence_threshold)
            if result:
                return result

            if not current_window_blocks and start_block_candidate.get('type') == 'paragraph':
                 logger.debug(f"No valid window formed starting with paragraph block {start_block_candidate['index']}.")

        logger.info("No cover page identified with sufficient confidence after checking all potential windows.")
        return None

    def display_results(self, cover_page_result: Optional[Dict[str, Any]], all_doc_blocks: Optional[List[Dict[str, Any]]] = None) -> None:

        super().display_results(classification_result=cover_page_result, all_doc_blocks=all_doc_blocks)

        if cover_page_result and logger.isEnabledFor(logging.DEBUG):

            if all_doc_blocks:
                logger.debug(f"\n  Identified cover page content (block by block):")
                block_content_map = {block['index']: block['content'] for block in all_doc_blocks}
                for block_idx in cover_page_result.get('block_indices', []):
                    content = block_content_map.get(block_idx, "Content not found for this block index.")
                    display_content = content.replace('\n', ' ').replace('\r', ' ').strip()
                    if len(display_content) > 100:
                        display_content = display_content[:97] + "..."
                    logger.debug(f"    Block {block_idx}: {display_content}")
        elif not cover_page_result: # if not found
            logger.info("  Suggestions for Cover Page detection:")
            logger.info("   - Adjust `confidence_threshold` (e.g., lower it).")
            logger.info("   - Verify `max_start_block_index_to_check`.")
            logger.info("   - Ensure `DocxParser` correctly extracts paragraph blocks at the document start.")
            logger.info("   - Check `cover_page_rules.json` for comprehensive keywords and appropriate scores.")
            logger.info("   - Consider adjusting `max_score_exemplar` in rules if scoring potential has changed.")

    def calculate_confidence_score_single_block(self, block_content: str, block_index: int,
                                                max_blocks_to_check: int = 30) -> Dict[str, Any]:
        """
        Calculate normalized confidence score (0-1) for a single block.
        (Renamed from calculate_confidence_score to be more specific in original code)
        This will benefit from the improved _calculate_score logic.
        """
        if block_index >= max_blocks_to_check:
            return {"confidence": 0.0, "raw_score": 0, "breakdown": {}, "is_candidate": False}

        score_result = self._calculate_score(block_content, block_index)
        raw_score = score_result["total"]

        capped_score = min(raw_score, self.max_score_exemplar)
        confidence = (capped_score / (self.max_score_exemplar + 1e-9))

        single_block_confidence_candidate_threshold = self.rules.get("single_block_candidate_min_confidence", 0.45)
        is_candidate = confidence >= single_block_confidence_candidate_threshold

        return {
            "confidence": confidence,
            "raw_score": raw_score,
            "breakdown": score_result["breakdown"],
            "is_candidate": is_candidate
        }
