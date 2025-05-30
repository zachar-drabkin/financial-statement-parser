# cover_page_classifier.py
import re
from typing import Dict, List, Optional, Tuple, Any
import logging
from utils.text_utils import normalize_text, find_exact_phrases, find_regex_patterns
from classifiers.base_classifier import BaseClassifier


logger = logging.getLogger(__name__)

# --- Constants ---
SECTION_NAME_COVER_PAGE = "Cover Page"
YEAR_PATTERN_REGEX = r'\b(19|20)\d{2}\b'

class CoverPageClassifier(BaseClassifier):
    """
    Rule-based Classifier for identifying Cover Page Sections in Financial Statements.
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


        if find_exact_phrases(text_upper_normalized, keywords):
            return max_score_for_type

        # check for ISO codes or symbols for partial score
        if find_exact_phrases(text_upper_normalized, iso_codes):
            score = max(score, single_indicator_score)

        # for symbols, use simple 'in' check as they are single characters mostly
        # and normalzation handles whitespace.
        for symbol in symbols:
            if symbol in text_upper_normalized: # Symbols are less reliant on \b word boundaries
                score = max(score, single_indicator_score)
                break # found one symbol, score assigned
        return score

    def _check_company_indicators(self, text_upper_normalized: str) -> int:
        rule_set = self.rules.get("company_indicators", {})
        suffixes = rule_set.get("suffixes", [])
        score_value = rule_set.get("score", 0)

        if find_exact_phrases(text_upper_normalized, suffixes):
            return score_value
        return 0

    def _check_year_pattern(self, text_upper_normalized: str) -> int:
        score_value = self.rules.get("year_pattern_score", 0)
        if re.search(YEAR_PATTERN_REGEX, text_upper_normalized):
            return score_value
        return 0

    def _calculate_score(self, combined_text: str, first_block_index: int) -> Dict[str, Any]:
        """
        Calculate raw score for a combined text of a window, with detailed breakdown.
        This method is the implementation for the abstract _calculate_score in BaseClassifier.
        """
        if not combined_text.strip():
            return {"total": 0, "breakdown": {}}

        text_upper_normalized = normalize_text(combined_text) # Use utility

        title_score = self._check_title_phrases(text_upper_normalized)
        date_score = self._check_date_phrases(text_upper_normalized)
        currency_score = self._check_currency_phrases(text_upper_normalized)
        company_score = self._check_company_indicators(text_upper_normalized)
        year_score = self._check_year_pattern(text_upper_normalized)

        block_bonus_max_idx = self.rules.get("block_bonus_max_index", 10)
        block_bonus_val = self.rules.get("block_bonus_score", 1)
        block_bonus = block_bonus_val if first_block_index < block_bonus_max_idx else 0

        combination_bonus = 0
        combo_min_indicators = self.rules.get("combination_bonus_min_strong_indicators", 3)
        combo_bonus_val = self.rules.get("combination_bonus_score", 1)

        strong_indicators = sum([
            1 if title_score > 0 else 0, 1 if date_score > 0 else 0,
            1 if currency_score > 0 else 0, 1 if company_score > 0 else 0
        ])
        if strong_indicators >= combo_min_indicators:
            combination_bonus = combo_bonus_val

        total_score = (title_score + date_score + currency_score +
                       company_score + year_score + block_bonus + combination_bonus)

        breakdown = {
            "title": title_score, "date": date_score, "currency": currency_score,
            "company": company_score, "year": year_score, "block_bonus": block_bonus,
            "combination_bonus": combination_bonus
        }
        return {"total": total_score, "breakdown": breakdown}

    def _expand_window(self, doc_blocks: List[Dict[str, Any]], start_list_idx: int) -> Tuple[List[Dict[str, Any]], str]:
        """
        Expands a window of sequential paragraph blocks from a starting index.
        Returns the list of blocks in the window and their combined content.
        """
        current_window_blocks = []
        current_window_content = ""
        for j in range(start_list_idx, len(doc_blocks)):
            current_block_in_expansion = doc_blocks[j]

            if current_block_in_expansion.get('type') != 'paragraph':
                # log if window expanson started
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
        """
        if not current_window_blocks:
            return None

        final_window_content = current_window_content.strip()
        first_block_in_window_idx = current_window_blocks[0]['index']

        score_result = self._calculate_score(final_window_content, first_block_in_window_idx)
        raw_score = score_result["total"]
        breakdown = score_result["breakdown"]

        capped_score = min(raw_score, self.max_score_exemplar)
        # ensure  not zero when dividing
        final_confidence = (capped_score / (self.max_score_exemplar + 1e-9)) # Add small epsilon for safety

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
                 confidence_threshold: float = 0.6,
                 max_start_block_index_to_check: int = 500) -> Optional[Dict[str, Any]]:
        """
        Identify Cover Page Section from document blocks using a sliding window.
        Implements the abstract 'classify' method from BaseClassifier.
        Removes 'debug' parameter as logging is handled globally.
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
                return result # Found the first suitable cover page

            if not current_window_blocks and start_block_candidate.get('type') == 'paragraph':
                 logger.debug(f"No valid window formed starting with paragraph block {start_block_candidate['index']}.")

        logger.info("No cover page identified with sufficient confidence after checking all potential windows.")
        return None

    # display_cover_page_results method inherited from BaseClassifier.
    def display_results(self, cover_page_result: Optional[Dict[str, Any]], all_doc_blocks: Optional[List[Dict[str, Any]]] = None) -> None:
        # call the parent display_results for common formatting
        super().display_results(classification_result=cover_page_result, all_doc_blocks=all_doc_blocks)

        if cover_page_result and logger.isEnabledFor(logging.DEBUG):
            # display block-by-block content if available
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
            logger.info("   - Ensure `DocxParser` correctly extracts blocks.")

    def calculate_confidence_score_single_block(self, block_content: str, block_index: int,
                                                max_blocks_to_check: int = 30) -> Dict[str, Any]:
        """
        Calculate normalized confidence score (0-1) for a single block.
        (Renamed from calculate_confidence_score to be more specific)
        """
        if block_index >= max_blocks_to_check:
            return {"confidence": 0.0, "raw_score": 0, "breakdown": {}, "is_candidate": False}

        score_result = self._calculate_score(block_content, block_index) # Uses the main scoring logic
        raw_score = score_result["total"]

        capped_score = min(raw_score, self.max_score_exemplar)
        confidence = (capped_score / (self.max_score_exemplar + 1e-9))

        is_candidate = confidence >= 0.45

        return {
            "confidence": confidence,
            "raw_score": raw_score,
            "breakdown": score_result["breakdown"],
            "is_candidate": is_candidate
        }
