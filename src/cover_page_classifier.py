# parsers/cover_page_classifier.py
import re
from typing import Dict, List, Optional, Any

class CoverPageClassifier:
    """
    Rule-based Classifier for identifying Cover Page Sections in Financial Statements.
    Implement sliding window for sequential paragraph blocks
    Upon finding the first suitable cover page - we can stop searching.
    """

    def __init__(self):
        self.keywords = self._initialize_keywords()
        # adjusted max_score_exemplar:
        # Title(4) + Date(2) + Currency(2) + Company(1) + Year(1) + Block bonus(1) + Combination bonus(1)
        self.max_score_exemplar = 12

    def _initialize_keywords(self) -> Dict[str, Any]:
        """Initialize comprehensive keyword dictionary with international coverage"""
        return {
            "title_phrases": {
                "keywords": [
                    # Core financial statement titles
                    "CONSOLIDATED FINANCIAL STATEMENTS", "FINANCIAL STATEMENTS",
                    "CONSOLIDATED FINANCIAL STATEMENT", "FINANCIAL STATEMENT",
                    # Report types
                    "ANNUAL REPORT", "QUARTERLY REPORT", "INTERIM REPORT",
                    "CONSOLIDATED ANNUAL FINANCIAL REPORT",
                    # Specific IFRS/GAAP statement titles
                    "STATEMENT OF FINANCIAL POSITION", "CONSOLIDATED STATEMENT OF FINANCIAL POSITION",
                    "BALANCE SHEET", "CONSOLIDATED BALANCE SHEET",
                    "INCOME STATEMENT", "CONSOLIDATED INCOME STATEMENT",
                    "STATEMENT OF PROFIT OR LOSS", "STATEMENT OF COMPREHENSIVE INCOME",
                    "CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME",
                    "STATEMENT OF CASH FLOWS", "CONSOLIDATED STATEMENT OF CASH FLOWS",
                    "STATEMENT OF CHANGES IN EQUITY", "CONSOLIDATED STATEMENT OF CHANGES IN EQUITY",
                    "STATEMENT OF STOCKHOLDERS' EQUITY", "CONSOLIDATED STATEMENT OF STOCKHOLDERS' EQUITY",
                    # Prefixed variations
                    "UNAUDITED FINANCIAL STATEMENTS", "AUDITED FINANCIAL STATEMENTS",
                    "INTERIM CONDENSED CONSOLIDATED FINANCIAL STATEMENTS",
                    "UNAUDITED CONSOLIDATED FINANCIAL STATEMENTS"
                ],
                "score": 4
            },
            "date_phrases": {
                "keywords": [
                    "FOR THE YEAR ENDED", "FOR THE YEARS ENDED", "FOR THE PERIOD ENDED",
                    "AS AT", "AS OF", "YEAR ENDED", "YEARS ENDED", "PERIOD ENDED", "ENDED",
                    "FOR THE THREE MONTHS ENDED", "FOR THE SIX MONTHS ENDED", "FOR THE NINE MONTHS ENDED",
                    "YEAR ENDING", "QUARTER ENDING", "MONTH ENDING"
                ],
                "patterns": [
                    r'\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{1,2},\s+(19|20)\d{2}\b',
                    r'\b\d{1,2}\s+(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(19|20)\d{2}\b',
                    r'\b(19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b',
                ],
                "score": 2
            },
            "currency_phrases": {
                "keywords": [
                    "EXPRESSED IN", "AMOUNTS IN", "REPORTING CURRENCY", "PRESENTATION CURRENCY",
                    "CANADIAN DOLLARS", "US DOLLARS", "UNITED STATES DOLLARS",
                    "EURO", "EUROS", "POUND STERLING", "POUNDS STERLING", "SWISS FRANC",
                    "SWISS FRANCS", "JAPANESE YEN", "AUSTRALIAN DOLLAR", "AUSTRALIAN DOLLARS",
                    "IN THOUSANDS", "IN MILLIONS", "IN BILLIONS", "THOUSANDS OF", "MILLIONS OF",
                    "BILLIONS OF", "IN THOUSANDS OF USD", "IN MILLIONS OF EUR",
                    "AMOUNTS IN THOUSANDS", "AMOUNTS IN MILLIONS", "(UNLESS OTHERWISE STATED)"
                ],
                "iso_codes": [
                    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "KRW",
                    "SGD", "HKD", "NZD", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "RUB",
                    "BRL", "MXN", "ZAR", "TRY", "ILS"
                ],
                "symbols": ["$", "€", "£", "¥", "₹", "₩", "₽"],
                "score": 2
            },
            "company_indicators": {
                "suffixes": [
                    "LTD", "LIMITED", "INC", "INCORPORATED", "CORP", "CORPORATION", "PLC",
                    "LLC", "LLP", "LP", "PC", "AG", "GMBH", "KGAA", "S\\.A\\.", "SAS",
                    "SARL", "S\\.L\\.", "S\\.R\\.L\\.", "N\\.V\\.", "B\\.V\\.", "S\\.P\\.A\\.",
                    "AB", "AS", "A/S", "OYJ", "OY", "SA", "SP\\. Z O\\.O\\.", "SRO",
                    "K\\.K\\.", "CO\\., LTD", "PTY LTD"
                ],
                "score": 1
            }
        }

    def _check_title_phrases(self, text_upper: str) -> int:
        for phrase in self.keywords["title_phrases"]["keywords"]:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            # if you found a regular expression exact match
            # of a keyword in text_upper, return the score
            # for title phrases.
            if re.search(pattern, text_upper):
                return self.keywords["title_phrases"]["score"]
        # if you did not find any match return score 0.
        return 0

    def _check_date_phrases(self, text_upper: str) -> int:
        for phrase in self.keywords["date_phrases"]["keywords"]:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            # if you found a regular expression exact match
            # of a keyword in text_upper, return the score
            # for date phrases.
            if re.search(pattern, text_upper):
                return self.keywords["date_phrases"]["score"]
        for pattern_str in self.keywords["date_phrases"]["patterns"]:
            # if you found a pattern match in text_upper,
            # return the score for date phrases.
            if re.search(pattern_str, text_upper):
                return self.keywords["date_phrases"]["score"]
        return 0

    def _check_currency_phrases(self, text_upper: str) -> int:
        score = 0
        for phrase in self.keywords["currency_phrases"]["keywords"]:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(pattern, text_upper):
                # return max score for currency phrases
                score = max(score, self.keywords["currency_phrases"]["score"])
                if score == self.keywords["currency_phrases"]["score"]:
                    # stop if we reached max score for currency phrases
                    break
        if score == self.keywords["currency_phrases"]["score"]:
            # stop if we reached max score for currency phrases
            return score

        for iso_code in self.keywords["currency_phrases"]["iso_codes"]:
            pattern = r'\b' + re.escape(iso_code) + r'\b'
            if re.search(pattern, text_upper):
                # a value of 1 will be assigned if only a iso-code is
                # found without any other currency context.
                score = max(score, 1)
        for symbol in self.keywords["currency_phrases"]["symbols"]:
            if symbol in text_upper: # Context check might be needed for some symbols
                # a value of 1 will be assigned if only a symbol is
                # found without any other currency context.
                score = max(score, 1)
        return score

    def _check_company_indicators(self, text_upper: str) -> int:
        for suffix in self.keywords["company_indicators"]["suffixes"]:
            pattern = r'\b' + suffix + r'\b'
            # if you found a match of a keyword in text_upper,
            # return the score for company indicators.
            if re.search(pattern, text_upper):
                return self.keywords["company_indicators"]["score"]
        return 0

    def _check_year_pattern(self, text_upper: str) -> int:
        year_pattern = r'\b(19|20)\d{2}\b'
        if re.search(year_pattern, text_upper):
            return 1
        return 0

    def _calculate_raw_score(self, combined_text: str, first_block_index: int) -> Dict[str, Any]:
        """Calculate raw score for a combined text of a window, with detailed breakdown."""
        if not combined_text.strip():
            return {"total": 0, "breakdown": {}}

        text_upper = combined_text.upper()

        title_score = self._check_title_phrases(text_upper)
        date_score = self._check_date_phrases(text_upper)
        currency_score = self._check_currency_phrases(text_upper)
        company_score = self._check_company_indicators(text_upper)
        year_score = self._check_year_pattern(text_upper)

        # give a bonus for the first block in the window
        # if it is within the first 10 blocks of the document.
        block_bonus = 1 if first_block_index < 10 else 0

        combination_bonus = 0
        strong_indicators = sum([
            1 if title_score > 0 else 0, 1 if date_score > 0 else 0,
            1 if currency_score > 0 else 0, 1 if company_score > 0 else 0
        ])
        if strong_indicators >= 3:
            # another bonus if at least 3 strong indicators are present
            combination_bonus = 1

        total_score = (title_score + date_score + currency_score +
                      company_score + year_score + block_bonus + combination_bonus)

        breakdown = {
            "title": title_score, "date": date_score, "currency": currency_score,
            "company": company_score, "year": year_score, "block_bonus": block_bonus,
            "combination_bonus": combination_bonus
        }
        return {"total": total_score, "breakdown": breakdown}

    def classify_cover_page(self, doc_blocks: List[Dict],
                            confidence_threshold: float = 0.6,
                            # Max document index for the start of a cover page window
                            max_start_block_index_to_check: int = 500,
                            debug: bool = False) -> Optional[Dict[str, Any]]:
        """
        Identify Cover Page Section from document blocks using a sliding window.
        The cover page consists of sequential 'paragraph' type blocks.
        Returns the first identified cover page meeting the threshold.

        Args:
            doc_blocks: List of document blocks. Each block is a dict with
                        'content' (string), 'index' (int), and 'type' (string) keys.
                        It's assumed that doc_blocks are arriving by order of 'index'.
            confidence_threshold: Minimum confidence score (0-1) for classification.
            max_start_block_index_to_check: Don't start new windows beyond this block index.
            debug: Whether to print debug information.

        Returns:
            Dictionary with Cover Page info or None if not found.
        """
        for i in range(len(doc_blocks)):
            start_block_candidate = doc_blocks[i]

            # from my research this check is not so good as we do not know
            # how many irrelevant blocks are before the cover page.
            # nonetheless, it is a good optimization to avoid
            # unnecessary checks for very deep blocks.
            if start_block_candidate['index'] >= max_start_block_index_to_check:
                if debug:
                    print(f"Stopping search: Start block index {start_block_candidate['index']} "
                          f"is >= max_start_block_index_to_check ({max_start_block_index_to_check}).")
                break

            # no point in checking table types as start of a cover page window.
            if start_block_candidate.get('type') != 'paragraph':
                if debug:
                    print(f"Skipping block {start_block_candidate['index']} as start of window: "
                          f"not a paragraph (type: {start_block_candidate.get('type')}).")
                continue

            current_window_blocks = []
            current_window_content = ""

            # expand window from start_block_candidate
            for j in range(i, len(doc_blocks)):
                current_block_in_expansion = doc_blocks[j]

                # window can only contain sequential paragraphs
                # assumes doc_blocks are in order of index, and we are checking for list contiguity
                # and paragraph type.
                if current_block_in_expansion.get('type') != 'paragraph':
                    # print only if window expansion had started
                    if debug and current_window_blocks:
                         print(f"Window ending at block {current_window_blocks[-1]['index']}: "
                               f"Next block {current_block_in_expansion['index']} is not a paragraph "
                               f"(type: {current_block_in_expansion.get('type')}).")
                    break

                # if we reached here there is a paragraph block
                # that can be added to the current window.
                current_window_blocks.append(current_block_in_expansion)
                current_window_content += current_block_in_expansion['content'] + "\n"

                if debug:
                    latest_block_preview = current_block_in_expansion['content'][:80].strip()
                    window_indices = [b['index'] for b in current_window_blocks]
                    print(f"Window expanded to include block {current_block_in_expansion['index']} "
                          f"(type: {current_block_in_expansion.get('type')}). "
                          f"Current window blocks indices: {window_indices}. "
                          f"Added content preview: '{latest_block_preview}...'.")

            # after the inner loop (window expansion) finishes or breaks:
            if current_window_blocks:
                # give a score to the fully formed window
                final_window_content = current_window_content.strip()
                first_block_in_window_idx = current_window_blocks[0]['index']

                score_result = self._calculate_raw_score(final_window_content, first_block_in_window_idx)
                raw_score = score_result["total"]
                breakdown = score_result["breakdown"]

                # score cannot be higher than max_score_exemplar
                # to prevent confidence > 1.0.
                capped_score = min(raw_score, self.max_score_exemplar)
                final_confidence = (capped_score / (self.max_score_exemplar + 1)) if self.max_score_exemplar > 0 else 0.0

                window_indices_str = str([b['index'] for b in current_window_blocks])
                if debug:
                    print(f"  EVALUATING WINDOW: Blocks {window_indices_str}, "
                          f"  Start Index: {first_block_in_window_idx}\n"
                          f"  Combined Content: '{final_window_content[:150].replace(chr(10), ' ')}...'\n"
                          f"  Raw Score: {raw_score}, Confidence: {final_confidence:.3f}, Breakdown: {breakdown}")

                # check if the window meets the confidence threshold
                # and if so, return the result immediately.
                if final_confidence >= confidence_threshold:
                    if debug:
                        print(f"  >>> COVER PAGE IDENTIFIED (Window): Blocks {window_indices_str}, "
                              f"Confidence: {final_confidence:.3f} >= Threshold: {confidence_threshold}")

                    result = {
                        "section_name": "Cover Page",
                        "start_block_index": current_window_blocks[0]['index'],
                        "end_block_index": current_window_blocks[-1]['index'],
                        "block_indices": [b['index'] for b in current_window_blocks],
                        "num_blocks": len(current_window_blocks),
                        "confidence": final_confidence,
                        "raw_score": raw_score,
                        "breakdown": breakdown,
                        "content": final_window_content,
                        # extract all relevant details for summary
                        "details": {
                             "confidence": final_confidence,
                             "raw_score": raw_score,
                             "breakdown": breakdown,
                             "content_preview": final_window_content[:200].strip()
                        }
                    }
                    # no need to go further.
                    return result

            # window was not formed.
            elif debug and start_block_candidate.get('type') == 'paragraph':
                 print(f"No valid window formed starting with paragraph block {start_block_candidate['index']} "
                       f"(it might have been a single block window that was then evaluated, or expansion failed immediately).")


        if debug:
            print("\nNo cover page identified with sufficient confidence after checking all potential windows.")
        # if nothing was found, return None
        return None

    def calculate_confidence_score(self, block_content: str, block_index: int,
                                max_blocks_to_check: int = 30) -> Dict[str, Any]:
        """
        Calculate normalized confidence score (0-1) for a single block.
        This method is for analyzing individual blocks, not for the main window-based classification.
        """
        # Only check blocks within a reasonable range from the start of the document
        # max_blocks_to_check here applies to how deep in the document an individual block can be
        if block_index >= max_blocks_to_check:
            return {
                "confidence": 0.0,
                "raw_score": 0,
                "breakdown": {},
                "is_candidate": False
            }

        # Calculate raw score for the single block's content
        # self._calculate_raw_score expects combined_text and first_block_index;
        # for a single block, block_content is the combined_text and block_index is the first_block_index.
        score_result = self._calculate_raw_score(block_content, block_index)
        raw_score = score_result["total"]
        breakdown = score_result["breakdown"]

        # Normalize to 0-1 confidence using min-max scaling
        # Cap raw score at max_score_exemplar to prevent confidence > 1.0
        capped_score = min(raw_score, self.max_score_exemplar)
        confidence = (capped_score / (self.max_score_exemplar + 1)) if self.max_score_exemplar > 0 else 0.0

        # Determine if this is a viable candidate (using a suitable threshold for single blocks)
        is_candidate = confidence >= 0.45  # This was an example threshold from original code

        return {
            "confidence": confidence,
            "raw_score": raw_score,
            "breakdown": breakdown,
            "is_candidate": is_candidate
        }

    def display_cover_page_results(slef, cover_page_result: Optional[Dict[str, Any]], all_doc_blocks: List[Dict]):
        """Display cover page classification results in a clear format."""

        print("\nCOVER PAGE CLASSIFICATION RESULTS:")
        print("-" * 40)

        if cover_page_result:
            print(f" Cover page identified!")
            print(f" Section: {cover_page_result['section_name']}")
            print(f" Confidence: {cover_page_result['confidence']:.3f}")
            print(f" Raw score: {cover_page_result['raw_score']}")
            print(f" Number of blocks in cover page: {cover_page_result['num_blocks']}")
            print(f" Start Block Index: {cover_page_result['start_block_index']}")
            print(f" End Block Index: {cover_page_result['end_block_index']}")

            print(f"\n Score breakdown for the identified cover page window (combined content):")
            breakdown = cover_page_result['breakdown']
            has_breakdown_contribution = False
            for category, score in breakdown.items():
                if score > 0:
                    print(f"   • {category.replace('_', ' ').title()}: +{score}")
                    has_breakdown_contribution = True
            if not has_breakdown_contribution:
                print("   (No specific keyword categories significantly contributed, or score is based on bonuses only)")

            print(f"\n Identified cover page content (block by block):")

            # Create a quick lookup for block content by index from the originally parsed blocks
            block_content_map = {block['index']: block['content'] for block in all_doc_blocks}

            for block_idx in cover_page_result['block_indices']:
                content = block_content_map.get(block_idx, "Content not found for this block index.")

                # Truncate long content for display and clean up newlines for single-line preview
                display_content = content.replace('\n', ' ').replace('\r', ' ').strip()
                if len(display_content) > 100:
                    display_content = display_content[:97] + "..."
                print(f"   Block {block_idx}: {display_content}")

        else:
            print("\n No cover page identified with sufficient confidence.")
            print("  Suggestions:")
            print("   - Adjust `confidence_threshold` in the `classify_cover_page` call (e.g., lower it).")
            print("   - Verify `max_start_block_index_to_check` is suitable for your document structures.")
            print("   - Ensure the `DocxParser` correctly extracts blocks with 'index', 'content', and 'type'.")
            print("   - Run with `debug=True` in `classify_cover_page` for detailed logs from the classifier.")
