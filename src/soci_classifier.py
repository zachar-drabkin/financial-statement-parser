import re
import math # For potential future use, not strictly needed in this revision
from typing import Dict, List, Optional, Tuple, Any

class SoCIClassifier:
    """
    Rule-based Classifier for identifying Statement of Comprehensive Income (SOCI)
    sections in Financial Statements.
    Implements a sliding window approach that can start with a title paragraph
    followed by tables, or directly with table(s) if they are content-rich.
    Handles IFRS single and two-statement presentations.
    Improved termination logic.
    """

    def __init__(self):
        self.keywords = self._initialize_keywords_soci()
        # max score based on illustrative scores for:
        # Title (10) + P&L Content (64) + OCI Content (22) + Structural (13) + EPS/Discontinued (10)
        # + IFRS 2-statement Bonus (15) + Positional Bonus (5) + Block Bonus (5) + Combination Bonus (10)
        self.max_score_exemplar_soci = 154
        # self.known_statement_titles = self._get_all_known_statement_titles() # Used by _is_termination_block implicitly

    def _normalize_text(self, text: str) -> str:
        """Convert text to uppercase and replace multiple whitespace chars with a single space."""
        if not text:
            return ""
        text = text.upper()
        text = re.sub(r'\s+', ' ', text) # Replace multiple whitespace (newline, tab, space) with single space
        return text.strip()

    def _get_all_known_statement_titles_normalized(self) -> List[str]:
        """Helper to get all known statement titles, normalized for termination checks."""
        titles = []
        for kw_list_name, kw_list_data in self.keywords["soci_titles"].items():
            for kw_item in kw_list_data:
                 titles.append(self._normalize_text(kw_item['text']))

        for kw_item_text in self.keywords["termination_keywords"]["statement_titles"]:
            titles.append(self._normalize_text(kw_item_text))
        return list(set(titles))

    def _initialize_keywords_soci(self) -> Dict[str, Any]:
        """Initialize comprehensive keyword dictionary for SOCI classification."""
        # keywords are uppercased here for consistent matching later.
        return {
            "soci_titles": {
                "ifrs_primary": [
                    {"text": "STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME", "score": 10, "type": "ifrs_combined_soci"},
                    {"text": "STATEMENT OF COMPREHENSIVE INCOME", "score": 9, "type": "ifrs_oci_focused_or_gaap_with_oci"},
                    {"text": "STATEMENT OF PROFIT OR LOSS", "score": 8, "type": "ifrs_pl"},
                ],
                "gaap_primary": [
                    {"text": "INCOME STATEMENT", "score": 9, "type": "gaap_income_statement"},
                    {"text": "STATEMENT OF OPERATIONS", "score": 8, "type": "gaap_operations"},
                    {"text": "STATEMENT OF EARNINGS", "score": 8, "type": "gaap_earnings"},
                    {"text": "CONSOLIDATED INCOME STATEMENT", "score": 9, "type": "gaap_income_statement_cons"},
                    {"text": "CONSOLIDATED STATEMENTS OF INCOME", "score": 9, "type": "gaap_income_statement_cons"}, # Plural
                    {"text": "CONSOLIDATED STATEMENT OF OPERATIONS", "score": 8, "type": "gaap_operations_cons"},
                    {"text": "CONSOLIDATED STATEMENTS OF OPERATIONS", "score": 8, "type": "gaap_operations_cons"}, # Plural
                    {"text": "CONSOLIDATED STATEMENT OF EARNINGS", "score": 8, "type": "gaap_earnings_cons"},
                    {"text": "CONSOLIDATED STATEMENTS OF EARNINGS", "score": 8, "type": "gaap_earnings_cons"}, # Plural
                    {"text": "CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME", "score": 9, "type": "ifrs_oci_focused_or_gaap_with_oci_cons"},
                    {"text": "CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME", "score": 9, "type": "ifrs_oci_focused_or_gaap_with_oci_cons"}, # Plural
                ],
                "shared_ambiguous": [
                    {"text": "STATEMENT OF (LOSS)", "score": 5, "type": "ambiguous_loss"},
                    {"text": "STATEMENT OF LOSS", "score": 5, "type": "ambiguous_loss_direct"},
                    {"text": "PROFIT AND LOSS ACCOUNT", "score": 6, "type": "legacy_pl"},
                    {"text": "STATEMENTS OF LOSS AND COMPREHENSIVE LOSS", "score": 7, "type": "gaap_loss_statement_cons"},
                ],
                 "length_check_buffer": 30 # Title block should not be excessively longer
            },
            "pl_keywords": { # Keywords are generally searched with \bword\b, so case here is for definition
                "revenue": [
                    {"text": "REVENUE", "score": 10}, {"text": "SALES", "score": 10}, {"text": "NET SALES", "score": 10},
                    {"text": "TURNOVER", "score": 9}, {"text": "TOTAL REVENUES", "score": 10}
                ],
                "cogs": [
                    {"text": "COST OF SALES", "score": 7}, {"text": "COST OF GOODS SOLD", "score": 7}, {"text": "COST OF REVENUE", "score": 7}
                ],
                "gross_profit": [
                    {"text": "GROSS PROFIT", "score": 8}, {"text": "GROSS MARGIN", "score": 7}, {"text": "GROSS LOSS", "score": 8}
                ],
                "operating_income": [ # Matches "Operating loss" from log due to "OPERATING" and "LOSS"
                    {"text": "OPERATING INCOME", "score": 8}, {"text": "INCOME FROM OPERATIONS", "score": 8},
                    {"text": "PROFIT FROM OPERATIONS", "score": 8}, {"text": "OPERATING PROFIT", "score": 8},
                    {"text": "LOSS FROM OPERATIONS", "score": 8}, {"text": "OPERATING LOSS", "score": 8} # Explicitly add
                ],
                "operating_expenses": [ # To ensure "OPERATING EXPENSES" is scored directly
                    {"text": "OPERATING EXPENSES", "score": 6},
                    {"text": "SELLING, GENERAL AND ADMINISTRATIVE EXPENSES", "score": 5}, {"text": "SG&A", "score": 5},
                    {"text": "RESEARCH AND DEVELOPMENT EXPENSES", "score": 5}, {"text": "R&D", "score": 5},
                ],
                "finance_costs_income": [ # finance income (expense), net
                    {"text": "FINANCE COSTS", "score": 6}, {"text": "INTEREST EXPENSE", "score": 6},
                    {"text": "FINANCE INCOME", "score": 6}, {"text": "INTEREST INCOME", "score": 6},
                    {"text": "FINANCE INCOME (EXPENSE), NET", "score": 6}
                ],
                "pre_tax_income": [
                    {"text": "INCOME BEFORE TAX", "score": 8}, {"text": "PROFIT BEFORE TAX", "score": 8},
                    {"text": "EARNINGS BEFORE INCOME TAXES", "score": 8}, {"text": "LOSS BEFORE TAX", "score": 8}
                ],
                "tax": [
                    {"text": "INCOME TAX EXPENSE", "score": 7}, {"text": "PROVISION FOR INCOME TAXES", "score": 7},
                    {"text": "TAX EXPENSE", "score": 7}, {"text": "INCOME TAX BENEFIT", "score": 7},
                    {"text": "DEFERRED INCOME TAX", "score": 5} # Often part of tax expense
                ],
                "net_income": [
                    {"text": "NET INCOME", "score": 10}, {"text": "NET EARNINGS", "score": 10}, {"text": "NET LOSS", "score": 10},
                    {"text": "PROFIT FOR THE PERIOD", "score": 10}, {"text": "LOSS FOR THE PERIOD", "score": 10},
                    {"text": "PROFIT ATTRIBUTABLE TO OWNERS", "score": 9},
                ],
                "other_pl_specific_items": [
                    {"text": "DEBT FORGIVENESS", "score": 3}, # From log, seems like a PL item
                    {"text": "GAIN ON FOREIGN EXCHANGE", "score": 3},
                    {"text": "UNREALIZED GAIN ON MARKETABLE SECURITIES", "score": 4}, # Often P&L if trading
                    {"text": "GAIN ON SALE OF TESTCO PROPERTIES", "score": 4}, # Could be specific name
                    {"text": "LOSS ON ROCKET PROJECT TRANSACTIONS", "score": 4}, # Could be specific name
                ],
                "other_pl_generic_items": [
                    {"text": "DEPRECIATION EXPENSE", "score": 4}, {"text": "AMORTIZATION EXPENSE", "score": 4},
                    {"text": "OTHER INCOME", "score": 3}, {"text": "OTHER EXPENSE", "score": 3},
                    {"text": "SHARE OF PROFIT OF ASSOCIATES", "score": 4}, {"text": "SHARE OF LOSS OF ASSOCIATES", "score": 4},
                ]
            },
            "oci_keywords": {
                "headers": [
                    {"text": "OTHER COMPREHENSIVE INCOME", "score": 8},
                    {"text": "OTHER COMPREHENSIVE (LOSS)", "score": 8},
                    {"text": "COMPONENTS OF OTHER COMPREHENSIVE INCOME", "score": 7},
                ],
                "classification_headers": [
                    {"text": "ITEMS THAT MAY BE RECLASSIFIED SUBSEQUENTLY TO PROFIT OR LOSS", "score": 6},
                    {"text": "ITEMS THAT WILL NOT BE RECLASSIFIED SUBSEQUENTLY TO PROFIT OR LOSS", "score": 6},
                ],
                "items": [
                    {"text": "EXCHANGE DIFFERENCES ON TRANSLATING FOREIGN OPERATIONS", "score": 7},
                    {"text": "FOREIGN CURRENCY TRANSLATION ADJUSTMENTS", "score": 7},
                    {"text": "CURRENCY TRANSLATION DIFFERENCES", "score": 7}, # From example
                    {"text": "UNREALIZED GAINS AND LOSSES ON DEBT INVESTMENTS AT FVOCI", "score": 7},
                    {"text": "UNREALIZED GAINS AND LOSSES ON EQUITY INVESTMENTS AT FVOCI", "score": 7},
                    {"text": "AVAILABLE-FOR-SALE FINANCIAL ASSETS", "score": 6},
                    {"text": "REVALUATION SURPLUS", "score": 6},
                    {"text": "REMEASUREMENTS OF DEFINED BENEFIT PENSION PLANS", "score": 7},
                    {"text": "ACTUARIAL GAINS AND LOSSES ON DEFINED BENEFIT PLANS", "score": 7},
                    {"text": "EFFECTIVE PORTION OF GAINS AND LOSSES ON HEDGING INSTRUMENTS", "score": 6},
                ],
                "aggregations": [
                    {"text": "TOTAL OTHER COMPREHENSIVE INCOME", "score": 7},
                    {"text": "TOTAL OTHER COMPREHENSIVE LOSS", "score": 7},
                    {"text": "TOTAL COMPREHENSIVE INCOME", "score": 8},
                    {"text": "TOTAL COMPREHENSIVE LOSS", "score": 8},
                    {"text": "COMPREHENSIVE INCOME", "score": 8},
                    {"text": "COMPREHENSIVE LOSS", "score": 8},
                ]
            },
            "structural_cues": {
                "period_indicators_keywords": [
                    {"text": "FOR THE YEAR ENDED", "score": 5}, {"text": "FOR THE YEARS ENDED", "score": 5},
                    {"text": "YEARS ENDED DECEMBER 31", "score": 5}, # From example table header
                    {"text": "YEAR ENDED", "score": 5},
                    {"text": "ENDED", "score": 3},
                    {"text": "FOR THE THREE MONTHS ENDED", "score": 4},
                ],
                "period_indicators_patterns": [
                    r'\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{1,2},\s+(?:19|20)\d{2}\b',
                    r'\b\d{1,2}\s+(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(?:19|20)\d{2}\b',
                    r'\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b',
                ],
                "year_pattern": r'\b(?:19|20)\d{2}\b',
                "currency_keywords": [
                    {"text": "EXPRESSED IN", "score": 2}, {"text": "AMOUNTS IN", "score": 2},
                    {"text": "THOUSANDS", "score": 1}, {"text": "MILLIONS", "score": 1},
                ],
                "currency_iso_codes": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"], # Score 2
                "currency_symbols": ["$", "€", "£", "¥"], # Score 1
            },
            "eps_discontinued_ops": [
                {"text": "EARNINGS PER SHARE", "score": 5},
                {"text": "NET LOSS PER SHARE", "score": 5}, # From example
                {"text": "BASIC AND DILUTED", "score": 2}, # Often accompanies EPS lines
                {"text": "BASIC EARNINGS PER SHARE", "score": 5},
                {"text": "DILUTED EARNINGS PER SHARE", "score": 5},
                {"text": "BASIC EPS", "score": 4},
                {"text": "WEIGHTED AVERAGE NUMBER OF COMMON SHARES", "score": 3}, # Context for EPS
                {"text": "DISCONTINUED OPERATIONS", "score": 5},
                {"text": "INCOME FROM DISCONTINUED OPERATIONS", "score": 5},
            ],
            "termination_keywords": {
                "statement_titles": [ # Titles of OTHER statements
                    "STATEMENT OF FINANCIAL POSITION", "CONSOLIDATED STATEMENT OF FINANCIAL POSITION", "BALANCE SHEET", "CONSOLIDATED BALANCE SHEETS",
                    "STATEMENT OF CASH FLOWS", "CONSOLIDATED STATEMENT OF CASH FLOWS", "CASH FLOW STATEMENT",
                    "STATEMENT OF CHANGES IN EQUITY", "CONSOLIDATED STATEMENT OF CHANGES IN EQUITY",
                    "STATEMENT OF CHANGES IN SHAREHOLDERS' EQUITY", "STATEMENT OF STOCKHOLDERS' EQUITY",
                    # Main "Notes" section headers
                    "NOTES TO THE CONSOLIDATED FINANCIAL STATEMENTS", "NOTES TO FINANCIAL STATEMENTS",
                    "NOTES TO THE FINANCIAL STATEMENTS", "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
                ],
                "other_sections": [ # Other major report sections
                    "AUDITOR'S REPORT", "INDEPENDENT AUDITOR'S REPORT",
                    "MANAGEMENT DISCUSSION AND ANALYSIS", "MANAGEMENT'S DISCUSSION AND ANALYSIS", "MD&A",
                ],
                "note_section_patterns": [ # Patterns to identify start of detailed notes
                    r"^\s*(?:NOTE\s+)?\d+\s*[\.:\-‐]?\s*[A-Z][A-Z0-9\s,'&\(\)\/\-]{5,}", # e.g., "1. NATURE OF OPERATIONS", "NOTE 1: BASIS OF PREP"
                    r"^\s*\d+\.\s*(NATURE OF OPERATIONS|BASIS OF PREPARATION|SIGNIFICANT ACCOUNTING POLICIES)", # Common first note titles
                ]
            }
        }

    def _find_keywords(self, normalized_text_upper: str, keyword_list: List[Dict], unique: bool = True) -> (int, List[str], int):
        found_score = 0
        found_items = []
        count = 0
        seen_keywords_texts = set()

        for item in keyword_list:
            keyword_text_upper = item["text"].upper() # ensure keyword from dict is upper for matching
            pattern = r'\b' + re.escape(keyword_text_upper) + r'\b'
            if re.search(pattern, normalized_text_upper):
                if unique and keyword_text_upper in seen_keywords_texts:
                    continue
                seen_keywords_texts.add(keyword_text_upper)
                found_score += item["score"]
                found_items.append(item["text"]) #  return orginal casing for display
                count +=1
        return found_score, found_items, count

    def _check_soci_titles(self, normalized_text_upper_block: str) -> (int, Optional[str], Optional[str]):
        max_score = 0
        best_title_text = None
        best_title_type = None
        buffer = self.keywords["soci_titles"]["length_check_buffer"]

        all_title_lists = list(self.keywords["soci_titles"].values())
        if "length_check_buffer" in self.keywords["soci_titles"]: # remove non-list item
             all_title_lists.remove(buffer)


        sorted_title_keywords = []
        for kw_list in all_title_lists:
            sorted_title_keywords.extend(sorted(kw_list, key=lambda x: len(x["text"]), reverse=True))

        for title_item in sorted_title_keywords:
            title_keyword_upper = title_item["text"].upper()
            pattern = r'\b' + re.escape(title_keyword_upper) + r'\b'
            match = re.search(pattern, normalized_text_upper_block)
            if match:
                # title should be a substantial part of the block, not buried in unrelated text
                # check if the block content length is reasonable compared to title length
                if len(normalized_text_upper_block) < (len(title_keyword_upper) + buffer):
                    if title_item["score"] > max_score: # priritize higher score if multiple found
                        max_score = title_item["score"]
                        best_title_text = title_item["text"]
                        best_title_type = title_item["type"]
        return max_score, best_title_text, best_title_type

    def _check_pl_keywords(self, normalized_text_upper: str) -> (int, List[str], bool, bool):
        total_score = 0
        all_found_keywords = set() #   use set for unique keywords
        has_revenue_flag = False
        has_net_income_flag = False

        # check for Revenue
        revenue_score, revenue_kw, _ = self._find_keywords(normalized_text_upper, self.keywords["pl_keywords"]["revenue"])
        if revenue_kw:
            total_score += revenue_score
            all_found_keywords.update(revenue_kw)
            has_revenue_flag = True

        #  check for Net Income / Profit for the period
        net_income_score, net_income_kw, _ = self._find_keywords(normalized_text_upper, self.keywords["pl_keywords"]["net_income"])
        if net_income_kw:
            total_score += net_income_score
            all_found_keywords.update(net_income_kw)
            has_net_income_flag = True

        for category_name, keywords_list in self.keywords["pl_keywords"].items():
            if category_name in ["revenue", "net_income"]: continue
            cat_score, cat_kw, _ = self._find_keywords(normalized_text_upper, keywords_list)
            if cat_kw:
                total_score += cat_score # sum all distinct keyword score
                all_found_keywords.update(cat_kw)
        return total_score, list(all_found_keywords), has_revenue_flag, has_net_income_flag

    def _check_oci_keywords(self, normalized_text_upper: str) -> (int, List[str]):
        total_score = 0
        all_found_keywords = set()
        for category_name, keywords_list in self.keywords["oci_keywords"].items():
            cat_score, cat_kw, _ = self._find_keywords(normalized_text_upper, keywords_list)
            if cat_kw:
                total_score += cat_score
                all_found_keywords.update(cat_kw)
        return total_score, list(all_found_keywords)

    def _check_structural_cues(self, normalized_text_upper: str) -> (int, List[str]):
        score = 0
        found_cues = []
        period_kw_score, period_kws, _ = self._find_keywords(normalized_text_upper, self.keywords["structural_cues"]["period_indicators_keywords"])
        if period_kws:
            score += period_kw_score
            found_cues.extend(period_kws)
        for pattern_str in self.keywords["structural_cues"]["period_indicators_patterns"]:
            if re.search(pattern_str, normalized_text_upper, re.IGNORECASE): # Patterns are already regex
                score += 5
                found_cues.append(f"Date Pattern: {pattern_str.split(':')[0]}" if ':' in pattern_str else f"Date Pattern: {pattern_str}") # Cleaner display
                break
        year_matches = re.findall(self.keywords["structural_cues"]["year_pattern"], normalized_text_upper)
        if len(set(year_matches)) >= 2: score += 5; found_cues.append(f"Comparative Years: {list(set(year_matches))}")
        elif year_matches: score += 2; found_cues.append(f"Year Found: {list(set(year_matches))}")

        currency_score_gained = False
        curr_kw_score, curr_kws, _ = self._find_keywords(normalized_text_upper, self.keywords["structural_cues"]["currency_keywords"])
        if curr_kws: score += curr_kw_score; found_cues.extend(curr_kws); currency_score_gained = True

        if not currency_score_gained:
            for iso_code in self.keywords["structural_cues"]["currency_iso_codes"]:
                if re.search(r'\b' + re.escape(iso_code) + r'\b', normalized_text_upper):
                    score += 2; found_cues.append(f"ISO Code: {iso_code}"); currency_score_gained = True; break
        if not currency_score_gained:
            for symbol in self.keywords["structural_cues"]["currency_symbols"]:
                if symbol in normalized_text_upper: # Symbol check is simpler
                    score += 1; found_cues.append(f"Symbol: {symbol}"); break
        return score, found_cues

    def _check_eps_discontinued_ops(self, normalized_text_upper: str) -> (int, List[str]):
        score, kws, _ = self._find_keywords(normalized_text_upper, self.keywords["eps_discontinued_ops"])
        return score, kws

    def _is_termination_block(self, block_content: str, block_type: Optional[str], debug: bool = False) -> bool:
        normalized_content = self._normalize_text(block_content)
        if not normalized_content: return False

        # check for explicit statement titles that terminate SOCI
        for title_text in self.keywords["termination_keywords"]["statement_titles"]:
            # use word boundaries, ensure title is prominent if found
            # termination titles should be normalized as they are stored
            pattern = r'\b' + re.escape(self._normalize_text(title_text)) + r'\b'
            match = re.search(pattern, normalized_content)
            if match:
                # check if the title is a major part of this block, not just a minor mention.
                # sofp titles were removed from here as soci can't be terminated by its predecessor.
                # If soci specific titles were in termination_keywords, they need to be handled.
                # This check assumes termination_keywords are for other major sections.
                if len(normalized_content) < (len(self._normalize_text(title_text)) + 60) and \
                   self._normalize_text(title_text) not in [self._normalize_text(k['text']) for k_list in self.keywords["soci_titles"].values() if isinstance(k_list, list) for k in k_list]:
                    if debug:
                        print(f"    Termination: Statement title '{title_text}' in block '{normalized_content[:100]}...'")
                    return True

        #  check for other major section headers
        for section_header_text in self.keywords["termination_keywords"]["other_sections"]:
             pattern = r'\b' + re.escape(self._normalize_text(section_header_text)) + r'\b'
             if re.search(pattern, normalized_content):
                if len(normalized_content) < (len(self._normalize_text(section_header_text)) + 80):
                    if debug:
                        print(f"    Termination: Other section header '{section_header_text}' in block '{normalized_content[:100]}...'")
                    return True

        # Check for start of detailed notes patterns (more robust)
        for note_pattern_regex in self.keywords["termination_keywords"]["note_section_patterns"]:
            if re.search(note_pattern_regex, block_content.strip().upper()): # Use original case for regex that might depend on it, but strip and upper for general patterns
                if debug:
                    print(f"    Termination: Note section pattern '{note_pattern_regex}' in block '{normalized_content[:100]}...'")
                return True
        return False

    def _calculate_raw_score_soci(self, combined_text_original_case: str, first_block_index: int,
                                  title_info: Optional[Tuple[int, str, str]] = None, # (score, text, type) from pre-check
                                  is_table_only_core: bool = False
                                  ) -> Dict[str, Any]:
        if not combined_text_original_case.strip():
            return {"total": 0, "breakdown": {}, "best_title_text": None, "best_title_type": None,
                    "is_potential_ifrs_pl": False, "is_missing_critical_pl": True}

        # normalize the entire combined text once for most keyword checks
        # title check might use original case block text if title_info is from a single block.
        normalized_combined_text = self._normalize_text(combined_text_original_case)
        breakdown = {}

        title_score, best_title_text, best_title_type = 0, None, None
        if title_info: # If title info passed from a specific title paragraph
            title_score, best_title_text, best_title_type = title_info
        elif not is_table_only_core : # If not table only, try to find title in combined text (less likely to be good)
             #  this case is for when window started with non-title paragraph but expanded.
             #  for table-only core, title_score remains 0.
             # this check is on the whole window, so it's less precise for titles.
             temp_title_score, temp_bt, temp_btt = self._check_soci_titles(normalized_combined_text)
             if temp_title_score > 0 : # Only assign if a decent title found in the whole mess
                  title_score, best_title_text, best_title_type = temp_title_score, temp_bt, temp_btt

        breakdown["title"] = title_score

        pl_score, found_pl_kws, has_revenue, has_net_income = self._check_pl_keywords(normalized_combined_text)
        is_missing_critical_pl = not (has_revenue and has_net_income)
        if is_missing_critical_pl and pl_score > 10 :
            original_pl_score = pl_score
            pl_score = max(0, pl_score // 3)
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

        ifrs_two_statement_bonus = breakdown.get("ifrs_two_statement_bonus", 0) # This bonus is applied later

        block_bonus = 5 if first_block_index < 30 else 0 # SOCI can start a bit deeper
        breakdown["block_bonus"] = block_bonus

        combination_bonus = 0
        title_is_strong_enough = title_score >= 5 # Relaxed slightly if other parts are very strong
        pl_is_strong = pl_score >= 40 and not is_missing_critical_pl
        oci_expected_from_title = best_title_type and ("soci" in best_title_type or "oci" in best_title_type or "comprehensive" in best_title_type.lower())
        oci_is_present_or_not_expected = (oci_expected_from_title and oci_score >= 10) or (not oci_expected_from_title and oci_score >=0) # OCI must be decent if title implies it

        if (title_is_strong_enough or is_table_only_core) and pl_is_strong and oci_is_present_or_not_expected:
            combination_bonus = 10
        breakdown["combination_bonus"] = combination_bonus

        total_score = (title_score + pl_score + oci_score + structural_score + eps_dis_score +
                       ifrs_two_statement_bonus + block_bonus + combination_bonus) # Positional bonus omitted for now

        is_potential_ifrs_pl = False
        if best_title_type == "ifrs_pl" and pl_is_strong and oci_score < 10 and not is_missing_critical_pl:
            is_potential_ifrs_pl = True

        return {
            "total": total_score, "breakdown": breakdown, "best_title_text": best_title_text,
            "best_title_type": best_title_type, "is_potential_ifrs_pl": is_potential_ifrs_pl,
            "is_missing_critical_pl": is_missing_critical_pl
        }

    def classify_soci_section(self, doc_blocks: List[Dict],
                              start_block_index: int = 0, # Index in doc_blocks to start search
                              confidence_threshold: float = 0.50, # Threshold
                              max_start_block_index_to_check: int = 500,
                              max_blocks_in_soci_window: int = 30, # Max blocks for a single statement attempt
                              debug: bool = False) -> Optional[Dict[str, Any]]:
        best_soci_candidate = None

        # Main loop to find the start of a potential SOCI section
        list_idx = start_block_index
        while list_idx < len(doc_blocks):
            start_block_candidate = doc_blocks[list_idx]
            current_doc_block_list_start_idx = list_idx # Save the list index of this attempt's start

            if start_block_candidate['index'] >= max_start_block_index_to_check:
                if debug:
                    print(f"Stopping search: Start block index {start_block_candidate['index']} >= {max_start_block_index_to_check}.")
                break

            # Skip if the start_block_candidate itself is a hard termination boundary for *another* section
            # (e.g. "Statement of Financial Position" title itself should not start an SOCI search)
            # This check is important to avoid trying to classify a different statement as SOCI.
            # However, a generic "Notes" header could be a valid start for a Notes classifier, but terminates SOCI.
            # For now, basic check: if it's a termination for SOCI, don't start an SOCI here.
            if self._is_termination_block(start_block_candidate['content'], start_block_candidate.get('type'), debug=debug) and \
               self._check_soci_titles(self._normalize_text(start_block_candidate['content']))[0] == 0: # If it's a terminator AND not an SOCI title itself
                if debug:
                    print(f"Skipping block {start_block_candidate['index']} (list idx {list_idx}) as start: It's a hard termination boundary for SOCI and not an SOCI title.")
                list_idx += 1
                continue

            current_window_blocks = []
            current_window_content_original_case = ""
            window_last_list_idx = list_idx -1 # Tracks the end of the current window attempt in doc_blocks LIST

            is_title_paragraph_start = False
            initial_title_info = None # Tuple (score, text, type)
            is_table_only_start = False

            # --- Attempt to form a candidate window ---
            block_type_of_start = start_block_candidate.get('type')

            if block_type_of_start == 'paragraph':
                normalized_start_content = self._normalize_text(start_block_candidate['content'])
                prelim_title_score, prelim_title_text, prelim_title_type = self._check_soci_titles(normalized_start_content)

                # Condition to start with a paragraph: It must be a decent title, OR very generic but followed by a strong table.
                # For now, require a decent title score from the paragraph itself.
                # A more complex logic could look ahead if title is weak.
                # Strict check for paragraph start: must have some title characteristic or indicative word.
                has_indicative_term = re.search(r'(INCOME|PROFIT|LOSS|EARNINGS|OPERATIONS|COMPREHENSIVE)', normalized_start_content)

                if prelim_title_score >= 4 or (prelim_title_score > 0 and has_indicative_term): # Min score for a para to be a title
                    if debug:
                        print(f"\nPotential SOCI start (paragraph title) at block {start_block_candidate['index']} ('{start_block_candidate['content'][:80].strip()}...'), prelim_title: '{prelim_title_text}' ({prelim_title_type}) score {prelim_title_score}")
                    current_window_blocks.append(start_block_candidate)
                    current_window_content_original_case += start_block_candidate['content'] + "\n"
                    is_title_paragraph_start = True
                    initial_title_info = (prelim_title_score, prelim_title_text, prelim_title_type)
                    window_last_list_idx = list_idx

                    # Try to append subsequent tables
                    for k_para_follow in range(list_idx + 1, min(list_idx + 1 + max_blocks_in_soci_window -1 , len(doc_blocks))):
                        block_to_add = doc_blocks[k_para_follow]
                        if self._is_termination_block(block_to_add['content'], block_to_add.get('type'), debug=debug):
                            if debug:
                                print(f"  Window (after title para) stopped by termination: Block {block_to_add['index']}")
                            break
                        if block_to_add.get('type') == 'table':
                            current_window_blocks.append(block_to_add)
                            current_window_content_original_case += block_to_add['content'] + "\n"
                            window_last_list_idx = k_para_follow
                        else: # non-table after initial table(s) for a paragraph-led SOCI
                            if debug:
                                print(f"  Window (after title para) sequence of tables stopped by non-table: Block {block_to_add['index']}")
                            #  vheck if this non-table is an IFRS OCI part later in IFRS 2-statement logic
                            break
                else:
                    if debug:
                        print(f"Skipping paragraph block {start_block_candidate['index']} as start: Low preliminary title score ({prelim_title_score}) or no key terms like INCOME/PROFIT/LOSS.")
                    pass # current_window_blocks remains empty

            # If no window started with a paragraph, OR if we want to check tables independently:
            # This allows a table itself to be the start of an SOCI if it's very characteristic
            if not current_window_blocks and block_type_of_start == 'table':
                if debug:
                    print(f"\nPotential SOCI start (table) at block {start_block_candidate['index']} ('{start_block_candidate['content'][:150].strip().replace(chr(10),' ')}...')")
                is_table_only_start = True
                # Window consists of contiguous tables
                for k_table_follow in range(list_idx, min(list_idx + max_blocks_in_soci_window, len(doc_blocks))):
                    block_to_add = doc_blocks[k_table_follow]
                    if self._is_termination_block(block_to_add['content'], block_to_add.get('type'), debug=debug):
                        if debug:
                            print(f"  Window (table-only) stopped by termination: Block {block_to_add['index']}")
                        break
                    if block_to_add.get('type') == 'table':
                        current_window_blocks.append(block_to_add)
                        current_window_content_original_case += block_to_add['content'] + "\n"
                        window_last_list_idx = k_table_follow
                    else: # Non-table block, sequence ends
                        if debug:
                            print(f"  Window (table-only) sequence of tables stopped by non-table: Block {block_to_add['index']}")
                        break

            # --- Evaluate the formed candidate window ---
            if current_window_blocks:
                final_window_content_original = current_window_content_original_case.strip()
                first_block_in_window_doc_idx = current_window_blocks[0]['index']

                # Initial scoring of the current window
                score_result = self._calculate_raw_score_soci(
                    final_window_content_original,
                    first_block_in_window_doc_idx,
                    title_info=initial_title_info,
                    is_table_only_core=is_table_only_start
                )

                # --- IFRS Two-Statement Logic ---
                if score_result["is_potential_ifrs_pl"] and score_result["total"] > 40 : # Min score for a P&L part
                    if debug:
                        print(f"  Potential IFRS P&L statement found (ends list_idx {window_last_list_idx}). Looking for OCI part...")

                    oci_window_blocks_extension = []
                    oci_window_content_extension_original = ""
                    idx_after_pl_part_in_list = window_last_list_idx + 1

                    for k_oci in range(idx_after_pl_part_in_list, min(idx_after_pl_part_in_list + 5, len(doc_blocks))):
                        oci_candidate_block = doc_blocks[k_oci]
                        if self._is_termination_block(oci_candidate_block['content'], oci_candidate_block.get('type'), debug=debug):
                            if debug:
                                print(f"    OCI search stopped: block {oci_candidate_block['index']} is termination.")
                            break

                        # Normalized content for OCI checks
                        normalized_oci_candidate_content = self._normalize_text(oci_candidate_block['content'])
                        oci_header_score, _ = self._check_oci_keywords(normalized_oci_candidate_content) # check for OCI headers/items

                        # Profit for the period carry-forward is harder to check without value parsing. Focus on OCI keywords.
                        # Heuristic: If this block has strong OCI headers or starts an OCI section
                        if oci_header_score > 5 or (len(oci_window_blocks_extension) > 0 and oci_candidate_block.get('type') == 'table'): # If already started OCI part and next is table
                            oci_window_blocks_extension.append(oci_candidate_block)
                            oci_window_content_extension_original += oci_candidate_block['content'] + "\n"
                            if debug:
                                print(f"    Added block {oci_candidate_block['index']} to IFRS OCI extension.")
                            # If OCI content in extension is substantial, break
                            temp_oci_score_ext, _ = self._check_oci_keywords(self._normalize_text(oci_window_content_extension_original))
                            if temp_oci_score_ext >= 15 : # Sufficient OCI items found in extension
                                if debug:
                                    print(f"    Sufficient OCI content found in extension (score {temp_oci_score_ext}).")
                                break
                        elif oci_window_blocks_extension : # Started extending but current block doesn't fit OCI pattern well
                            break
                        else: # First block for OCI extension not fitting
                            if debug:
                                print(f"    Block {oci_candidate_block['index']} not a strong start for OCI part (OCI score: {oci_header_score}).")
                            break

                    if oci_window_blocks_extension:
                        if debug:
                            print(f"  IFRS OCI part identified. Merging with P&L part.")
                        merged_window_blocks = current_window_blocks + oci_window_blocks_extension
                        merged_window_content_original = final_window_content_original + "\n" + oci_window_content_extension_original.strip()

                        # Rescore the merged window, passing the IFRS 2-statement flag (handled by setting bonus in breakdown)
                        breakdown_for_merged = score_result["breakdown"] # Start with original breakdown
                        breakdown_for_merged["ifrs_two_statement_bonus"] = 15 # Add bonus directly for rescore

                        # Recalculate OCI score for the merged content
                        updated_oci_score, updated_oci_kws = self._check_oci_keywords(self._normalize_text(merged_window_content_original))
                        breakdown_for_merged["oci_content"] = updated_oci_score
                        breakdown_for_merged["found_oci_keywords"] = updated_oci_kws

                        # Recalculate total with the new OCI and bonus
                        score_result["total"] = sum(v for k, v in breakdown_for_merged.items() if isinstance(v, (int, float)) and k not in ["pl_penalty_critical_items_missing"])
                        if "pl_penalty_critical_items_missing" in breakdown_for_merged: # re-apply penalty if it exists
                             score_result["total"] += breakdown_for_merged["pl_penalty_critical_items_missing"]

                        score_result["breakdown"] = breakdown_for_merged
                        score_result["is_ifrs_two_statement_confirmed"] = True # Custom flag in result
                        current_window_blocks = merged_window_blocks
                        final_window_content_original = merged_window_content_original
                        window_last_list_idx += len(oci_window_blocks_extension) # Update end of window
                # --- End of IFRS Two-Statement Logic ---

                raw_score = score_result["total"]
                breakdown_details = score_result["breakdown"]
                normalized_display_title = score_result["best_title_text"] or "Statement of Comprehensive Income (Inferred)"

                capped_score = min(raw_score, self.max_score_exemplar_soci)
                final_confidence = (capped_score / self.max_score_exemplar_soci) if self.max_score_exemplar_soci > 0 else 0.0

                if score_result["is_missing_critical_pl"] and raw_score > 10:
                    final_confidence = min(final_confidence, 0.2)
                    if debug:
                        print(f"  WARNING: Critical P&L items missing. Confidence capped for window starting at {current_window_blocks[0]['index']}.")

                window_indices_str = str([b['index'] for b in current_window_blocks])
                if debug:
                    print(f"  EVALUATING SOCI WINDOW: Blocks Doc Indices {window_indices_str} (List indices {current_doc_block_list_start_idx}-{window_last_list_idx})")
                    print(f"  Raw Score: {raw_score}, Confidence: {final_confidence:.3f}, Breakdown Sum: {sum(b for b in breakdown_details.values() if isinstance(b, (int,float)) )}")
                    print(f"  Breakdown: {breakdown_details}")
                    print(f"  Title: '{score_result['best_title_text']}' ({score_result['best_title_type']}), IFRS PL Potential: {score_result['is_potential_ifrs_pl']}")

                if final_confidence >= confidence_threshold:
                    if debug:
                        print(f"  >>> SOCI SECTION CANDIDATE MEETS THRESHOLD (Window Doc Indices {window_indices_str}), Confidence: {final_confidence:.3f}")

                    candidate_result = {
                        "section_name": "Statement of Comprehensive Income",
                        "normalized_section_name": normalized_display_title,
                        "start_block_index": current_window_blocks[0]['index'],
                        "end_block_index": current_window_blocks[-1]['index'],
                        "block_indices": [b['index'] for b in current_window_blocks],
                        "num_blocks": len(current_window_blocks),
                        "confidence": final_confidence,
                        "raw_score": raw_score,
                        "breakdown": breakdown_details,
                        "content_preview": self._normalize_text(final_window_content_original)[:300],
                        "is_ifrs_two_statement": score_result.get("is_ifrs_two_statement_confirmed", False)
                    }
                    if best_soci_candidate is None or final_confidence > best_soci_candidate["confidence"] or \
                       (abs(final_confidence - best_soci_candidate["confidence"]) < 0.01 and len(current_window_blocks) > best_soci_candidate["num_blocks"]): # Prioritize higher confidence, then more blocks
                        best_soci_candidate = candidate_result
                        if debug:
                            print(f"    Updated best_soci_candidate (Confidence: {final_confidence:.3f})")
                        # if final_confidence > 0.85: # Very high confidence, could return early
                        #     if debug: print("Very high confidence SOCI found. Returning.")
                        #     return best_soci_candidate

                # Advance main loop iterator past the blocks considered in this window attempt
                list_idx = window_last_list_idx + 1

            else: # No window blocks were formed from this start_block_candidate
                list_idx += 1

        if best_soci_candidate:
            if debug:
                print(f"\nReturning best SOCI candidate found with confidence {best_soci_candidate['confidence']:.3f}")
            return best_soci_candidate

        if debug:
            print("\nNo SOCI section identified with sufficient confidence after checking all potential windows.")
        return None

    def display_soci_results(self, soci_result: Optional[Dict[str, Any]], all_doc_blocks: Optional[List[Dict]] = None):
        """Display SOCI classification results in a clear format."""
        print("\nSOCI CLASSIFICATION RESULTS:")
        print("-" * 40)

        if soci_result:
            print(f" SOCI section identified!")
            print(f"  Section Name: {soci_result['section_name']}")
            print(f"  Normalized Title: {soci_result.get('normalized_section_name', 'N/A')}")
            print(f"  Confidence: {soci_result['confidence']:.3f}")
            print(f"  Raw score: {soci_result['raw_score']} (out of {self.max_score_exemplar_soci})")
            print(f"  Number of blocks: {soci_result['num_blocks']}")
            print(f"  Start Block Index: {soci_result['start_block_index']}")
            print(f"  End Block Index: {soci_result['end_block_index']}")
            print(f"  IFRS Two-Statement: {soci_result.get('is_ifrs_two_statement', False)}")

            print(f"\n  Score breakdown for the identified SOCI window:")
            breakdown = soci_result['breakdown']
            # ensure all expected numeric keys are present in breakdown for summation
            numeric_score_sum = 0
            keys_for_sum = ["title", "p&l_content", "oci_content", "structural_cues",
                            "eps_discontinued_ops", "ifrs_two_statement_bonus",
                            "block_bonus", "combination_bonus"] # Add other scored numeric keys if any

            for category, score_val in breakdown.items():
                if isinstance(score_val, (int, float)) and score_val != 0:
                    print(f"    • {category.replace('_', ' ').title()}: {score_val}")
                    if category in keys_for_sum : numeric_score_sum +=score_val
                elif isinstance(score_val, list) and score_val and "keywords" in category : # Show lists of keywords
                     print(f"    • {category.replace('_', ' ').title()}: {', '.join(str(s) for s in score_val[:5])}{'...' if len(score_val) > 5 else ''}")
            # print(f"    Calculated sum from breakdown (for verification): {numeric_score_sum}")


            if all_doc_blocks:
                print(f"\n  Identified SOCI content (first 3 blocks preview):")
                block_content_map = {block['index']: block['content'] for block in all_doc_blocks}
                for i, block_idx in enumerate(soci_result['block_indices']):
                    if i >= 3 and len(soci_result['block_indices']) > 3:
                        print(f"    ... and {len(soci_result['block_indices']) - 3} more blocks.")
                        break
                    content = block_content_map.get(block_idx, "Content not found.")
                    display_content = self._normalize_text(content) # Show normalized
                    if len(display_content) > 100:
                        display_content = display_content[:97] + "..."
                    print(f"    Block {block_idx}: {display_content}")
            else:
                print(f"\n  Content Preview (normalized): {soci_result['content_preview']}")
        else:
            print("\n  No SOCI section identified with sufficient confidence.")
