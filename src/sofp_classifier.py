import re
import math
from typing import Dict, List, Optional, Any

class SoFPClassifier:
    """
    Rule-based Classifier for identifying Statement of Financial Position (SoFP)
    sections in Financial Statements.
    It handles SoFPs that might be a paragraph (title) followed by table(s),
    or just table(s).
    """

    def __init__(self):
        self.keywords = self._initialize_sofp_keywords()
        # Max score exemplar (approximate, adjust as features are fully implemented):
        # BalancingEquation(5) + Title(4) + MajorSections(3) + AssetItems(3) +
        # LiabilityItems(3) + EquityItems(3) + TotalIndicators(3) + StructuralCues(2) +
        # BlockBonus(1) + CombinationBonus(2)
        self.max_score_exemplar = 29 # Max score without balancing equation
                                     # balancing_equation score is 5
                                     # title_phrases score is 4
                                     # major_sections score is 3 (1 per section, up to 3)
                                     # asset_keywords score is 3
                                     # liability_keywords score is 3
                                     # equity_keywords score is 3
                                     # total_indicator_keywords score is 3
                                     # structural_cues (comparative_periods_notes_column) score is 2
                                     # block_bonus score is 1
                                     # combination_bonus score is 2

    def _initialize_sofp_keywords(self) -> Dict[str, Any]:
        """Initialize comprehensive SoFP keyword dictionary."""
        return {
            "title_phrases": {
                "keywords": [
                    # Primary
                    "STATEMENT OF FINANCIAL POSITION", "BALANCE SHEET",
                    "CONSOLIDATED STATEMENT OF FINANCIAL POSITION", "CONSOLIDATED BALANCE SHEETS", # Typo in report corrected: "SHEETS"
                    "CONDENSED CONSOLIDATED STATEMENT OF FINANCIAL POSITION", "CONDENSED CONSOLIDATED BALANCE SHEET",
                    "CONSOLIDATED STATEMENT OF CONDITION", "STATEMENT OF CONDITION", # Often for Banks
                    # Less common / specific
                    "STATEMENT OF NET WORTH",
                    # Variations indicating interim
                    "UNAUDITED INTERIM CONDENSED CONSOLIDATED FINANCIAL STATEMENTS" # Broader, but SoFP is inside
                ],
                "score": 4, # Max score for finding a title
                "length_check_buffer": 20 # Title should not be excessively longer than keyword
            },
            "major_section_keywords": {
                # Keywords for "ASSETS", "LIABILITIES", "EQUITY" and their variants
                "assets": ["ASSETS", "TOTAL ASSETS"], # "TOTAL ASSETS" can act as a strong header
                "liabilities": ["LIABILITIES", "TOTAL LIABILITIES"],
                "equity": [
                    "EQUITY", "SHAREHOLDERS' EQUITY", "STOCKHOLDERS' EQUITY",
                    "OWNERS' EQUITY", "MEMBERS' EQUITY", "PARTNERS' CAPITAL",
                    "CAPITAL", "CAPITAL AND RESERVES", "NET ASSETS", "FUND BALANCE",
                    "EQUITY ATTRIBUTABLE TO OWNERS OF THE PARENT",
                    "TOTAL EQUITY", "TOTAL SHAREHOLDERS' EQUITY", "TOTAL STOCKHOLDERS' EQUITY"
                ],
                "combined_liabilities_equity": [
                    "LIABILITIES AND EQUITY", "LIABILITIES AND STOCKHOLDERS' EQUITY",
                    "LIABILITIES AND SHAREHOLDERS' EQUITY"
                ],
                # Score 1 for each unique major section found (Assets, Liabilities, Equity), max 3.
                "score_per_section": 1,
                "max_score": 3
            },
            "asset_keywords": { # Specific line items under Assets
                "keywords": [
                    "CASH AND CASH EQUIVALENTS", "CASH", "ACCOUNTS RECEIVABLE", "TRADE RECEIVABLES", "RECEIVABLES", "DEBTORS",
                    "INVENTORY", "INVENTORIES", "STOCKS", "MARKETABLE SECURITIES", "SHORT-TERM INVESTMENTS",
                    "PREPAID EXPENSES", "PREPAYMENTS", "OTHER CURRENT ASSETS", "PROPERTY, PLANT, AND EQUIPMENT", "PPE",
                    "FIXED ASSETS", "TANGIBLE FIXED ASSETS", "LAND", "BUILDINGS", "EQUIPMENT", "ACCUMULATED DEPRECIATION",
                    "LONG-TERM INVESTMENTS", "INVESTMENTS IN ASSOCIATES", "INTANGIBLE ASSETS", "GOODWILL", "PATENTS",
                    "TRADEMARKS", "DEFERRED TAX ASSETS", "RIGHT-OF-USE ASSETS", "LEASED ASSETS", "INVESTMENT PROPERTY",
                    "BIOLOGICAL ASSETS", "ASSETS HELD FOR SALE", "CURRENT ASSETS", "NON-CURRENT ASSETS", "LONG-TERM ASSETS",
                    "LOANS AND ADVANCES TO CUSTOMERS", "TRADING ASSETS", "CASH AND BALANCES WITH CENTRAL BANKS" # Banking
                ],
                "max_score": 3, # Score based on number of unique items found, capped
                "count_for_max_score": 5 # e.g., 5 distinct items give max score
            },
            "liability_keywords": { # Specific line items under Liabilities
                "keywords": [
                    "ACCOUNTS PAYABLE", "TRADE PAYABLES", "CREDITORS", "SHORT-TERM DEBT", "SHORT-TERM BORROWINGS",
                    "NOTES PAYABLE", "ACCRUED EXPENSES", "ACCRUED LIABILITIES", "UNEARNED REVENUE", "DEFERRED REVENUE",
                    "INCOME TAXES PAYABLE", "CURRENT TAX PAYABLE", "CURRENT PORTION OF LONG-TERM DEBT",
                    "DIVIDENDS PAYABLE", "OTHER CURRENT LIABILITIES", "LONG-TERM DEBT", "BONDS PAYABLE",
                    "DEFERRED TAX LIABILITIES", "PENSION OBLIGATIONS", "LEASE OBLIGATIONS", "PROVISIONS",
                    "CURRENT LIABILITIES", "NON-CURRENT LIABILITIES", "LONG-TERM LIABILITIES",
                    "DEPOSITS FROM CUSTOMERS", "DEBT SECURITIES IN ISSUE", # Banking
                    "INSURANCE CONTRACT LIABILITIES" # Insurance
                ],
                "max_score": 3,
                "count_for_max_score": 5
            },
            "equity_keywords": { # Specific line items under Equity
                "keywords": [
                    "SHARE CAPITAL", "COMMON STOCK", "PREFERRED STOCK", "CAPITAL STOCK", "ISSUED CAPITAL",
                    "ADDITIONAL PAID-IN CAPITAL", "SHARE PREMIUM", "RETAINED EARNINGS", "ACCUMULATED PROFIT",
                    "ACCUMULATED DEFICIT", "REVALUATION RESERVE", "HEDGING RESERVE",
                    "FOREIGN CURRENCY TRANSLATION RESERVE", "OTHER RESERVES",
                    "ACCUMULATED OTHER COMPREHENSIVE INCOME", "AOCI",
                    "TREASURY STOCK", "TREASURY SHARES", "NON-CONTROLLING INTERESTS", "NCI", "MINORITY INTEREST",
                    "OWNER'S CAPITAL ACCOUNT", "PARTNER'S CAPITAL ACCOUNT"
                ],
                "max_score": 3,
                "count_for_max_score": 4 # Fewer distinct items typically in Equity main section
            },
            "total_indicator_keywords": {
                "keywords": [
                    "TOTAL ASSETS", "TOTAL LIABILITIES", "TOTAL EQUITY", "TOTAL SHAREHOLDERS' EQUITY",
                    "TOTAL STOCKHOLDERS' EQUITY", "TOTAL NET ASSETS", "TOTAL CAPITAL AND RESERVES",
                    "TOTAL LIABILITIES AND EQUITY", "TOTAL LIABILITIES AND SHAREHOLDERS' EQUITY",
                    "TOTAL LIABILITIES AND STOCKHOLDERS' EQUITY",
                    "TOTAL CURRENT ASSETS", "TOTAL NON-CURRENT ASSETS", "TOTAL LONG-TERM ASSETS",
                    "TOTAL CURRENT LIABILITIES", "TOTAL NON-CURRENT LIABILITIES", "TOTAL LONG-TERM LIABILITIES"
                ],
                "critical_totals_for_balance_check": ["TOTAL ASSETS", "TOTAL LIABILITIES", "TOTAL EQUITY", "TOTAL LIABILITIES AND EQUITY"],
                "max_score": 3, # Score based on number of unique critical totals found
                "count_for_max_score": 3 # Finding 3 distinct (sub)totals
            },
            "structural_cues": {
                "comparative_year_pattern": r'\b(19|20)\d{2}\b', # Find at least two such years
                "note_column_keywords": [r'\bNOTE[S]?\b', r'\bREF\.?\b'],
                "currency_indicators": [
                    r'\$\s*\d+', r'€\s*\d+', r'£\s*\d+', r'¥\s*\d+', # Common currency symbols near numbers
                    r'\(\s*IN\s+(THOUSANDS|MILLIONS|BILLIONS)\s*\)', # (in thousands)
                    r'\b(USD|EUR|GBP|CAD|AUD)\b' # Common ISO codes
                ],
                "score": 2 # 1 for comparative years, 1 for note/currency
            },
            "hard_termination_section_starters": {
                "keywords": [
                    # Next primary face statement titles
                    "STATEMENT OF INCOME", "INCOME STATEMENT", "STATEMENT OF EARNINGS",
                    "STATEMENT OF OPERATIONS", "PROFIT AND LOSS ACCOUNT", "STATEMENT OF PROFIT OR LOSS",
                    "CONSOLIDATED STATEMENT OF PROFIT OR LOSS",
                    "CONSOLIDATED STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME",
                    "STATEMENT OF COMPREHENSIVE INCOME", "STATEMENT OF OTHER COMPREHENSIVE INCOME",
                    "CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME",
                    "STATEMENT OF CASH FLOWS", "CASH FLOW STATEMENT", "CONSOLIDATED STATEMENT OF CASH FLOWS",
                    "STATEMENT OF CHANGES IN EQUITY", "STATEMENT OF SHAREHOLDERS' EQUITY",
                    "STATEMENT OF STOCKHOLDERS' EQUITY", "STATEMENT OF RETAINED EARNINGS",
                    "CONSOLIDATED STATEMENT OF CHANGES IN EQUITY",

                    # Main "Notes to Financial Statements" section header (now a hard stop for SoFP section)
                    "NOTES TO FINANCIAL STATEMENTS", "NOTES TO THE FINANCIAL STATEMENTS",
                    "NOTES TO THE CONSOLIDATED FINANCIAL STATEMENTS",
                    "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
                    "ACCOMPANYING NOTES TO THE FINANCIAL STATEMENTS", # Common variation

                    # Other major distinct report sections
                    "INDEPENDENT AUDITOR'S REPORT", "AUDITOR'S REPORT", "REPORT OF INDEPENDENT AUDITOR",
                    "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM",
                    "MANAGEMENT DISCUSSION AND ANALYSIS", "MD&A", "MANAGEMENT'S DISCUSSION AND ANALYSIS",
                    "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
                    "REPORT OF THE DIRECTORS", "DIRECTORS' REPORT", "CHAIRMAN'S STATEMENT", "CEO REPORT",
                    "STATUTORY DECLARATION"
                ],
                "regex_patterns": [
                    # More robust regex for key section titles
                    r"STATEMENTS?\s+OF\s+(FINANCIAL\s+POSITION|CONDITION)\b", # To avoid self-terminating on SoFP variations if used broadly
                    r"STATEMENTS?\s+OF\s+(OPERATIONS|EARNINGS|INCOME)\b",
                    r"STATEMENTS?\s+OF\s+COMPREHENSIVE\s+INCOME\b",
                    r"STATEMENTS?\s+OF\s+CASH\s+FLOWS?\b",
                    r"STATEMENTS?\s+OF\s+CHANGES\s+IN\s+(SHAREHOLDERS['’]?\s+|STOCKHOLDERS['’]?\s+)?EQUITY\b",
                    r"NOTES\s+TO\s+(THE\s+)?(CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS\b",
                    r"^\s*(?:NOTE\s+)?(?:\d+|(?:\([a-zA-Z0-9]+\)))\s*[\.:\-‐]?\s*(?:[A-Z][A-Za-z\s,'&\(\)\/\-]{2,})"
                ],
                "content_indicator_regex_for_period_statements": [ # NEW SUBCATEGORY
                    r"FOR\s+THE\s+YEAR(S)?\s+ENDED",
                    r"YEAR(S)?\s+ENDED",  # This will catch "Years ended December 31,"
                    r"FOR\s+THE\s+PERIOD\s+ENDED",
                    r"PERIOD\s+ENDED",
                    r"FOR\s+THE\s+\w+\s+MONTHS?\s+ENDED" # e.g., "FOR THE THREE MONTHS ENDED"
                ]
            }
        }

    # Classify the SoFP section in the document
    def _is_hard_termination_block(self, text_content: str, block_type: Optional[str] = None, debug: bool = False) -> bool:
        """
        Checks if the block content signals a "hard stop" indicating the start of a
        new primary statement or another major distinct report section.
        """
        text_upper_normalized = self._normalize_text(text_content) # Normalize for robust matching

        # 1. Check keywords (explicit titles of other statements, "NOTES TO FINANCIAL STATEMENTS" main header, etc.)
        for kw in self.keywords["hard_termination_section_starters"]["keywords"]:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_upper_normalized):
                if debug: print(f"      DEBUG[_is_hard_termination_block]: Matched keyword '{kw}'")
                return True

        # 2. Check regex patterns (more complex titles, numbered notes for main Notes section start)
        for pattern_str in self.keywords["hard_termination_section_starters"].get("regex_patterns", []):
            if re.search(pattern_str, text_upper_normalized, re.IGNORECASE):
                if debug: print(f"      DEBUG[_is_hard_termination_block]: Matched regex_pattern '{pattern_str}'")
                return True

        # 3. NEW: Check for strong content indicators of a period-based statement (primarily for tables)
        # This helps catch untitled period-based statements like an Income Statement based on header content.
        # Block 10 (`| | Years ended December 31,...`) is a 'table'.
        if block_type == 'table': # Apply this check specifically to tables for now
            for pattern_str in self.keywords["hard_termination_section_starters"].get("content_indicator_regex_for_period_statements", []):
                # Search for these phrases; they don't need strict word boundaries if part of a line
                if re.search(pattern_str, text_upper_normalized):
                    # Heuristic: if it's a period statement indicator, and we also see multiple years in the table content, it's a strong signal.
                    year_pattern_hdr = r'\b(19|20)\d{2}\b' # Same as in structural_cues
                    years_found_hdr = re.findall(year_pattern_hdr, text_upper_normalized)

                    # If a strong period phrase like "FOR THE YEAR(S) ENDED" or "YEAR(S) ENDED" is found in a table header context
                    # and at least one year is present, it's a good indicator.
                    if len(years_found_hdr) >= 1: # Check for at least one year
                        if debug: print(f"      DEBUG[_is_hard_termination_block]: Matched period content indicator '{pattern_str}' AND >=1 year(s) in a table.")
                        return True
        return False

    def _normalize_text(self, text: str) -> str:
        # Simple normalization: uppercase and replace multiple spaces/newlines
        text = text.upper()
        text = re.sub(r'\s+', ' ', text)
        return text

    def _check_keywords_in_text(self, text_upper_normalized: str, keyword_list: List[str], is_regex_pattern: bool = False) -> List[str]:
        found = []
        for keyword in keyword_list:
            pattern = keyword if is_regex_pattern else r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_upper_normalized):
                found.append(keyword)
        return found

    def _check_sofp_title_phrases(self, text_upper: str) -> int:
        """Checks for SoFP titles, considers length heuristic."""
        phrases = self.keywords["title_phrases"]["keywords"]
        buffer = self.keywords["title_phrases"]["length_check_buffer"]
        for phrase in phrases:
            # Exact phrase match
            pattern = r'\b' + re.escape(phrase) + r'\b'
            match = re.search(pattern, text_upper)
            if match:
                # Check length: matched text part should not be excessively longer than the phrase itself
                # This is tricky if the title is part of a longer heading.
                # A simpler check: if the entire block text is not too long relative to the title.
                if len(text_upper) < (len(phrase) + buffer):
                    return self.keywords["title_phrases"]["score"]
        return 0

    def _check_major_sections(self, text_upper_normalized: str) -> int:
        """Checks for major section headers: Assets, Liabilities, Equity."""
        found_sections = set()
        max_score = self.keywords["major_section_keywords"]["max_score"]
        score_per_section = self.keywords["major_section_keywords"]["score_per_section"]
        score = 0

        if self._check_keywords_in_text(text_upper_normalized, self.keywords["major_section_keywords"]["assets"]):
            found_sections.add("assets")
        if self._check_keywords_in_text(text_upper_normalized, self.keywords["major_section_keywords"]["liabilities"]):
            found_sections.add("liabilities")
        if self._check_keywords_in_text(text_upper_normalized, self.keywords["major_section_keywords"]["equity"]):
            found_sections.add("equity")

        # If combined "Liabilities and Equity" is found, it implies both.
        if self._check_keywords_in_text(text_upper_normalized, self.keywords["major_section_keywords"]["combined_liabilities_equity"]):
            found_sections.add("liabilities")
            found_sections.add("equity")

        score = len(found_sections) * score_per_section
        return min(score, max_score)

    def _count_and_score_items(self, text_upper_normalized: str, item_category: str) -> int:
        """Generic function to count unique keywords for assets, liabilities, equity, totals."""
        keywords = self.keywords[item_category]["keywords"]
        max_score = self.keywords[item_category]["max_score"]
        count_for_max_score = self.keywords[item_category]["count_for_max_score"]

        found_items = self._check_keywords_in_text(text_upper_normalized, keywords)
        unique_found_count = len(set(found_items))

        if unique_found_count == 0:
            return 0
        if unique_found_count >= count_for_max_score:
            return max_score
        # Scale score linearly for counts between 1 and count_for_max_score
        return max(1, int(round((unique_found_count / count_for_max_score) * max_score)))


    def _check_structural_cues(self, text_upper_normalized: str) -> int:
        """Checks for comparative years, note column keywords, and currency indicators."""
        score = 0
        # Check for comparative years (at least two distinct years)
        year_pattern = self.keywords["structural_cues"]["comparative_year_pattern"]
        years_found = re.findall(year_pattern, text_upper_normalized)
        if len(set(years_found)) >= 2:
            score += 1

        # Check for note column keywords or currency indicators
        note_keywords_found = any(
            self._check_keywords_in_text(text_upper_normalized, [pattern], is_regex_pattern=True)
            for pattern in self.keywords["structural_cues"]["note_column_keywords"]
        )
        currency_indicators_found = any(
            self._check_keywords_in_text(text_upper_normalized, [pattern], is_regex_pattern=True)
            for pattern in self.keywords["structural_cues"]["currency_indicators"]
        )

        if note_keywords_found or currency_indicators_found:
            score +=1

        return min(score, self.keywords["structural_cues"]["score"]) # Max score for structural cues

    def _parse_financial_number(self, value_str: str, debug: bool = False) -> Optional[float]:
        """
        Parses a string representing a financial number (handles commas and parentheses for negatives).
        """
        if debug:
            print(f"      DEBUG[_parse_financial_number]: Parsing value string: '{value_str}'")
        if not value_str:
            return None

        cleaned_value = str(value_str).strip().replace(',', '')
        if debug:
            print(f"      DEBUG[_parse_financial_number]: Cleaned value: '{cleaned_value}'")
        is_negative = False

        if cleaned_value.startswith('(') and cleaned_value.endswith(')'):
            is_negative = True
            cleaned_value = cleaned_value[1:-1]
            if debug:
                print(f"      DEBUG[_parse_financial_number]: Detected negative value in parentheses: '{cleaned_value}'")

        # Check if it's just a standalone hyphen or empty after stripping
        if cleaned_value == '-' or not cleaned_value:
            return None # Or handle as zero if appropriate for your context like '-' representing nil

        # Ensure there's at least one digit if it's not a hyphen representing nil
        if not re.search(r'\d', cleaned_value):
            return None

        try:
            number = float(cleaned_value)
            if is_negative:
                number = -number

            if debug:
                print(f"      DEBUG[_parse_financial_number]: Parsed number: {number}")
            return number
        except ValueError:
            return None

    def _find_value_for_label(self, text_content_upper_normalized: str, label_variants: List[str], year_column_idx: int = 0, debug: bool = False) -> Optional[float]:
        """
        Finds a numerical value associated with given label variants in the text.
        It assumes the target number is the `year_column_idx`-th number found on the line after the label.
        year_column_idx = 0 usually means the most recent year's figure.
        """
        for label in label_variants:
            # regex to find the line containing the label (case-insensitive due to pre-normalized text).
            # assume label is at/near the beginning of a line.
            line_pattern = re.compile(r"^\s*" + re.escape(label) + r"\b(.*)$", re.MULTILINE) # Ensure label is whole word
            match = line_pattern.search(text_content_upper_normalized)

            if match:
                rest_of_line = match.group(1)
                if debug:
                    print(f"      DEBUG[_find_value_for_label]: Found line for '{label}': '{rest_of_line[:100]}'")

                # find all sequences that look like numbers or parts of numbers (digits, commas, dots, parens, hyphen for negative)
                potential_value_strings = re.findall(r"[\(\)\d,\.\-]+", rest_of_line)
                if debug:
                    print(f"      DEBUG[_find_value_for_label]: Potential value strings: {potential_value_strings}")


                parsed_numbers_on_line = []
                for s_val in potential_value_strings:
                    # filter out strings that are likely note references (e.g., single/double digits not part of a larger number)
                    # or just punctuation.
                    # A number should have at least two digits, or one digit if it's part of a decimal or a larger structure.
                    #
                    # this heuristic tries to avoid small note numbers if they are isolated.
                    # e.g. for "Trade and other receivables,8,"13,260","10,928" -> "8" should be skipped.
                    # A string like "1,234" will pass. A string like "(500)" will pass. "8" might be skipped.
                    # If s_val contains multiple digits or a structure like (digits) or digits,digits
                    # it's more likely a financial value.
                    if not (re.search(r"\d.*\d", s_val) or (s_val.count('(') + s_val.count(')') + s_val.count(',')) > 0 or len(s_val) > 2) and s_val.strip('-').isdigit() and len(s_val.strip('-')) <= 2 :
                        if debug: print(f"      DEBUG[_find_value_for_label]: Skipping potential note ref/small isolated number: '{s_val}'")
                        continue

                    num = self._parse_financial_number(s_val, debug=debug)
                    if debug:
                        print(f"      DEBUG[_find_value_for_label]: Parsing '{s_val}' -> {num}")
                    if num is not None:
                        parsed_numbers_on_line.append(num)

                if debug:
                    print(f"      DEBUG[_find_value_for_label]: Parsed numbers from line for '{label}': {parsed_numbers_on_line}")

                if len(parsed_numbers_on_line) > year_column_idx:
                    # example: Label,,Val2023,Val2022. parsed_numbers_on_line = [Val2023_num, Val2022_num]
                    # year_column_idx = 0 gives Val2023_num.
                    return parsed_numbers_on_line[year_column_idx]
                # If only one number found on the line and we are looking for the first column (most recent year)
                elif year_column_idx == 0 and len(parsed_numbers_on_line) == 1:
                    return parsed_numbers_on_line[0]

        if debug:
            print(f"      DEBUG[_find_value_for_label]: Value not found for labels '{label_variants}' (year_column_idx {year_column_idx}).")
        return None

    def _check_balancing_equation(self, raw_text_content: str, debug: bool = False) -> int:
        if not raw_text_content:
            return 0

        # uppercase the text but keep newlines for line-based regex
        text_content_for_search = raw_text_content.upper()

        if debug:
            # show the normalized-for-display version
            normalized_preview = self._normalize_text(raw_text_content[:300])
            if debug:
                print(f"    DEBUG[_check_balancing_equation]: Checking content (first 300 chars, display normalized): {normalized_preview.replace(chr(10),' ')}")

        BALANCE_EQUATION_SCORE = 5
        REL_TOL = 1e-5
        ABS_TOL = 0.51

        total_assets_labels = ["TOTAL ASSETS"]
        total_liabilities_labels = ["TOTAL LIABILITIES"]
        total_equity_labels = [
            "TOTAL SHAREHOLDERS’ EQUITY (DEFICIENCY)", "TOTAL SHAREHOLDERS' EQUITY (DEFICIENCY)",
            "TOTAL SHAREHOLDERS’ EQUITY", "TOTAL SHAREHOLDERS' EQUITY",
            "TOTAL STOCKHOLDERS’ EQUITY (DEFICIENCY)", "TOTAL STOCKHOLDERS' EQUITY (DEFICIENCY)",
            "TOTAL STOCKHOLDERS’ EQUITY", "TOTAL STOCKHOLDERS' EQUITY",
            "TOTAL EQUITY (DEFICIENCY)", "TOTAL EQUITY"
        ]
        total_liab_equity_labels = [
            "TOTAL LIABILITIES AND SHAREHOLDERS’ EQUITY (DEFICIENCY)", "TOTAL LIABILITIES AND SHAREHOLDERS' EQUITY (DEFICIENCY)",
            "TOTAL LIABILITIES AND SHAREHOLDERS’ EQUITY", "TOTAL LIABILITIES AND SHAREHOLDERS' EQUITY",
            "TOTAL LIABILITIES AND STOCKHOLDERS’ EQUITY (DEFICIENCY)", "TOTAL LIABILITIES AND STOCKHOLDERS' EQUITY (DEFICIENCY)",
            "TOTAL LIABILITIES AND STOCKHOLDERS’ EQUITY", "TOTAL LIABILITIES AND STOCKHOLDERS' EQUITY",
            "TOTAL LIABILITIES AND EQUITY (DEFICIENCY)", "TOTAL LIABILITIES AND EQUITY"
        ]

        # pass the text_content_for_search (uppercase, newlines preserved) to _find_value_for_label
        assets = self._find_value_for_label(text_content_for_search, total_assets_labels, 0, debug=debug)

        total_liab_and_equity = self._find_value_for_label(text_content_for_search, total_liab_equity_labels, 0, debug=debug)

        if assets is not None and total_liab_and_equity is not None:
            if debug:
                print(f"    DEBUG[_check_balancing_equation]: Option 1 Check -> Assets: {assets}, Total L+E (combined): {total_liab_and_equity}")
            if math.isclose(assets, total_liab_and_equity, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                if debug: print(f"    DEBUG[_check_balancing_equation]: Balances! (Assets vs Total L+E combined). Score: {BALANCE_EQUATION_SCORE}")
                return BALANCE_EQUATION_SCORE
            elif debug:
                print(f"    DEBUG[_check_balancing_equation]: Mismatch (Assets vs Total L+E combined). Diff: {abs(assets - total_liab_and_equity)}")

        liabilities = self._find_value_for_label(text_content_for_search, total_liabilities_labels, 0, debug=debug)
        equity = self._find_value_for_label(text_content_for_search, total_equity_labels, 0, debug=debug)

        if assets is not None and liabilities is not None and equity is not None:
            sum_liab_equity = liabilities + equity
            if debug:
                print(f"    DEBUG[_check_balancing_equation]: Option 2 Check -> Assets: {assets}, Liabilities: {liabilities}, Equity: {equity}, Sum L+E: {sum_liab_equity}")
            if math.isclose(assets, sum_liab_equity, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                if debug:
                    print(f"    DEBUG[_check_balancing_equation]: Balances! (Assets vs L + E separate). Score: {BALANCE_EQUATION_SCORE}")
                return BALANCE_EQUATION_SCORE
            elif debug:
                print(f"    DEBUG[_check_balancing_equation]: Mismatch (Assets vs L + E separate). Diff: {abs(assets - sum_liab_equity)}")

        if debug:
            print(f"    DEBUG[_check_balancing_equation]: Equation does not balance or required values not found. Score: 0")
        return 0

    def _calculate_raw_score(self, combined_text: str, first_block_index: int,
                             is_title_paragraph_present: bool, num_table_blocks: int) -> Dict[str, Any]:
        """Calculate raw score for combined text, considering if it's a title + table scenario."""
        if not combined_text.strip():
            return {"total": 0, "breakdown": {}}

        text_upper = combined_text.upper() # For title check that doesn't use normalization yet
        text_upper_normalized = self._normalize_text(combined_text) # For most keyword checks

        title_score = 0
        # if the first block was a paragraph AND it's being considered as a title for subsequent tables
        # the actual title score for the paragraph part of a combined window
        # could be pre-calculated or derived from is_title_paragraph_present
        if is_title_paragraph_present: # Assume the first block's content (if para) contributed to title
             # re-check title on the combined text, or use a passed-in score for the title paragraph.
             # for simplicity, we can infer that a title was found if is_title_paragraph_present is true.
             # a more accurate way would be to score the title part separately.
             # let's assume if is_title_paragraph_present, we assign some title points.
             # the _check_sofp_title_phrases should be called on the title paragraph itself.
             # here we are scoring the *entire window*.
             # a strong title found in a paragraph preceding tables is valuable.
             # we'll use the is_title_paragraph_present to add to combination_bonus later.
             # the actual text of the title is already in combined_text if it was a Paragraph->Table window.
             title_score = self._check_sofp_title_phrases(text_upper)


        major_sections_score = self._check_major_sections(text_upper_normalized)
        asset_items_score = self._count_and_score_items(text_upper_normalized, "asset_keywords")
        liability_items_score = self._count_and_score_items(text_upper_normalized, "liability_keywords")
        equity_items_score = self._count_and_score_items(text_upper_normalized, "equity_keywords")
        total_indicators_score = self._count_and_score_items(text_upper_normalized, "total_indicator_keywords")
        structural_cues_score = self._check_structural_cues(text_upper_normalized)
        balancing_equation_score = self._check_balancing_equation(combined_text)

        block_bonus = 1 if first_block_index < 60 else 0 # SoFP can be a bit deeper than cover page

        combination_bonus = 0
        # bonus if a title paragraph is confirmed and followed by actual table content
        if is_title_paragraph_present and num_table_blocks > 0 and title_score > 0:
            combination_bonus += 1
        # bonus for strong table characteristics
        if major_sections_score >= 2 and total_indicators_score >= 1 and structural_cues_score >=1 and num_table_blocks > 0:
            combination_bonus += 1
        combination_bonus = min(combination_bonus, 2) # Max 2 for combination

        total_score = (title_score + major_sections_score + asset_items_score +
                       liability_items_score + equity_items_score + total_indicators_score +
                       structural_cues_score + balancing_equation_score +
                       block_bonus + combination_bonus)

        breakdown = {
            "title": title_score, "major_sections": major_sections_score,
            "asset_items": asset_items_score, "liability_items": liability_items_score,
            "equity_items": equity_items_score, "total_indicators": total_indicators_score,
            "structural_cues": structural_cues_score, "balancing_equation": balancing_equation_score,
            "block_bonus": block_bonus, "combination_bonus": combination_bonus
        }
        return {"total": total_score, "breakdown": breakdown}

    def classify_sofp_section(self, doc_blocks: List[Dict],
                              start_block_index: int = 0,
                              confidence_threshold: float = 0.6,
                              max_start_block_index_to_check: int = 600,
                              debug: bool = False) -> Optional[Dict[str, Any]]:
        # This is the index for iterating through the doc_blocks LIST
        i = start_block_index
        while i < len(doc_blocks):
            start_block_candidate = doc_blocks[i]
            # keep track of the start of this attempt
            current_doc_block_list_idx = i

            if start_block_candidate['index'] >= max_start_block_index_to_check:
                if debug:
                    print(f"Stopping search: Start block index {start_block_candidate['index']} >= max_start_block_index_to_check ({max_start_block_index_to_check}).")
                break

            # skip if the start_block_candidate itself is a hard termination boundary
            if self._is_hard_termination_block(start_block_candidate['content'], start_block_candidate.get('type'), debug=debug):
                if debug:
                    print(f"Skipping block {start_block_candidate['index']} (list idx {current_doc_block_list_idx}) as start: It's a hard termination boundary.")
                i += 1
                continue

            # --- stage 1: Identify a potential CORE SoFP Candidate ---
            current_core_sofp_blocks = []
            current_core_sofp_content = ""
            is_potential_title_paragraph_for_core = False
            num_table_blocks_in_core = 0

            # this index will track the end of the core SoFP within the doc_blocks LIST
            core_window_last_list_idx = current_doc_block_list_idx -1

            block_type_of_start_candidate = start_block_candidate.get('type')

            if block_type_of_start_candidate == 'paragraph':
                temp_title_score = self._check_sofp_title_phrases(start_block_candidate['content'].upper())
                # heuristic for candidate title
                if temp_title_score >= self.keywords["title_phrases"]["score"] * 0.5:
                    current_core_sofp_blocks.append(start_block_candidate)
                    current_core_sofp_content += start_block_candidate['content'] + "\n"
                    is_potential_title_paragraph_for_core = True
                    core_window_last_list_idx = current_doc_block_list_idx

                    start_table_list_idx = current_doc_block_list_idx + 1
                    if start_table_list_idx < len(doc_blocks):
                        for k_core in range(start_table_list_idx, len(doc_blocks)):
                            core_block_to_add = doc_blocks[k_core]
                            # core stops if it hits a hard terminator OR if it's no longer a table
                            if self._is_hard_termination_block(core_block_to_add['content'], core_block_to_add.get('type'), debug=debug):
                                if debug:
                                    print(f"  Core SoFP (after para) table sequence stopped by HARD termination: Block {core_block_to_add['index']}")
                                break
                            if core_block_to_add.get('type') == 'table':
                                current_core_sofp_blocks.append(core_block_to_add)
                                current_core_sofp_content += core_block_to_add['content'] + "\n"
                                num_table_blocks_in_core += 1
                                core_window_last_list_idx = k_core
                            else: # not a table, so core SoFP table sequence ends
                                if debug: print(f"  Core SoFP (after para) table sequence stopped by non-table: Block {core_block_to_add['index']}")
                                break
                    if is_potential_title_paragraph_for_core and num_table_blocks_in_core == 0:
                        # discard if title para has no tables
                        current_core_sofp_blocks = []
                # if paragraph wasn't a good title, current_core_sofp_blocks remains empty.

            elif block_type_of_start_candidate == 'table':
                for k_core in range(current_doc_block_list_idx, len(doc_blocks)):
                    core_block_to_add = doc_blocks[k_core]
                    # core stops if it hits a hard terminator OR if it's no longer a table
                    if self._is_hard_termination_block(core_block_to_add['content'], core_block_to_add.get('type'), debug=debug):
                        if debug:
                            print(f"  Core SoFP (table-only) sequence stopped by HARD termination: Block {core_block_to_add['index']}")
                        break
                    if core_block_to_add.get('type') == 'table':
                        current_core_sofp_blocks.append(core_block_to_add)
                        current_core_sofp_content += core_block_to_add['content'] + "\n"
                        num_table_blocks_in_core += 1
                        core_window_last_list_idx = k_core
                    else: # not a table, so core SoFP table sequence ends
                        if debug:
                            print(f"  Core SoFP (table-only) sequence stopped by non-table: Block {core_block_to_add['index']}")
                        break

            # --- Evaluate the CORE SoFP Candidate ---
            next_i_candidate = current_doc_block_list_idx + 1 # Default next starting point

            if current_core_sofp_blocks: # If any blocks were collected for a potential core SoFP
                final_core_sofp_content = current_core_sofp_content.strip()
                first_block_in_core_doc_idx = current_core_sofp_blocks[0]['index']

                score_result = self._calculate_raw_score(
                    final_core_sofp_content,
                    first_block_in_core_doc_idx,
                    is_potential_title_paragraph_for_core,
                    num_table_blocks_in_core
                )
                raw_core_score = score_result["total"]
                core_breakdown = score_result["breakdown"]
                capped_core_score = min(raw_core_score, self.max_score_exemplar)
                final_core_confidence = (capped_core_score / self.max_score_exemplar) if self.max_score_exemplar > 0 else 0.0

                core_indices_str = str([b['index'] for b in current_core_sofp_blocks])
                if debug:
                    print(f"\nEVALUATING CORE SoFP: Blocks {core_indices_str} (List indices up to {core_window_last_list_idx})")
                    print(f"  Core Content Preview: '{final_core_sofp_content[:150].replace(chr(10), ' ')}...'")
                    print(f"  Core Raw Score: {raw_core_score}, Core Confidence: {final_core_confidence:.3f}")

                if final_core_confidence >= confidence_threshold:
                    if debug:
                        print(f"  Core SoFP QUALIFIES (Confidence {final_core_confidence:.3f}). Proceeding to expand.")

                    expanded_window_blocks = list(current_core_sofp_blocks)
                    expanded_window_content = str(final_core_sofp_content)

                    # --- stage 2: Expand the CORE SoFP with Notes and other subsequent content ---
                    idx_after_core_sofp_in_list = core_window_last_list_idx + 1

                    if idx_after_core_sofp_in_list < len(doc_blocks):
                        for j_expand in range(idx_after_core_sofp_in_list, len(doc_blocks)):
                            block_to_add_for_expansion = doc_blocks[j_expand]
                            if self._is_hard_termination_block(block_to_add_for_expansion['content'], block_to_add_for_expansion.get('type'), debug=debug):
                                if debug:
                                    print(f"  Expansion stopped by HARD termination boundary: Block {block_to_add_for_expansion['index']} ('{block_to_add_for_expansion['content'][:60].strip().replace(chr(10), ' ')}...')")
                                break
                            else:
                                if debug:
                                    print(f"  Expanding with block {block_to_add_for_expansion['index']} ('{block_to_add_for_expansion['content'][:60].strip().replace(chr(10), ' ')}...')")
                                expanded_window_blocks.append(block_to_add_for_expansion)
                                expanded_window_content += "\n" + block_to_add_for_expansion['content']

                    final_expanded_content = expanded_window_content.strip()

                    expanded_indices_str = str([b['index'] for b in expanded_window_blocks])
                    if debug:
                        print(f"  >>> EXPANDED SoFP SECTION CANDIDATE (Window): Blocks {expanded_indices_str}")
                        print(f"      Based on Core Confidence: {final_core_confidence:.3f}")

                    result = {
                        "section_name": "Statement of Financial Position",
                        "start_block_index": expanded_window_blocks[0]['index'],
                        "end_block_index": expanded_window_blocks[-1]['index'],
                        "block_indices": [b['index'] for b in expanded_window_blocks],
                        "num_blocks": len(expanded_window_blocks),
                        "confidence": final_core_confidence,
                        "raw_score": raw_core_score,
                        "breakdown": core_breakdown,
                        "content_preview": final_expanded_content[:300].strip(),
                        "full_content": final_expanded_content
                    }
                    return result # Return the first qualifying expanded section
                else:
                    if debug:
                        print(f"  Core SoFP (Blocks {core_indices_str}) did NOT qualify (Confidence {final_core_confidence:.3f}).")
                    # If core was attempted but didn't qualify, advance past the blocks considered for this core.
                    next_i_candidate = core_window_last_list_idx + 1

            i = next_i_candidate # Advance outer loop counter

        if debug: print("\nNo SoFP (core meeting threshold for expansion) identified after checking all potential windows.")
        return None

    def display_sofp_results(self, sofp_result: Optional[Dict[str, Any]], all_doc_blocks: Optional[List[Dict]] = None):
        """Display SoFP classification results."""
        print("\nSoFP CLASSIFICATION RESULTS:")
        print("-" * 40)

        if sofp_result:
            print(f" Statement of Financial Position identified!")
            print(f" Section: {sofp_result['section_name']}")
            print(f" Confidence: {sofp_result['confidence']:.3f}")
            print(f" Raw score: {sofp_result['raw_score']}")
            print(f" Number of blocks: {sofp_result['num_blocks']}")
            print(f" Start Block Index: {sofp_result['start_block_index']}")
            print(f" End Block Index: {sofp_result['end_block_index']}")
            print(f" Block Indices: {sofp_result['block_indices']}")

            print(f"\n Score breakdown:")
            breakdown = sofp_result['breakdown']
            for category, score in breakdown.items():
                if score > 0:
                    print(f"   • {category.replace('_', ' ').title()}: +{score}")

            print(f"\n Content Preview (first 300 chars):")
            print(sofp_result['content_preview'])
            # Optionally print full content or block-by-block if all_doc_blocks provided
        else:
            print("\n No Statement of Financial Position identified with sufficient confidence.")
            print("  Suggestions:")
            print("   - Adjust `confidence_threshold`.")
            print("   - Verify `max_start_block_index_to_check`.")
            print("   - Ensure document parser correctly extracts blocks with 'index', 'content', and 'type' ('paragraph', 'table').")
            print("   - Run with `debug=True` for detailed logs.")
