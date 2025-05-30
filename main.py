import os
import logging
import argparse
from typing import Dict, List, Optional, Any
from docx_parser import DocxParser
from utils.section_generator import SectionGenerator
from classifiers.cover_page_classifier import CoverPageClassifier
from classifiers.sofp_classifier import SoFPClassifier
from classifiers.soci_classifier import SoCIClassifier
from classifiers.financial_notes_classifier import FinancialNotesClassifier

# Logger for this module (main script)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Function Definitions for Each Processing Stage ---
def setup_logging(debug_mode_enabled: bool) -> None:
    """Configures the global logging settings."""
    log_level = logging.DEBUG if debug_mode_enabled else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info(f"Logging initialized. Debug mode: {debug_mode_enabled}")

def load_and_parse_docx(filepath: str, debug_mode_enabled: bool) -> Optional[List[Dict[str, Any]]]:
    """Loads and parses the DOCX file into meaningful blocks."""
    logger.info(f"Processing document: {filepath}")
    logger.debug("=" * 60)

    parser = DocxParser()
    meaningful_blocks = parser.parse_document(filepath)

    if not meaningful_blocks:
        logger.warning("No meaningful blocks found or error processing document.")
        return None

    logger.info(f"Found {len(meaningful_blocks)} meaningful blocks from parser.")

    if debug_mode_enabled and meaningful_blocks:
        first_block = meaningful_blocks[0]
        if not all(key in first_block for key in ['index', 'content', 'type']):
            logger.error("\nERROR: Parsed blocks do not have required keys: 'index', 'content', 'type'.")
            logger.error(f"Example of first block's keys: {list(first_block.keys()) if isinstance(first_block, dict) else 'Not a dict'}")
            logger.error("Please ensure your DocxParser provides these keys for each block.\n")
            return None
        else:
            logger.debug("Structure check for first few parsed blocks:")
            for idx, block_item in enumerate(meaningful_blocks[:min(3, len(meaningful_blocks))]):
                 logger.debug(f"  Block {idx}: Index={block_item.get('index')}, Type='{block_item.get('type')}', "
                              f"Content='{str(block_item.get('content'))[:30].replace(chr(10),' ').strip()}...'")
    logger.debug("-" * 60)
    return meaningful_blocks

def classify_cover_page_section(doc_blocks: List[Dict[str, Any]], rules_dir: str) -> Optional[Dict[str, Any]]:
    """Identifies and classifies the Cover Page section."""
    logger.info("Starting Cover Page classification...")
    rules_file_path = os.path.join(rules_dir, 'cover_page_rules.json')
    cp_classifier = CoverPageClassifier(rules_file_path=rules_file_path)
    cover_page_info = cp_classifier.classify(
        doc_blocks,
        confidence_threshold=0.55, # Consider making these configurable too
        max_start_block_index_to_check=500
    )
    cp_classifier.display_results(cover_page_info, doc_blocks)
    logger.info("Cover Page analysis complete!")
    logger.debug("\n" + "="*60)
    return cover_page_info

def classify_sofp_section(doc_blocks: List[Dict[str, Any]], rules_dir: str, start_list_idx: int) -> Optional[Dict[str, Any]]:
    """Identifies and classifies the Statement of Financial Position section."""
    logger.info(f"Starting SoFP classification from list index {start_list_idx}...")
    rules_file_path = os.path.join(rules_dir, 'sofp_rules.json')
    sofp_classifier = SoFPClassifier(rules_file_path=rules_file_path)
    sofpc_info = sofp_classifier.classify(
        doc_blocks,
        start_block_index_in_list=start_list_idx,
        confidence_threshold=0.5,
        max_start_block_index_to_check=500
    )
    sofp_classifier.display_results(sofpc_info, doc_blocks)
    logger.info("Statement of Financial Position analysis complete!")
    logger.debug("\n" + "="*60)
    return sofpc_info

def classify_soci_section(doc_blocks: List[Dict[str, Any]], rules_dir: str, start_list_idx: int) -> Optional[Dict[str, Any]]:
    """Identifies and classifies the Statement of Comprehensive Income section."""
    logger.info(f"Starting SoCI classification from list index {start_list_idx}...")
    rules_file_path = os.path.join(rules_dir, 'soci_rules.json')
    soci_classifier = SoCIClassifier(rules_file_path=rules_file_path)
    socic_info = soci_classifier.classify(
        doc_blocks,
        start_block_index_in_list=start_list_idx,
        confidence_threshold=0.5,
        max_start_block_index_to_check=500,
        max_blocks_in_soci_window=30
    )
    soci_classifier.display_results(socic_info, doc_blocks)
    logger.info("Statement of Comprehensive Income analysis complete!")
    logger.debug("\n" + "="*60)
    return socic_info

def classify_financial_notes_section(doc_blocks: List[Dict[str, Any]], rules_dir: str, start_list_idx: int) -> List[Dict[str, Any]]:
    """Identifies and classifies Financial Notes sections."""
    logger.info(f"Starting Financial Notes classification from list index {start_list_idx}...")
    rules_file_path = os.path.join(rules_dir, 'financial_notes_rules.json')
    fn_classifier = FinancialNotesClassifier(rules_file_path=rules_file_path)
    identified_notes_list = fn_classifier.classify(
        doc_blocks,
        start_block_index_in_list=start_list_idx,
        confidence_threshold=0.3,
        max_start_block_index_to_check=1000 # Example, can be None
    )
    fn_classifier.display_results(identified_notes_list, doc_blocks)
    logger.info("Financial Notes analysis complete!")
    logger.debug("\n" + "="*60)
    return identified_notes_list

def get_start_list_index(meaningful_blocks: List[Dict[str, Any]], previous_section_info: Optional[Dict[str, Any]]) -> int:
    """Calculates the starting list index for the next classifier."""
    start_doc_idx = 0
    if previous_section_info and 'end_block_index' in previous_section_info:
        start_doc_idx = previous_section_info['end_block_index'] + 1

    for list_idx, block in enumerate(meaningful_blocks):
        if block['index'] >= start_doc_idx:
            return list_idx
    return len(meaningful_blocks) # No blocks left, or all consumed

# --- Main Function ---
def process_financial_document(docx_filepath: str, rules_directory: str) -> None:
    """
    Orchestrates the loading, parsing, and classification of a financial document.
    """
    meaningful_blocks = load_and_parse_docx(docx_filepath, logger.isEnabledFor(logging.DEBUG))
    if not meaningful_blocks:
        return

    generator = SectionGenerator()

    # Cover Page
    cover_page_info = classify_cover_page_section(meaningful_blocks, rules_directory)
    if cover_page_info:
        generator.add_normalized_section(
            section_header="Cover Page", normalized_section="cover_page",
            start_block=cover_page_info['start_block_index'], end_block=cover_page_info['end_block_index'],
            confidence_rate=cover_page_info['confidence']
        )
    else:
        logger.warning("Cover page not identified; cannot add to section generator.")

    # Statement of Financial Position (SoFP)
    start_list_idx_sofp = get_start_list_index(meaningful_blocks, cover_page_info)
    sofpc_info = classify_sofp_section(meaningful_blocks, rules_directory, start_list_idx_sofp)
    if sofpc_info:
        generator.add_normalized_section(
            section_header="Statement of Financial Position", normalized_section="statement_of_financial_position",
            start_block=sofpc_info['start_block_index'], end_block=sofpc_info['end_block_index'],
            confidence_rate=sofpc_info['confidence']
        )
    else:
        logger.warning("SoFP not identified; cannot add to section generator.")

    # Statement of Comprehensive Income (SoCI)
    start_list_idx_soci = get_start_list_index(meaningful_blocks, sofpc_info or cover_page_info) # Use SoFP if available, else CoverPage
    socic_info = classify_soci_section(meaningful_blocks, rules_directory, start_list_idx_soci)
    if socic_info:
        generator.add_normalized_section(
            section_header="Statement of Comprehensive Income", normalized_section="statement_of_comprehensive_income",
            start_block=socic_info['start_block_index'], end_block=socic_info['end_block_index'],
            confidence_rate=socic_info['confidence']
        )
    else:
        logger.warning("SoCI not identified; cannot add to section generator.")

    # Financial Notes
    start_list_idx_notes = get_start_list_index(meaningful_blocks, socic_info or sofpc_info or cover_page_info)
    identified_notes_list = classify_financial_notes_section(meaningful_blocks, rules_directory, start_list_idx_notes)
    if identified_notes_list:
        for note in identified_notes_list:
            generator.add_note_section(
                section_header=note.get('section_header', "Notes to Financial Statements"),
                note_number=note['note_number'], note_title=note['note_title'],
                start_block=note['start_block'], end_block=note['end_block'],
                confidence_rate=note['confidence_rate']
            )
    else:
        logger.info("No financial notes identified to add to generator.")

    # Final Output
    financial_sections = generator.get_sections()
    logger.info(f"Final Financial Sections JSON:\n{financial_sections}")
    # print(financial_sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and classify sections in financial DOCX files.")
    parser.add_argument("docx_filepath", help="Path to the .docx financial statement file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging mode.")
    parser.add_argument("--rules_dir", default=os.path.join(SCRIPT_DIR, 'rules'),
                        help="Directory containing the JSON rule files for classifiers. Defaults to './rules/'.")
    args = parser.parse_args()

    setup_logging(args.debug)

    # check if the provided docx_filepath exists
    if not os.path.exists(args.docx_filepath):
        logger.critical(f"Error: The DOCX file was not found at '{args.docx_filepath}'")
        exit(1)
    if not os.path.isdir(args.rules_dir):
        logger.critical(f"Error: The rules directory was not found at '{args.rules_dir}'")
        exit(1)

    process_financial_document(args.docx_filepath, args.rules_dir)
