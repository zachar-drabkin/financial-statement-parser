import os
import logging # Import logging
from typing import Dict, List, Optional, Any
from docx_parser import DocxParser
from utils.section_generator import SectionGenerator
from classifiers.cover_page_classifier import CoverPageClassifier
from classifiers.sofp_classifier import SoFPClassifier
from classifiers.soci_classifier import SoCIClassifier
from classifiers.financial_notes_classifier import FinancialNotesClassifier

# --- Global Configuration ---
# TODO: set this via a command-line argument in a more advanced setup.
DEBUG_MODE_ENABLED = False

def main():
    """Main function to run the enhanced document parser and classifier"""

    # --- Logging Configuration ---
    log_level = logging.DEBUG if DEBUG_MODE_ENABLED else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    generator = SectionGenerator()

    # --- Document Processing ---
    # TODO: Move this to command line argument
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docx_file = os.path.join(base_dir, 'data', 'financial_statements', 'BestCo Work Sample v1.0.docx')
    #docx_file = os.path.join(base_dir, 'data', 'financial_statements', 'PVI Work Sample.docx')
    #docx_file = os.path.join(base_dir, 'data', 'financial_statements', '1933 Work Sample.docx')

    logger.info(f"Processing document: {docx_file}")
    logger.debug("=" * 60)

    # --- DOCX Parsing ---
    parser = DocxParser()
    meaningful_blocks = parser.parse_document(docx_file)

    if not meaningful_blocks:
        logger.warning("No meaningful blocks found or error processing document.")
        return

    logger.info(f"Found {len(meaningful_blocks)} meaningful blocks from parser.")

    # --- Block Structure Validation ---
    if DEBUG_MODE_ENABLED and meaningful_blocks:
        first_block = meaningful_blocks[0]
        if not all(key in first_block for key in ['index', 'content', 'type']):
            logger.error("\nERROR: Parsed blocks do not have required keys: 'index', 'content', 'type'.")
            logger.error(f"Example of first block's keys: {list(first_block.keys()) if isinstance(first_block, dict) else 'Not a dict'}")
            logger.error("Please ensure your DocxParser provides these keys for each block.\n")
            return
        else:
            logger.debug("Structure check for first few parsed blocks:")
            for idx, block_item in enumerate(meaningful_blocks[:min(3, len(meaningful_blocks))]):
                 logger.debug(f"  Block {idx}: Index={block_item.get('index')}, Type='{block_item.get('type')}', "
                              f"Content='{str(block_item.get('content'))[:30].replace(chr(10),' ').strip()}...'")
    logger.debug("-" * 60)

    # --- Cover Page Classification ---
    rules_file_path_cover_page = os.path.join(base_dir, 'rules', 'cover_page_rules.json')
    cpClassifier = CoverPageClassifier(rules_file_path=rules_file_path_cover_page)

    cover_page_info = cpClassifier.classify(
        meaningful_blocks,
        confidence_threshold=0.55,
        max_start_block_index_to_check=500
    )

    cpClassifier.display_results(cover_page_info, meaningful_blocks)
    logger.info("Cover Page analysis complete!")
    logger.debug("\n" + "="*60)

    # --- Cover Page Section Generation ---
    if cover_page_info:
        generator.add_normalized_section(
            section_header="Cover Page",
            normalized_section="cover_page",
            start_block=cover_page_info['start_block_index'],
            end_block=cover_page_info['end_block_index'],
            confidence_rate=cover_page_info['confidence']
        )
    else:
        logger.warning("Cover page not identified; cannot add to section generator.")

    # --- Statement of Financial Position (SoFP) Classification ---
    start_index_for_sofp_doc = 0 # Document index where SoFP starts
    start_index_for_sofp_list = 0 # List index in meaningful_blocks
    if cover_page_info:
        start_index_for_sofp_doc = cover_page_info['end_block_index'] + 1
        # find the corresponding list index for the doc index
        for idx, block in enumerate(meaningful_blocks):
            if block['index'] >= start_index_for_sofp_doc:
                start_index_for_sofp_list = idx
                break
        else: # i all blocks are part of cover page or no blocks left
            start_index_for_sofp_list = len(meaningful_blocks)

    rules_file_path_sofp = os.path.join(base_dir, 'rules', 'sofp_rules.json')
    sofpClassifier = SoFPClassifier(rules_file_path=rules_file_path_sofp)
    sofpc_info = sofpClassifier.classify( # Changed method name
        meaningful_blocks,
        # index to start in doc_blocks
        start_block_index_in_list=start_index_for_sofp_list,
        confidence_threshold=0.5,
        max_start_block_index_to_check=500
    )

    sofpClassifier.display_results(sofpc_info, meaningful_blocks) # Display results
    logger.info("Statement of Financial Position analysis complete!")
    logger.debug("\n" + "="*60)

    # --- Statement of Financial Position Generation ---
    if sofpc_info:
        generator.add_normalized_section(
            section_header="Statement of Financial Position",
            normalized_section="statement_of_financial_position",
            start_block=sofpc_info['start_block_index'],
            end_block=sofpc_info['end_block_index'],
            confidence_rate=sofpc_info['confidence']
        )
    else:
        logger.warning("SoFP not identified; cannot add to section generator.")

     # --- Statement of Comprehensive Income (SoCI) Classification ---
    start_index_for_soci_doc = 0
    start_index_for_soci_list = 0

    if sofpc_info:
        start_index_for_soci_doc = sofpc_info['end_block_index'] + 1
    elif cover_page_info: # fallback if SoFP not found - but Cover Page was
        start_index_for_soci_doc = cover_page_info['end_block_index'] + 1
    # else: start_index_for_soci_doc remains 0 if no prior section found

    # find corresponding list index for the doc index
    for idx, block in enumerate(meaningful_blocks):
        if block['index'] >= start_index_for_soci_doc:
            start_index_for_soci_list = idx
            break
    else: # if all blocks are part of previous sections or no blocks left
        start_index_for_soci_list = len(meaningful_blocks)

    # initialize the refactored SoCIClassifier
    rules_file_path_soci = os.path.join(base_dir, 'rules', 'soci_rules.json')
    sociClassifier = SoCIClassifier(rules_file_path=rules_file_path_soci)
    socicInfo = sociClassifier.classify(
        meaningful_blocks,
        start_block_index_in_list=start_index_for_soci_list,
        confidence_threshold=0.5,
        max_start_block_index_to_check=500,
        max_blocks_in_soci_window=30
    )

    sociClassifier.display_results(socicInfo, meaningful_blocks)
    logger.info("Statement of Comprehensive Income analysis complete!")
    logger.debug("\n" + "="*60)

    # --- Statement of Comprehensive Income Generation ---
    if socicInfo:
        generator.add_normalized_section(
            section_header="Statement of Comprehensive Income",
            normalized_section="statement_of_comprehensive_income",
            start_block=socicInfo['start_block_index'],
            end_block=socicInfo['end_block_index'],
            confidence_rate=socicInfo['confidence']
        )
    else:
        logger.warning("SoCI not identified; cannot add to section generator.")

    # --- Financial Notes Classification ---
    start_index_for_notes_doc = 0
    start_index_for_notes_list = 0 # List index in meaningful_blocks

    # start index based on previous successful classifications
    if socicInfo: # SoCI index if found
        start_index_for_notes_doc = socicInfo['end_block_index'] + 1
    elif sofpc_info: # fallback to SoFP
        start_index_for_notes_doc = sofpc_info['end_block_index'] + 1
    elif cover_page_info: # fallback to Cover Page
        start_index_for_notes_doc = cover_page_info['end_block_index'] + 1
    # else: start_index_for_notes_doc remains 0

    # find the corresponding list index for the doc index
    for idx, block in enumerate(meaningful_blocks):
        if block['index'] >= start_index_for_notes_doc:
            start_index_for_notes_list = idx
            break
    else: # iff all blocks are part of previous sections or no blocks left
        start_index_for_notes_list = len(meaningful_blocks)

    rules_file_path_financial_notes = os.path.join(base_dir, 'rules', 'financial_notes_rules.json')
    fnClassifier = FinancialNotesClassifier(rules_file_path=rules_file_path_financial_notes)
    identifiedNotesList = fnClassifier.classify(
        meaningful_blocks,
        start_block_index_in_list=start_index_for_notes_list, # list index
        confidence_threshold=0.3,
        max_start_block_index_to_check=1000
    )

    fnClassifier.display_results(identifiedNotesList, meaningful_blocks) # Display results
    logger.info("Notes to Financial Statement analysis complete!")
    logger.debug("\n" + "="*60)

    # --- Financial Notes Section Generation ---
    if identifiedNotesList:
        for note in identifiedNotesList:
            generator.add_note_section(
                section_header=note.get('section_header', "Notes to Financial Statements"),
                note_number=note['note_number'],
                note_title=note['note_title'],
                start_block=note['start_block'],
                end_block=note['end_block'],
                confidence_rate=note['confidence_rate']
            )
    else:
        logger.info("No financial notes identified to add to generator.")


    # --- Final Output ---
    financialSections = generator.get_sections()
    logger.info(f"Final Financial Sections JSON:\n{financialSections}")

if __name__ == "__main__":
    main()
