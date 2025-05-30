from typing import Dict, List, Optional, Any
from docx_parser import DocxParser
from cover_page_classifier import CoverPageClassifier
from section_generator import SectionGenerator
from sofp_classifier import SoFPClassifier
from soci_classifier import SoCIClassifier

debug = False  # Set to True for debugging output

def main():
    """Main function to run the enhanced document parser and classifier"""

    # Testing file paths - ensure this path is correct or make it configurable
    docx_file = 'data/financial_statements/BestCo Work Sample v1.0.docx'
    #docx_file = 'data/financial_statements/1933 Work Sample.docx'
    #docx_file = 'data/financial_statements/PVI Work Sample.docx'

    if debug:
        print(f"Processing document: {docx_file}")
        print("=" * 60)

    # DocxParser MUST return blocks with 'index', 'content', and 'type'
    parser = DocxParser(debug)
    meaningful_blocks = parser.parse_document(docx_file)

    if not meaningful_blocks:
        if debug:
            print("No meaningful blocks found or error processing document.")
        return

    if debug:
        print(f"Found {len(meaningful_blocks)} meaningful blocks from parser.")

    # validation of block structure from parser
    if meaningful_blocks:
        first_block = meaningful_blocks[0]
        if not all(key in first_block for key in ['index', 'content', 'type']):
            if debug:
                print("\nERROR: Parsed blocks do not seem to have the required keys: 'index', 'content', 'type'.")
                print(f"Example of first block's keys: {list(first_block.keys()) if isinstance(first_block, dict) else 'Not a dict'}")
                print("Please ensure your DocxParser provides these keys for each block.\n")
                return
        # structure of first few blocks if valid
        else:
            if debug:
                print("Structure check for first few parsed blocks:")
            for idx, block_item in enumerate(meaningful_blocks[:min(3, len(meaningful_blocks))]):
                 if debug:
                    print(f"  Block {idx}: Index={block_item.get('index')}, Type='{block_item.get('type')}', "
                        f"Content='{str(block_item.get('content'))[:30].replace(chr(10),' ').strip()}...'")
    if debug:
        print("-" * 60)

    # initialize classifier and classify cover page
    cpClassifier = CoverPageClassifier()

    # call classify_cover_page with desired parameters.
    cover_page_info = cpClassifier.classify_cover_page(
        meaningful_blocks,
        # threshold for cover page section (0.0 to 1.0)
        confidence_threshold=0.55,
        # how far into the doc to start looking for a cover page
        max_start_block_index_to_check=500,
        debug=debug
    )

    if debug:
        # display results using the updated function
        cpClassifier.display_cover_page_results(cover_page_info, meaningful_blocks)
        print("\n" + "="*60)
        print("Cover Page analysis complete!")

    generator = SectionGenerator()
    generator.add_normalized_section(
        section_header="Cover Page",
        normalized_section="cover_page",
        start_block=cover_page_info['start_block_index'],
        end_block=cover_page_info['end_block_index'],
        confidence_rate=cover_page_info['confidence']
    )

    sofpClassifier = SoFPClassifier()
    sofpc_info = sofpClassifier.classify_sofp_section(
        meaningful_blocks,
        start_block_index=cover_page_info['end_block_index'] + 1,  # Start after cover page section
        confidence_threshold=0.5,
        max_start_block_index_to_check=500,
        debug=debug
    )

    if debug:
        # display results using the updated function
        sofpClassifier.display_sofp_results(sofpc_info, meaningful_blocks)
        print("\n" + "="*60)
        print("Statement of Financial Position analysis complete!")

    generator.add_normalized_section(
        section_header="Statement of Financial Position",
        normalized_section="statement_of_financial_position",
        start_block=sofpc_info['start_block_index'],
        end_block=sofpc_info['end_block_index'],
        confidence_rate=sofpc_info['confidence']
    )

    sociClassifier = SoCIClassifier()
    socicInfo = sociClassifier.classify_soci_section(
        meaningful_blocks,
        start_block_index=sofpc_info['end_block_index'] + 1,  # Start after SoFP section
        confidence_threshold=0.5,
        max_start_block_index_to_check=500,
        debug=debug
    )

    if debug:
        # display results using the updated function
        sociClassifier.display_soci_results(socicInfo, meaningful_blocks)
        print("\n" + "="*60)
        print("Statement of Financial Position analysis complete!")

    generator.add_normalized_section(
        section_header="Statement of Comprehensive Income",
        normalized_section="statement_of_comprehensive_income",
        start_block=socicInfo['start_block_index'],
        end_block=socicInfo['end_block_index'],
        confidence_rate=socicInfo['confidence']
    )

    financialSections = generator.get_sections()
    print(financialSections)


if __name__ == "__main__":
    main()
