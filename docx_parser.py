import logging
from typing import List, Dict, Any, Optional

from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.table import _Cell

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- Constants ---
BLOCK_TYPE_PARAGRAPH = "paragraph"
BLOCK_TYPE_TABLE = "table"
SUMMARY_HEADER_DOC_PARSING = "DOCUMENT PARSING SUMMARY"


class DocxParser:
    """
    DOCX parser that extracts document blocks and provides meaningful block indexing.
    - Filters out empty blocks
    - Provides indexing for non-empty blocks
    - Handles paragraphs and tables
    """

    def __init__(self):
        """
        Initialize the parser.
        """
        self.document: Optional[_Document] = None
        self.all_blocks: List[Dict[str, Any]] = []
        self.meaningful_blocks: List[Dict[str, Any]] = []

    def _iter_block_items(self, parent: Any):
        """
        Generate a reference to each paragraph and table child within the parent,
        in document order. Each returned value is an instance of either
        Paragraph or Table.
        """
        if isinstance(parent, _Document):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            logger.error(f"Parent must be a Document or Cell object, got {type(parent)}")
            raise ValueError("Parent must be a Document or Cell object")

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def _extract_block_content(self, block: Any) -> str:
        """
        Extract text content from a block (paragraph or table).
        """
        if isinstance(block, Paragraph):
            return block.text.strip()
        elif isinstance(block, Table):
            table_texts = []
            for row in block.rows:
                row_texts = [cell.text.strip() for cell in row.cells]
                table_texts.append(" | ".join(row_texts))
            return "\n".join(table_texts).strip()
        else:
            logger.warning(f"Attempted to extract content from unsupported block type: {type(block)}")
            return ""

    def _is_meaningful_block(self, content: str) -> bool:
        """
        Determine if a block has meaningful content.
        """
        return bool(content.strip())

    def parse_document(self, docx_path: str) -> List[Dict[str, Any]]:
        """
        Parse the docx file and return meaningful blocks with proper indexing.
        """
        try:
            self.document = Document(docx_path)
        except Exception as ex:
            logger.error(f"Could not open document {docx_path}. Exception: {ex}", exc_info=True)
            return []

        logger.info(f"Starting parsing of document: {docx_path}")
        logger.debug("=" * 60)

        # --- Pass 1: Extract all blocks from the document ---
        self.all_blocks = []
        logger.debug("Starting Pass 1: Extracting all blocks...")
        for original_index, block in enumerate(self._iter_block_items(self.document)):
            block_type = BLOCK_TYPE_PARAGRAPH if isinstance(block, Paragraph) else BLOCK_TYPE_TABLE
            content = self._extract_block_content(block)

            self.all_blocks.append({
                "original_index": original_index,
                "type": block_type,
                "content": content,
                "raw_block": block
            })
            logger.debug(f"  Extracted raw block {original_index}: type='{block_type}', content_preview='{content[:50].replace(chr(10),' ')}...'")
        logger.debug(f"Pass 1 Complete: Total {len(self.all_blocks)} raw blocks extracted.")


        # --- Pass 2: Filter meaningful blocks and assign new 0-based 'index' ---
        self.meaningful_blocks = []
        meaningful_idx_counter = 0
        logger.debug("Starting Pass 2: Filtering meaningful blocks and re-indexing...")

        for raw_block_data in self.all_blocks:
            if self._is_meaningful_block(raw_block_data["content"]):
                meaningful_block = {
                    "index": meaningful_idx_counter,  # 0-based meaningful index
                    "type": raw_block_data["type"],
                    "content": raw_block_data["content"],
                }
                self.meaningful_blocks.append(meaningful_block)

                # Log only meaningful blocks if logger level is DEBUG
                content_preview = meaningful_block["content"][:80].replace(chr(10),' ') + \
                                  ("..." if len(meaningful_block["content"]) > 80 else "")
                logger.debug(f"  Meaningful Block {meaningful_idx_counter} (orig_idx {raw_block_data['original_index']}): type='{meaningful_block['type']}', content='{content_preview}'")
                meaningful_idx_counter += 1
        logger.debug(f"Pass 2 Complete: Total {len(self.meaningful_blocks)} meaningful blocks identified.")

        # --- Summary Logging ---
        logger.info(f"Finished parsing document: {docx_path}")
        logger.info(f"  Total raw blocks found in document: {len(self.all_blocks)}")
        logger.info(f"  Meaningful blocks extracted: {len(self.meaningful_blocks)}")
        logger.info(f"  Empty blocks filtered out: {len(self.all_blocks) - len(self.meaningful_blocks)}")
        logger.debug("=" * 60)

        return self.meaningful_blocks

    def get_all_blocks(self) -> List[Dict[str, Any]]:
        """
        Get all blocks (including empty ones) from the last parsed document.
        """
        return self.all_blocks

    def get_meaningful_blocks(self) -> List[Dict[str, Any]]:
        """
        Get only meaningful blocks from the last parsed document.

        """
        return self.meaningful_blocks

    def get_block_by_meaningful_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get a meaningful block by its 0-based meaningful index.

        """
        # safety
        if 0 <= index < len(self.meaningful_blocks):
            return self.meaningful_blocks[index]
        logger.warning(f"Attempted to get meaningful block by out-of-bounds index: {index}")
        return None

    def print_summary(self):
        """Log a summary of the parsed document."""
        if not self.document: 
            logger.info("No document has been parsed yet. Call parse_document() first.")
            return

        logger.info(SUMMARY_HEADER_DOC_PARSING)
        logger.info("=" * 40)
        logger.info(f"Total raw blocks in document: {len(self.all_blocks)}")
        logger.info(f"Meaningful blocks extracted: {len(self.meaningful_blocks)}")
        logger.info(f"Empty blocks filtered out: {len(self.all_blocks) - len(self.meaningful_blocks)}")
        logger.info("")

        if self.meaningful_blocks:
            logger.info("First 5 meaningful blocks:")
            for block in self.meaningful_blocks[:5]:
                content_preview = block["content"][:50].replace(chr(10),' ') + \
                                  ("..." if len(block["content"]) > 50 else "")
                logger.info(f"  Block {block['index']}: Type='{block['type']}', Content='{content_preview}'")

            if len(self.meaningful_blocks) > 5:
                logger.info(f"  ... and {len(self.meaningful_blocks) - 5} more meaningful blocks.")
        else:
            logger.info("No meaningful blocks to display in summary.")
