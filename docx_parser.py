from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.table import _Cell
from typing import List, Dict, Any, Optional

class DocxParser:
    """
    DOCX parser that extracts document blocks and provides meaningful block indexing.

    Features:
    - Filters out empty blocks
    - Prvides indexing for non empty blocks
    - Handles paragraphs and tables
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the parser.

        Args:
            debug: Whether to print block information during parsing
        """
        self.debug = debug
        self.document = None
        self.all_blocks = []
        self.meaningful_blocks = []

    def _iter_block_items(self, parent):
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
            raise ValueError("Parent must be a Document or Cell object")

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def _extract_block_content(self, block) -> str:
        """
        Extract text content from a block (paragraph or table).

        Args:
            block: Paragraph or Table object

        Returns:
            String content of the block
        """
        if isinstance(block, Paragraph):
            return block.text.strip()
        elif isinstance(block, Table):
            table_texts = []
            for row in block.rows:
                row_texts = [cell.text.strip() for cell in row.cells]
                table_texts.append(" | ".join(row_texts))
            return "\n".join(table_texts)
        else:
            return ""

    def _is_meaningful_block(self, content: str) -> bool:
        """
        Determine if a block has meaningful content.

        Args:
            content: Block content string

        Returns:
            True if block has meaningful content, False otherwise
        """
        return bool(content.strip())

    def parse_document(self, docx_path: str) -> List[Dict[str, Any]]:
        """
        Parse the docx file and return meaningful blocks with proper indexing.

        Args:
            docx_path: Path to the DOCX file

        Returns:
            List of meaningful block dictionaries with 1-based indexing
        """
        try:
            self.document = Document(docx_path)
        except Exception as ex:
            print(f"Could not open document {docx_path}. Got Exception: {ex}")
            return []

        if self.debug:
            print(f"Parsing document: {docx_path}")
            print("=" * 60)

        # first pass: extract all blocks
        self.all_blocks = []
        for original_index, block in enumerate(self._iter_block_items(self.document)):
            block_type = "paragraph" if isinstance(block, Paragraph) else "table"
            content = self._extract_block_content(block)

            self.all_blocks.append({
                "original_index": original_index,
                "type": block_type,
                "content": content,
                "raw_block": block
            })

        # second pass: filter meaningful blocks and re-index starting from 1
        self.meaningful_blocks = []
        meaningful_index = 0  # Start from 1 as requested

        for block in self.all_blocks:
            if self._is_meaningful_block(block["content"]):
                meaningful_block = {
                    "index": meaningful_index,  # 1-based meaningful index
                    "type": block["type"],
                    "content": block["content"],
                    "raw_block": block["raw_block"]
                }

                self.meaningful_blocks.append(meaningful_block)

                # print only meaningful blocks
                if self.debug:
                    content_preview = block["content"][:80] + "..." if len(block["content"]) > 80 else block["content"]
                    print(f"Block {meaningful_index} ({block['type']}): {content_preview}")

                meaningful_index += 1

        if self.debug:
            print()
            print(f"Summary:")
            print(f"  Total blocks in document: {len(self.all_blocks)}")
            print(f"  Meaningful blocks: {len(self.meaningful_blocks)}")
            print(f"  Empty blocks filtered out: {len(self.all_blocks) - len(self.meaningful_blocks)}")
            print()

        return self.meaningful_blocks

    def get_all_blocks(self) -> List[Dict[str, Any]]:
        """
        Get all blocks (including empty ones) from the last parsed document.

        Returns:
            List of all block dictionaries
        """
        return self.all_blocks

    def get_meaningful_blocks(self) -> List[Dict[str, Any]]:
        """
        Get only meaningful blocks from the last parsed document.

        Returns:
            List of meaningful block dictionaries with 1-based indexing
        """
        return self.meaningful_blocks

    def get_block_by_meaningful_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get a meaningful block by its 1-based meaningful index.

        Args:
            index: 1-based meaningful block index

        Returns:
            Block dictionary or None if not found
        """
        for block in self.meaningful_blocks:
            if block["index"] == index:
                return block
        return None

    def print_summary(self):
        """Print a summary of the parsed document."""
        if not self.meaningful_blocks:
            print("No document has been parsed yet.")
            return

        print("DOCUMENT PARSING SUMMARY")
        print("=" * 40)
        print(f"Total blocks: {len(self.all_blocks)}")
        print(f"Meaningful blocks: {len(self.meaningful_blocks)}")
        print(f"Empty blocks filtered: {len(self.all_blocks) - len(self.meaningful_blocks)}")
        print()

        print("First 5 meaningful blocks:")
        for block in self.meaningful_blocks[:5]:
            content_preview = block["content"][:50] + "..." if len(block["content"]) > 50 else block["content"]
            print(f"  Block {block['index']}: {content_preview}")

        if len(self.meaningful_blocks) > 5:
            print(f"  ... and {len(self.meaningful_blocks) - 5} more blocks")

