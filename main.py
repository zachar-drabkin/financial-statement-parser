from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.table import _Cell

def iter_block_items(parent):
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

def get_document_blocks(docx_path):
    """
    Parse the docx file and return a list of block objects (Paragraphs or Tables).
    Each block will have its sequence index and type.
    """
    try:
        document = Document(docx_path)
    except Exception as ex:
        print(f"Could not open document {docx_path}. Got Exception: {ex}")
        return []

    blocks = []
    for i, block in enumerate(iter_block_items(document)):
        block_type = "paragraph" if isinstance(block, Paragraph) else "table"
        block_text = ""
        if isinstance(block, Paragraph):
            block_text = block.text.strip()
        elif isinstance(block, Table):
            table_texts = []
            for row in block.rows:
                row_texts = [cell.text.strip() for cell in row.cells]
                table_texts.append(" | ".join(row_texts))
            block_text = "\n".join(table_texts)

        blocks.append({
            "index": i,
            "type": block_type,
            "content": block_text,
            "raw_block": block
        })
        print(f"Block {i} ({block_type}): {block_text[:100]}...")
    return blocks

def main():
    """Main function to run the document parser"""
    docx_file = 'data/financial_statements/BestCo Work Sample v1.0.docx'

    print(f"Processing document: {docx_file}")
    doc_blocks = get_document_blocks(docx_file)

    if not doc_blocks:
        print("No blocks found or error processing document")
        return

    print(f"\nFound {len(doc_blocks)} blocks total")
    print("-" * 50)

    # Process the blocks
    for block_data in doc_blocks:
        print(f"Index: {block_data['index']}, Type: {block_data['type']}")

        print(f"Content: {block_data['content']}")
        print()


if __name__ == "__main__":
    main()
