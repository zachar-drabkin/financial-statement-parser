import pytest
from unittest.mock import MagicMock, patch
import logging

from docx_parser import DocxParser, BLOCK_TYPE_PARAGRAPH, BLOCK_TYPE_TABLE
# --- Tests for DocxParser class ---
class DummyParagraphTypeForTest:
    def __init__(self, text_content=""):
        self.text = text_content

class DummyTableTypeForTest:
    def __init__(self, rows_data=None):
        self.rows = []
        if rows_data:
            for row_texts in rows_data:
                mock_row = MagicMock()
                mock_row.cells = []
                for cell_text in row_texts:
                    mock_cell = MagicMock()
                    mock_cell.text = cell_text
                    mock_row.cells.append(mock_cell)
                self.rows.append(mock_row)

# --- Tests for _is_meaningful_block ---
def test_is_meaningful_block_true():
    parser = DocxParser()
    assert parser._is_meaningful_block("Hello World") is True

def test_is_meaningful_block_false_empty():
    parser = DocxParser()
    assert parser._is_meaningful_block("") is False

def test_is_meaningful_block_false_whitespace():
    parser = DocxParser()
    assert parser._is_meaningful_block("   \n \t ") is False

# --- Tests for _extract_block_content ---
def test_extract_block_content_paragraph():
    parser = DocxParser()
    paragraph_instance = DummyParagraphTypeForTest(text_content="Test paragraph content.")

    with patch('docx_parser.Paragraph', new=DummyParagraphTypeForTest):
        content = parser._extract_block_content(paragraph_instance)
    assert content == "Test paragraph content."

def test_extract_block_content_table():
    parser = DocxParser()
    rows_cell_texts = [["R1C1", "R1C2"], ["R2C1", "R2C2"]]
    table_instance = DummyTableTypeForTest(rows_data=rows_cell_texts)

    with patch('docx_parser.Table', new=DummyTableTypeForTest):

        with patch('docx_parser.Paragraph', new=DummyParagraphTypeForTest):
             content = parser._extract_block_content(table_instance)
    assert content == "R1C1 | R1C2\nR2C1 | R2C2"

# --- Tests for parse_document ---
@patch('docx_parser.Document')
def test_parse_document_empty_file(MockDocument, mocker):
    mock_doc_instance = MockDocument.return_value
    mocker.patch.object(DocxParser, '_iter_block_items', return_value=[])

    parser = DocxParser()
    result = parser.parse_document("dummy_empty.docx")
    assert result == []
    MockDocument.assert_called_once_with("dummy_empty.docx")

@patch('docx_parser.Document')
def test_parse_document_only_empty_blocks(MockDocument, mocker):
    mock_doc_instance = MockDocument.return_value

    with patch('docx_parser.Paragraph', new=DummyParagraphTypeForTest) as PatchedDummyParagraph, \
         patch('docx_parser.Table', new=DummyTableTypeForTest) as PatchedDummyTable:

        p_instance_empty = PatchedDummyParagraph(text_content="   ")

        t_instance_empty = PatchedDummyTable(rows_data=[["\n", "\t "]])

        mocker.patch.object(DocxParser, '_iter_block_items', return_value=[p_instance_empty, t_instance_empty])

        parser = DocxParser()
        result = parser.parse_document("dummy_with_empty.docx")

        assert len(result) == 1
        assert result[0]['type'] == BLOCK_TYPE_TABLE
        assert result[0]['content'] == "|"
        assert result[0]['index'] == 0


@patch('docx_parser.Document')
def test_parse_document_mixed_content(MockDocument, mocker):
    mock_doc_instance = MockDocument.return_value

    with patch('docx_parser.Paragraph', new=DummyParagraphTypeForTest) as PatchedDummyParagraph, \
         patch('docx_parser.Table', new=DummyTableTypeForTest) as PatchedDummyTable:

        mock_p1_content = "Meaningful Paragraph 1"
        mock_p2_content = "  "
        mock_t1_rows_data = [["R1C1", "R1C2"], ["", "R2C2"]]
        mock_p3_content = "Meaningful Paragraph 2"

        p1 = PatchedDummyParagraph(text_content=mock_p1_content)
        p2 = PatchedDummyParagraph(text_content=mock_p2_content)
        t1 = PatchedDummyTable(rows_data=mock_t1_rows_data)
        p3 = PatchedDummyParagraph(text_content=mock_p3_content)

        mocker.patch.object(DocxParser, '_iter_block_items', return_value=[p1, p2, t1, p3])

        parser = DocxParser()
        result = parser.parse_document("dummy_mixed.docx")

        assert len(result) == 3
        assert result[0]['index'] == 0
        assert result[0]['type'] == BLOCK_TYPE_PARAGRAPH
        assert result[0]['content'] == mock_p1_content

        assert result[1]['index'] == 1
        assert result[1]['type'] == BLOCK_TYPE_TABLE
        assert result[1]['content'] == "R1C1 | R1C2\n | R2C2"

        assert result[2]['index'] == 2
        assert result[2]['type'] == BLOCK_TYPE_PARAGRAPH
        assert result[2]['content'] == mock_p3_content

@patch('docx_parser.Document', side_effect=FileNotFoundError("File not found"))
def test_parse_document_file_not_found(MockDocument, caplog):
    parser = DocxParser()
    with caplog.at_level(logging.ERROR):
        result = parser.parse_document("non_existent.docx")

    assert result == []
    MockDocument.assert_called_once_with("non_existent.docx")
    assert "Could not open document non_existent.docx" in caplog.text
    assert "FileNotFoundError: File not found" in caplog.text
