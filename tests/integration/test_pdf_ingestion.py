"""Integration tests: real PDFs (generated with reportlab) through the pdfplumber backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from ips_r.core.exceptions import PDFParseError
from ips_r.data.models import ElementType
from ips_r.data.pdf import PDFParser, PdfPlumberBackend
from tests import data_factories as factories


@pytest.fixture(scope="module")
def pdf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pdfs")
    factories.paper_pdf(root / "paper.pdf")
    factories.paper_pdf(root / "no-info.pdf", info=False)
    factories.two_column_pdf(root / "two-column.pdf")
    factories.blank_pdf(root / "blank.pdf")
    factories.corrupted_pdf(root / "corrupt.pdf")
    return root


def test_backend_extracts_words_tables_and_info(pdf_dir: Path) -> None:
    raw = PdfPlumberBackend().read(pdf_dir / "paper.pdf")
    assert len(raw.pages) == 2
    assert raw.info["Title"] == "Intent Preservation in RAG"
    assert any(word.bold for word in raw.pages[0].words)
    assert raw.pages[0].tables[0].rows == (("Model", "Score"), ("BM25", "0.41"), ("Dense", "0.52"))
    no_tables = PdfPlumberBackend({"extract_tables": False}).read(pdf_dir / "paper.pdf")
    assert no_tables.pages[0].tables == ()


def test_parser_preserves_structure(pdf_dir: Path) -> None:
    doc = PDFParser.from_config({}).parse(pdf_dir / "paper.pdf", "paper")
    kinds = [e.type for e in doc.elements]
    for kind in ElementType:
        assert kind in kinds, kind
    headings = [(e.text, e.level) for e in doc.elements if e.type is ElementType.HEADING]
    assert headings == [
        ("Abstract", 1),
        ("1 Introduction", 1),
        ("1.1 Problem Setting", 2),
        ("References", 1),
    ]
    references = [e for e in doc.elements if e.type is ElementType.REFERENCE]
    assert len(references) == 2
    assert all(e.page == 2 and e.section == "References" for e in references)
    table = next(e for e in doc.elements if e.type is ElementType.TABLE)
    assert table.rows[1] == ("BM25", "0.41")
    captions = [e.text for e in doc.elements if e.type is ElementType.CAPTION]
    assert captions == ["Figure 1: Overview of the system.", "Table 1: Results on the dev set."]
    assert doc.num_pages == 2
    assert doc.extra["sections"] == [h for h, _ in headings]


def test_metadata_extraction(pdf_dir: Path) -> None:
    meta = PDFParser.from_config({}).parse(pdf_dir / "paper.pdf", "paper").metadata
    assert meta.title == "Intent Preservation in RAG"
    assert meta.authors == ("Ada Lovelace", "Alan Turing")
    assert meta.year == 2024
    assert meta.venue == "Proceedings of the 47th SIGIR Conference 2024"
    assert meta.doi == "10.1234/ipsr.2024.001"
    assert meta.keywords == ("rag", "retrieval")
    assert meta.abstract is not None and meta.abstract.startswith("We study query reformulation")
    assert "Keywords" not in meta.abstract
    assert meta.language == "en"


def test_metadata_falls_back_to_typeset_content(pdf_dir: Path) -> None:
    meta = PDFParser.from_config({}).parse(pdf_dir / "no-info.pdf", "x").metadata
    assert meta.title == "Intent Preservation in RAG"
    assert meta.authors == ("Ada Lovelace", "Alan Turing")
    assert meta.keywords == ("retrieval", "query reformulation", "intent")


def test_two_column_reading_order(pdf_dir: Path) -> None:
    doc = PDFParser.from_config({}).parse(pdf_dir / "two-column.pdf", "tc")
    assert doc.elements[0].type is ElementType.TITLE
    body = doc.elements[1].text
    assert body.index("left column line 5") < body.index("right column line 0")
    assert not any(e.text == "1" for e in doc.elements)


@pytest.mark.parametrize(
    ("name", "message"), [("blank.pdf", "no extractable text"), ("corrupt.pdf", "Cannot parse")]
)
def test_unusable_pdfs_raise(pdf_dir: Path, name: str, message: str) -> None:
    with pytest.raises(PDFParseError, match=message):
        PDFParser.from_config({}).parse(pdf_dir / name, "x")
