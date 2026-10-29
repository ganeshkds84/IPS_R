"""Unit tests for layout analysis and metadata extraction on synthetic page content."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ips_r.core.exceptions import ConfigurationError, PDFParseError
from ips_r.data.models import ElementType
from ips_r.data.pdf import (
    PDF_BACKENDS,
    AbstractPDFBackend,
    LayoutAnalyzer,
    LayoutSettings,
    MetadataExtractor,
    MetadataSettings,
    PDFParser,
    RawPage,
    RawPDF,
    RawTable,
    RawWord,
    split_authors,
)
from ips_r.data.pdf.backend import _info_value


class _Page:
    """Builds words line by line with a running y position."""

    def __init__(self, number: int = 1) -> None:
        self.number = number
        self.words: list[RawWord] = []
        self.tables: list[RawTable] = []
        self.y = 50.0

    def line(
        self,
        text: str,
        *,
        size: float = 10,
        bold: bool = False,
        x: float = 72,
        gap: float | None = None,
    ) -> None:
        self.y += gap if gap is not None else size * 1.2
        cursor = x
        for token in text.split():
            width = len(token) * size * 0.5
            font = "Helvetica-Bold" if bold else "Helvetica"
            self.words.append(
                RawWord(token, cursor, cursor + width, self.y, self.y + size, size, font)
            )
            cursor += width + size * 0.3

    def build(self) -> RawPage:
        return RawPage(self.number, 595.0, 842.0, tuple(self.words), tuple(self.tables))


def _paper() -> RawPDF:
    p1 = _Page(1)
    p1.line("A Study of Query Intent", size=20, bold=True)
    p1.line("Grace Hopper and Alan Turing*", gap=26)
    p1.line("grace@example.org, Example University")
    p1.line("Abstract: We examine reformulation of queries in retrieval systems.", gap=20)
    p1.line("Index Terms: retrieval, intent", gap=18)
    p1.line("1. Introduction", size=12, bold=True, gap=20)
    p1.line("Retrieval systems are widely used in practice and in research.", gap=16)
    p1.line("They rely on queries writ-")
    p1.line("ten by people.")
    p1.line("A new paragraph begins here after a larger gap.", gap=20)
    p1.line("Fig. 2: The architecture of the system.", gap=20)
    p1.line("continued caption text")
    p1.line("2.1 Setup Details", size=10, gap=20)
    p1.line("Experiments use standard data.", gap=14)
    p1.tables.append(RawTable((300, 460, 400, 480), (("A", "B"), ("1", "2"))))
    p1.words.append(RawWord("inside", 310, 330, 465, 475, 10, "Helvetica"))
    p1.line("7", gap=40)
    p2 = _Page(2)
    p2.line("References", size=12, bold=True)
    p2.line("[1] First reference text.")
    p2.line("continues here.")
    p2.line("[2] Second reference.")
    return RawPDF(
        pages=(p1.build(), p2.build()),
        info={
            "Title": "untitled",
            "Author": "",
            "CreationDate": "D:20190101",
            "Subject": "doi 10.5555/abc.1).",
        },
    )


def _blocks(raw: RawPDF, **settings: Any) -> list[tuple[str, str, int, int | None, str | None]]:
    result = LayoutAnalyzer(LayoutSettings(**settings)).analyze(raw)
    return [(b.type.value, b.text, b.page, b.level, b.section) for b in result.blocks]


def test_layout_classifies_structure() -> None:
    blocks = _blocks(_paper())
    kinds = [b[0] for b in blocks]
    assert blocks[0][:2] == ("title", "A Study of Query Intent")
    assert blocks[1][:2] == ("authors", "Grace Hopper and Alan Turing*")
    assert blocks[2][:2] == ("paragraph", "grace@example.org, Example University")
    assert blocks[3][0] == "abstract"
    assert ("heading", "1. Introduction", 1, 1, "1. Introduction") in blocks
    assert any(b[1].endswith("They rely on queries written by people.") for b in blocks)
    assert (
        "paragraph",
        "A new paragraph begins here after a larger gap.",
        1,
        None,
        "1. Introduction",
    ) in blocks
    assert (
        "caption",
        "Fig. 2: The architecture of the system. continued caption text",
        1,
        None,
        "1. Introduction",
    ) in blocks
    assert ("heading", "2.1 Setup Details", 1, 2, "2.1 Setup Details") in blocks
    table = next(b for b in blocks if b[0] == "table")
    assert table[1] == "A | B\n1 | 2"
    assert kinds.index("table") > kinds.index("heading")
    assert not any("inside" in b[1] for b in blocks)
    assert not any(b[1] == "7" for b in blocks)
    references = [b for b in blocks if b[0] == "reference"]
    assert [r[1] for r in references] == [
        "[1] First reference text. continues here.",
        "[2] Second reference.",
    ]
    assert all(r[2] == 2 and r[4] == "References" for r in references)


def test_layout_settings_toggle_behaviour() -> None:
    blocks = _blocks(_paper(), drop_page_numbers=False, title_size_ratio=5.0)
    assert any(b[1] == "7" for b in blocks)
    assert not any(b[0] == "title" for b in blocks)
    assert not any(b[0] == "authors" for b in blocks)


def test_reference_entries_split_on_gaps_without_markers() -> None:
    page = _Page()
    page.line("Bibliography", size=14)
    page.line("Smith, J. A book")
    page.line("about retrieval.")
    page.line("Doe, J. Another book", gap=24)
    page.line("about generation.")
    blocks = _blocks(RawPDF(pages=(page.build(),)), title_size_ratio=10.0)
    assert [b[1] for b in blocks if b[0] == "reference"] == [
        "Smith, J. A book about retrieval.",
        "Doe, J. Another book about generation.",
    ]


def test_two_column_reading_order() -> None:
    page = _Page()
    page.line("Header Spanning The Page Width", size=16)
    for i in range(3):
        page.y += 12
        for x, label in ((72, "left"), (330, "right")):
            for j, token in enumerate((label, f"line{i}", "text")):
                page.words.append(
                    RawWord(
                        token, x + j * 40, x + j * 40 + 30, page.y, page.y + 10, 10, "Helvetica"
                    )
                )
    raw = RawPDF(pages=(page.build(),))
    lines = LayoutAnalyzer().page_lines(raw.pages[0])
    texts = [line.text for line in lines]
    assert texts[0] == "Header Spanning The Page Width"
    assert texts[1:4] == [f"left line{i} text" for i in range(3)]
    assert texts[4:] == [f"right line{i} text" for i in range(3)]
    merged = LayoutAnalyzer(LayoutSettings(detect_columns=False)).page_lines(raw.pages[0])
    assert merged[1].text == "left line0 text right line0 text"


def test_heading_levels_from_font_size_and_known_names() -> None:
    page = _Page()
    page.line("Introduction", size=16)
    page.line("Some body text that is long enough to dominate the size estimate here.")
    page.line("Details", size=13, gap=20)
    page.line("More body text that is long enough to dominate the size estimate too.")
    page.line("conclusion", gap=20)
    page.line("Final words that conclude the body text of this synthetic document.")
    blocks = _blocks(RawPDF(pages=(page.build(),)), title_size_ratio=10.0)
    headings = [(b[1], b[3]) for b in blocks if b[0] == "heading"]
    assert headings == [("Introduction", 1), ("Details", 2), ("conclusion", 3)]


def test_invalid_layout_settings() -> None:
    with pytest.raises(ConfigurationError, match="Unknown"):
        LayoutSettings.from_mapping({"nope": 1})
    with pytest.raises(ConfigurationError, match="boolean"):
        LayoutSettings.from_mapping({"detect_columns": "yes"})
    with pytest.raises(ConfigurationError, match="number"):
        LayoutSettings.from_mapping({"column_gap": "wide"})
    with pytest.raises(ConfigurationError, match="list"):
        LayoutSettings.from_mapping({"known_headings": "abstract"})
    with pytest.raises(ConfigurationError, match="str"):
        LayoutSettings.from_mapping({"caption_pattern": 5})
    with pytest.raises(ConfigurationError, match="pattern"):
        LayoutAnalyzer(LayoutSettings(caption_pattern="("))
    settings = LayoutSettings.from_mapping(
        {"known_headings": ["x"], "line_tolerance": 3, "detect_columns": False}
    )
    assert settings.known_headings == ("x",)
    assert settings.line_tolerance == 3.0


def test_metadata_extraction_from_layout_and_info() -> None:
    raw = _paper()
    layout = LayoutAnalyzer().analyze(raw)
    meta = MetadataExtractor().extract(raw.info, layout, "the and of " * 20)
    assert meta.title == "A Study of Query Intent"
    assert meta.authors == ("Grace Hopper", "Alan Turing")
    assert meta.abstract == "We examine reformulation of queries in retrieval systems."
    assert meta.keywords == ("retrieval", "intent")
    assert meta.doi == "10.5555/abc.1"
    assert meta.year == 2019
    assert meta.language == "en"
    assert meta.venue is None


def test_metadata_venue_and_year_sources() -> None:
    extractor = MetadataExtractor()
    assert extractor._venue("Accepted at EMNLP; main") == "Accepted at EMNLP"
    assert extractor._venue("EMNLP '23 main track") == "EMNLP 2023"
    assert (
        extractor._venue("Appeared in Journal of Testing; vol 2")
        == "Appeared in Journal of Testing"
    )
    assert extractor._venue("arXiv:2101.00001v2 [cs.IR]") == "arXiv"
    assert extractor._venue("SIGIR is great") == "SIGIR"
    assert extractor._year("© 2021 ACM", None, None) == 2021
    assert extractor._year("arXiv:1912.01234", None, None) == 2019
    assert extractor._year("copyright 3000", "Conf 2018", None) == 2018
    no_creation = MetadataExtractor(MetadataSettings(use_creation_date_year=False))
    assert no_creation._year("", None, "D:2015") is None
    prefer_info = MetadataExtractor(MetadataSettings(prefer_info_title=True))
    assert prefer_info._title("Info Title", "Layout Title") == "Info Title"
    assert prefer_info._title("x.pdf", "Layout Title") == "Layout Title"
    assert extractor._keywords_of(None, "no keywords here") == ()
    assert extractor._keywords_of(" ; ", "Keywords: a, b") == ("a", "b")
    assert extractor._authors("Anonymous", "Ada Lovelace") == ("Ada Lovelace",)
    assert extractor._authors(None, None) == ()
    with pytest.raises(ConfigurationError):
        MetadataExtractor(MetadataSettings(keyword_pattern="("))
    with pytest.raises(ConfigurationError):
        MetadataSettings.from_mapping({"unknown": True})


def test_split_authors() -> None:
    assert split_authors("A. Smith1, B. Jones*† and C. Lee; C. Lee & D") == (
        "A. Smith",
        "B. Jones",
        "C. Lee",
    )


def test_info_value_normalisation() -> None:
    class Literal:
        name = "Lit"

    assert _info_value(None) is None
    assert _info_value(b"\xfe\xff\x00H\x00i") == "Hi"
    assert _info_value(b"caf\xe9") == "café"
    assert _info_value(Literal()) == "Lit"
    assert _info_value("  ") is None


class _StaticBackend(AbstractPDFBackend):
    def __init__(self, options: Any = None) -> None:
        super().__init__(options)
        self.raw = RawPDF(pages=(RawPage(1, 595, 842),))

    def read(self, path: Path) -> RawPDF:
        return self.raw


@pytest.fixture
def static_backend() -> Iterator[None]:
    PDF_BACKENDS.add("static", _StaticBackend)
    yield
    PDF_BACKENDS.remove("static")


def test_parser_configuration(static_backend: None) -> None:
    parser = PDFParser.from_config({"backend": "static", "layout": None})
    with pytest.raises(PDFParseError, match="no extractable text"):
        parser.parse(Path("x.pdf"), "x")
    with pytest.raises(ConfigurationError, match="Unknown parser"):
        PDFParser.from_config({"bogus": 1})
    with pytest.raises(ConfigurationError, match="Unknown pdf_backend"):
        PDFParser.from_config({"backend": "nope"})
    with pytest.raises(ConfigurationError, match="mapping"):
        PDFParser.from_config({"layout": [1]})
    assert ElementType.TABLE.value == "table"
