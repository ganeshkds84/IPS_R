"""Rule-based layout analysis: positioned words → typed structural blocks.

The analysis is pure (no I/O) and deterministic. Pipeline:

1. Remove words inside detected tables; group the rest into lines.
2. Split two-column pages and order lines in reading order.
3. Estimate the body font size and line pitch.
4. Classify lines (title, authors, heading, caption, reference, equation, footnote,
   paragraph) using font size, weight, position on the page, numbering, known heading
   names, caption patterns and the density of mathematical symbols.
5. Merge consecutive lines into blocks (paragraph breaks are detected from vertical
   gaps), insert tables, and attach the enclosing section heading to each block.

Blocks never span pages, so every block carries an exact 1-based page number.
All thresholds come from :class:`LayoutSettings` (``params.parser.layout`` in config).
Settings live in :mod:`.layout_settings`, line grouping and the block types in
:mod:`.lines`; both are re-exported here.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import Final

from ips_r.core.exceptions import ConfigurationError
from ips_r.data.models import ElementType
from ips_r.data.pdf.backend import RawPage, RawPDF
from ips_r.data.pdf.layout_settings import LayoutSettings, settings_from_mapping
from ips_r.data.pdf.lines import (
    LayoutBlock,
    LayoutResult,
    TextLine,
    dominant,
    follows,
    group_lines,
    interleave_tables,
    join_lines,
)

_SAME_SIZE: Final = 0.5  # font sizes closer than this (points) count as equal
_MAX_AUTHOR_LINE_WORDS: Final = 30

__all__ = [
    "LayoutAnalyzer",
    "LayoutBlock",
    "LayoutResult",
    "LayoutSettings",
    "TextLine",
    "settings_from_mapping",
]


class LayoutAnalyzer:
    """Converts a :class:`RawPDF` into :class:`LayoutBlock` objects.

    Args:
        settings: Analysis thresholds and patterns.
    """

    def __init__(self, settings: LayoutSettings | None = None) -> None:
        self._s = settings or LayoutSettings()
        try:
            self._page_number = re.compile(self._s.page_number_pattern, re.IGNORECASE)
            self._caption = re.compile(self._s.caption_pattern, re.IGNORECASE)
            self._numbered = re.compile(self._s.numbered_heading_pattern)
            self._ref_marker = re.compile(self._s.reference_marker_pattern)
            self._abstract_prefix = re.compile(self._s.abstract_prefix_pattern, re.IGNORECASE)
            self._keywords_prefix = re.compile(self._s.keywords_prefix_pattern, re.IGNORECASE)
            self._equation_number = re.compile(self._s.equation_number_pattern)
            self._footnote_marker = re.compile(self._s.footnote_marker_pattern)
        except re.error as exc:
            raise ConfigurationError(f"Invalid parser.layout pattern: {exc}") from exc
        self._known = frozenset(self._s.known_headings)
        self._reference_headings = frozenset(self._s.reference_headings)
        self._math = frozenset(self._s.equation_symbols)

    # ------------------------------------------------------------------ #
    # Lines
    # ------------------------------------------------------------------ #
    def page_lines(self, page: RawPage) -> list[TextLine]:
        """Group a page's non-table words into lines in reading order."""
        return group_lines(page, self._s, self._page_number)

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def analyze(self, raw: RawPDF) -> LayoutResult:
        """Run the full analysis."""
        page_lines = {page.number: self.page_lines(page) for page in raw.pages}
        lines = [line for page in raw.pages for line in page_lines[page.number]]
        body_size = dominant((line.size, len(line.text)) for line in lines)
        body_bold = sum(len(line.text) for line in lines if line.bold) * 2 > sum(
            len(line.text) for line in lines
        )
        pitch = self._line_pitch(lines, body_size)
        heights = {page.number: page.height for page in raw.pages}
        kinds = self._classify(lines, body_size, body_bold, heights)
        placed = self._assemble(lines, kinds, pitch, raw.pages)
        blocks = self._assign_levels_and_sections(placed)
        title = next((b.text for b in blocks if b.type is ElementType.TITLE), None)
        authors = next((b.text for b in blocks if b.type is ElementType.AUTHORS), None)
        first_page = [line.text for line in page_lines.get(raw.pages[0].number, [])]
        return LayoutResult(
            blocks=tuple(blocks),
            num_pages=len(raw.pages),
            body_size=body_size,
            title=title,
            authors=authors,
            front_text="\n".join(first_page),
            sections=tuple(b.text for b in blocks if b.type is ElementType.HEADING),
        )

    @staticmethod
    def _line_pitch(lines: Sequence[TextLine], body_size: float) -> float:
        """Median vertical distance between consecutive body lines (fallback: 1.2 x size)."""
        deltas = [
            b.top - a.top
            for a, b in pairwise(lines)
            if a.page == b.page
            and round(a.size, 1) == round(b.size, 1) == body_size
            and 0 < b.top - a.top < 3 * body_size
        ]
        return statistics.median(deltas) if deltas else 1.2 * max(body_size, 1.0)

    def _normalized_heading(self, text: str) -> str:
        """Heading text without numbering and trailing punctuation, lower case."""
        stripped = self._numbered.sub("", text).strip().rstrip(".:").strip()
        return stripped.lower()

    def _is_heading(self, line: TextLine, body_size: float, body_bold: bool) -> bool:
        """Decide whether ``line`` is a section heading."""
        text = line.text.strip()
        words = len(text.split())
        if words == 0 or words > self._s.max_heading_words or text.endswith((",", ";")):
            return False
        if self._caption.match(text):
            return False
        large = line.size >= body_size * self._s.heading_size_ratio
        bold = line.bold and not body_bold
        known = self._normalized_heading(text) in self._known
        numbered = (
            bool(self._numbered.match(text))
            and words <= self._s.max_numbered_heading_words
            and not text.endswith(".")
        )
        return large or bold or known or numbered

    def _is_footnote(self, line: TextLine, body_size: float, page_height: float) -> bool:
        """Small text in the bottom zone of a page."""
        return (
            self._s.detect_footnotes
            and line.top >= page_height * self._s.footnote_zone
            and line.size < body_size * self._s.footnote_size_ratio
        )

    def _is_equation(self, line: TextLine) -> bool:
        """Short line dense in mathematical symbols, or numbered ``(n)`` with an operator."""
        if not self._s.detect_equations:
            return False
        text = line.text.strip()
        if len(text.split()) > self._s.max_equation_words:
            return False
        visible = [ch for ch in text if not ch.isspace()]
        symbols = sum(1 for ch in visible if ch in self._math)
        if visible and symbols / len(visible) >= self._s.equation_symbol_ratio:
            return True
        return bool(self._equation_number.search(text)) and symbols > 0

    def _classify(
        self,
        lines: Sequence[TextLine],
        body_size: float,
        body_bold: bool,
        heights: Mapping[int, float] | None = None,
    ) -> list[ElementType]:
        """Assign a provisional element type to every line."""
        heights = heights or {}
        kinds = [ElementType.PARAGRAPH] * len(lines)
        title_end = self._mark_title(lines, kinds, body_size)
        in_references = False
        in_abstract = False
        front_matter = title_end > 0
        for i in range(title_end, len(lines)):
            line = lines[i]
            if self._is_footnote(line, body_size, heights.get(line.page, float("inf"))):
                kinds[i] = ElementType.FOOTNOTE
            elif self._is_heading(line, body_size, body_bold):
                kinds[i] = ElementType.HEADING
                name = self._normalized_heading(line.text)
                in_references = name in self._reference_headings
                in_abstract = name == "abstract"
                front_matter = False
            elif (
                front_matter
                and line.page == lines[0].page
                and self._abstract_prefix.match(line.text)
            ):
                kinds[i] = ElementType.ABSTRACT
                in_abstract, front_matter = True, False
            elif (
                front_matter and line.page == lines[0].page and self._looks_like_authors(line.text)
            ):
                kinds[i] = ElementType.AUTHORS
            elif self._keywords_prefix.match(line.text):
                in_abstract = front_matter = False  # keywords end the abstract
            elif in_references:
                kinds[i] = ElementType.REFERENCE
            elif self._caption.match(line.text):
                kinds[i] = ElementType.CAPTION
            elif self._is_equation(line):
                kinds[i] = ElementType.EQUATION
            elif in_abstract:
                kinds[i] = ElementType.ABSTRACT
        return kinds

    def _mark_title(
        self, lines: Sequence[TextLine], kinds: list[ElementType], body_size: float
    ) -> int:
        """Mark the title lines on the first page; return the index after the title."""
        if not lines:
            return 0
        first_page = lines[0].page
        window = [
            (i, line)
            for i, line in enumerate(lines[: self._s.title_search_lines])
            if line.page == first_page
        ]
        largest = max(line.size for _, line in window)
        if largest < body_size * self._s.title_size_ratio:
            return 0
        start = next(i for i, line in window if abs(line.size - largest) < _SAME_SIZE)
        end = start
        while (
            end < len(lines)
            and lines[end].page == first_page
            and abs(lines[end].size - largest) < _SAME_SIZE
        ):
            kinds[end] = ElementType.TITLE
            end += 1
        return end

    def _looks_like_authors(self, text: str) -> bool:
        """Heuristic test for an author line in the front matter."""
        lowered = text.lower()
        if any(marker in lowered for marker in self._s.affiliation_markers):
            return False
        names = [part for part in re.split(r",|;|\band\b|&", text) if part.strip()]
        capitalised = [p for p in names if re.match(r"^\s*[A-Z][\w.'\-]*(?:\s+[A-Z][\w.'\-]*)+", p)]
        return (
            bool(names)
            and len(capitalised) * 2 >= len(names)
            and len(text.split()) <= _MAX_AUTHOR_LINE_WORDS
        )

    # ------------------------------------------------------------------ #
    # Blocks
    # ------------------------------------------------------------------ #
    def _assemble(
        self,
        lines: Sequence[TextLine],
        kinds: Sequence[ElementType],
        pitch: float,
        pages: Sequence[RawPage],
    ) -> list[tuple[LayoutBlock, float]]:
        """Merge lines into blocks and interleave tables; returns ``(block, top)`` pairs."""
        threshold = pitch * self._s.paragraph_gap_ratio
        blocks: list[tuple[LayoutBlock, float]] = []
        current: list[TextLine] = []
        current_kind: ElementType | None = None
        references_numbered = any(
            k is ElementType.REFERENCE and self._ref_marker.match(line.text)
            for line, k in zip(lines, kinds, strict=True)
        )

        def flush() -> None:
            if current and current_kind is not None:
                text = join_lines(line.text for line in current)
                bbox = (
                    min(line.x0 for line in current),
                    min(line.top for line in current),
                    max(line.x1 for line in current),
                    max(line.bottom for line in current),
                )
                block = LayoutBlock(
                    current_kind, text, current[0].page, size=current[0].size, bbox=bbox
                )
                blocks.append((block, current[0].top))

        for line, kind in zip(lines, kinds, strict=True):
            effective = kind
            if kind is ElementType.PARAGRAPH and current_kind is ElementType.CAPTION:
                effective = ElementType.CAPTION
            if current and self._continues(
                current[-1],
                line,
                current_kind,
                effective,
                threshold=threshold,
                references_numbered=references_numbered,
            ):
                current.append(line)
                continue
            flush()
            current, current_kind = [line], kind
        flush()

        for page in pages:
            for table in page.tables:
                text = "\n".join(" | ".join(row) for row in table.rows).strip()
                if text:
                    blocks.append(
                        (
                            LayoutBlock(
                                ElementType.TABLE,
                                text,
                                page.number,
                                rows=table.rows,
                                bbox=table.bbox,
                            ),
                            table.bbox[1],
                        )
                    )
        return blocks

    def _continues(
        self,
        previous: TextLine,
        line: TextLine,
        current_kind: ElementType | None,
        kind: ElementType,
        *,
        threshold: float,
        references_numbered: bool,
    ) -> bool:
        """Return whether ``line`` continues the block that ends with ``previous``."""
        if kind is not current_kind or line.page != previous.page:
            return False
        if kind in (ElementType.TITLE, ElementType.AUTHORS):
            return True
        if kind is ElementType.HEADING:
            same_size = abs(line.size - previous.size) < _SAME_SIZE
            return same_size and 0 < line.top - previous.top <= threshold
        if kind is ElementType.REFERENCE and references_numbered:
            return not self._ref_marker.match(line.text)
        if self._opens_block(line, kind):
            return False
        return follows(previous, line, threshold)

    def _opens_block(self, line: TextLine, kind: ElementType) -> bool:
        """Whether ``line`` starts a new caption or footnote by its leading marker."""
        if kind is ElementType.CAPTION:
            return bool(self._caption.match(line.text))
        if kind is ElementType.FOOTNOTE:
            return bool(self._footnote_marker.match(line.text))
        return False

    def _assign_levels_and_sections(
        self, placed: Sequence[tuple[LayoutBlock, float]]
    ) -> list[LayoutBlock]:
        """Interleave tables by position, then compute heading levels and section names.

        Numbered headings take their level from the numbering depth (``2.1`` → 2); other
        headings from the rank of their font size among all heading sizes.
        """
        blocks = interleave_tables(placed)
        heading_sizes = sorted(
            {round(b.size, 1) for b in blocks if b.type is ElementType.HEADING}, reverse=True
        )
        result: list[LayoutBlock] = []
        section: str | None = None
        for block in blocks:
            if block.type is ElementType.HEADING:
                section = block.text
                match = self._numbered.match(block.text)
                if match and match.group(0)[0].isdigit():
                    level = match.group(0).strip().rstrip(".").count(".") + 1
                else:
                    level = heading_sizes.index(round(block.size, 1)) + 1
                result.append(replace(block, section=section, level=level))
            else:
                result.append(replace(block, section=section))
        return result
