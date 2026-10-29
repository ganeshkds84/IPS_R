"""Rule-based layout analysis: positioned words → typed structural blocks.

The analysis is pure (no I/O) and deterministic. Pipeline:

1. Remove words inside detected tables; group the rest into lines.
2. Split two-column pages and order lines in reading order.
3. Estimate the body font size and line pitch.
4. Classify lines (title, authors, heading, caption, reference, paragraph) using font
   size, weight, numbering, known heading names and caption patterns.
5. Merge consecutive lines into blocks (paragraph breaks are detected from vertical
   gaps), insert tables, and attach the enclosing section heading to each block.

Blocks never span pages, so every block carries an exact 1-based page number.
All thresholds come from :class:`LayoutSettings` (``params.parser.layout`` in config).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

from ips_r.core.exceptions import ConfigurationError
from ips_r.data.models import ElementType
from ips_r.data.pdf.backend import RawPage, RawPDF, RawTable, RawWord


@dataclass(frozen=True, slots=True)
class LayoutSettings:
    """Thresholds and patterns of the layout analysis (all lengths in PDF points)."""

    line_tolerance: float = 2.0
    column_gap: float = 12.0
    detect_columns: bool = True
    heading_size_ratio: float = 1.15
    title_size_ratio: float = 1.3
    title_search_lines: int = 15
    max_heading_words: int = 15
    max_numbered_heading_words: int = 8
    paragraph_gap_ratio: float = 1.4
    drop_page_numbers: bool = True
    page_number_pattern: str = r"^(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?$"
    caption_pattern: str = r"^(?:fig(?:ure)?|tab(?:le)?)\.?\s*\d+[.:]"
    numbered_heading_pattern: str = r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+(?=[A-Z])"
    reference_marker_pattern: str = r"^(?:\[\d+\]|\d+\.\s)"
    abstract_prefix_pattern: str = r"^abstract\s*[:.—–-]\s*"
    keywords_prefix_pattern: str = r"^(?:keywords|key words|index terms)\s*[:.—–-]"
    reference_headings: tuple[str, ...] = ("references", "bibliography", "works cited")
    known_headings: tuple[str, ...] = (
        "abstract",
        "introduction",
        "related work",
        "background",
        "method",
        "methods",
        "methodology",
        "experiments",
        "experimental setup",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "limitations",
        "acknowledgments",
        "acknowledgements",
        "appendix",
        "references",
        "bibliography",
    )
    affiliation_markers: tuple[str, ...] = (
        "@",
        "university",
        "institute",
        "department",
        "school",
        "laboratory",
        "college",
        "faculty",
        "inc.",
        "ltd",
        "corporation",
        "research",
        "centre",
        "center",
    )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> LayoutSettings:
        """Build settings from a config mapping, rejecting unknown keys.

        Raises:
            ConfigurationError: On unknown keys or values of the wrong type.
        """
        return settings_from_mapping(cls, mapping, "parser.layout")


def settings_from_mapping[S](cls: type[S], mapping: Mapping[str, Any] | None, where: str) -> S:
    """Instantiate a settings dataclass from a mapping, coercing lists to tuples."""
    values = dict(mapping or {})
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigurationError(f"Unknown {where} settings: {unknown}")
    defaults = cls()
    for key, value in values.items():
        default = getattr(defaults, key)
        if isinstance(default, tuple):
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise ConfigurationError(f"{where}.{key} must be a list, got {value!r}")
            values[key] = tuple(value)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigurationError(f"{where}.{key} must be a boolean, got {value!r}")
        elif isinstance(default, int | float):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigurationError(f"{where}.{key} must be a number, got {value!r}")
            values[key] = type(default)(value)
        elif not isinstance(value, type(default)):
            raise ConfigurationError(
                f"{where}.{key} must be {type(default).__name__}, got {value!r}"
            )
    return replace(defaults, **values)  # type: ignore[type-var]


@dataclass(frozen=True, slots=True)
class TextLine:
    """A line of text in reading order."""

    page: int
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    bold: bool


@dataclass(frozen=True, slots=True)
class LayoutBlock:
    """A structural block produced by the analysis."""

    type: ElementType
    text: str
    page: int
    section: str | None = None
    level: int | None = None
    rows: tuple[tuple[str, ...], ...] = ()
    size: float = 0.0


@dataclass(frozen=True, slots=True)
class LayoutResult:
    """Outcome of :meth:`LayoutAnalyzer.analyze`."""

    blocks: tuple[LayoutBlock, ...]
    num_pages: int
    body_size: float
    title: str | None = None
    authors: str | None = None
    front_text: str = ""
    sections: tuple[str, ...] = field(default=())

    def text_of(self, kind: ElementType) -> list[str]:
        """Texts of all blocks of type ``kind``."""
        return [block.text for block in self.blocks if block.type is kind]


def _dominant(values: Iterable[tuple[float, int]]) -> float:
    """Return the value with the largest total weight (ties → smaller value)."""
    weights: Counter[float] = Counter()
    for value, weight in values:
        weights[round(value, 1)] += weight
    if not weights:
        return 0.0
    return min(weights, key=lambda v: (-weights[v], v))


def _make_line(page: int, words: Sequence[RawWord]) -> TextLine:
    """Combine words (already in x order) into a line."""
    text = " ".join(word.text for word in words)
    return TextLine(
        page=page,
        text=text,
        x0=min(w.x0 for w in words),
        x1=max(w.x1 for w in words),
        top=min(w.top for w in words),
        bottom=max(w.bottom for w in words),
        size=_dominant((w.size, len(w.text)) for w in words),
        bold=sum(len(w.text) for w in words if w.bold) * 2 > sum(len(w.text) for w in words),
    )


def _inside(word: RawWord, table: RawTable, margin: float = 1.0) -> bool:
    """Return whether the centre of ``word`` lies inside ``table``."""
    x0, top, x1, bottom = table.bbox
    cx, cy = (word.x0 + word.x1) / 2, (word.top + word.bottom) / 2
    return x0 - margin <= cx <= x1 + margin and top - margin <= cy <= bottom + margin


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
        except re.error as exc:
            raise ConfigurationError(f"Invalid parser.layout pattern: {exc}") from exc
        self._known = frozenset(self._s.known_headings)
        self._reference_headings = frozenset(self._s.reference_headings)

    # ------------------------------------------------------------------ #
    # Lines
    # ------------------------------------------------------------------ #
    def page_lines(self, page: RawPage) -> list[TextLine]:
        """Group a page's non-table words into lines in reading order."""
        words = [w for w in page.words if not any(_inside(w, t) for t in page.tables)]
        rows: list[list[RawWord]] = []
        for word in sorted(words, key=lambda w: (w.top, w.x0)):
            if rows and abs(word.top - rows[-1][0].top) <= self._s.line_tolerance:
                rows[-1].append(word)
            else:
                rows.append([word])
        segments: list[TextLine] = []
        middle = page.width / 2
        for row in rows:
            row.sort(key=lambda w: w.x0)
            start = 0
            for i in range(1, len(row)):
                gap = row[i].x0 - row[i - 1].x1
                crosses = row[i - 1].x1 <= middle <= row[i].x0
                if self._s.detect_columns and gap >= self._s.column_gap and crosses:
                    segments.append(_make_line(page.number, row[start:i]))
                    start = i
            segments.append(_make_line(page.number, row[start:]))
        segments = [line for line in segments if line.text.strip()]
        if self._s.drop_page_numbers:
            segments = [line for line in segments if not self._page_number.match(line.text)]
        return self._reading_order(segments, middle) if self._s.detect_columns else segments

    @staticmethod
    def _reading_order(lines: list[TextLine], middle: float) -> list[TextLine]:
        """Order a two-column page as: header, left column, right column, footer."""
        left = [line for line in lines if line.x1 <= middle]
        right = [line for line in lines if line.x0 >= middle]
        if not left or not right:
            return lines
        full = [line for line in lines if line.x0 < middle < line.x1]
        column_top = min(line.top for line in (*left, *right))
        header = [line for line in full if line.top < column_top]
        footer = [line for line in full if line.top >= column_top]
        return header + left + right + footer

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def analyze(self, raw: RawPDF) -> LayoutResult:
        """Run the full analysis."""
        page_lines = {page.number: self.page_lines(page) for page in raw.pages}
        lines = [line for page in raw.pages for line in page_lines[page.number]]
        body_size = _dominant((line.size, len(line.text)) for line in lines)
        body_bold = sum(len(line.text) for line in lines if line.bold) * 2 > sum(
            len(line.text) for line in lines
        )
        pitch = self._line_pitch(lines, body_size)
        kinds = self._classify(lines, body_size, body_bold)
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
        """Median vertical distance between consecutive body lines (fallback: 1.2 × size)."""
        deltas = [
            b.top - a.top
            for a, b in zip(lines, lines[1:], strict=False)
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

    def _classify(
        self, lines: Sequence[TextLine], body_size: float, body_bold: bool
    ) -> list[ElementType]:
        """Assign a provisional element type to every line."""
        kinds = [ElementType.PARAGRAPH] * len(lines)
        title_end = self._mark_title(lines, kinds, body_size)
        in_references = False
        in_abstract = False
        front_matter = title_end > 0
        for i in range(title_end, len(lines)):
            line = lines[i]
            if self._is_heading(line, body_size, body_bold):
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
        start = next(i for i, line in window if abs(line.size - largest) < 0.5)
        end = start
        while (
            end < len(lines)
            and lines[end].page == first_page
            and abs(lines[end].size - largest) < 0.5
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
        return bool(names) and len(capitalised) * 2 >= len(names) and len(text.split()) <= 30

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
                text = _join_lines(line.text for line in current)
                block = LayoutBlock(current_kind, text, current[0].page, size=current[0].size)
                blocks.append((block, current[0].top))

        for line, kind in zip(lines, kinds, strict=True):
            effective = kind
            if kind is ElementType.PARAGRAPH and current_kind is ElementType.CAPTION:
                effective = ElementType.CAPTION
            if current and self._continues(
                current[-1], line, current_kind, effective, threshold, references_numbered
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
                            LayoutBlock(ElementType.TABLE, text, page.number, rows=table.rows),
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
        threshold: float,
        references_numbered: bool,
    ) -> bool:
        """Return whether ``line`` continues the block that ends with ``previous``."""
        if kind is not current_kind or line.page != previous.page:
            return False
        if kind in (ElementType.TITLE, ElementType.AUTHORS):
            return True
        if kind is ElementType.HEADING:
            return abs(line.size - previous.size) < 0.5 and 0 < line.top - previous.top <= threshold
        if kind is ElementType.CAPTION and self._caption.match(line.text):
            return False
        if kind is ElementType.REFERENCE and references_numbered:
            return not self._ref_marker.match(line.text)
        delta = line.top - previous.top
        if delta < 0:  # column jump: continue unless the previous line ended a sentence
            return not previous.text.rstrip().endswith((".", "!", "?", ":"))
        return delta <= threshold

    def _assign_levels_and_sections(
        self, placed: Sequence[tuple[LayoutBlock, float]]
    ) -> list[LayoutBlock]:
        """Interleave tables by position, then compute heading levels and section names.

        Numbered headings take their level from the numbering depth (``2.1`` → 2); other
        headings from the rank of their font size among all heading sizes.
        """
        blocks = self._interleave(placed)
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

    @staticmethod
    def _interleave(placed: Sequence[tuple[LayoutBlock, float]]) -> list[LayoutBlock]:
        """Place table blocks among text blocks of the same page by vertical position."""
        text_blocks = [(b, top) for b, top in placed if b.type is not ElementType.TABLE]
        tables = [(b, top) for b, top in placed if b.type is ElementType.TABLE]
        result: list[LayoutBlock] = []
        pending = sorted(tables, key=lambda item: (item[0].page, item[1]))
        for block, top in text_blocks:
            while pending and (
                pending[0][0].page < block.page
                or (pending[0][0].page == block.page and pending[0][1] < top)
            ):
                result.append(pending.pop(0)[0])
            result.append(block)
        result.extend(block for block, _ in pending)
        return result


def _join_lines(texts: Iterable[str]) -> str:
    """Join line texts, repairing end-of-line hyphenation."""
    out = ""
    for raw in texts:
        text = raw.strip()
        if not text:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and text[:1].islower():
            out = out[:-1] + text
        elif out:
            out = f"{out} {text}"
        else:
            out = text
    return out
