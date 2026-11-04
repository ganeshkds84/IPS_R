"""Text lines and layout blocks: grouping positioned words into lines in reading order."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ips_r.data.models import BBox, ElementType
from ips_r.data.pdf.backend import RawPage, RawTable, RawWord
from ips_r.data.pdf.layout_settings import LayoutSettings


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
    bbox: BBox | None = None


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


def dominant(values: Iterable[tuple[float, int]]) -> float:
    """Return the value with the largest total weight (ties → smaller value)."""
    weights: Counter[float] = Counter()
    for value, weight in values:
        weights[round(value, 1)] += weight
    if not weights:
        return 0.0
    return min(weights, key=lambda v: (-weights[v], v))


def make_line(page: int, words: Sequence[RawWord]) -> TextLine:
    """Combine words (already in x order) into a line."""
    text = " ".join(word.text for word in words)
    return TextLine(
        page=page,
        text=text,
        x0=min(w.x0 for w in words),
        x1=max(w.x1 for w in words),
        top=min(w.top for w in words),
        bottom=max(w.bottom for w in words),
        size=dominant((w.size, len(w.text)) for w in words),
        bold=sum(len(w.text) for w in words if w.bold) * 2 > sum(len(w.text) for w in words),
    )


def inside_table(word: RawWord, table: RawTable, margin: float = 1.0) -> bool:
    """Return whether the centre of ``word`` lies inside ``table``."""
    x0, top, x1, bottom = table.bbox
    cx, cy = (word.x0 + word.x1) / 2, (word.top + word.bottom) / 2
    return x0 - margin <= cx <= x1 + margin and top - margin <= cy <= bottom + margin


def group_lines(
    page: RawPage, settings: LayoutSettings, page_number: re.Pattern[str]
) -> list[TextLine]:
    """Group a page's non-table words into lines in reading order."""
    words = [w for w in page.words if not any(inside_table(w, t) for t in page.tables)]
    rows: list[list[RawWord]] = []
    for word in sorted(words, key=lambda w: (w.top, w.x0)):
        if rows and abs(word.top - rows[-1][0].top) <= settings.line_tolerance:
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
            if settings.detect_columns and gap >= settings.column_gap and crosses:
                segments.append(make_line(page.number, row[start:i]))
                start = i
        segments.append(make_line(page.number, row[start:]))
    segments = [line for line in segments if line.text.strip()]
    if settings.drop_page_numbers:
        segments = [line for line in segments if not page_number.match(line.text)]
    return reading_order(segments, middle) if settings.detect_columns else segments


def reading_order(lines: list[TextLine], middle: float) -> list[TextLine]:
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


def follows(previous: TextLine, line: TextLine, threshold: float) -> bool:
    """Whether ``line`` follows ``previous`` closely enough to share its block."""
    delta = line.top - previous.top
    if delta < 0:  # column jump: continue unless the previous line ended a sentence
        return not previous.text.rstrip().endswith((".", "!", "?", ":"))
    return delta <= threshold


def interleave_tables(placed: Sequence[tuple[LayoutBlock, float]]) -> list[LayoutBlock]:
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


def join_lines(texts: Iterable[str]) -> str:
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
