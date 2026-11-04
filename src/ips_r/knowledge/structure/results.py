"""Outputs and per-document state of structural analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

from ips_r.data.models import Document, DocumentElement
from ips_r.knowledge.models import (
    ChunkKind,
    Citation,
    CrossReference,
    Equation,
    Figure,
    Footnote,
    Paragraph,
    Reference,
    Section,
    SourceSpan,
    Table,
)
from ips_r.knowledge.normalization import TrackedText

FLOW_SEPARATOR: Final = "\n\n"


@dataclass(frozen=True, slots=True)
class TextUnit:
    """A normalised text region that chunk processing cuts into chunks.

    Attributes:
        id: Id of the knowledge object the unit belongs to (its chunks' ``parent_id``).
        kind: Kind of chunks produced from the unit.
        section_id: Enclosing section.
        order: Document position (units are chunked in this order).
        text: Normalised text with its original-offset map.
        atomic: Never split (tables, captions, footnotes, references), except oversized
            tables, which are split by rows.
        table_rows: Normalised table rows, for row-wise splitting of oversized tables.
        header: Caption/header lines repeated in every row-wise table chunk.
    """

    id: str
    kind: ChunkKind
    section_id: str
    order: int
    text: TrackedText
    atomic: bool = False
    table_rows: tuple[str, ...] = ()
    header: str = ""


@dataclass(slots=True)
class StructureResult:
    """Knowledge objects and text units of one document."""

    sections: list[Section] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    cross_references: list[CrossReference] = field(default_factory=list)
    footnotes: list[Footnote] = field(default_factory=list)
    units: list[TextUnit] = field(default_factory=list)


@dataclass(slots=True)
class Mention:
    """An unresolved citation or cross reference found in running text."""

    host_id: str
    section_id: str
    order: int
    match: re.Match[str]
    span: SourceSpan
    is_citation: bool
    style: str = ""


class AnalysisState:
    """Mutable state of one structural analysis run."""

    def __init__(self, document: Document) -> None:
        self.document = document
        self.result = StructureResult()
        self.order = 0
        self.flows: dict[str, list[TrackedText]] = {}
        self.flow_order: dict[str, int] = {}
        self.mentions: list[Mention] = []
        self.counters: dict[str, int] = {}

    def next_order(self) -> int:
        """Return the next document position."""
        self.order += 1
        return self.order

    def new_id(self, prefix: str) -> str:
        """Return the next deterministic object id with ``prefix``."""
        count = self.counters.get(prefix, 0)
        self.counters[prefix] = count + 1
        return f"{self.document.id}#{prefix}{count:04d}"

    def span(
        self, tracked: TrackedText, element: DocumentElement | None, index: int | None
    ) -> SourceSpan:
        """Original-document span of a whole normalised element."""
        start, end = tracked.source_span(0, len(tracked))
        return SourceSpan(
            self.document.id,
            start,
            end,
            page=element.page if element else None,
            bbox=element.bbox if element else None,
            element_index=index,
        )
