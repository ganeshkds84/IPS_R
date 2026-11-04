"""Primitive knowledge types: object kinds, :class:`SourceSpan` and shared validators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from ips_r.data.models import BBox


class KnowledgeKind(StrEnum):
    """Kind of a knowledge object (also the artifact table name stem)."""

    SECTION = "section"
    PARAGRAPH = "paragraph"
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    REFERENCE = "reference"
    CITATION = "citation"
    CROSS_REFERENCE = "cross_reference"
    FOOTNOTE = "footnote"
    CHUNK = "chunk"


class ChunkKind(StrEnum):
    """What a chunk contains."""

    FRONT_MATTER = "front_matter"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"


def check(condition: bool, message: str) -> None:
    """Raise :class:`ValueError` with ``message`` unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def nonempty(model: str, name: str, value: Any) -> None:
    """Require ``value`` (field ``model.name``) to be a non-empty string."""
    check(isinstance(value, str) and bool(value), f"{model}.{name} must be a non-empty string")


def freeze(instance: object, name: str) -> None:
    """Replace a mapping field of a frozen dataclass by a read-only copy."""
    object.__setattr__(instance, name, MappingProxyType(dict(getattr(instance, name))))


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Location of a knowledge object in its original document.

    Attributes:
        document_id: Source document.
        start: Start offset in the original ``Document.text`` (inclusive).
        end: End offset (exclusive).
        page: 1-based page (paginated sources only).
        page_end: Last page covered (defaults to ``page``).
        bbox: Bounding box on ``page`` (PDF points).
        element_index: Index of the originating element in ``Document.elements``.
    """

    document_id: str
    start: int
    end: int
    page: int | None = None
    page_end: int | None = None
    bbox: BBox | None = None
    element_index: int | None = None

    def __post_init__(self) -> None:
        nonempty("SourceSpan", "document_id", self.document_id)
        check(0 <= self.start <= self.end, f"SourceSpan [{self.start}, {self.end}) is invalid")
        if self.page_end is None:
            object.__setattr__(self, "page_end", self.page)
        check(
            self.page is None or (self.page >= 1 and (self.page_end or 0) >= self.page),
            "SourceSpan page range is invalid",
        )
        if self.bbox is not None:
            object.__setattr__(self, "bbox", tuple(float(v) for v in self.bbox))
