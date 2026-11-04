"""Structural document elements: element types, bounding boxes, drafts and spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from ips_r.data.fields import (
    require_optional_int,
    require_optional_str,
    require_str,
    require_str_tuple,
    set_field,
)


class ElementType(StrEnum):
    """Kind of a structural document element."""

    TITLE = "title"
    AUTHORS = "authors"
    ABSTRACT = "abstract"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CAPTION = "caption"
    TABLE = "table"
    REFERENCE = "reference"
    EQUATION = "equation"
    FOOTNOTE = "footnote"


BBox = tuple[float, float, float, float]
"""``(x0, top, x1, bottom)`` in PDF points, origin at the top-left of the page."""

_BBOX_LENGTH: Final = 4


def _require_bbox(model: str, value: Any) -> BBox | None:
    """Validate an optional bounding box and normalise it to a float 4-tuple."""
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != _BBOX_LENGTH:
        raise ValueError(f"{model}.bbox must be a 4-tuple (x0, top, x1, bottom), got {value!r}")
    if any(isinstance(v, bool) or not isinstance(v, int | float) for v in value):
        raise ValueError(f"{model}.bbox must contain numbers, got {value!r}")
    x0, top, x1, bottom = (float(v) for v in value)
    if x1 < x0 or bottom < top:
        raise ValueError(f"{model}.bbox must satisfy x0 <= x1 and top <= bottom, got {value!r}")
    return (x0, top, x1, bottom)


@dataclass(frozen=True, slots=True)
class ElementDraft:
    """A structural element before it is placed in a document's text.

    Use :meth:`Document.from_elements` to assemble drafts into a :class:`Document`
    whose element offsets are computed automatically.
    """

    type: ElementType
    text: str
    page: int | None = None
    section: str | None = None
    level: int | None = None
    rows: tuple[tuple[str, ...], ...] = ()
    bbox: BBox | None = None

    def __post_init__(self) -> None:
        set_field(self, "type", ElementType(self.type))
        set_field(self, "bbox", _require_bbox("ElementDraft", self.bbox))
        require_str("ElementDraft", "text", self.text, allow_empty=False)
        require_optional_int("ElementDraft", "page", self.page, minimum=1)
        require_optional_int("ElementDraft", "level", self.level, minimum=1)
        require_optional_str("ElementDraft", "section", self.section)
        set_field(
            self, "rows", tuple(require_str_tuple("ElementDraft", "rows", r) for r in self.rows)
        )


@dataclass(frozen=True, slots=True)
class DocumentElement:
    """A structural element occupying ``text[start:end]`` of its document.

    Attributes:
        type: Element kind.
        text: Element text (equals the document text slice).
        start: Start character offset in the document text (inclusive).
        end: End character offset (exclusive).
        page: 1-based page number for paginated sources.
        section: Heading of the section that contains the element.
        level: Heading level (1 = top level) for headings.
        rows: Table cells, row-major, for tables.
        bbox: Bounding box on the page (paginated sources only).
    """

    type: ElementType
    text: str
    start: int
    end: int
    page: int | None = None
    section: str | None = None
    level: int | None = None
    rows: tuple[tuple[str, ...], ...] = ()
    bbox: BBox | None = None

    def __post_init__(self) -> None:
        set_field(self, "type", ElementType(self.type))
        set_field(self, "bbox", _require_bbox("DocumentElement", self.bbox))
        require_str("DocumentElement", "text", self.text, allow_empty=False)
        require_optional_int("DocumentElement", "start", self.start, minimum=0)
        require_optional_int("DocumentElement", "page", self.page, minimum=1)
        require_optional_int("DocumentElement", "level", self.level, minimum=1)
        require_optional_str("DocumentElement", "section", self.section)
        if self.end - self.start != len(self.text):
            raise ValueError(
                f"DocumentElement span [{self.start}, {self.end}) does not match text length "
                f"{len(self.text)}"
            )
        set_field(
            self, "rows", tuple(require_str_tuple("DocumentElement", "rows", r) for r in self.rows)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "type": self.type.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "section": self.section,
            "level": self.level,
            "rows": [list(row) for row in self.rows],
            "bbox": list(self.bbox) if self.bbox is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DocumentElement:
        """Rebuild an element from :meth:`to_dict` output.

        Raises:
            ValueError: If the data is invalid. ``KeyError`` if a required key is missing.
        """
        return cls(
            type=ElementType(data["type"]),
            text=data["text"],
            start=data["start"],
            end=data["end"],
            page=data.get("page"),
            section=data.get("section"),
            level=data.get("level"),
            rows=tuple(tuple(row) for row in data.get("rows", ())),
            bbox=tuple(data["bbox"]) if data.get("bbox") is not None else None,
        )
