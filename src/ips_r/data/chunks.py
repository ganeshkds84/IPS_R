"""Chunks: contiguous document spans produced by a chunker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ips_r.core import types as core
from ips_r.data.fields import (
    frozen_mapping,
    require_optional_int,
    require_optional_str,
    require_str,
    set_field,
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous span ``document.text[start:end]`` prepared for indexing.

    Attributes:
        id: ``"<document_id>::<index>"``, unique within a prepared dataset.
        document_id: Id of the parent document.
        text: Chunk text, exactly the parent text slice.
        index: 0-based position of the chunk within its document.
        start: Start character offset in the parent text (inclusive).
        end: End character offset (exclusive).
        strategy: Name of the chunking strategy that produced the chunk.
        page_start: First page covered (paginated sources only).
        page_end: Last page covered (paginated sources only).
        section: Heading of the section where the chunk starts.
        metadata: Free-form, read-only extras.
    """

    id: str
    document_id: str
    text: str
    index: int
    start: int
    end: int
    strategy: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_str("Chunk", "id", self.id, allow_empty=False)
        require_str("Chunk", "document_id", self.document_id, allow_empty=False)
        require_str("Chunk", "text", self.text, allow_empty=False)
        require_str("Chunk", "strategy", self.strategy, allow_empty=False)
        require_optional_int("Chunk", "index", self.index, minimum=0)
        require_optional_int("Chunk", "start", self.start, minimum=0)
        require_optional_int("Chunk", "page_start", self.page_start, minimum=1)
        require_optional_int("Chunk", "page_end", self.page_end, minimum=1)
        require_optional_str("Chunk", "section", self.section)
        if self.end - self.start != len(self.text):
            raise ValueError(
                f"Chunk span [{self.start}, {self.end}) does not match text length {len(self.text)}"
            )
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("Chunk.page_start and Chunk.page_end must both be set or both None")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("Chunk.page_end must be >= Chunk.page_start")
        set_field(self, "metadata", frozen_mapping("Chunk", "metadata", self.metadata))

    @property
    def num_chars(self) -> int:
        """Length of the chunk in characters."""
        return len(self.text)

    def to_core(self) -> core.Document:
        """Convert to :class:`ips_r.core.types.Document`, the unit retrievers index."""
        return core.Document(
            id=self.id,
            text=self.text,
            metadata={
                "document_id": self.document_id,
                "index": self.index,
                "start": self.start,
                "end": self.end,
                "page_start": self.page_start,
                "page_end": self.page_end,
                "section": self.section,
                **self.metadata,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "strategy": self.strategy,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Chunk:
        """Rebuild a chunk from :meth:`to_dict` output."""
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            text=data["text"],
            index=data["index"],
            start=data["start"],
            end=data["end"],
            strategy=data["strategy"],
            page_start=data.get("page_start"),
            page_end=data.get("page_end"),
            section=data.get("section"),
            metadata=data.get("metadata", {}),
        )
