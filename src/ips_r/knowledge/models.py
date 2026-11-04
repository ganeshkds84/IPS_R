"""Knowledge objects: the structured, traceable representation of processed documents.

Every object carries:

- ``id``: deterministic, derived from the document id and the object's position or content;
- ``document_id`` and ``order``: the object's position in document reading order;
- ``section_id``: the enclosing :class:`Section` (the parent section);
- ``source``: a :class:`SourceSpan` with the exact character span in the *original*
  :class:`~ips_r.data.models.Document` text, the page, the bounding box and the
  index of the originating document element.

Objects are frozen, validate their invariants, and convert to and from flat
JSON-compatible records (:func:`to_record` / :func:`from_record`) for serialization.

Definitions live in :mod:`~ips_r.knowledge.primitives` (kinds, spans),
:mod:`~ips_r.knowledge.objects` (structural objects) and
:mod:`~ips_r.knowledge.records` (record codec); this module adds chunks and
documents and re-exports everything.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from ips_r.data.models import DocumentMetadata
from ips_r.knowledge.objects import (
    Citation,
    CrossReference,
    Equation,
    Figure,
    Footnote,
    Paragraph,
    Reference,
    Section,
    Table,
)
from ips_r.knowledge.primitives import ChunkKind, KnowledgeKind, SourceSpan, check, freeze, nonempty
from ips_r.knowledge.records import field_types, from_record, to_record

__all__ = [
    "KNOWLEDGE_SCHEMA_VERSION",
    "KNOWLEDGE_TYPES",
    "ChunkKind",
    "Citation",
    "CrossReference",
    "Equation",
    "Figure",
    "Footnote",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "KnowledgeKind",
    "Paragraph",
    "Reference",
    "Section",
    "SourceSpan",
    "Table",
    "field_types",
    "from_record",
    "to_record",
]

KNOWLEDGE_SCHEMA_VERSION: Final = "1.0"
"""Version of the knowledge artifact schema; bump on any incompatible field change."""


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """An index-ready chunk with full lineage.

    Attributes:
        id: Deterministic id derived from the document id, parent unit and content.
        index: Position among the document's chunks (reading order).
        parent_id: Knowledge object the chunk was cut from (section, table, figure, ...).
        unit_start: Start offset within the parent unit's normalised text.
        unit_end: End offset within the parent unit's normalised text.
        prev_id: Previous chunk of the same document (adjacent link).
        next_id: Next chunk of the same document (adjacent link).
        duplicate_of: Id of an earlier chunk with identical normalised text, if any.
        content_hash: Hash of ``text``.
        lineage_hash: Hash of ``text`` + source document hash + processing fingerprint.
        metadata: Enrichment fields (see :class:`~ips_r.knowledge.enrichment.MetadataEnricher`).
    """

    id: str
    document_id: str
    index: int
    text: str
    kind: ChunkKind
    section_id: str
    parent_id: str
    unit_start: int
    unit_end: int
    source: SourceSpan
    section_path: tuple[str, ...] = ()
    prev_id: str | None = None
    next_id: str | None = None
    duplicate_of: str | None = None
    content_hash: str = ""
    lineage_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nonempty("KnowledgeChunk", "id", self.id)
        nonempty("KnowledgeChunk", "text", self.text)
        check(self.index >= 0, "KnowledgeChunk.index must be >= 0")
        check(0 <= self.unit_start < self.unit_end, "KnowledgeChunk unit span is invalid")
        object.__setattr__(self, "kind", ChunkKind(self.kind))
        object.__setattr__(self, "section_path", tuple(self.section_path))
        freeze(self, "metadata")


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """All knowledge objects derived from one source document."""

    id: str
    source_hash: str
    knowledge_hash: str
    metadata: DocumentMetadata
    sections: tuple[Section, ...]
    paragraphs: tuple[Paragraph, ...] = ()
    figures: tuple[Figure, ...] = ()
    tables: tuple[Table, ...] = ()
    equations: tuple[Equation, ...] = ()
    references: tuple[Reference, ...] = ()
    citations: tuple[Citation, ...] = ()
    cross_references: tuple[CrossReference, ...] = ()
    footnotes: tuple[Footnote, ...] = ()
    chunks: tuple[KnowledgeChunk, ...] = ()
    num_pages: int | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        nonempty("KnowledgeDocument", "id", self.id)
        check(bool(self.sections) and self.sections[0].level == 0, "first section must be the root")
        for name, items in self.objects().items():
            ids = [item.id for item in items]
            check(len(ids) == len(set(ids)), f"KnowledgeDocument {self.id!r}: duplicate {name} ids")

    def objects(self) -> dict[str, tuple[Any, ...]]:
        """Knowledge objects by table name (``sections``, ``paragraphs``, ...)."""
        return {
            "sections": self.sections,
            "paragraphs": self.paragraphs,
            "figures": self.figures,
            "tables": self.tables,
            "equations": self.equations,
            "references": self.references,
            "citations": self.citations,
            "cross_references": self.cross_references,
            "footnotes": self.footnotes,
            "chunks": self.chunks,
        }

    def summary_record(self) -> dict[str, Any]:
        """Document-level record (the ``documents`` artifact table)."""
        return {
            "id": self.id,
            "source_hash": self.source_hash,
            "knowledge_hash": self.knowledge_hash,
            "title": self.metadata.title,
            "metadata": json.dumps(self.metadata.to_dict(), sort_keys=True, ensure_ascii=False),
            "num_pages": self.num_pages,
            "source": self.source,
            **{f"num_{name}": len(items) for name, items in self.objects().items()},
        }


KNOWLEDGE_TYPES: Final[Mapping[str, type[Any]]] = {
    "sections": Section,
    "paragraphs": Paragraph,
    "figures": Figure,
    "tables": Table,
    "equations": Equation,
    "references": Reference,
    "citations": Citation,
    "cross_references": CrossReference,
    "footnotes": Footnote,
    "chunks": KnowledgeChunk,
}
"""Artifact table name → knowledge object class."""
