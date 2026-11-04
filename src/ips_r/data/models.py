"""Canonical, strongly typed data models of the dataset layer.

Every dataset adapter converts its source format into these models, so the rest of the
system sees one representation:

- :class:`Document`: a source document (QA passage, web page, academic paper) with
  bibliographic :class:`DocumentMetadata` and optional structural :class:`DocumentElement`
  spans (headings, paragraphs, captions, tables, references, ...).
- :class:`Chunk`: a contiguous span of a document produced by a chunker.
- :class:`Question`: a question with gold answers and supporting document ids.
- :class:`DatasetMetadata`: provenance and size of a dataset.

Models are frozen and validate *structural* invariants in ``__post_init__`` (raising
:class:`ValueError`). *Semantic* checks that should be reported rather than raised
(empty text, implausible years, malformed DOIs, ...) live in
:mod:`ips_r.data.validation`.

All models round-trip through ``to_dict`` / ``from_dict`` (JSON-serialisable).

Elements live in :mod:`ips_r.data.elements`, chunks in :mod:`ips_r.data.chunks` and
shared field validators in :mod:`ips_r.data.fields`; every model is re-exported here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from ips_r.core import types as core
from ips_r.data.chunks import Chunk
from ips_r.data.elements import BBox, DocumentElement, ElementDraft, ElementType
from ips_r.data.fields import (
    frozen_mapping,
    require_optional_int,
    require_optional_str,
    require_str,
    require_str_tuple,
    set_field,
)

__all__ = [
    "ELEMENT_SEPARATOR",
    "BBox",
    "Chunk",
    "DatasetMetadata",
    "Document",
    "DocumentElement",
    "DocumentMetadata",
    "ElementDraft",
    "ElementType",
    "Question",
]

ELEMENT_SEPARATOR: Final = "\n\n"
"""Separator inserted between elements when a document's text is assembled from elements."""


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Bibliographic metadata of a document. Every field is optional."""

    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    keywords: tuple[str, ...] = ()
    abstract: str | None = None
    language: str | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        for name in ("title", "venue", "doi", "abstract", "language", "url"):
            require_optional_str("DocumentMetadata", name, getattr(self, name))
        if self.year is not None and (
            isinstance(self.year, bool) or not isinstance(self.year, int)
        ):
            raise ValueError(f"DocumentMetadata.year must be an integer, got {self.year!r}")
        set_field(self, "authors", require_str_tuple("DocumentMetadata", "authors", self.authors))
        set_field(
            self, "keywords", require_str_tuple("DocumentMetadata", "keywords", self.keywords)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "keywords": list(self.keywords),
            "abstract": self.abstract,
            "language": self.language,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DocumentMetadata:
        """Rebuild metadata from :meth:`to_dict` output."""
        return cls(
            title=data.get("title"),
            authors=tuple(data.get("authors", ())),
            year=data.get("year"),
            venue=data.get("venue"),
            doi=data.get("doi"),
            keywords=tuple(data.get("keywords", ())),
            abstract=data.get("abstract"),
            language=data.get("language"),
            url=data.get("url"),
        )


@dataclass(frozen=True, slots=True)
class Document:
    """A canonical source document.

    Attributes:
        id: Stable, dataset-unique identifier derived from the source.
        text: Full plain text. Empty text is structurally allowed but reported by validation.
        metadata: Bibliographic metadata.
        elements: Structural spans, sorted and non-overlapping, each equal to its text slice.
        source: Origin of the document (file path, URL, ...).
        num_pages: Page count for paginated sources.
        extra: Free-form, adapter-specific information (read-only).
    """

    id: str
    text: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    elements: tuple[DocumentElement, ...] = ()
    source: str | None = None
    num_pages: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_str("Document", "id", self.id, allow_empty=False)
        require_str("Document", "text", self.text)
        require_optional_str("Document", "source", self.source)
        require_optional_int("Document", "num_pages", self.num_pages, minimum=1)
        if not isinstance(self.metadata, DocumentMetadata):
            raise ValueError("Document.metadata must be a DocumentMetadata instance")
        elements = tuple(self.elements)
        previous_end = 0
        for element in elements:
            if element.start < previous_end or element.end > len(self.text):
                raise ValueError(
                    f"Document {self.id!r}: element spans must be sorted, non-overlapping "
                    f"and within the text, got [{element.start}, {element.end})"
                )
            if self.text[element.start : element.end] != element.text:
                raise ValueError(f"Document {self.id!r}: element text does not match its span")
            if (
                self.num_pages is not None
                and element.page is not None
                and element.page > self.num_pages
            ):
                raise ValueError(
                    f"Document {self.id!r}: element page {element.page} exceeds "
                    f"num_pages={self.num_pages}"
                )
            previous_end = element.end
        set_field(self, "elements", elements)
        set_field(self, "extra", frozen_mapping("Document", "extra", self.extra))

    @property
    def title(self) -> str | None:
        """Shortcut for ``metadata.title``."""
        return self.metadata.title

    @classmethod
    def from_elements(
        cls,
        id: str,
        drafts: Sequence[ElementDraft],
        *,
        metadata: DocumentMetadata | None = None,
        source: str | None = None,
        num_pages: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Document:
        """Assemble a document whose text is the drafts joined by :data:`ELEMENT_SEPARATOR`.

        Raises:
            ValueError: If any field is invalid.
        """
        parts: list[str] = []
        elements: list[DocumentElement] = []
        offset = 0
        for draft in drafts:
            if parts:
                parts.append(ELEMENT_SEPARATOR)
                offset += len(ELEMENT_SEPARATOR)
            parts.append(draft.text)
            elements.append(
                DocumentElement(
                    type=draft.type,
                    text=draft.text,
                    start=offset,
                    end=offset + len(draft.text),
                    page=draft.page,
                    section=draft.section,
                    level=draft.level,
                    rows=draft.rows,
                    bbox=draft.bbox,
                )
            )
            offset += len(draft.text)
        return cls(
            id=id,
            text="".join(parts),
            metadata=metadata or DocumentMetadata(),
            elements=tuple(elements),
            source=source,
            num_pages=num_pages,
            extra=extra or {},
        )

    def elements_in(self, start: int, end: int) -> tuple[DocumentElement, ...]:
        """Return the elements overlapping the character range ``[start, end)``."""
        return tuple(e for e in self.elements if e.start < end and e.end > start)

    def to_core(self) -> core.Document:
        """Convert to the provisional retrieval contract :class:`ips_r.core.types.Document`."""
        return core.Document(
            id=self.id,
            text=self.text,
            metadata={"title": self.title, "source": self.source, **self.extra},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "elements": [element.to_dict() for element in self.elements],
            "source": self.source,
            "num_pages": self.num_pages,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Document:
        """Rebuild a document from :meth:`to_dict` output."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=DocumentMetadata.from_dict(data.get("metadata", {})),
            elements=tuple(DocumentElement.from_dict(e) for e in data.get("elements", ())),
            source=data.get("source"),
            num_pages=data.get("num_pages"),
            extra=data.get("extra", {}),
        )


@dataclass(frozen=True, slots=True)
class Question:
    """A question with gold annotations.

    Attributes:
        id: Stable, dataset-unique identifier.
        text: Question text. Empty text is structurally allowed but reported by validation.
        answers: Gold answer strings (empty for ranking-only datasets).
        supporting_document_ids: Ids of documents that support the answer.
        question_type: Dataset-specific category (``bridge``, ``extractive``, ...).
        metadata: Free-form, read-only extras (answer offsets, difficulty, ...).
    """

    id: str
    text: str
    answers: tuple[str, ...] = ()
    supporting_document_ids: tuple[str, ...] = ()
    question_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_str("Question", "id", self.id, allow_empty=False)
        require_str("Question", "text", self.text)
        require_optional_str("Question", "question_type", self.question_type)
        set_field(self, "answers", require_str_tuple("Question", "answers", self.answers))
        set_field(
            self,
            "supporting_document_ids",
            require_str_tuple("Question", "supporting_document_ids", self.supporting_document_ids),
        )
        set_field(self, "metadata", frozen_mapping("Question", "metadata", self.metadata))

    def to_query(self) -> core.Query:
        """Convert to :class:`ips_r.core.types.Query`."""
        return core.Query(id=self.id, text=self.text, metadata=dict(self.metadata))

    def to_reference(self) -> core.Reference:
        """Convert to :class:`ips_r.core.types.Reference`."""
        return core.Reference(
            query_id=self.id,
            answers=self.answers,
            relevant_ids=self.supporting_document_ids,
            metadata={"question_type": self.question_type},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.id,
            "text": self.text,
            "answers": list(self.answers),
            "supporting_document_ids": list(self.supporting_document_ids),
            "question_type": self.question_type,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Question:
        """Rebuild a question from :meth:`to_dict` output."""
        return cls(
            id=data["id"],
            text=data["text"],
            answers=tuple(data.get("answers", ())),
            supporting_document_ids=tuple(data.get("supporting_document_ids", ())),
            question_type=data.get("question_type"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Provenance and size of a dataset.

    Counts and ``content_hash`` are ``None`` until the dataset has been fully processed
    (see :func:`ips_r.data.preparation.prepare_dataset`).
    """

    name: str
    adapter: str
    description: str = ""
    split: str | None = None
    version: str | None = None
    homepage: str | None = None
    license: str | None = None
    language: str | None = None
    sources: tuple[str, ...] = ()
    num_documents: int | None = None
    num_questions: int | None = None
    content_hash: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_str("DatasetMetadata", "name", self.name, allow_empty=False)
        require_str("DatasetMetadata", "adapter", self.adapter, allow_empty=False)
        require_str("DatasetMetadata", "description", self.description)
        for name in ("split", "version", "homepage", "license", "language", "content_hash"):
            require_optional_str("DatasetMetadata", name, getattr(self, name))
        require_optional_int("DatasetMetadata", "num_documents", self.num_documents, minimum=0)
        require_optional_int("DatasetMetadata", "num_questions", self.num_questions, minimum=0)
        set_field(self, "sources", require_str_tuple("DatasetMetadata", "sources", self.sources))
        set_field(self, "extra", frozen_mapping("DatasetMetadata", "extra", self.extra))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "adapter": self.adapter,
            "description": self.description,
            "split": self.split,
            "version": self.version,
            "homepage": self.homepage,
            "license": self.license,
            "language": self.language,
            "sources": list(self.sources),
            "num_documents": self.num_documents,
            "num_questions": self.num_questions,
            "content_hash": self.content_hash,
            "extra": dict(self.extra),
        }
