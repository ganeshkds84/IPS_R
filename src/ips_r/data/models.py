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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

from ips_r.core import types as core

ELEMENT_SEPARATOR: Final = "\n\n"
"""Separator inserted between elements when a document's text is assembled from elements."""


def _require_str(model: str, name: str, value: Any, *, allow_empty: bool = True) -> None:
    """Validate that ``value`` is a string (optionally non-empty)."""
    if not isinstance(value, str):
        raise ValueError(f"{model}.{name} must be a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ValueError(f"{model}.{name} must be a non-empty string")


def _require_optional_str(model: str, name: str, value: Any) -> None:
    """Validate that ``value`` is ``None`` or a string."""
    if value is not None:
        _require_str(model, name, value)


def _require_str_tuple(model: str, name: str, value: Any) -> tuple[str, ...]:
    """Validate and normalise a sequence of strings into a tuple."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{model}.{name} must be a sequence of strings, got {value!r}")
    items = tuple(value)
    for item in items:
        _require_str(model, name, item)
    return items


def _require_optional_int(model: str, name: str, value: Any, *, minimum: int) -> None:
    """Validate that ``value`` is ``None`` or an integer ``>= minimum``."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{model}.{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{model}.{name} must be >= {minimum}, got {value}")


def _frozen(model: str, name: str, value: Any) -> Mapping[str, Any]:
    """Return a read-only copy of a mapping field."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{model}.{name} must be a mapping, got {type(value).__name__}")
    return MappingProxyType(dict(value))


def _set(instance: object, name: str, value: Any) -> None:
    """Assign a field of a frozen dataclass during ``__post_init__``."""
    object.__setattr__(instance, name, value)


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

    def __post_init__(self) -> None:
        _set(self, "type", ElementType(self.type))
        _require_str("ElementDraft", "text", self.text, allow_empty=False)
        _require_optional_int("ElementDraft", "page", self.page, minimum=1)
        _require_optional_int("ElementDraft", "level", self.level, minimum=1)
        _require_optional_str("ElementDraft", "section", self.section)
        _set(self, "rows", tuple(_require_str_tuple("ElementDraft", "rows", r) for r in self.rows))


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
    """

    type: ElementType
    text: str
    start: int
    end: int
    page: int | None = None
    section: str | None = None
    level: int | None = None
    rows: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        _set(self, "type", ElementType(self.type))
        _require_str("DocumentElement", "text", self.text, allow_empty=False)
        _require_optional_int("DocumentElement", "start", self.start, minimum=0)
        _require_optional_int("DocumentElement", "page", self.page, minimum=1)
        _require_optional_int("DocumentElement", "level", self.level, minimum=1)
        _require_optional_str("DocumentElement", "section", self.section)
        if self.end - self.start != len(self.text):
            raise ValueError(
                f"DocumentElement span [{self.start}, {self.end}) does not match text length "
                f"{len(self.text)}"
            )
        _set(
            self, "rows", tuple(_require_str_tuple("DocumentElement", "rows", r) for r in self.rows)
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
        )


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
            _require_optional_str("DocumentMetadata", name, getattr(self, name))
        if self.year is not None and (
            isinstance(self.year, bool) or not isinstance(self.year, int)
        ):
            raise ValueError(f"DocumentMetadata.year must be an integer, got {self.year!r}")
        _set(self, "authors", _require_str_tuple("DocumentMetadata", "authors", self.authors))
        _set(self, "keywords", _require_str_tuple("DocumentMetadata", "keywords", self.keywords))

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
        _require_str("Document", "id", self.id, allow_empty=False)
        _require_str("Document", "text", self.text)
        _require_optional_str("Document", "source", self.source)
        _require_optional_int("Document", "num_pages", self.num_pages, minimum=1)
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
        _set(self, "elements", elements)
        _set(self, "extra", _frozen("Document", "extra", self.extra))

    @property
    def title(self) -> str | None:
        """Shortcut for ``metadata.title``."""
        return self.metadata.title

    @classmethod
    def from_elements(
        cls,
        id: str,  # noqa: A002 - mirrors the field name
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
        _require_str("Chunk", "id", self.id, allow_empty=False)
        _require_str("Chunk", "document_id", self.document_id, allow_empty=False)
        _require_str("Chunk", "text", self.text, allow_empty=False)
        _require_str("Chunk", "strategy", self.strategy, allow_empty=False)
        _require_optional_int("Chunk", "index", self.index, minimum=0)
        _require_optional_int("Chunk", "start", self.start, minimum=0)
        _require_optional_int("Chunk", "page_start", self.page_start, minimum=1)
        _require_optional_int("Chunk", "page_end", self.page_end, minimum=1)
        _require_optional_str("Chunk", "section", self.section)
        if self.end - self.start != len(self.text):
            raise ValueError(
                f"Chunk span [{self.start}, {self.end}) does not match text length {len(self.text)}"
            )
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("Chunk.page_start and Chunk.page_end must both be set or both None")
        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                raise ValueError("Chunk.page_end must be >= Chunk.page_start")
        _set(self, "metadata", _frozen("Chunk", "metadata", self.metadata))

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
        _require_str("Question", "id", self.id, allow_empty=False)
        _require_str("Question", "text", self.text)
        _require_optional_str("Question", "question_type", self.question_type)
        _set(self, "answers", _require_str_tuple("Question", "answers", self.answers))
        _set(
            self,
            "supporting_document_ids",
            _require_str_tuple("Question", "supporting_document_ids", self.supporting_document_ids),
        )
        _set(self, "metadata", _frozen("Question", "metadata", self.metadata))

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
        _require_str("DatasetMetadata", "name", self.name, allow_empty=False)
        _require_str("DatasetMetadata", "adapter", self.adapter, allow_empty=False)
        _require_str("DatasetMetadata", "description", self.description)
        for name in ("split", "version", "homepage", "license", "language", "content_hash"):
            _require_optional_str("DatasetMetadata", name, getattr(self, name))
        _require_optional_int("DatasetMetadata", "num_documents", self.num_documents, minimum=0)
        _require_optional_int("DatasetMetadata", "num_questions", self.num_questions, minimum=0)
        _set(self, "sources", _require_str_tuple("DatasetMetadata", "sources", self.sources))
        _set(self, "extra", _frozen("DatasetMetadata", "extra", self.extra))

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
