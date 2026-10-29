"""Abstract chunker: splits documents into :class:`~ips_r.data.models.Chunk` objects.

Strategies implement one pure hook, :meth:`AbstractChunker._split`, returning character
spans of the document text. The base class turns spans into chunks and enforces the
contract every strategy shares:

- ``chunk.text == document.text[chunk.start:chunk.end]`` (exact provenance),
- spans are trimmed of surrounding whitespace and empty spans are dropped,
- chunk ids are ``"<document_id>::<index>"`` and indices are contiguous from 0,
- page range and section are derived from the document's structural elements.

No embeddings are computed here; chunking is purely textual.

Common parameter: ``min_chunk_chars`` (drop chunks shorter than this; default 1).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, ClassVar, Final

from ips_r.components.base import AbstractComponent
from ips_r.core.exceptions import ComponentError
from ips_r.core.registry import Registry
from ips_r.data.models import Chunk, Document
from ips_r.data.text import strip_span

Span = tuple[int, int]
"""``(start, end)`` character offsets, end exclusive."""

CHUNK_ID_SEPARATOR: Final = "::"


def make_chunk_id(document_id: str, index: int) -> str:
    """Return the canonical chunk id ``"<document_id>::<index:05d>"``."""
    return f"{document_id}{CHUNK_ID_SEPARATOR}{index:05d}"


class AbstractChunker(AbstractComponent):
    """Splits documents into chunks; implementations provide :meth:`_split`."""

    component_type: ClassVar[str] = "chunker"
    strategy: ClassVar[str] = "abstract"

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._min_chunk_chars = self.int_param("min_chunk_chars", 1, minimum=1)

    def chunk(self, document: Document) -> list[Chunk]:
        """Split ``document`` into chunks.

        Raises:
            ComponentError: If splitting fails or the strategy returns invalid spans.
        """
        if not document.text.strip():
            return []
        spans = self._invoke(
            "chunk", lambda: list(self._split(document.text)), document_id=document.id
        )
        chunks: list[Chunk] = []
        for start, end in spans:
            if not (0 <= start <= end <= len(document.text)):
                raise ComponentError(
                    self.name,
                    f"invalid span [{start}, {end}) for document {document.id!r} "
                    f"of length {len(document.text)}",
                )
            trimmed = strip_span(document.text, start, end)
            if trimmed is None or trimmed[1] - trimmed[0] < self._min_chunk_chars:
                continue
            chunks.append(self._make_chunk(document, len(chunks), *trimmed))
        self._logger.debug("chunker.document", document_id=document.id, chunks=len(chunks))
        return chunks

    def chunk_documents(self, documents: Iterable[Document]) -> Iterator[Chunk]:
        """Lazily chunk a stream of documents."""
        for document in documents:
            yield from self.chunk(document)

    def _make_chunk(self, document: Document, index: int, start: int, end: int) -> Chunk:
        """Build a chunk with page range and section derived from document elements."""
        elements = document.elements_in(start, end)
        pages = [element.page for element in elements if element.page is not None]
        return Chunk(
            id=make_chunk_id(document.id, index),
            document_id=document.id,
            text=document.text[start:end],
            index=index,
            start=start,
            end=end,
            strategy=self.strategy,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            section=elements[0].section if elements else None,
        )

    @abstractmethod
    def _split(self, text: str) -> Sequence[Span]:
        """Return the character spans of the chunks of ``text`` (may overlap)."""
        raise NotImplementedError


CHUNKERS: Registry[AbstractChunker] = Registry("chunker", AbstractChunker)
"""Registry of chunking strategies."""
