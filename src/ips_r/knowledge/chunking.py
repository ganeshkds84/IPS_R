"""Section-, table- and caption-aware chunk processing with full lineage.

:class:`KnowledgeChunker` turns :class:`~ips_r.knowledge.structure.TextUnit` objects into
:class:`~ips_r.knowledge.models.KnowledgeChunk` objects:

- **section-aware**: units never span sections, so chunks never cross a section boundary;
- **table-aware**: a table (with its caption) is one chunk; oversized tables are split
  by rows, repeating the caption and header row in every piece;
- **caption-aware**: figure captions are standalone, never-split chunks;
- running text is split by any Milestone 1 chunking strategy (``chunker=...``), or
  kept whole when no chunker is configured;
- **deterministic ids** from document id, parent unit, unit span and text;
- **duplicate detection** across the whole corpus (``mark`` / ``drop`` / ``off``);
- **overlap validation** between consecutive pieces of a unit (excessive overlap,
  containment, uncovered gaps);
- **parent-child** (``parent_id``, ``section_id``) and **adjacent** (``prev_id`` /
  ``next_id``) links.

Every chunk keeps an exact original-document :class:`SourceSpan`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from typing import Final

from ips_r.config.schema import ChunkProcessingConfig
from ips_r.core.exceptions import ConfigurationError
from ips_r.data.chunking.base import AbstractChunker
from ips_r.data.models import Document
from ips_r.data.text import normalize_whitespace, strip_span
from ips_r.knowledge.hashing import combine_hashes, content_hash, short_hash, text_hash
from ips_r.knowledge.models import KnowledgeChunk, Section, SourceSpan
from ips_r.knowledge.structure import TextUnit

DEDUP_MODES: Final = ("mark", "drop", "off")
CHUNK_ID_HASH_LENGTH: Final = 16


@dataclass(frozen=True, slots=True)
class ChunkIssue:
    """A problem found by overlap validation."""

    kind: str
    chunk_id: str
    message: str


@dataclass(slots=True)
class ChunkValidation:
    """Accumulated chunk-validation outcome (counts are complete, examples capped)."""

    max_examples: int = 50
    counts: Counter[str] = field(default_factory=Counter)
    examples: list[ChunkIssue] = field(default_factory=list)
    duplicates: int = 0
    dropped_duplicates: int = 0

    def add(self, issue: ChunkIssue) -> None:
        """Count ``issue`` and keep it as an example while below the cap."""
        self.counts[issue.kind] += 1
        if len(self.examples) < self.max_examples:
            self.examples.append(issue)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "issue_counts": dict(sorted(self.counts.items())),
            "duplicates": self.duplicates,
            "dropped_duplicates": self.dropped_duplicates,
            "examples": [
                {"kind": i.kind, "chunk_id": i.chunk_id, "message": i.message}
                for i in self.examples
            ],
        }


@dataclass(frozen=True, slots=True)
class _Piece:
    unit: TextUnit
    start: int
    end: int
    text: str


class KnowledgeChunker:
    """Builds linked, deduplicated, validated chunks for one document at a time.

    Args:
        settings: Chunk processing options.
        base_chunker: Strategy for splitting running text (``None`` = one chunk per unit).

    Raises:
        ConfigurationError: If ``deduplicate`` or ``max_table_chars`` is invalid.
    """

    def __init__(
        self,
        settings: ChunkProcessingConfig | None = None,
        base_chunker: AbstractChunker | None = None,
    ) -> None:
        self._s = settings or ChunkProcessingConfig()
        if self._s.deduplicate not in DEDUP_MODES:
            raise ConfigurationError(
                f"processing.chunking.deduplicate must be one of {DEDUP_MODES}, got {self._s.deduplicate!r}"
            )
        if self._s.max_table_chars < 1:
            raise ConfigurationError("processing.chunking.max_table_chars must be >= 1")
        if not 0 <= self._s.max_overlap_ratio <= 1:
            raise ConfigurationError("processing.chunking.max_overlap_ratio must be in [0, 1]")
        self._base = base_chunker
        self._seen: dict[str, str] = {}
        self.validation = ChunkValidation()

    @property
    def has_base_chunker(self) -> bool:
        """Whether running text is split by a chunking strategy."""
        return self._base is not None

    @property
    def fingerprint(self) -> str:
        """Hash of chunk settings and base strategy (part of artifact hashes)."""
        base = None
        if self._base is not None:
            base = {"strategy": self._base.strategy, "params": dict(self._base.config)}
        return content_hash({"chunking": {**asdict(self._s), "base": base}})

    def reset(self) -> None:
        """Forget corpus-level duplicate state and validation results."""
        self._seen.clear()
        self.validation = ChunkValidation()

    def chunk(
        self,
        document: Document,
        units: Sequence[TextUnit],
        sections: Sequence[Section],
        *,
        source_hash: str,
        fingerprint: str,
    ) -> list[KnowledgeChunk]:
        """Chunk all units of ``document`` (in unit order) and link the result."""
        paths = {section.id: section.path for section in sections}
        chunks: list[KnowledgeChunk] = []
        for unit in units:
            pieces = self._pieces(unit)
            self._validate_overlaps(document.id, pieces)
            for piece in pieces:
                chunk = self._make_chunk(document, piece, paths, source_hash, fingerprint)
                if chunk is not None:
                    chunks.append(chunk)
        return _link(chunks)

    # ------------------------------------------------------------------ #
    # Splitting
    # ------------------------------------------------------------------ #
    def _pieces(self, unit: TextUnit) -> list[_Piece]:
        text = unit.text.text
        if unit.atomic:
            if unit.table_rows and len(text) > self._s.max_table_chars:
                return self._table_pieces(unit)
            return self._whole(unit)
        if self._base is None:
            return self._whole(unit)
        probe = Document(id=unit.id, text=text)
        return [_Piece(unit, c.start, c.end, c.text) for c in self._base.chunk(probe)]

    @staticmethod
    def _whole(unit: TextUnit) -> list[_Piece]:
        span = strip_span(unit.text.text, 0, len(unit.text.text))
        if span is None:
            return []
        return [_Piece(unit, span[0], span[1], unit.text.text[span[0] : span[1]])]

    def _table_pieces(self, unit: TextUnit) -> list[_Piece]:
        """Split an oversized table into row groups, each prefixed by caption and header."""
        text = unit.text.text
        budget = max(self._s.max_table_chars - len(unit.header) - 1, 1)
        pieces: list[_Piece] = []
        cursor = text.find(unit.table_rows[0]) if unit.table_rows else len(text)
        group: list[tuple[int, int]] = []
        for row in unit.table_rows:
            start = text.find(row, cursor)
            end = start + len(row)
            cursor = end
            if group and end - group[0][0] > budget:
                pieces.append(self._row_piece(unit, group))
                group = []
            group.append((start, end))
        if group:
            pieces.append(self._row_piece(unit, group))
        return pieces

    @staticmethod
    def _row_piece(unit: TextUnit, group: list[tuple[int, int]]) -> _Piece:
        start, end = group[0][0], group[-1][1]
        body = unit.text.text[start:end]
        return _Piece(unit, start, end, f"{unit.header}\n{body}" if unit.header else body)

    # ------------------------------------------------------------------ #
    # Validation, ids, dedup
    # ------------------------------------------------------------------ #
    def _validate_overlaps(self, document_id: str, pieces: Sequence[_Piece]) -> None:
        for previous, current in pairwise(pieces):
            ident = f"{document_id}:{current.unit.id}@{current.start}"
            overlap = previous.end - current.start
            # Relative to the longer neighbour: a short tail window must not count as excessive.
            longest = max(previous.end - previous.start, current.end - current.start)
            if current.start >= previous.start and current.end <= previous.end:
                self.validation.add(
                    ChunkIssue("contained", ident, "chunk is contained in its predecessor")
                )
            elif overlap > self._s.max_overlap_ratio * longest:
                self.validation.add(
                    ChunkIssue(
                        "excessive_overlap", ident, f"overlaps predecessor by {overlap} chars"
                    )
                )
            elif overlap < 0 and current.unit.text.text[previous.end : current.start].strip():
                self.validation.add(
                    ChunkIssue("gap", ident, f"{-overlap} uncovered chars before chunk")
                )

    def _make_chunk(
        self,
        document: Document,
        piece: _Piece,
        paths: dict[str, tuple[str, ...]],
        source_hash: str,
        fingerprint: str,
    ) -> KnowledgeChunk | None:
        unit = piece.unit
        chunk_hash = text_hash(piece.text)
        chunk_id = (
            f"{document.id}::"
            f"{short_hash(content_hash([unit.id, piece.start, piece.end, chunk_hash]), CHUNK_ID_HASH_LENGTH)}"
        )
        duplicate_of = None
        if self._s.deduplicate != "off":
            key = text_hash(normalize_whitespace(piece.text).casefold())
            duplicate_of = self._seen.get(key)
            if duplicate_of is None:
                self._seen[key] = chunk_id
            else:
                self.validation.duplicates += 1
                if self._s.deduplicate == "drop":
                    self.validation.dropped_duplicates += 1
                    return None
        return KnowledgeChunk(
            id=chunk_id,
            document_id=document.id,
            index=0,
            text=piece.text,
            kind=unit.kind,
            section_id=unit.section_id,
            parent_id=unit.id,
            unit_start=piece.start,
            unit_end=piece.end,
            source=_source_span(document, unit, piece),
            section_path=paths.get(unit.section_id, ()),
            duplicate_of=duplicate_of,
            content_hash=chunk_hash,
            lineage_hash=combine_hashes(source_hash, fingerprint, chunk_hash),
        )


def _source_span(document: Document, unit: TextUnit, piece: _Piece) -> SourceSpan:
    """Original-document span, pages and element of a piece."""
    start, end = unit.text.source_span(piece.start, piece.end)
    elements = [
        (i, e)
        for i, e in enumerate(document.elements)
        if e.start < max(end, start + 1) and e.end > start
    ]
    pages = [e.page for _, e in elements if e.page is not None]
    single = elements[0][1] if len(elements) == 1 else None
    return SourceSpan(
        document.id,
        start,
        end,
        page=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        bbox=single.bbox if single is not None else None,
        element_index=elements[0][0] if elements else None,
    )


def _link(chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
    """Assign indices and adjacent links in reading order."""
    linked: list[KnowledgeChunk] = []
    for index, chunk in enumerate(chunks):
        linked.append(
            replace(
                chunk,
                index=index,
                prev_id=chunks[index - 1].id if index else None,
                next_id=chunks[index + 1].id if index + 1 < len(chunks) else None,
            )
        )
    return linked
