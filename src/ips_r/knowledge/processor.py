"""Per-document knowledge processing: normalisation, structure, chunking, enrichment.

:class:`KnowledgeProcessor` turns one canonical :class:`~ips_r.data.models.Document`
into a :class:`~ips_r.knowledge.models.KnowledgeDocument` whose ``knowledge_hash``
covers the source content, every processing setting and every derived object.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

from ips_r.config.schema import NormalizationConfig, ProcessingConfig
from ips_r.data.chunking.base import AbstractChunker
from ips_r.data.models import Document
from ips_r.knowledge.chunking import KnowledgeChunker
from ips_r.knowledge.enrichment import MetadataEnricher
from ips_r.knowledge.hashing import combine_hashes, content_hash
from ips_r.knowledge.models import KNOWLEDGE_SCHEMA_VERSION, KnowledgeDocument, to_record
from ips_r.knowledge.normalization import TextNormalizer
from ips_r.knowledge.structure import StructuralAnalyzer


def document_source_hash(document: Document) -> str:
    """Content hash of a source document, independent of where its file lives.

    ``Document.source`` (a file path or URL) is excluded so that moving the data
    directory does not invalidate artifacts; every content field is included.
    """
    payload = document.to_dict()
    payload.pop("source", None)
    return content_hash(payload)


class KnowledgeProcessor:
    """Turns one canonical document into a :class:`KnowledgeDocument`.

    Args:
        normalization: Text normalisation settings.
        processing: Structure, chunking and enrichment settings.
        base_chunker: Running-text chunking strategy (``None`` = one chunk per unit).
    """

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        processing: ProcessingConfig | None = None,
        base_chunker: AbstractChunker | None = None,
    ) -> None:
        self.processing = processing or ProcessingConfig()
        self.normalizer = TextNormalizer(normalization)
        self.analyzer = StructuralAnalyzer(
            self.normalizer, self.processing.structure, self.processing.chunking
        )
        self.chunker = KnowledgeChunker(self.processing.chunking, base_chunker)
        self.enricher = MetadataEnricher(self.processing.enrichment)
        self.timings: dict[str, float] = {}

    @property
    def fingerprints(self) -> dict[str, str]:
        """Hashes of every setting that influences the output."""
        return {
            "normalization": self.normalizer.fingerprint,
            "structure": content_hash(asdict(self.processing.structure)),
            "chunking": self.chunker.fingerprint,
            "enrichment": content_hash(asdict(self.processing.enrichment)),
            "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        }

    @property
    def fingerprint(self) -> str:
        """Combined processing fingerprint."""
        return content_hash(self.fingerprints)

    def reset(self) -> None:
        """Clear corpus-level state (duplicate index, chunk validation, timings) before a run."""
        self.chunker.reset()
        self.timings.clear()

    @contextmanager
    def _timed(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings[stage] = self.timings.get(stage, 0.0) + time.perf_counter() - started

    def process(self, document: Document) -> KnowledgeDocument:
        """Run normalisation, structure analysis, chunking and enrichment."""
        source_hash = document_source_hash(document)
        fingerprint = self.fingerprint
        with self._timed("normalization_structure"):
            structure = self.analyzer.analyze(document)
        with self._timed("chunking"):
            chunks = self.chunker.chunk(
                document,
                structure.units,
                structure.sections,
                source_hash=source_hash,
                fingerprint=fingerprint,
            )
        with self._timed("enrichment"):
            chunks = self.enricher.enrich(
                chunks,
                document.metadata,
                structure.sections,
                structure.citations,
                structure.cross_references,
            )
        objects: dict[str, tuple[Any, ...]] = {
            "sections": tuple(structure.sections),
            "paragraphs": tuple(structure.paragraphs),
            "figures": tuple(structure.figures),
            "tables": tuple(structure.tables),
            "equations": tuple(structure.equations),
            "references": tuple(structure.references),
            "citations": tuple(structure.citations),
            "cross_references": tuple(structure.cross_references),
            "footnotes": tuple(structure.footnotes),
            "chunks": tuple(chunks),
        }
        knowledge_hash = combine_hashes(
            source_hash,
            fingerprint,
            content_hash(
                {name: [to_record(item) for item in items] for name, items in objects.items()}
            ),
        )
        return KnowledgeDocument(
            id=document.id,
            source_hash=source_hash,
            knowledge_hash=knowledge_hash,
            metadata=document.metadata,
            num_pages=document.num_pages,
            source=document.source,
            sections=tuple(structure.sections),
            paragraphs=tuple(structure.paragraphs),
            figures=tuple(structure.figures),
            tables=tuple(structure.tables),
            equations=tuple(structure.equations),
            references=tuple(structure.references),
            citations=tuple(structure.citations),
            cross_references=tuple(structure.cross_references),
            footnotes=tuple(structure.footnotes),
            chunks=tuple(chunks),
        )
