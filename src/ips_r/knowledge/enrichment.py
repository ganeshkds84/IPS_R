"""Metadata enrichment: attach document, section and cross-link context to chunks.

The enricher only *adds* ``KnowledgeChunk.metadata`` fields. It never changes chunk
text, so content hashes stay stable. Available fields:

========================  =====================================================
``title`` ``authors``     document bibliographic metadata
``year`` ``venue``
``doi`` ``language``
``section_path``          titles from the root to the chunk's section
``section_number``        number of the chunk's section (``2.1``)
``is_appendix``           whether the chunk's section is in an appendix
``pages``                 ``[page_start, page_end]`` of the chunk
``citations``             citation markers inside the chunk + resolved reference ids
``cross_references``      cross-reference labels inside the chunk + resolved targets
``token_count``           number of tokens (``token_pattern``)
========================  =====================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Final

from ips_r.config.schema import EnrichmentConfig
from ips_r.core.exceptions import ConfigurationError
from ips_r.data.models import DocumentMetadata
from ips_r.data.text import compile_pattern
from ips_r.knowledge.models import Citation, CrossReference, KnowledgeChunk, Section, SourceSpan

ENRICHMENT_FIELDS: Final = frozenset(
    {
        "title",
        "authors",
        "year",
        "venue",
        "doi",
        "language",
        "section_path",
        "section_number",
        "is_appendix",
        "pages",
        "citations",
        "cross_references",
        "token_count",
    }
)


def _inside(span: SourceSpan | None, chunk: KnowledgeChunk) -> bool:
    """Whether ``span`` lies within the chunk's original span."""
    return span is not None and chunk.source.start <= span.start and span.end <= chunk.source.end


class MetadataEnricher:
    """Adds the configured context fields to chunk metadata.

    Args:
        settings: Field selection and token pattern.

    Raises:
        ConfigurationError: On unknown fields or an invalid token pattern.
    """

    def __init__(self, settings: EnrichmentConfig | None = None) -> None:
        self._s = settings or EnrichmentConfig()
        unknown = sorted(set(self._s.fields) - ENRICHMENT_FIELDS)
        if unknown:
            raise ConfigurationError(
                f"Unknown processing.enrichment fields {unknown}; available: {sorted(ENRICHMENT_FIELDS)}"
            )
        try:
            self._token = compile_pattern(self._s.token_pattern)
        except Exception as exc:
            raise ConfigurationError(f"Invalid processing.enrichment.token_pattern: {exc}") from exc
        self._fields = tuple(self._s.fields)

    def enrich(
        self,
        chunks: Sequence[KnowledgeChunk],
        metadata: DocumentMetadata,
        sections: Sequence[Section],
        citations: Sequence[Citation],
        cross_references: Sequence[CrossReference],
    ) -> list[KnowledgeChunk]:
        """Return copies of ``chunks`` with enrichment metadata."""
        by_section = {section.id: section for section in sections}
        document_fields: dict[str, Any] = {
            "title": metadata.title,
            "authors": list(metadata.authors),
            "year": metadata.year,
            "venue": metadata.venue,
            "doi": metadata.doi,
            "language": metadata.language,
        }
        enriched: list[KnowledgeChunk] = []
        for chunk in chunks:
            section = by_section.get(chunk.section_id)
            values: dict[str, Any] = dict(document_fields)
            values["section_path"] = list(chunk.section_path)
            values["section_number"] = section.number if section else None
            values["is_appendix"] = section.is_appendix if section else False
            values["pages"] = (
                [chunk.source.page, chunk.source.page_end]
                if chunk.source.page is not None
                else None
            )
            values["citations"] = [
                {"marker": c.marker, "targets": list(c.target_ids)}
                for c in citations
                if _inside(c.source, chunk)
            ]
            values["cross_references"] = [
                {"label": x.label, "kind": x.target_kind, "target": x.target_id}
                for x in cross_references
                if _inside(x.source, chunk)
            ]
            values["token_count"] = sum(1 for _ in self._token.finditer(chunk.text))
            selected = {name: values[name] for name in self._fields}
            enriched.append(replace(chunk, metadata={**chunk.metadata, **selected}))
        return enriched
