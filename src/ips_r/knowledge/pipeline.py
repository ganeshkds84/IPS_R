"""Knowledge processing pipeline: validated dataset → index-ready knowledge artifacts.

Stages (per document, streaming):

1. **Validation**: :class:`~ips_r.data.validation.DatasetValidator`; invalid documents
   are dropped (``knowledge.drop_invalid``).
2. **PDF parsing**: already done by the dataset adapter (``academic_pdf``).
3. **Text cleaning + normalisation**: :class:`TextNormalizer` (offset-preserving).
4. **Structural analysis**: :class:`StructuralAnalyzer`.
5. **Chunk generation**: :class:`KnowledgeChunker`.
6. **Metadata enrichment**: :class:`MetadataEnricher`.
   (Stages 3-6 run per document in
   :class:`~ips_r.knowledge.processor.KnowledgeProcessor`.)
7. **Embedding** / **sparse index** / **vector store**: run only when a component is
   configured (interfaces only in Milestone 2; stages are reported as ``skipped``).
8. **Artifact serialization**: :class:`ArtifactWriter`, manifest, checksums, statistics
   and the processing report.

No retrieval happens here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ips_r.config.schema import KnowledgeConfig, SerializationConfig
from ips_r.core.exceptions import ComponentError, DataValidationError
from ips_r.core.log import get_logger
from ips_r.data.base import AbstractDataset
from ips_r.data.io import write_json, write_text
from ips_r.data.issues import ValidationReport
from ips_r.data.models import Document
from ips_r.data.validation import DatasetValidator, has_errors
from ips_r.knowledge.embedding import AbstractEmbeddingModel
from ips_r.knowledge.hashing import artifact_hash
from ips_r.knowledge.indexing import (
    AbstractHybridIndexer,
    AbstractSparseIndexer,
    AbstractVectorStore,
    IndexArtifact,
    VectorRecord,
)
from ips_r.knowledge.models import KNOWLEDGE_SCHEMA_VERSION, KnowledgeDocument
from ips_r.knowledge.processor import KnowledgeProcessor, document_source_hash
from ips_r.knowledge.serialization import ArtifactWriter, write_manifest
from ips_r.knowledge.statistics import (
    KnowledgeStatisticsAccumulator,
    ProcessingReport,
    statistics_markdown,
)

logger = get_logger(__name__)

__all__ = [
    "STAGES",
    "IndexComponents",
    "KnowledgeBuildResult",
    "KnowledgeProcessor",
    "build_knowledge_artifacts",
    "document_source_hash",
]

STAGES: Final = (
    "validation",
    "normalization_structure",
    "chunking",
    "enrichment",
    "embedding",
    "sparse_index",
    "vector_store",
    "serialization",
)


@dataclass(slots=True)
class IndexComponents:
    """Optional embedding/index components (none are implemented in Milestone 2)."""

    embedding: AbstractEmbeddingModel | None = None
    vector_store: AbstractVectorStore | None = None
    sparse: AbstractSparseIndexer | None = None
    hybrid: AbstractHybridIndexer | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeBuildResult:
    """Outcome of :func:`build_knowledge_artifacts`."""

    output_dir: Path
    artifact_hash: str
    manifest_path: Path
    statistics: dict[str, Any]
    report: ProcessingReport
    document_hashes: dict[str, str] = field(default_factory=dict)
    indexes: tuple[IndexArtifact, ...] = ()

    def summary(self) -> dict[str, Any]:
        """Compact JSON-serialisable summary."""
        return {
            "output_dir": str(self.output_dir),
            "artifact_hash": self.artifact_hash,
            "artifact_id": self.artifact_hash.removeprefix("sha256:")[:12],
            "documents": self.report.documents_processed,
            "documents_dropped": self.report.documents_dropped,
            "documents_failed": self.report.documents_failed,
            "chunks": self.statistics["chunks"]["count"],
            "sections": self.statistics["sections"]["count"],
            "citations": self.statistics["citations"]["count"],
            "references": self.statistics["references"]["count"],
            "stages": dict(self.report.stages),
        }


def _index_stages(
    components: IndexComponents, document: KnowledgeDocument, report: ProcessingReport
) -> None:
    """Run the configured embedding / index stages for one document."""
    chunks = [c for c in document.chunks if c.duplicate_of is None]
    if not chunks:
        return
    records: list[VectorRecord] = []
    if components.embedding is not None:
        started = time.perf_counter()
        results = components.embedding.embed_chunks(chunks)
        records = [
            VectorRecord(
                r.request_id, r.vector, {"document_id": document.id, "content_hash": r.content_hash}
            )
            for r in results
        ]
        report.add_time("embedding", time.perf_counter() - started)
    if components.hybrid is not None and records:
        started = time.perf_counter()
        components.hybrid.index(chunks, records)
        report.add_time("vector_store", time.perf_counter() - started)
        return
    if components.sparse is not None:
        started = time.perf_counter()
        components.sparse.index_chunks(chunks)
        report.add_time("sparse_index", time.perf_counter() - started)
    if components.vector_store is not None and records:
        started = time.perf_counter()
        components.vector_store.upsert(records)
        report.add_time("vector_store", time.perf_counter() - started)


def build_knowledge_artifacts(
    dataset: AbstractDataset,
    output_dir: Path,
    *,
    processor: KnowledgeProcessor | None = None,
    settings: KnowledgeConfig | None = None,
    serialization: SerializationConfig | None = None,
    components: IndexComponents | None = None,
) -> KnowledgeBuildResult:
    """Run the full knowledge pipeline and write all artifacts to ``output_dir``.

    Raises:
        DataValidationError: If ``settings.fail_on_error`` and validation or processing failed
            (raised after all reports are written).
        ComponentError: If a dataset or index component fails.
        DataError: If an artifact cannot be written.
    """
    processor = processor or KnowledgeProcessor()
    settings = settings or KnowledgeConfig()
    serialization = serialization or SerializationConfig()
    components = components or IndexComponents()
    processor.reset()
    validator = DatasetValidator(settings.validation)
    stats = KnowledgeStatisticsAccumulator(dataset.name, settings.statistics)
    report = ProcessingReport(dataset=dataset.name)
    writer = ArtifactWriter(output_dir, serialization)
    document_hashes: dict[str, str] = {}
    logger.info("knowledge.started", dataset=dataset.name, output_dir=str(output_dir))

    for document in dataset.iter_documents():
        report.documents_seen += 1
        started = time.perf_counter()
        invalid = has_errors(validator.check_document(document))
        report.add_time("validation", time.perf_counter() - started)
        if invalid and settings.drop_invalid:
            report.documents_dropped += 1
            continue
        knowledge = _process_document(processor, document, report)
        if knowledge is None:
            continue
        document_hashes[knowledge.id] = knowledge.knowledge_hash
        stats.add(knowledge)
        _index_stages(components, knowledge, report)
        started = time.perf_counter()
        writer.write_document(knowledge)
        report.add_time("serialization", time.perf_counter() - started)

    for question in dataset.iter_questions():
        if not (has_errors(validator.check_question(question)) and settings.drop_invalid):
            writer.write_question(question)
            report.questions_written += 1

    started = time.perf_counter()
    tables = writer.close()
    fingerprints = processor.fingerprints
    digest = artifact_hash(document_hashes.items(), fingerprints, KNOWLEDGE_SCHEMA_VERSION)
    indexes = _index_artifacts(components, digest)
    validation = validator.report(dataset.name, dataset.ingestion_issues())
    metadata = dataset.describe().to_dict() | {
        "num_documents": report.documents_processed,
        "num_questions": report.questions_written,
    }
    manifest = write_manifest(
        output_dir,
        tables=tables,
        artifact_hash=digest,
        fingerprints=fingerprints,
        document_hashes=document_hashes,
        dataset=metadata,
        indexes=[i.to_dict() for i in indexes],
        write_checksums=serialization.write_checksums,
    )
    report.add_time("serialization", time.perf_counter() - started)
    for stage, seconds in processor.timings.items():
        report.add_time(stage, seconds)
    report.stages = _stage_status(processor, components)
    report.chunk_validation = processor.chunker.validation.to_dict()
    report.validation = _validation_summary(validation)
    statistics = stats.result()
    _write_reports(output_dir, statistics, report, validation.to_dict(), validation.to_markdown())
    result = KnowledgeBuildResult(
        output_dir=output_dir,
        artifact_hash=digest,
        manifest_path=manifest,
        statistics=statistics,
        report=report,
        document_hashes=document_hashes,
        indexes=indexes,
    )
    logger.info("knowledge.completed", **result.summary())
    if settings.fail_on_error and (not validation.is_valid or report.documents_failed):
        raise DataValidationError(
            f"Knowledge build for '{dataset.name}' had {validation.num_errors} validation errors "
            f"and {report.documents_failed} failed documents; see {output_dir / 'reports'}"
        )
    return result


def _process_document(
    processor: KnowledgeProcessor, document: Document, report: ProcessingReport
) -> KnowledgeDocument | None:
    """Process one document, recording a failure (and returning ``None``) on error."""
    try:
        knowledge = processor.process(document)
    except (ValueError, ComponentError) as exc:
        report.documents_failed += 1
        report.failures.append(
            {"document_id": document.id, "error": f"{type(exc).__name__}: {exc}"}
        )
        logger.warning("knowledge.document_failed", document_id=document.id, error=str(exc))
        return None
    report.documents_processed += 1
    report.chars_in += len(document.text)
    report.chars_out += sum(len(p.text) for p in knowledge.paragraphs)
    return knowledge


def _index_artifacts(components: IndexComponents, digest: str) -> tuple[IndexArtifact, ...]:
    """Provenance records of every configured index, bound to the knowledge artifact hash."""
    return tuple(
        component.artifact(digest)
        for component in (components.vector_store, components.sparse, components.hybrid)
        if component is not None
    )


def _validation_summary(validation: ValidationReport) -> dict[str, Any]:
    """Compact validation outcome for the processing report."""
    return {
        "is_valid": validation.is_valid,
        "errors": validation.num_errors,
        "warnings": validation.num_warnings,
        "counts_by_code": dict(validation.counts_by_code),
    }


def _stage_status(processor: KnowledgeProcessor, components: IndexComponents) -> dict[str, str]:
    """Human-readable status of every pipeline stage."""
    return {
        "validation": "completed",
        "normalization_structure": "completed",
        "chunking": "completed" if processor.chunker.has_base_chunker else "completed (unit-level)",
        "enrichment": "completed",
        "embedding": (
            "completed" if components.embedding else "skipped (no embedding model configured)"
        ),
        "sparse_index": (
            "completed" if components.sparse else "skipped (no sparse indexer configured)"
        ),
        "vector_store": (
            "completed"
            if components.vector_store or components.hybrid
            else "skipped (no vector store configured)"
        ),
        "serialization": "completed",
    }


def _write_reports(
    output_dir: Path,
    statistics: dict[str, Any],
    report: ProcessingReport,
    validation: dict[str, Any],
    validation_md: str,
) -> None:
    """Write statistics, processing and validation reports under ``reports/``.

    Reports carry wall-clock timings, so they are deliberately excluded from the
    manifest and checksums (which cover only the deterministic data tables).

    Raises:
        DataError: If a report cannot be written.
    """
    reports = output_dir / "reports"
    write_json(reports / "knowledge_statistics.json", statistics)
    write_text(reports / "knowledge_statistics.md", statistics_markdown(statistics))
    write_json(reports / "processing_report.json", report.to_dict())
    write_text(reports / "processing_report.md", report.to_markdown())
    write_json(reports / "validation_report.json", validation)
    write_text(reports / "validation_report.md", validation_md)
