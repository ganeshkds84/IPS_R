"""Dataset preparation service: ingest → validate → chunk → profile → write artefacts.

A single streaming pass over the dataset, so memory stays bounded by ids and
statistics rather than by corpus size. Artefacts written to ``output_dir``:

=========================  ===============================================
``metadata.json``          :class:`DatasetMetadata` incl. counts and content hash
``documents.jsonl``        accepted documents (``Document.to_dict``)
``questions.jsonl``        accepted questions (``Question.to_dict``)
``chunks.jsonl``           chunks (``Chunk.to_dict``), when a chunker is given
``validation_report.json`` / ``.md``   :class:`ValidationReport`
``statistics.json`` / ``.md``          :class:`DatasetStatistics`
=========================  ===============================================
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from ips_r.config.schema import PreparationConfig
from ips_r.core.exceptions import DataValidationError
from ips_r.core.log import get_logger
from ips_r.data.base import AbstractDataset
from ips_r.data.chunking.base import AbstractChunker
from ips_r.data.io import JsonlWriter, write_json, write_text
from ips_r.data.issues import ValidationReport
from ips_r.data.models import DatasetMetadata, Document, Question
from ips_r.data.statistics import DatasetStatistics, StatisticsAccumulator
from ips_r.data.validation import DatasetValidator, has_errors

logger = get_logger(__name__)

METADATA_FILE: Final = "metadata.json"
DOCUMENTS_FILE: Final = "documents.jsonl"
QUESTIONS_FILE: Final = "questions.jsonl"
CHUNKS_FILE: Final = "chunks.jsonl"
VALIDATION_JSON: Final = "validation_report.json"
VALIDATION_MD: Final = "validation_report.md"
STATISTICS_JSON: Final = "statistics.json"
STATISTICS_MD: Final = "statistics.md"


@dataclass(frozen=True, slots=True)
class PreparationResult:
    """Everything produced by :func:`prepare_dataset`."""

    output_dir: Path
    metadata: DatasetMetadata
    report: ValidationReport
    statistics: DatasetStatistics
    dropped_documents: int
    dropped_questions: int
    artifacts: Mapping[str, Path] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-serialisable summary (for ``results.json`` and the CLI)."""
        return {
            "dataset": self.metadata.name,
            "output_dir": str(self.output_dir),
            "num_documents": self.statistics.num_documents,
            "num_questions": self.statistics.num_questions,
            "num_chunks": self.statistics.num_chunks,
            "num_pages": self.statistics.num_pages,
            "dropped_documents": self.dropped_documents,
            "dropped_questions": self.dropped_questions,
            "validation": {
                "is_valid": self.report.is_valid,
                "errors": self.report.num_errors,
                "warnings": self.report.num_warnings,
            },
            "content_hash": self.metadata.content_hash,
            "artifacts": {name: str(path) for name, path in self.artifacts.items()},
        }


class _ContentHasher:
    """SHA-256 over accepted records, in stream order (stable across machines)."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def add(self, *parts: str) -> None:
        for part in parts:
            self._hash.update(part.encode("utf-8", errors="surrogatepass"))
            self._hash.update(b"\x1f")
        self._hash.update(b"\x1e")

    def add_document(self, document: Document) -> None:
        self.add("D", document.id, document.text)

    def add_question(self, question: Question) -> None:
        self.add(
            "Q",
            question.id,
            question.text,
            *question.answers,
            "|",
            *question.supporting_document_ids,
        )

    def hexdigest(self) -> str:
        return f"sha256:{self._hash.hexdigest()}"


def prepare_dataset(
    dataset: AbstractDataset,
    output_dir: Path,
    *,
    chunker: AbstractChunker | None = None,
    settings: PreparationConfig | None = None,
) -> PreparationResult:
    """Run the full dataset-layer pipeline and write all artefacts to ``output_dir``.

    Args:
        dataset: Source dataset.
        output_dir: Directory receiving the artefacts (created if missing).
        chunker: Chunking strategy; ``None`` skips chunk preparation.
        settings: Preparation options; defaults match ``configs/preparation/default.yaml``.

    Returns:
        The prepared metadata, validation report, statistics and artefact paths.

    Raises:
        DataValidationError: If ``settings.fail_on_error`` and validation found errors
            (raised after all reports have been written).
        ComponentError: If the dataset or chunker fails.
        DataError: If an artefact cannot be written.
    """
    settings = settings or PreparationConfig()
    validator = DatasetValidator(settings.validation)
    stats = StatisticsAccumulator(dataset.name, settings.statistics)
    hasher = _ContentHasher()
    artifacts: dict[str, Path] = {}
    dropped_docs = dropped_questions = 0
    logger.info(
        "preparation.started", dataset=dataset.name, chunker=chunker.name if chunker else None
    )

    with ExitStack() as stack:
        docs_out = _writer(
            stack, output_dir / DOCUMENTS_FILE, settings.write_documents, artifacts, "documents"
        )
        chunks_out = _writer(
            stack,
            output_dir / CHUNKS_FILE,
            chunker is not None and settings.write_chunks,
            artifacts,
            "chunks",
        )
        questions_out = _writer(
            stack, output_dir / QUESTIONS_FILE, settings.write_questions, artifacts, "questions"
        )

        for document in dataset.iter_documents():
            if has_errors(validator.check_document(document)) and settings.drop_invalid:
                dropped_docs += 1
                continue
            hasher.add_document(document)
            stats.add_document(document)
            if docs_out is not None:
                docs_out.write(document.to_dict())
            if chunker is not None:
                for chunk in chunker.chunk(document):
                    stats.add_chunk(chunk)
                    if chunks_out is not None:
                        chunks_out.write(chunk.to_dict())

        for question in dataset.iter_questions():
            if has_errors(validator.check_question(question)) and settings.drop_invalid:
                dropped_questions += 1
                continue
            hasher.add_question(question)
            stats.add_question(question)
            if questions_out is not None:
                questions_out.write(question.to_dict())

    report = validator.report(dataset.name, dataset.ingestion_issues())
    statistics = stats.result()
    metadata = replace(
        dataset.describe(),
        num_documents=statistics.num_documents,
        num_questions=statistics.num_questions,
        content_hash=hasher.hexdigest(),
        extra={
            "chunker": chunker.name if chunker else None,
            "chunk_strategy": chunker.strategy if chunker else None,
            "chunker_params": dict(chunker.config) if chunker else {},
            "dropped_documents": dropped_docs,
            "dropped_questions": dropped_questions,
        },
    )
    artifacts["metadata"] = write_json(output_dir / METADATA_FILE, metadata.to_dict())
    artifacts["validation_report"] = write_json(output_dir / VALIDATION_JSON, report.to_dict())
    artifacts["validation_report_md"] = write_text(output_dir / VALIDATION_MD, report.to_markdown())
    artifacts["statistics"] = write_json(output_dir / STATISTICS_JSON, statistics.to_dict())
    artifacts["statistics_md"] = write_text(output_dir / STATISTICS_MD, statistics.to_markdown())

    result = PreparationResult(
        output_dir=output_dir,
        metadata=metadata,
        report=report,
        statistics=statistics,
        dropped_documents=dropped_docs,
        dropped_questions=dropped_questions,
        artifacts=artifacts,
    )
    logger.info(
        "preparation.completed",
        dataset=dataset.name,
        documents=statistics.num_documents,
        questions=statistics.num_questions,
        chunks=statistics.num_chunks,
        errors=report.num_errors,
        warnings=report.num_warnings,
    )
    if settings.fail_on_error and not report.is_valid:
        raise DataValidationError(
            f"Dataset '{dataset.name}' failed validation with {report.num_errors} errors; "
            f"see {artifacts['validation_report_md']}"
        )
    return result


def _writer(
    stack: ExitStack, path: Path, enabled: bool, artifacts: dict[str, Path], key: str
) -> JsonlWriter | None:
    """Open a JSON-lines writer on ``stack`` if ``enabled`` and register its path."""
    if not enabled:
        return None
    artifacts[key] = path
    return stack.enter_context(JsonlWriter(path))
