"""Integration tests: dataset → knowledge pipeline → artifacts, tracked runs and the CLI."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ips_r.config.schema import KnowledgeConfig, NormalizationConfig, SerializationConfig
from ips_r.core.exceptions import DataValidationError
from ips_r.data.models import Document, Question
from ips_r.knowledge import (
    IndexComponents,
    KnowledgeBuildResult,
    KnowledgeDocument,
    KnowledgeProcessor,
    build_knowledge_artifacts,
    diff_document_hashes,
    load_manifest,
    read_table,
    verify_artifacts,
)
from ips_r.knowledge.pipeline import STAGES
from ips_r.knowledge.serialization import CHECKSUM_FILE, MANIFEST_FILE
from tests.knowledge_factories import InMemoryDataset, paper
from tests.knowledge_fakes import HashEmbedding, MemorySparseIndexer, MemoryVectorStore, SumHybrid

QUESTIONS = (Question("q1", "What is studied?", answers=("intent",)), Question("q2", "   "))
REPORTS = (
    "knowledge_statistics.json",
    "knowledge_statistics.md",
    "processing_report.json",
    "processing_report.md",
    "validation_report.json",
    "validation_report.md",
)


def _corpus() -> list[Document]:
    return [paper("p1"), paper("p2", sections=3), paper("p3", body_repeats=1)]


def _build(
    out: Path, documents: list[Document] | None = None, **kwargs: Any
) -> KnowledgeBuildResult:
    dataset = InMemoryDataset(_corpus() if documents is None else documents, QUESTIONS)
    return build_knowledge_artifacts(dataset, out, **kwargs)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def test_pipeline_writes_complete_verifiable_artifacts(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result.manifest_path == tmp_path / MANIFEST_FILE
    assert verify_artifacts(tmp_path) == []
    assert all((tmp_path / "reports" / name).is_file() for name in REPORTS)
    assert sorted(result.document_hashes) == ["p1", "p2", "p3"]
    assert result.indexes == ()

    report = result.report
    assert (report.documents_seen, report.documents_processed, report.documents_dropped) == (
        3,
        3,
        0,
    )
    assert report.questions_written == 1  # the whitespace-only question is dropped
    assert list(report.stages) == list(STAGES)
    assert report.stages["chunking"] == "completed (unit-level)"
    assert report.stages["embedding"].startswith("skipped")
    assert set(report.stage_seconds) >= {"validation", "chunking", "serialization"}

    summary = result.summary()
    assert summary["artifact_id"] == result.artifact_hash.removeprefix("sha256:")[:12]
    assert summary["documents"] == 3
    assert summary["chunks"] == result.statistics["chunks"]["count"] > 0
    json.dumps(summary)

    manifest = load_manifest(tmp_path)
    assert manifest["artifact_hash"] == result.artifact_hash
    assert manifest["documents"] == result.document_hashes
    assert manifest["dataset"]["num_documents"] == 3
    assert manifest["dataset"]["num_questions"] == 1
    assert manifest["tables"]["chunks"]["records"] == summary["chunks"]


def test_chunks_are_traceable_to_source_documents(tmp_path: Path) -> None:
    corpus = {d.id: d for d in _corpus()}
    _build(tmp_path)
    chunks = list(read_table(tmp_path, "chunks"))
    sections = {r["id"] for r in read_table(tmp_path, "sections")}
    by_id = {c["id"]: c for c in chunks}
    for chunk in chunks:
        document = corpus[chunk["document_id"]]
        source = chunk["source"]
        assert source["document_id"] == document.id
        assert 0 <= source["start"] <= source["end"] <= len(document.text)
        assert chunk["section_id"] in sections
        for link in ("prev_id", "next_id"):
            assert chunk[link] is None or by_id[chunk[link]]["document_id"] == document.id
        if chunk["duplicate_of"] is not None:
            assert chunk["duplicate_of"] in by_id


def test_validation_drops_invalid_documents(tmp_path: Path) -> None:
    documents = [*_corpus(), Document(id="blank", text="   ")]
    result = _build(tmp_path, documents)
    assert result.report.documents_dropped == 1
    assert "blank" not in result.document_hashes
    assert result.report.validation["errors"] >= 1
    assert not result.report.validation["is_valid"]


def test_keeping_invalid_documents(tmp_path: Path) -> None:
    documents = [*_corpus(), Document(id="blank", text="   ")]
    result = _build(tmp_path, documents, settings=KnowledgeConfig(drop_invalid=False))
    assert result.report.documents_dropped == 0
    assert result.report.questions_written == 2
    assert "blank" in result.document_hashes


def test_fail_on_error_raises_after_writing_reports(tmp_path: Path) -> None:
    documents = [*_corpus(), Document(id="blank", text="   ")]
    with pytest.raises(DataValidationError, match=r"had \d+ validation errors"):
        _build(tmp_path, documents, settings=KnowledgeConfig(fail_on_error=True))
    assert verify_artifacts(tmp_path) == []
    assert (tmp_path / "reports" / "validation_report.md").is_file()


class _FailingProcessor(KnowledgeProcessor):
    def process(self, document: Document) -> KnowledgeDocument:
        if document.id == "p2":
            raise ValueError("cannot analyse")
        return super().process(document)


def test_processing_failures_are_recorded_and_skipped(tmp_path: Path) -> None:
    result = _build(tmp_path, processor=_FailingProcessor())
    assert result.report.documents_failed == 1
    assert result.report.failures == [{"document_id": "p2", "error": "ValueError: cannot analyse"}]
    assert sorted(result.document_hashes) == ["p1", "p3"]
    report = (tmp_path / "reports" / "processing_report.md").read_text(encoding="utf-8")
    assert "`p2`: ValueError: cannot analyse" in report
    with pytest.raises(DataValidationError, match="1 failed documents"):
        _build(
            tmp_path / "strict",
            processor=_FailingProcessor(),
            settings=KnowledgeConfig(fail_on_error=True),
        )


# --------------------------------------------------------------------------- #
# Determinism and invalidation (regression)
# --------------------------------------------------------------------------- #
def _data_files(directory: Path) -> dict[str, bytes]:
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and "reports" not in p.parts
    }


def test_rebuild_is_byte_identical(tmp_path: Path) -> None:
    settings = SerializationConfig(formats=["jsonl", "arrow", "parquet"], batch_size=7)
    first = _build(tmp_path / "a", serialization=settings)
    second = _build(tmp_path / "b", serialization=settings)
    assert first.artifact_hash == second.artifact_hash
    assert first.statistics == second.statistics
    files = _data_files(tmp_path / "a")
    assert MANIFEST_FILE in files and CHECKSUM_FILE in files
    assert files == _data_files(tmp_path / "b")


def test_changing_a_document_invalidates_only_that_document(tmp_path: Path) -> None:
    base = _build(tmp_path / "base")
    corpus = _corpus()
    corpus[1] = paper("p2", sections=3, body_repeats=3)
    changed = _build(tmp_path / "changed", corpus)
    assert changed.artifact_hash != base.artifact_hash
    diff = diff_document_hashes(base.document_hashes, changed.document_hashes)
    assert diff.changed == ("p2",) and diff.invalidated == ("p2",)

    removed = _build(tmp_path / "removed", _corpus()[:2])
    assert diff_document_hashes(base.document_hashes, removed.document_hashes).removed == ("p3",)


def test_moving_the_source_file_does_not_invalidate(tmp_path: Path) -> None:
    base = _build(tmp_path / "base")
    moved = [replace(d, source=f"/elsewhere/{d.id}.pdf") for d in _corpus()]
    assert _build(tmp_path / "moved", moved).artifact_hash == base.artifact_hash


def test_changing_processing_settings_invalidates_everything(tmp_path: Path) -> None:
    base = _build(tmp_path / "base")
    lower = _build(
        tmp_path / "lower", processor=KnowledgeProcessor(NormalizationConfig(lowercase=True))
    )
    assert lower.artifact_hash != base.artifact_hash
    assert diff_document_hashes(base.document_hashes, lower.document_hashes).changed == (
        "p1",
        "p2",
        "p3",
    )


# --------------------------------------------------------------------------- #
# Embedding / index interface stages
# --------------------------------------------------------------------------- #
def test_embedding_and_separate_index_stages(tmp_path: Path) -> None:
    components = IndexComponents(
        embedding=HashEmbedding({"batch_size": 5}),
        vector_store=MemoryVectorStore(name="vs"),
        sparse=MemorySparseIndexer(name="sp"),
    )
    documents = [*_corpus(), Document(id="blank", text="   ")]
    result = _build(
        tmp_path, documents, components=components, settings=KnowledgeConfig(drop_invalid=False)
    )
    unique = [c for c in read_table(tmp_path, "chunks") if c["duplicate_of"] is None]
    assert components.vector_store is not None and components.sparse is not None
    assert components.vector_store.count() == components.sparse.count() == len(unique)
    stages = result.report.stages
    assert stages["embedding"] == stages["sparse_index"] == stages["vector_store"] == "completed"
    assert [i.name for i in result.indexes] == ["vs", "sp"]
    assert all(not i.is_stale(result.artifact_hash) for i in result.indexes)
    assert load_manifest(tmp_path)["indexes"] == [i.to_dict() for i in result.indexes]


def test_hybrid_index_stage(tmp_path: Path) -> None:
    store, sparse = MemoryVectorStore(), MemorySparseIndexer()
    components = IndexComponents(
        embedding=HashEmbedding(),
        vector_store=store,
        sparse=sparse,
        hybrid=SumHybrid(store, sparse),
    )
    result = _build(tmp_path, components=components)
    assert store.count() == sparse.count() > 0
    assert [i.kind for i in result.indexes] == ["vector", "sparse", "hybrid"]
    assert "sparse_index" not in result.report.stage_seconds


# --------------------------------------------------------------------------- #
# Stress: large synthetic corpora
# --------------------------------------------------------------------------- #
def test_large_synthetic_corpus(tmp_path: Path) -> None:
    from ips_r.data import CHUNKERS

    documents = [paper(f"doc{i:03d}", sections=4, body_repeats=6) for i in range(80)]
    processor = KnowledgeProcessor(
        base_chunker=CHUNKERS.create("token", {"chunk_size": 32, "chunk_overlap": 8})
    )
    settings = SerializationConfig(formats=["jsonl", "parquet"], batch_size=250)
    result = _build(tmp_path, documents, processor=processor, serialization=settings)
    assert result.report.documents_processed == 80
    assert result.report.chunk_validation["issue_counts"] == {}
    chunks = result.statistics["chunks"]["count"]
    assert chunks > 80 * 10
    assert sum(1 for _ in read_table(tmp_path, "chunks", "parquet")) == chunks
    assert verify_artifacts(tmp_path) == []


def test_single_very_large_document(tmp_path: Path) -> None:
    from ips_r.data import CHUNKERS

    document = paper("huge", sections=80, body_repeats=10)
    processor = KnowledgeProcessor(
        base_chunker=CHUNKERS.create("sentence", {"sentences_per_chunk": 3, "sentence_overlap": 1})
    )
    result = _build(tmp_path, [document], processor=processor)
    assert result.statistics["sections"]["count"] > 160
    chunks = list(read_table(tmp_path, "chunks"))
    assert [c["index"] for c in chunks] == list(range(len(chunks)))
    assert len({c["id"] for c in chunks}) == len(chunks)
