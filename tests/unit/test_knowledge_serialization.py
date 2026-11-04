"""Tests for artifact serialization: JSONL / Arrow / Parquet, manifest, checksums (Part G)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ips_r.config.schema import SerializationConfig
from ips_r.core.exceptions import ConfigurationError, DataError
from ips_r.data.models import Question
from ips_r.knowledge import (
    KNOWLEDGE_SCHEMA_VERSION,
    ArtifactWriter,
    KnowledgeDocument,
    KnowledgeProcessor,
    load_manifest,
    read_table,
    verify_artifacts,
    write_manifest,
)
from ips_r.knowledge.artifact_schema import _column_kind
from ips_r.knowledge.serialization import (
    CHECKSUM_FILE,
    MANIFEST_FILE,
    TABLES,
    _TableSink,
    arrow_schema,
    table_columns,
)
from tests.knowledge_factories import paper

ALL_FORMATS = SerializationConfig(formats=["jsonl", "arrow", "parquet"], batch_size=3)
QUESTION = Question("q1", "What is studied?", answers=("intent",), metadata={"k": [1, 2]})


@pytest.fixture(scope="module")
def documents() -> list[KnowledgeDocument]:
    processor = KnowledgeProcessor()
    return [processor.process(paper("p1")), processor.process(paper("p2", sections=3))]


def _write(
    directory: Path,
    documents: list[KnowledgeDocument],
    settings: SerializationConfig = ALL_FORMATS,
) -> dict[str, dict[str, Any]]:
    writer = ArtifactWriter(directory, settings)
    for document in documents:
        writer.write_document(document)
    writer.write_question(QUESTION)
    return writer.close()


def _manifest(directory: Path, tables: dict[str, dict[str, Any]], **kwargs: Any) -> Path:
    return write_manifest(
        directory,
        tables=tables,
        artifact_hash="sha256:" + "ab" * 32,
        fingerprints={"normalization": "sha256:n"},
        document_hashes={"p2": "sha256:2", "p1": "sha256:1"},
        dataset={"name": "memory"},
        **kwargs,
    )


def test_every_table_is_written_in_every_format(
    tmp_path: Path, documents: list[KnowledgeDocument]
) -> None:
    tables = _write(tmp_path, documents)
    assert sorted(tables) == sorted(TABLES)
    assert tables["documents"]["records"] == 2
    assert tables["questions"]["records"] == 1
    assert tables["chunks"]["records"] == sum(len(d.chunks) for d in documents)
    for info in tables.values():
        assert set(info["files"]) == {"jsonl", "arrow", "parquet"}
        for entry in info["files"].values():
            assert (tmp_path / entry["path"]).is_file()
            assert entry["sha256"].startswith("sha256:") and entry["bytes"] >= 0


@pytest.mark.parametrize("table", ["documents", "questions", "chunks", "citations", "sections"])
def test_formats_round_trip_to_identical_records(
    tmp_path: Path, documents: list[KnowledgeDocument], table: str
) -> None:
    _write(tmp_path, documents)
    jsonl = list(read_table(tmp_path, table, "jsonl"))
    assert jsonl
    assert list(read_table(tmp_path, table, "arrow")) == _normalised(jsonl, table)
    assert list(read_table(tmp_path, table, "parquet")) == _normalised(jsonl, table)


def _normalised(records: list[dict[str, Any]], table: str) -> list[dict[str, Any]]:
    """JSONL records with every schema column present (Arrow fills missing ones with null)."""
    columns = table_columns(table)
    return [{name: record.get(name) for name in columns} for record in records]


def test_empty_tables_keep_their_schema(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, SerializationConfig(formats=["arrow", "parquet"]))
    tables = writer.close()
    assert tables["chunks"]["records"] == 0
    assert list(read_table(tmp_path, "chunks", "parquet")) == []
    pq = pytest.importorskip("pyarrow.parquet")
    schema = pq.read_schema(str(tmp_path / "parquet" / "chunks.parquet"))
    assert schema.names == arrow_schema("chunks").names


def test_output_is_byte_identical_across_runs(
    tmp_path: Path, documents: list[KnowledgeDocument]
) -> None:
    first = _write(tmp_path / "a", documents)
    second = _write(tmp_path / "b", documents)
    assert first == second
    _manifest(tmp_path / "a", first)
    _manifest(tmp_path / "b", second)
    for name in (MANIFEST_FILE, CHECKSUM_FILE):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_column_kinds() -> None:
    chunks = table_columns("chunks")
    assert chunks["text"] == "string"
    assert chunks["index"] == "int"
    assert chunks["duplicate_of"] == "string"  # Optional[str] → string
    assert chunks["section_path"] == "strings"
    assert table_columns("questions")["answers"] == "strings"
    assert table_columns("documents")["num_chunks"] == "int"
    assert table_columns("sections")["is_appendix"] == "bool"
    assert table_columns("chunks")["metadata"] == "json"
    assert _column_kind(float | None) == "float"
    assert _column_kind(tuple[int, ...]) == "json"
    with pytest.raises(KeyError):
        table_columns("bogus")


@pytest.mark.parametrize(
    "settings",
    [
        SerializationConfig(formats=[]),
        SerializationConfig(formats=["jsonl", "csv"]),
        SerializationConfig(batch_size=0),
    ],
)
def test_invalid_settings_are_rejected(tmp_path: Path, settings: SerializationConfig) -> None:
    with pytest.raises(ConfigurationError):
        ArtifactWriter(tmp_path, settings)


def test_writer_cannot_close_twice(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    writer.close()
    with pytest.raises(DataError, match="already closed"):
        writer.close()


def test_writer_wraps_hashing_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: a failure on one table used to leave every later table's file open.
    writer = ArtifactWriter(tmp_path, SerializationConfig(formats=["jsonl"]))

    def broken(path: Path) -> str:
        raise OSError("disk gone")

    monkeypatch.setattr("ips_r.knowledge.serialization.file_hash", broken)
    with pytest.raises(DataError, match="disk gone"):
        writer.close()


def test_writer_closes_every_sink_before_reporting_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = ArtifactWriter(tmp_path, SerializationConfig(formats=["jsonl"]))
    closed: list[str] = []
    original = _TableSink.close

    def flaky(sink: _TableSink) -> None:
        original(sink)
        closed.append(sink.name)
        if sink.name in {"sections", "chunks"}:
            raise OSError(f"{sink.name} failed")

    monkeypatch.setattr(_TableSink, "close", flaky)
    with pytest.raises(DataError, match="sections: sections failed; chunks: chunks failed") as info:
        writer.close()
    assert closed == list(TABLES)
    assert isinstance(info.value.__cause__, OSError)


def test_manifest_contents(tmp_path: Path, documents: list[KnowledgeDocument]) -> None:
    tables = _write(tmp_path, documents)
    path = _manifest(tmp_path, tables, indexes=[{"name": "vs"}])
    manifest = load_manifest(tmp_path)
    assert path == tmp_path / MANIFEST_FILE
    assert manifest["schema_version"] == KNOWLEDGE_SCHEMA_VERSION
    assert manifest["artifact_id"] == "ab" * 6
    assert list(manifest["documents"]) == ["p1", "p2"]
    assert list(manifest["tables"]) == sorted(TABLES)
    assert manifest["indexes"] == [{"name": "vs"}]
    assert "pyarrow_version" in manifest and "package_version" in manifest
    lines = (tmp_path / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 * len(TABLES) + 1
    assert any(line.endswith(f"  {MANIFEST_FILE}") for line in lines)


def test_checksums_can_be_disabled(tmp_path: Path) -> None:
    tables = ArtifactWriter(tmp_path).close()
    _manifest(tmp_path, tables, write_checksums=False)
    assert not (tmp_path / CHECKSUM_FILE).exists()
    assert verify_artifacts(tmp_path) == []


def test_verify_detects_tampering(tmp_path: Path, documents: list[KnowledgeDocument]) -> None:
    _manifest(tmp_path, _write(tmp_path, documents))
    assert verify_artifacts(tmp_path) == []
    chunks = tmp_path / "jsonl" / "chunks.jsonl"
    chunks.write_text(chunks.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    (tmp_path / "parquet" / "sections.parquet").unlink()
    problems = verify_artifacts(tmp_path)
    assert "chunks/jsonl: checksum mismatch for jsonl/chunks.jsonl" in problems
    assert "sections/parquet: missing file parquet/sections.parquet" in problems
    assert "checksums: mismatch for jsonl/chunks.jsonl" in problems
    assert "checksums: missing file parquet/sections.parquet" in problems


def test_verify_detects_schema_version_mismatch(tmp_path: Path) -> None:
    _manifest(tmp_path, ArtifactWriter(tmp_path).close(), write_checksums=False)
    manifest = load_manifest(tmp_path)
    manifest["schema_version"] = "0"
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_artifacts(tmp_path) == [f"schema_version '0' != {KNOWLEDGE_SCHEMA_VERSION!r}"]


def test_load_manifest_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Cannot read manifest"):
        load_manifest(tmp_path)
    (tmp_path / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(DataError, match="Cannot read manifest"):
        load_manifest(tmp_path)
    (tmp_path / MANIFEST_FILE).write_text("[]", encoding="utf-8")
    with pytest.raises(DataError, match="not an object"):
        load_manifest(tmp_path)


def test_read_table_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Unknown artifact format"):
        list(read_table(tmp_path, "chunks", "csv"))
    with pytest.raises(DataError, match="Cannot read artifact table"):
        list(read_table(tmp_path, "chunks", "jsonl"))
    with pytest.raises(DataError, match="Cannot read artifact table"):
        list(read_table(tmp_path, "chunks", "arrow"))
    (tmp_path / "parquet").mkdir()
    (tmp_path / "parquet" / "chunks.parquet").write_bytes(b"not parquet")
    with pytest.raises(DataError, match="Cannot read artifact table"):
        list(read_table(tmp_path, "chunks", "parquet"))
