"""Deterministic artifact export: JSONL, Apache Arrow (IPC), Parquet, manifest, checksums.

Layout of an artifact directory::

    <dir>/
      manifest.json         schema version, artifact id/hash, fingerprints, per-table files
                            (records, bytes, sha256), per-document hashes, index artifacts
      checksums.sha256      "<sha256>  <relative path>" for every data file and the manifest
                            (sha256sum format; reports/ carry timings and are not covered)
      jsonl/<table>.jsonl
      arrow/<table>.arrow
      parquet/<table>.parquet

Tables: ``documents``, ``questions`` and one per knowledge object kind (``sections``,
``paragraphs``, ``figures``, ``tables``, ``equations``, ``references``, ``citations``,
``cross_references``, ``footnotes``, ``chunks``). Every table is written even when empty,
with the fixed schema of :mod:`ips_r.knowledge.artifact_schema`, so consumers can rely
on it.

Determinism: records are written in stream order (documents arrive in a
deterministic order), JSON keys are sorted, Arrow/Parquet schemas are fixed, and the
manifest contains no timestamps. Byte-identical outputs require the same pyarrow
version, which the manifest records.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ips_r import __version__
from ips_r.config.schema import SerializationConfig
from ips_r.core.exceptions import ConfigurationError, DataError
from ips_r.data.io import JsonlWriter, write_json, write_text
from ips_r.data.models import Question
from ips_r.knowledge.artifact_schema import (
    DOCUMENT_TABLE,
    QUESTION_TABLE,
    TABLES,
    arrow_row,
    arrow_schema,
    parquet_module,
    pyarrow_module,
    table_columns,
)
from ips_r.knowledge.hashing import file_hash
from ips_r.knowledge.models import KNOWLEDGE_SCHEMA_VERSION, KnowledgeDocument, to_record

FORMATS: Final = ("jsonl", "arrow", "parquet")
MANIFEST_FILE: Final = "manifest.json"
CHECKSUM_FILE: Final = "checksums.sha256"
_EXTENSIONS: Final = {"jsonl": "jsonl", "arrow": "arrow", "parquet": "parquet"}

__all__ = [
    "CHECKSUM_FILE",
    "DOCUMENT_TABLE",
    "FORMATS",
    "MANIFEST_FILE",
    "QUESTION_TABLE",
    "TABLES",
    "ArtifactWriter",
    "arrow_schema",
    "load_manifest",
    "read_table",
    "table_columns",
    "verify_artifacts",
    "write_manifest",
]


@dataclass(slots=True)
class _TableSink:
    """Writes one table in every configured format."""

    name: str
    directory: Path
    formats: tuple[str, ...]
    batch_size: int
    compression: str
    count: int = 0
    _buffer: list[dict[str, Any]] = field(default_factory=list)
    _jsonl: JsonlWriter | None = None
    _arrow: Any = None
    _parquet: Any = None

    def path(self, fmt: str) -> Path:
        return self.directory / fmt / f"{self.name}.{_EXTENSIONS[fmt]}"

    def open(self) -> None:
        if "jsonl" in self.formats:
            self._jsonl = JsonlWriter(self.path("jsonl")).__enter__()

    def write(self, record: dict[str, Any]) -> None:
        self.count += 1
        if self._jsonl is not None:
            self._jsonl.write(record)
        if "arrow" in self.formats or "parquet" in self.formats:
            self._buffer.append(record)
            if len(self._buffer) >= self.batch_size:
                self._flush()

    def _flush(self) -> None:
        pa = pyarrow_module()
        schema = arrow_schema(self.name)
        columns = table_columns(self.name)
        table = pa.Table.from_pylist([arrow_row(r, columns) for r in self._buffer], schema=schema)
        self._buffer.clear()
        if "arrow" in self.formats:
            if self._arrow is None:
                self.path("arrow").parent.mkdir(parents=True, exist_ok=True)
                self._arrow = pa.ipc.new_file(str(self.path("arrow")), schema)
            self._arrow.write_table(table)
        if "parquet" in self.formats:
            if self._parquet is None:
                pq = parquet_module()
                self.path("parquet").parent.mkdir(parents=True, exist_ok=True)
                self._parquet = pq.ParquetWriter(
                    str(self.path("parquet")), schema, compression=self.compression
                )
            self._parquet.write_table(table)

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.__exit__(None, None, None)
            self._jsonl = None
        if "arrow" in self.formats or "parquet" in self.formats:
            if self._buffer or (self._arrow is None and self._parquet is None):
                self._flush()
            for writer in (self._arrow, self._parquet):
                if writer is not None:
                    writer.close()
            self._arrow = self._parquet = None


class ArtifactWriter:
    """Streams knowledge documents and questions into artifact tables.

    Args:
        directory: Artifact directory (created).
        settings: Formats, batch size and compression.

    Raises:
        ConfigurationError: On unknown formats or invalid sizes.
    """

    def __init__(self, directory: Path, settings: SerializationConfig | None = None) -> None:
        self._s = settings or SerializationConfig()
        formats = tuple(dict.fromkeys(self._s.formats))
        unknown = sorted(set(formats) - set(FORMATS))
        if unknown or not formats:
            raise ConfigurationError(
                f"serialization.formats must be a non-empty subset of {FORMATS}, got {list(self._s.formats)}"
            )
        if self._s.batch_size < 1:
            raise ConfigurationError("serialization.batch_size must be >= 1")
        self.directory = directory
        self.formats = formats
        self._sinks = {
            name: _TableSink(
                name, directory, formats, self._s.batch_size, self._s.parquet_compression
            )
            for name in TABLES
        }
        self._closed = False
        for sink in self._sinks.values():
            sink.open()

    def write_document(self, document: KnowledgeDocument) -> None:
        """Append a knowledge document to every table."""
        self._sinks[DOCUMENT_TABLE].write(document.summary_record())
        for name, items in document.objects().items():
            sink = self._sinks[name]
            for item in items:
                sink.write(to_record(item))

    def write_question(self, question: Question) -> None:
        """Append a question."""
        self._sinks[QUESTION_TABLE].write(question.to_dict())

    def close(self) -> dict[str, dict[str, Any]]:
        """Finish all files; return per-table file info (records, paths, bytes, sha256).

        Raises:
            DataError: If a file cannot be written or hashed.
        """
        if self._closed:
            raise DataError("ArtifactWriter is already closed")
        self._closed = True
        # Close every sink before hashing, so a failure cannot leave other files open.
        failures: list[tuple[str, Exception]] = []
        for name, sink in self._sinks.items():
            try:
                sink.close()
            except (OSError, ValueError) as exc:
                failures.append((name, exc))
        if failures:
            detail = "; ".join(f"{name}: {exc}" for name, exc in failures)
            raise DataError(f"Cannot finalise artifact tables: {detail}") from failures[0][1]
        info: dict[str, dict[str, Any]] = {}
        for name, sink in self._sinks.items():
            try:
                files = {
                    fmt: {
                        "path": sink.path(fmt).relative_to(self.directory).as_posix(),
                        "bytes": sink.path(fmt).stat().st_size,
                        "sha256": file_hash(sink.path(fmt)),
                    }
                    for fmt in self.formats
                }
            except (OSError, ValueError) as exc:
                raise DataError(f"Cannot finalise artifact table {name}: {exc}") from exc
            info[name] = {"records": sink.count, "files": files}
        return info


def write_manifest(
    directory: Path,
    *,
    tables: Mapping[str, Mapping[str, Any]],
    artifact_hash: str,
    fingerprints: Mapping[str, str],
    document_hashes: Mapping[str, str],
    dataset: Mapping[str, Any],
    indexes: Sequence[Mapping[str, Any]] = (),
    write_checksums: bool = True,
) -> Path:
    """Write ``manifest.json`` (and ``checksums.sha256``); return the manifest path.

    Raises:
        DataError: If a file cannot be written.
    """
    pa = pyarrow_module()
    manifest = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "artifact_id": artifact_hash.removeprefix("sha256:")[:12],
        "artifact_hash": artifact_hash,
        "package_version": __version__,
        "pyarrow_version": pa.__version__,
        "fingerprints": dict(fingerprints),
        "dataset": dict(dataset),
        "tables": {name: dict(tables[name]) for name in sorted(tables)},
        "documents": dict(sorted(document_hashes.items())),
        "indexes": [dict(i) for i in indexes],
    }
    path = write_json(directory / MANIFEST_FILE, manifest)
    if write_checksums:
        relative = sorted(
            [entry["path"] for info in tables.values() for entry in info["files"].values()]
            + [MANIFEST_FILE]
        )
        lines = [f"{file_hash(directory / rel).removeprefix('sha256:')}  {rel}" for rel in relative]
        write_text(directory / CHECKSUM_FILE, "\n".join(lines) + "\n")
    return path


def load_manifest(directory: Path) -> dict[str, Any]:
    """Read ``manifest.json``.

    Raises:
        DataError: If it is missing or invalid.
    """
    try:
        data = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"Cannot read manifest in {directory}: {exc}") from exc
    if not isinstance(data, dict):
        raise DataError(f"Manifest in {directory} is not an object")
    return data


def verify_artifacts(directory: Path) -> list[str]:
    """Check every file listed in the manifest and ``checksums.sha256``.

    Returns:
        Human-readable problems; empty when the artifact is intact.

    Raises:
        DataError: If the manifest cannot be read.
    """
    manifest = load_manifest(directory)
    problems: list[str] = []
    if manifest.get("schema_version") != KNOWLEDGE_SCHEMA_VERSION:
        problems.append(
            f"schema_version {manifest.get('schema_version')!r} != {KNOWLEDGE_SCHEMA_VERSION!r}"
        )
    for table, info in manifest.get("tables", {}).items():
        for fmt, entry in info.get("files", {}).items():
            path = directory / entry["path"]
            if not path.is_file():
                problems.append(f"{table}/{fmt}: missing file {entry['path']}")
            elif file_hash(path) != entry["sha256"]:
                problems.append(f"{table}/{fmt}: checksum mismatch for {entry['path']}")
    checksums = directory / CHECKSUM_FILE
    if checksums.is_file():
        for line in checksums.read_text(encoding="utf-8").splitlines():
            digest, _, relative = line.partition("  ")
            path = directory / relative
            if not path.is_file():
                problems.append(f"checksums: missing file {relative}")
            elif file_hash(path).removeprefix("sha256:") != digest:
                problems.append(f"checksums: mismatch for {relative}")
    return problems


def read_table(directory: Path, table: str, fmt: str = "jsonl") -> Iterator[dict[str, Any]]:
    """Iterate the records of an artifact table (JSON columns decoded for Arrow/Parquet).

    Raises:
        DataError: If the format is unknown or the file cannot be read.
    """
    if fmt not in FORMATS:
        raise DataError(f"Unknown artifact format {fmt!r}")
    path = directory / fmt / f"{table}.{_EXTENSIONS[fmt]}"
    try:
        if fmt == "jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    yield json.loads(line)
            return
        pa = pyarrow_module()
        if fmt == "arrow":
            with pa.memory_map(str(path)) as source:
                rows = pa.ipc.open_file(source).read_all().to_pylist()
        else:
            rows = parquet_module().read_table(str(path)).to_pylist()
    except (OSError, ValueError) as exc:
        raise DataError(f"Cannot read artifact table {path}: {exc}") from exc
    columns = table_columns(table)
    for row in rows:
        yield {
            name: json.loads(value) if columns.get(name) == "json" and value is not None else value
            for name, value in row.items()
        }
