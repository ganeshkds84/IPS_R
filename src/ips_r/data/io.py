"""Fault-tolerant readers and writers for dataset files.

Readers never crash on bad *content*: undecodable bytes and unparseable records are
reported through an :data:`IssueSink` and skipped (or decoded with replacement
characters). I/O failures (missing or unreadable files) still raise, because they are
configuration problems rather than data problems.

Files ending in ``.gz`` are decompressed transparently.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

from ips_r.core.exceptions import DataError
from ips_r.core.log import get_logger
from ips_r.data.issues import IssueCode, Severity, ValidationIssue

logger = get_logger(__name__)

IssueSink = Callable[[ValidationIssue], None]
"""Callback receiving issues found while reading."""

ENCODING = "utf-8"


def open_binary(path: Path) -> IO[bytes]:
    """Open ``path`` for binary reading, decompressing ``.gz`` files.

    Raises:
        DataError: If the file cannot be opened.
    """
    try:
        if path.suffix == ".gz":
            return gzip.open(path, "rb")  # type: ignore[return-value]  # GzipFile is IO[bytes]
        return path.open("rb")
    except OSError as exc:
        raise DataError(f"Cannot open {path}: {exc}") from exc


def decode(raw: bytes, *, location: str, sink: IssueSink) -> str:
    """Decode UTF-8 (BOM tolerated), reporting undecodable bytes as ``invalid_encoding``.

    Undecodable bytes are replaced with U+FFFD so the remaining content stays usable.
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        sink(
            ValidationIssue(
                IssueCode.INVALID_ENCODING,
                Severity.ERROR,
                f"invalid {ENCODING} byte sequence at offset {exc.start}: {exc.reason}",
                location=location,
            )
        )
        return raw.decode("utf-8-sig", errors="replace")


def read_text(path: Path, *, sink: IssueSink) -> str:
    """Read a whole text file.

    Raises:
        DataError: If the file cannot be read.
    """
    with open_binary(path) as handle:
        try:
            raw = handle.read()
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise DataError(f"Cannot read {path}: {exc}") from exc
    return decode(raw, location=str(path), sink=sink)


def load_json(path: Path, *, sink: IssueSink) -> Any | None:
    """Parse a whole JSON file; report and return ``None`` if it is not valid JSON.

    Raises:
        DataError: If the file cannot be read.
    """
    text = read_text(path, sink=sink)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        sink(
            ValidationIssue(
                IssueCode.MALFORMED_RECORD,
                Severity.ERROR,
                f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})",
                location=str(path),
            )
        )
        return None


def iter_lines(path: Path, *, sink: IssueSink) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, line)`` pairs (1-based, newline stripped, blank lines skipped).

    Raises:
        DataError: If the file cannot be read.
    """
    with open_binary(path) as handle:
        try:
            for number, raw in enumerate(handle, start=1):
                line = decode(raw, location=f"{path}:{number}", sink=sink).rstrip("\r\n")
                if line.strip():
                    yield number, line
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise DataError(f"Cannot read {path}: {exc}") from exc


def iter_jsonl(path: Path, *, sink: IssueSink) -> Iterator[tuple[str, Any]]:
    """Yield ``(location, record)`` for each parseable line of a JSON-lines file.

    Unparseable lines are reported as ``malformed_record`` errors and skipped.
    """
    for number, line in iter_lines(path, sink=sink):
        location = f"{path}:{number}"
        try:
            yield location, json.loads(line)
        except json.JSONDecodeError as exc:
            sink(
                ValidationIssue(
                    IssueCode.MALFORMED_RECORD,
                    Severity.ERROR,
                    f"invalid JSON line: {exc.msg}",
                    location=location,
                )
            )


def iter_tsv(path: Path, *, sink: IssueSink) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(location, columns)`` for each line of a tab-separated file (no quoting)."""
    for number, line in iter_lines(path, sink=sink):
        yield f"{path}:{number}", line.split("\t")


class JsonlWriter:
    """Write records as JSON lines, counting what was written.

    Use as a context manager; parent directories are created.

    Raises:
        DataError: If the file cannot be opened or written.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: IO[str] | None = None
        self._count = 0

    @property
    def path(self) -> Path:
        """Destination file."""
        return self._path

    @property
    def count(self) -> int:
        """Number of records written so far."""
        return self._count

    def __enter__(self) -> Self:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("w", encoding=ENCODING, newline="\n")
        except OSError as exc:
            raise DataError(f"Cannot open {self._path} for writing: {exc}") from exc
        return self

    def write(self, record: Mapping[str, Any]) -> None:
        """Append one record.

        Raises:
            DataError: If the writer is not open or the write fails.
        """
        if self._handle is None:
            raise DataError(f"JsonlWriter for {self._path} is not open")
        try:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            try:
                line.encode(ENCODING)
            except UnicodeEncodeError:
                # Lone surrogates cannot be written as UTF-8; escape them instead.
                line = json.dumps(record, ensure_ascii=True, sort_keys=True)
            self._handle.write(line)
            self._handle.write("\n")
        except (OSError, TypeError, ValueError) as exc:
            raise DataError(f"Cannot write to {self._path}: {exc}") from exc
        self._count += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        logger.debug("jsonl.written", path=str(self._path), records=self._count)


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` as pretty-printed, key-sorted JSON.

    Raises:
        DataError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding=ENCODING,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise DataError(f"Cannot write {path}: {exc}") from exc
    return path


def write_text(path: Path, content: str) -> Path:
    """Write ``content`` to ``path`` as UTF-8.

    Raises:
        DataError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=ENCODING)
    except OSError as exc:
        raise DataError(f"Cannot write {path}: {exc}") from exc
    return path
