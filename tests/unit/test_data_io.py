"""Tests for fault-tolerant dataset readers and writers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from ips_r.core.exceptions import DataError
from ips_r.data.io import (
    JsonlWriter,
    iter_jsonl,
    iter_tsv,
    load_json,
    open_binary,
    read_text,
    write_json,
    write_text,
)
from ips_r.data.issues import IssueCode, ValidationIssue


@pytest.fixture
def sink() -> list[ValidationIssue]:
    return []


def test_read_text_reports_invalid_utf8_and_strips_bom(
    tmp_path: Path, sink: list[ValidationIssue]
) -> None:
    good = tmp_path / "good.txt"
    good.write_bytes(b"\xef\xbb\xbfhello")
    assert read_text(good, sink=sink.append) == "hello"
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"caf\xe9")
    assert read_text(bad, sink=sink.append) == "caf�"
    assert sink[0].code is IssueCode.INVALID_ENCODING


def test_gzip_is_transparent(tmp_path: Path, sink: list[ValidationIssue]) -> None:
    path = tmp_path / "x.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"a": 1}\n\n{"a": 2}\n'))
    assert [r for _, r in iter_jsonl(path, sink=sink.append)] == [{"a": 1}, {"a": 2}]
    assert sink == []


def test_iter_jsonl_reports_bad_lines(tmp_path: Path, sink: list[ValidationIssue]) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{oops\n', encoding="utf-8")
    records = list(iter_jsonl(path, sink=sink.append))
    assert records == [(f"{path}:1", {"a": 1})]
    assert sink[0].code is IssueCode.MALFORMED_RECORD
    assert sink[0].location == f"{path}:2"


def test_load_json(tmp_path: Path, sink: list[ValidationIssue]) -> None:
    path = tmp_path / "x.json"
    path.write_text('{"a": [1]}', encoding="utf-8")
    assert load_json(path, sink=sink.append) == {"a": [1]}
    path.write_text("{bad", encoding="utf-8")
    assert load_json(path, sink=sink.append) is None
    assert sink[0].code is IssueCode.MALFORMED_RECORD


def test_iter_tsv(tmp_path: Path, sink: list[ValidationIssue]) -> None:
    path = tmp_path / "x.tsv"
    path.write_text("1\ta\r\n2\tb\tc\n", encoding="utf-8")
    assert [cols for _, cols in iter_tsv(path, sink=sink.append)] == [["1", "a"], ["2", "b", "c"]]


def test_missing_and_corrupt_files_raise(tmp_path: Path, sink: list[ValidationIssue]) -> None:
    with pytest.raises(DataError, match="Cannot open"):
        open_binary(tmp_path / "missing.txt")
    corrupt = tmp_path / "corrupt.gz"
    corrupt.write_bytes(b"not gzip at all")
    with pytest.raises(DataError, match="Cannot read"):
        read_text(corrupt, sink=sink.append)
    with pytest.raises(DataError, match="Cannot read"):
        list(iter_jsonl(corrupt, sink=sink.append))


def test_jsonl_writer(tmp_path: Path) -> None:
    path = tmp_path / "out" / "x.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"b": 1, "a": "é"})
        writer.write({"s": "lone \ud800"})
        assert writer.count == 2
        assert writer.path == path
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"a": "é", "b": 1}'
    assert json.loads(lines[1])["s"] == "lone \ud800"
    with pytest.raises(DataError, match="not open"):
        writer.write({"a": 1})
    with pytest.raises(DataError, match="Cannot write"), JsonlWriter(tmp_path / "y.jsonl") as other:
        other.write({"bad": object()})


def test_jsonl_writer_open_failure(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(DataError, match="Cannot open"), JsonlWriter(blocker / "sub" / "x.jsonl"):
        pass


def test_write_json_and_text(tmp_path: Path) -> None:
    path = write_json(tmp_path / "a" / "x.json", {"b": 1, "a": 2})
    assert path.read_text(encoding="utf-8").startswith('{\n  "a": 2')
    assert write_text(tmp_path / "b" / "x.md", "# hi").read_text(encoding="utf-8") == "# hi"
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(DataError):
        write_json(blocker / "x.json", {})
    with pytest.raises(DataError):
        write_json(tmp_path / "bad.json", {"x": object()})
    with pytest.raises(DataError):
        write_text(blocker / "x.md", "")
