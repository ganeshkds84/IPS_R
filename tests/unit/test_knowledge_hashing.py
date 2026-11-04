"""Tests for deterministic hashing and invalidation (Part H)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pytest

from ips_r.knowledge.hashing import (
    artifact_hash,
    canonical_json,
    combine_hashes,
    content_hash,
    diff_document_hashes,
    file_hash,
    short_hash,
    text_hash,
)


class Color(StrEnum):
    RED = "red"


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [1, (2, 3)]}) == '{"a":[1,[2,3]],"b":1}'
    assert canonical_json({"x": {3, 1, 2}, "c": Color.RED}) == '{"c":"red","x":[1,2,3]}'
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"a": 1}).startswith("sha256:")


def test_hashing_rejects_unhashable_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        content_hash({"x": float("nan")})
    with pytest.raises(TypeError, match="object"):
        content_hash({"x": object()})
    assert content_hash(1.5) != content_hash(1)


def test_text_file_and_combined_hashes(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes("café".encode())
    assert file_hash(path) == text_hash("café")
    assert combine_hashes("a", "b") != combine_hashes("b", "a")
    assert short_hash("sha256:abcdef0123456789", 6) == "abcdef"


def test_diff_document_hashes_reports_invalidation() -> None:
    old = {"a": "1", "b": "2", "c": "3"}
    new = {"a": "1", "b": "X", "d": "4"}
    diff = diff_document_hashes(old, new)
    assert (diff.added, diff.removed, diff.changed) == (("d",), ("c",), ("b",))
    assert diff.invalidated == ("b", "c", "d")
    assert not diff.is_empty
    assert diff_document_hashes(old, old).is_empty


def test_artifact_hash_depends_on_documents_config_and_schema() -> None:
    base = artifact_hash([("a", "1")], {"norm": "x"}, "1.0")
    assert base == artifact_hash([("a", "1")], {"norm": "x"}, "1.0")
    assert base != artifact_hash([("a", "2")], {"norm": "x"}, "1.0")
    assert base != artifact_hash([("a", "1")], {"norm": "y"}, "1.0")
    assert base != artifact_hash([("a", "1")], {"norm": "x"}, "2.0")
