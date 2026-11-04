"""Deterministic content hashing and artifact invalidation.

Hashes form a chain, so a change anywhere upstream changes every hash downstream:

``source document hash`` + ``processing fingerprint``
    → ``knowledge document hash`` (covers every derived object)
    → ``chunk content hash`` / ``chunk lineage hash``
    → ``artifact hash`` (all document hashes + fingerprints + schema version)
    → ``IndexArtifact.source_artifact_hash`` (later milestones)

All hashes are SHA-256 over *canonical JSON* (sorted keys, no insignificant
whitespace, UTF-8), so they are stable across processes, machines and Python hash
seeds. :func:`diff_document_hashes` reports which documents changed between two
manifests, i.e. which downstream artifacts are invalidated.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

HASH_PREFIX: Final = "sha256:"
_CHUNK_SIZE: Final = 1 << 20


def _canonical(value: Any) -> Any:
    """Convert ``value`` to a JSON-compatible structure with a single canonical form."""
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot hash non-finite float {value!r}")
        return value
    if value is None or isinstance(value, bool | int | str):
        return value
    raise TypeError(f"cannot hash value of type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialise ``value`` to canonical JSON (sorted keys, compact separators).

    Raises:
        TypeError: If ``value`` contains an unsupported type.
        ValueError: If ``value`` contains a non-finite float.
    """
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(data: bytes) -> str:
    return HASH_PREFIX + hashlib.sha256(data).hexdigest()


def content_hash(value: Any) -> str:
    """Return ``"sha256:<hex>"`` of the canonical JSON of ``value``."""
    return _digest(canonical_json(value).encode("utf-8", errors="surrogatepass"))


def text_hash(text: str) -> str:
    """Return ``"sha256:<hex>"`` of a string."""
    return _digest(text.encode("utf-8", errors="surrogatepass"))


def combine_hashes(*hashes: str) -> str:
    """Hash an ordered sequence of hashes (order matters)."""
    return content_hash(list(hashes))


def file_hash(path: Path) -> str:
    """Return ``"sha256:<hex>"`` of a file's bytes (streamed).

    Raises:
        OSError: If the file cannot be read.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            hasher.update(chunk)
    return HASH_PREFIX + hasher.hexdigest()


def short_hash(value: str, length: int = 12) -> str:
    """Return the first ``length`` hex characters of a ``sha256:`` hash."""
    return value.removeprefix(HASH_PREFIX)[:length]


@dataclass(frozen=True, slots=True)
class HashDiff:
    """Difference between two ``{document_id: hash}`` maps."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def invalidated(self) -> tuple[str, ...]:
        """Documents whose downstream artifacts must be rebuilt (sorted)."""
        return tuple(sorted({*self.added, *self.removed, *self.changed}))

    @property
    def is_empty(self) -> bool:
        """``True`` when nothing changed."""
        return not (self.added or self.removed or self.changed)


def diff_document_hashes(old: Mapping[str, str], new: Mapping[str, str]) -> HashDiff:
    """Compare per-document hashes of two artifact manifests."""
    return HashDiff(
        added=tuple(sorted(set(new) - set(old))),
        removed=tuple(sorted(set(old) - set(new))),
        changed=tuple(sorted(k for k in set(old) & set(new) if old[k] != new[k])),
    )


def artifact_hash(
    document_hashes: Iterable[tuple[str, str]], fingerprints: Mapping[str, str], schema_version: str
) -> str:
    """Hash of a whole knowledge artifact: ordered document hashes + config + schema."""
    return content_hash(
        {
            "documents": [list(pair) for pair in document_hashes],
            "fingerprints": dict(fingerprints),
            "schema_version": schema_version,
        }
    )
