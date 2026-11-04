"""Index interfaces (Milestone 2: interfaces only; no vector store, BM25 or hybrid index).

- :class:`AbstractVectorStore`: stores :class:`VectorRecord` objects (``upsert``,
  ``delete``, ``count``) and declares the ``search`` contract later milestones implement.
- :class:`AbstractSparseIndexer`: indexes knowledge chunks for lexical search.
- :class:`AbstractHybridIndexer`: composes a vector store and a sparse indexer and
  declares the score-fusion hook.
- :class:`IndexArtifact`: provenance record of a built index. Its
  ``source_artifact_hash`` ties it to the knowledge artifact it was built from, so a
  changed document makes the index detectably stale (:meth:`IndexArtifact.is_stale`).
- :class:`RetrievalArtifact`: the record format of a ranked result list.

Public methods validate arguments and results (template method); implementations
override only the ``_hooks``.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ips_r.components.base import AbstractComponent
from ips_r.core.exceptions import ComponentError
from ips_r.core.registry import Registry
from ips_r.knowledge.hashing import content_hash
from ips_r.knowledge.models import KnowledgeChunk

INDEX_KINDS: Final = ("vector", "sparse", "hybrid")


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """A vector with its id and filterable metadata."""

    id: str
    vector: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("VectorRecord.id must be non-empty")
        vector = tuple(float(v) for v in self.vector)
        if not vector or not all(math.isfinite(v) for v in vector):
            raise ValueError("VectorRecord.vector must be non-empty and finite")
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ScoredId:
    """An item id with its score (higher is better)."""

    id: str
    score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("ScoredId.score must be finite")


@dataclass(frozen=True, slots=True)
class IndexArtifact:
    """Provenance of a built index.

    Attributes:
        name: Index name.
        kind: ``vector``, ``sparse`` or ``hybrid``.
        backend: Implementation name.
        num_items: Indexed items.
        source_artifact_hash: Hash of the knowledge artifact the index was built from.
        config_hash: Hash of the index configuration.
        dimension: Vector dimension (vector indexes).
        location: Where the index lives (path, URI, namespace).
        metadata: Free-form extras.
    """

    name: str
    kind: str
    backend: str
    num_items: int
    source_artifact_hash: str
    config_hash: str
    dimension: int | None = None
    location: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in INDEX_KINDS:
            raise ValueError(f"IndexArtifact.kind must be one of {INDEX_KINDS}")
        if self.num_items < 0 or (self.dimension is not None and self.dimension < 1):
            raise ValueError("IndexArtifact.num_items must be >= 0 and dimension >= 1")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def artifact_hash(self) -> str:
        """Deterministic identity of the index build."""
        return content_hash(self.to_dict())

    def is_stale(self, knowledge_artifact_hash: str) -> bool:
        """Whether the index was built from a different knowledge artifact."""
        return self.source_artifact_hash != knowledge_artifact_hash

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "kind": self.kind,
            "backend": self.backend,
            "num_items": self.num_items,
            "source_artifact_hash": self.source_artifact_hash,
            "config_hash": self.config_hash,
            "dimension": self.dimension,
            "location": self.location,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RetrievalArtifact:
    """A ranked result list for one query, tied to the index it came from."""

    query_id: str
    results: tuple[ScoredId, ...]
    index_artifact_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        results = tuple(self.results)
        ids = [r.id for r in results]
        if len(ids) != len(set(ids)):
            raise ValueError("RetrievalArtifact result ids must be unique")
        if any(a.score < b.score for a, b in pairwise(results)):
            raise ValueError("RetrievalArtifact results must be sorted by descending score")
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "query_id": self.query_id,
            "results": [{"id": r.id, "score": r.score} for r in self.results],
            "index_artifact_hash": self.index_artifact_hash,
            "metadata": dict(self.metadata),
        }


def _check_ranking(
    component: AbstractComponent, results: Sequence[ScoredId], top_k: int
) -> list[ScoredId]:
    """Validate a ranking returned by a search hook."""
    ranking = list(results)
    if len(ranking) > top_k:
        raise ComponentError(component.name, f"returned {len(ranking)} results for top_k={top_k}")
    if any(a.score < b.score for a, b in pairwise(ranking)):
        raise ComponentError(component.name, "results are not sorted by descending score")
    return ranking


class _IndexBase(AbstractComponent):
    """Shared artifact bookkeeping."""

    index_kind: ClassVar[str] = "vector"

    def artifact(self, source_artifact_hash: str, *, location: str | None = None) -> IndexArtifact:
        """Describe the current index state."""
        return IndexArtifact(
            name=self.name,
            kind=self.index_kind,
            backend=type(self).__name__,
            num_items=self.count(),
            source_artifact_hash=source_artifact_hash,
            config_hash=content_hash(dict(self.config)),
            dimension=self._dimension(),
            location=location,
        )

    def count(self) -> int:
        """Number of indexed items."""
        return int(self._invoke("count", self._count))

    def _dimension(self) -> int | None:
        return None

    @abstractmethod
    def _count(self) -> int:
        raise NotImplementedError


class AbstractVectorStore(_IndexBase):
    """Persists vectors and (in later milestones) answers nearest-neighbour searches."""

    component_type: ClassVar[str] = "vector_store"
    index_kind: ClassVar[str] = "vector"

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert or replace records.

        Raises:
            ComponentError: If ids repeat or dimensions are inconsistent.
        """
        if not records:
            return
        if len({r.id for r in records}) != len(records):
            raise ComponentError(self.name, "upsert received duplicate ids")
        if len({len(r.vector) for r in records}) != 1:
            raise ComponentError(self.name, "upsert received vectors of different dimensions")
        self._invoke("upsert", lambda: self._upsert(records), num_records=len(records))

    def delete(self, ids: Sequence[str]) -> None:
        """Remove records by id (unknown ids are ignored)."""
        if ids:
            self._invoke("delete", lambda: self._delete(ids), num_ids=len(ids))

    def search(
        self, vector: Sequence[float], top_k: int = 10, filters: Mapping[str, Any] | None = None
    ) -> list[ScoredId]:
        """Return up to ``top_k`` nearest records (contract only in Milestone 2).

        Raises:
            ComponentError: If ``top_k`` is invalid or the ranking violates the contract.
        """
        self._require_positive("top_k", top_k)
        results = self._invoke("search", lambda: self._search(tuple(vector), top_k, filters or {}))
        return _check_ranking(self, results, top_k)

    @abstractmethod
    def _upsert(self, records: Sequence[VectorRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _delete(self, ids: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _search(
        self, vector: tuple[float, ...], top_k: int, filters: Mapping[str, Any]
    ) -> Sequence[ScoredId]:
        raise NotImplementedError


class AbstractSparseIndexer(_IndexBase):
    """Builds a lexical index over chunk text (BM25 etc. arrive in later milestones)."""

    component_type: ClassVar[str] = "sparse_indexer"
    index_kind: ClassVar[str] = "sparse"

    def index_chunks(self, chunks: Sequence[KnowledgeChunk]) -> None:
        """Add chunks to the index (duplicates of already-indexed ids are replaced)."""
        if chunks:
            self._invoke("index", lambda: self._index(chunks), num_chunks=len(chunks))

    def search(self, query: str, top_k: int = 10) -> list[ScoredId]:
        """Return up to ``top_k`` chunk ids for ``query`` (contract only in Milestone 2).

        Raises:
            ComponentError: If arguments or the ranking are invalid.
        """
        self._require_positive("top_k", top_k)
        if not query.strip():
            raise ComponentError(self.name, "query must be non-empty")
        results = self._invoke("search", lambda: self._search(query, top_k))
        return _check_ranking(self, results, top_k)

    @abstractmethod
    def _index(self, chunks: Sequence[KnowledgeChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _search(self, query: str, top_k: int) -> Sequence[ScoredId]:
        raise NotImplementedError


class AbstractHybridIndexer(_IndexBase):
    """Combines a vector store and a sparse indexer; implementations define score fusion.

    Args:
        vector_store: Dense component.
        sparse_indexer: Lexical component.
        config: Fusion parameters.
        name: Instance name.
    """

    component_type: ClassVar[str] = "hybrid_indexer"
    index_kind: ClassVar[str] = "hybrid"

    def __init__(
        self,
        vector_store: AbstractVectorStore,
        sparse_indexer: AbstractSparseIndexer,
        config: Any = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(config, name=name)
        self.vector_store = vector_store
        self.sparse_indexer = sparse_indexer

    def _setup(self) -> None:
        self.vector_store.setup()
        self.sparse_indexer.setup()

    def _teardown(self) -> None:
        self.vector_store.teardown()
        self.sparse_indexer.teardown()

    def index(self, chunks: Sequence[KnowledgeChunk], records: Sequence[VectorRecord]) -> None:
        """Index chunks lexically and their vectors densely.

        Raises:
            ComponentError: If chunk ids and vector ids differ.
        """
        if {c.id for c in chunks} != {r.id for r in records}:
            raise ComponentError(self.name, "chunk ids and vector record ids must match")
        self.sparse_indexer.index_chunks(chunks)
        self.vector_store.upsert(records)

    def fuse(
        self, dense: Sequence[ScoredId], sparse: Sequence[ScoredId], top_k: int = 10
    ) -> list[ScoredId]:
        """Fuse two rankings with the implementation's strategy (contract only)."""
        self._require_positive("top_k", top_k)
        return _check_ranking(
            self, self._invoke("fuse", lambda: self._fuse(dense, sparse, top_k)), top_k
        )

    def _count(self) -> int:
        return self.vector_store.count()

    @abstractmethod
    def _fuse(
        self, dense: Sequence[ScoredId], sparse: Sequence[ScoredId], top_k: int
    ) -> Sequence[ScoredId]:
        raise NotImplementedError


VECTOR_STORES: Registry[AbstractVectorStore] = Registry("vector_store", AbstractVectorStore)
SPARSE_INDEXERS: Registry[AbstractSparseIndexer] = Registry("sparse_indexer", AbstractSparseIndexer)
HYBRID_INDEXERS: Registry[AbstractHybridIndexer] = Registry("hybrid_indexer", AbstractHybridIndexer)
"""Registries of index implementations (all empty in Milestone 2)."""
