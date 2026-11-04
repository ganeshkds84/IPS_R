"""In-memory embedding / index doubles exercising the Milestone 2 interfaces (tests only).

They are deliberately trivial: they exist to drive the template methods, not to
retrieve anything.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from ips_r.knowledge import (
    AbstractEmbeddingModel,
    AbstractHybridIndexer,
    AbstractSparseIndexer,
    AbstractVectorStore,
    EmbeddingCache,
    EmbeddingRequest,
    KnowledgeChunk,
    ScoredId,
    VectorRecord,
)
from ips_r.knowledge.embedding import Vector


class DictCache(EmbeddingCache):
    """Cache backed by a dict."""

    def __init__(self) -> None:
        self.store: dict[str, Vector] = {}

    def _get(self, key: str) -> Vector | None:
        return self.store.get(key)

    def _put(self, key: str, vector: Vector) -> None:
        self.store[key] = vector


class HashEmbedding(AbstractEmbeddingModel):
    """Deterministic pseudo-vectors derived from the text hash.

    ``params.mode`` makes the model misbehave: ``short`` (one vector too few),
    ``wrong_dim``, ``nan`` or ``zero``.
    """

    def __init__(
        self, config: Any = None, *, name: str | None = None, cache: EmbeddingCache | None = None
    ) -> None:
        super().__init__(config, name=name, cache=cache)
        self._dim = self.int_param("dimension", 4, minimum=1)
        self._mode = self.str_param("mode", "ok")
        self.calls: list[int] = []

    @property
    def dimension(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255 + 0.01 for i in range(self._dim)]

    def _embed(self, requests: Sequence[EmbeddingRequest]) -> Sequence[Sequence[float]]:
        self.calls.append(len(requests))
        vectors = [self._vector(r.text) for r in requests]
        match self._mode:
            case "short":
                return vectors[:-1]
            case "wrong_dim":
                return [[*v, 0.0] for v in vectors]
            case "nan":
                return [[float("nan")] * self._dim for _ in vectors]
            case "zero":
                return [[0.0] * self._dim for _ in vectors]
        return vectors


class MemoryVectorStore(AbstractVectorStore):
    """Dict of vectors; ``search`` returns ids by dot product."""

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self.records: dict[str, VectorRecord] = {}
        self.unsorted = self.bool_param("unsorted", False)

    def _count(self) -> int:
        return len(self.records)

    def _dimension(self) -> int | None:
        return len(next(iter(self.records.values())).vector) if self.records else None

    def _upsert(self, records: Sequence[VectorRecord]) -> None:
        self.records.update({r.id: r for r in records})

    def _delete(self, ids: Sequence[str]) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)

    def _search(
        self, vector: tuple[float, ...], top_k: int, filters: Mapping[str, Any]
    ) -> Sequence[ScoredId]:
        scored = [
            ScoredId(r.id, sum(a * b for a, b in zip(vector, r.vector, strict=True)))
            for r in self.records.values()
            if all(r.metadata.get(k) == v for k, v in filters.items())
        ]
        ranked = sorted(scored, key=lambda s: (-s.score, s.id))
        return ranked[::-1] if self.unsorted else ranked[:top_k]


class MemorySparseIndexer(AbstractSparseIndexer):
    """Dict of chunk texts; ``search`` counts shared lower-cased words."""

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self.texts: dict[str, str] = {}
        self.overflow = self.bool_param("overflow", False)

    def _count(self) -> int:
        return len(self.texts)

    def _index(self, chunks: Sequence[KnowledgeChunk]) -> None:
        self.texts.update({c.id: c.text for c in chunks})

    def _search(self, query: str, top_k: int) -> Sequence[ScoredId]:
        words = set(query.lower().split())
        scored = [
            ScoredId(i, float(len(words & set(t.lower().split())))) for i, t in self.texts.items()
        ]
        ranked = sorted(scored, key=lambda s: (-s.score, s.id))
        return ranked if self.overflow else ranked[:top_k]


class SumHybrid(AbstractHybridIndexer):
    """Fuses rankings by a weighted sum of scores."""

    def __init__(
        self,
        vector_store: AbstractVectorStore,
        sparse_indexer: AbstractSparseIndexer,
        config: Any = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(vector_store, sparse_indexer, config, name=name)
        self._dense_weight = self.float_param("dense_weight", 1.0)

    def _fuse(
        self, dense: Sequence[ScoredId], sparse: Sequence[ScoredId], top_k: int
    ) -> Sequence[ScoredId]:
        totals: dict[str, float] = {}
        for weight, ranking in ((self._dense_weight, dense), (1.0, sparse)):
            for item in ranking:
                totals[item.id] = totals.get(item.id, 0.0) + weight * item.score
        ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
        return [ScoredId(i, s) for i, s in ranked[:top_k]]
