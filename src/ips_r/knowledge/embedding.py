"""Embedding interfaces (Milestone 2: interfaces only; no embedding model is provided).

- :class:`EmbeddingRequest`: one text to embed, identified by a chunk id, carrying the
  content hash that keys the cache.
- :class:`EmbeddingBatch`: an ordered, duplicate-free group of requests with a
  deterministic batch id.
- :class:`EmbeddingResult`: a finite vector of the model's dimension.
- :class:`EmbeddingCache`: abstract content-addressed cache keyed by
  ``(model, kind, content hash)``.
- :class:`AbstractEmbeddingModel`: template method :meth:`~AbstractEmbeddingModel.embed`
  validates the batch, serves cache hits, calls :meth:`~AbstractEmbeddingModel._embed`
  for misses only, checks counts, dimensions and finiteness, optionally L2-normalises,
  and stores new vectors in the cache.

Implementations (later milestones) register with :data:`EMBEDDING_MODELS` and are
selected with ``embedding=<name>``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ips_r.components.base import AbstractComponent
from ips_r.core.exceptions import ComponentError
from ips_r.core.registry import Registry
from ips_r.knowledge.hashing import content_hash, text_hash
from ips_r.knowledge.models import KnowledgeChunk

EMBEDDING_KINDS: Final = ("document", "query")

Vector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """Text to embed.

    Attributes:
        id: Caller id (chunk id or query id).
        text: Text to embed.
        kind: ``document`` or ``query`` (models may embed them differently).
        content_hash: Hash of ``text`` (computed when omitted).
        metadata: Free-form, read-only extras.
    """

    id: str
    text: str
    kind: str = "document"
    content_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.text:
            raise ValueError("EmbeddingRequest.id and text must be non-empty")
        if self.kind not in EMBEDDING_KINDS:
            raise ValueError(f"EmbeddingRequest.kind must be one of {EMBEDDING_KINDS}")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", text_hash(self.text))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_chunk(cls, chunk: KnowledgeChunk) -> EmbeddingRequest:
        """Build a document request for a knowledge chunk."""
        return cls(id=chunk.id, text=chunk.text, kind="document", content_hash=chunk.content_hash)


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """An ordered group of requests with unique ids."""

    requests: tuple[EmbeddingRequest, ...]

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        if not requests:
            raise ValueError("EmbeddingBatch must not be empty")
        ids = [r.id for r in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("EmbeddingBatch request ids must be unique")
        object.__setattr__(self, "requests", requests)

    @property
    def batch_id(self) -> str:
        """Deterministic id from the ordered request ids and content hashes."""
        return content_hash([[r.id, r.content_hash] for r in self.requests])

    def __len__(self) -> int:
        return len(self.requests)

    @classmethod
    def batches(
        cls, requests: Sequence[EmbeddingRequest], batch_size: int
    ) -> Iterator[EmbeddingBatch]:
        """Split ``requests`` into consecutive batches of at most ``batch_size``.

        Raises:
            ValueError: If ``batch_size`` < 1.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        for start in range(0, len(requests), batch_size):
            yield cls(tuple(requests[start : start + batch_size]))


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Vector produced for one request."""

    request_id: str
    vector: Vector
    model: str
    content_hash: str
    cached: bool = False

    def __post_init__(self) -> None:
        vector = tuple(float(v) for v in self.vector)
        if not vector or not all(math.isfinite(v) for v in vector):
            raise ValueError("EmbeddingResult.vector must be non-empty and finite")
        object.__setattr__(self, "vector", vector)

    @property
    def dimension(self) -> int:
        """Vector length."""
        return len(self.vector)


class EmbeddingCache(ABC):
    """Content-addressed vector cache (implementations: memory, disk, KV store, ...)."""

    @staticmethod
    def key(model: str, kind: str, content_hash_: str) -> str:
        """Cache key: identical text embedded by the same model is reused across runs."""
        return content_hash([model, kind, content_hash_])

    def get(self, key: str) -> Vector | None:
        """Return the cached vector, or ``None``."""
        return self._get(key)

    def put(self, key: str, vector: Vector) -> None:
        """Store ``vector`` under ``key``."""
        self._put(key, tuple(vector))

    @abstractmethod
    def _get(self, key: str) -> Vector | None:
        raise NotImplementedError

    @abstractmethod
    def _put(self, key: str, vector: Vector) -> None:
        raise NotImplementedError


class AbstractEmbeddingModel(AbstractComponent):
    """Maps texts to dense vectors.

    Parameters (``params`` of ``configs/embedding/<name>.yaml``):
        ``normalize`` (bool, default ``false``): L2-normalise vectors.
        ``batch_size`` (int, default 32): requests per :class:`EmbeddingBatch`.

    Args:
        config: Component parameters.
        name: Instance name.
        cache: Optional vector cache.
    """

    component_type: ClassVar[str] = "embedding_model"

    def __init__(
        self, config: Any = None, *, name: str | None = None, cache: EmbeddingCache | None = None
    ) -> None:
        super().__init__(config, name=name)
        self._normalize = self.bool_param("normalize", False)
        self.batch_size = self.int_param("batch_size", 32, minimum=1)
        self._cache = cache

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension produced by the model."""
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        """Identifier recorded with results and used in cache keys (default: instance name)."""
        return self.name

    def embed(self, batch: EmbeddingBatch) -> list[EmbeddingResult]:
        """Embed a batch, serving cache hits and validating the model output.

        Raises:
            ComponentError: If the model returns the wrong number of vectors, the wrong
                dimension or non-finite values.
        """
        results: dict[str, EmbeddingResult] = {}
        misses: list[EmbeddingRequest] = []
        for request in batch.requests:
            cached = self._cache.get(self._key(request)) if self._cache else None
            if cached is not None:
                results[request.id] = EmbeddingResult(
                    request.id, cached, self.model_id, request.content_hash, cached=True
                )
            else:
                misses.append(request)
        if misses:
            vectors = self._invoke(
                "embed", lambda: list(self._embed(misses)), batch_size=len(misses)
            )
            if len(vectors) != len(misses):
                raise ComponentError(
                    self.name, f"returned {len(vectors)} vectors for {len(misses)} requests"
                )
            for request, raw in zip(misses, vectors, strict=True):
                vector = self._check(raw)
                if self._cache is not None:
                    self._cache.put(self._key(request), vector)
                results[request.id] = EmbeddingResult(
                    request.id, vector, self.model_id, request.content_hash
                )
        return [results[request.id] for request in batch.requests]

    def embed_chunks(self, chunks: Sequence[KnowledgeChunk]) -> list[EmbeddingResult]:
        """Embed knowledge chunks in batches of ``batch_size``."""
        requests = [EmbeddingRequest.from_chunk(chunk) for chunk in chunks]
        results: list[EmbeddingResult] = []
        for batch in EmbeddingBatch.batches(requests, self.batch_size):
            results.extend(self.embed(batch))
        return results

    def _key(self, request: EmbeddingRequest) -> str:
        return EmbeddingCache.key(self.model_id, request.kind, request.content_hash)

    def _check(self, raw: Sequence[float]) -> Vector:
        vector = tuple(float(v) for v in raw)
        if len(vector) != self.dimension:
            raise ComponentError(self.name, f"vector dimension {len(vector)} != {self.dimension}")
        if not all(math.isfinite(v) for v in vector):
            raise ComponentError(self.name, "vector contains non-finite values")
        if self._normalize:
            norm = math.sqrt(sum(v * v for v in vector))
            if norm == 0:
                raise ComponentError(self.name, "cannot normalise a zero vector")
            vector = tuple(v / norm for v in vector)
        return vector

    @abstractmethod
    def _embed(self, requests: Sequence[EmbeddingRequest]) -> Sequence[Sequence[float]]:
        """Return one vector per request, in order."""
        raise NotImplementedError


EMBEDDING_MODELS: Registry[AbstractEmbeddingModel] = Registry("embedding", AbstractEmbeddingModel)
"""Registry of embedding model implementations (empty in Milestone 2)."""
