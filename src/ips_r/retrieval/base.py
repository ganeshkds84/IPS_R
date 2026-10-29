"""Abstract retriever: fetches documents relevant to a query from an index."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from ips_r.components.base import AbstractComponent
from ips_r.core.exceptions import ComponentError
from ips_r.core.registry import Registry
from ips_r.core.types import Document, Query, ScoredDocument


class AbstractRetriever(AbstractComponent):
    """Indexes a document collection and returns ranked documents for queries.

    Contract: :meth:`retrieve` returns at most ``top_k`` results sorted by
    descending score.
    """

    component_type: ClassVar[str] = "retriever"

    def index(self, documents: Sequence[Document]) -> None:
        """Build (or rebuild) the index over ``documents``.

        Raises:
            ComponentError: If ``documents`` is empty or indexing fails.
        """
        if len(documents) == 0:
            raise ComponentError(self.name, "cannot index an empty document collection")
        self._invoke("index", lambda: self._index(documents), num_documents=len(documents))
        self._logger.info("retriever.indexed", num_documents=len(documents))

    def retrieve(self, query: Query, top_k: int = 10) -> list[ScoredDocument]:
        """Return up to ``top_k`` documents for ``query``.

        Raises:
            ComponentError: If ``top_k`` is invalid, retrieval fails, or the
                implementation violates the result contract.
        """
        self._require_positive("top_k", top_k)
        results = self._invoke(
            "retrieve", lambda: self._retrieve(query, top_k), query_id=query.id, top_k=top_k
        )
        if len(results) > top_k:
            raise ComponentError(self.name, f"returned {len(results)} results for top_k={top_k}")
        return results

    def retrieve_batch(
        self, queries: Sequence[Query], top_k: int = 10
    ) -> list[list[ScoredDocument]]:
        """Retrieve for several queries. Override for vectorised implementations."""
        return [self.retrieve(query, top_k) for query in queries]

    @abstractmethod
    def _index(self, documents: Sequence[Document]) -> None:
        """Implementation hook for :meth:`index`."""
        raise NotImplementedError

    @abstractmethod
    def _retrieve(self, query: Query, top_k: int) -> list[ScoredDocument]:
        """Implementation hook for :meth:`retrieve`."""
        raise NotImplementedError


RETRIEVERS: Registry[AbstractRetriever] = Registry("retriever", AbstractRetriever)
"""Registry of retriever implementations."""
