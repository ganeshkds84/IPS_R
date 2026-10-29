"""Abstract reranker: reorders a list of scored documents for a query."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from ips_r.components.base import AbstractComponent
from ips_r.core.exceptions import ComponentError
from ips_r.core.registry import Registry
from ips_r.core.types import Query, ScoredDocument


class AbstractReranker(AbstractComponent):
    """Re-scores and reorders documents.

    Contract: the output is a subset of the input documents (matched by id),
    of length at most ``top_k`` when given.
    """

    component_type: ClassVar[str] = "reranker"

    def rerank(
        self,
        query: Query,
        documents: Sequence[ScoredDocument],
        top_k: int | None = None,
    ) -> list[ScoredDocument]:
        """Rerank ``documents`` for ``query``.

        Raises:
            ComponentError: If ``top_k`` is invalid, reranking fails, or the output
                violates the contract.
        """
        if top_k is not None:
            self._require_positive("top_k", top_k)
        if not documents:
            self._logger.debug("reranker.empty_input", query_id=query.id)
            return []
        results = self._invoke(
            "rerank",
            lambda: self._rerank(query, documents, top_k),
            query_id=query.id,
            num_documents=len(documents),
            top_k=top_k,
        )
        input_ids = {item.document.id for item in documents}
        foreign = [item.document.id for item in results if item.document.id not in input_ids]
        if foreign:
            raise ComponentError(self.name, f"returned documents not in the input: {foreign}")
        if top_k is not None and len(results) > top_k:
            raise ComponentError(self.name, f"returned {len(results)} results for top_k={top_k}")
        return results

    @abstractmethod
    def _rerank(
        self,
        query: Query,
        documents: Sequence[ScoredDocument],
        top_k: int | None,
    ) -> list[ScoredDocument]:
        """Implementation hook for :meth:`rerank`."""
        raise NotImplementedError


RERANKERS: Registry[AbstractReranker] = Registry("reranker", AbstractReranker)
"""Registry of reranker implementations."""
