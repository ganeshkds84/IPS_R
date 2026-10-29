"""Abstract dataset: source of queries, documents and references."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from ips_r.components.base import AbstractComponent
from ips_r.core.registry import Registry
from ips_r.core.types import Document, Query, Reference


class AbstractDataset(AbstractComponent):
    """Provides the evaluation material for an experiment.

    Implementations override the ``_load_*`` hooks; caching, splitting and
    download logic are implementation concerns.
    """

    component_type: ClassVar[str] = "dataset"

    def queries(self) -> list[Query]:
        """Return the queries of this dataset.

        Raises:
            ComponentError: If loading fails.
        """
        queries = self._invoke("queries", self._load_queries)
        self._logger.info("dataset.queries.loaded", count=len(queries))
        return queries

    def documents(self) -> list[Document]:
        """Return the document collection.

        Raises:
            ComponentError: If loading fails.
        """
        documents = self._invoke("documents", self._load_documents)
        self._logger.info("dataset.documents.loaded", count=len(documents))
        return documents

    def references(self) -> list[Reference]:
        """Return gold references aligned with :meth:`queries` by ``query_id``.

        Raises:
            ComponentError: If loading fails.
        """
        references = self._invoke("references", self._load_references)
        self._logger.info("dataset.references.loaded", count=len(references))
        return references

    @abstractmethod
    def _load_queries(self) -> list[Query]:
        """Load and return queries."""
        raise NotImplementedError

    @abstractmethod
    def _load_documents(self) -> list[Document]:
        """Load and return documents."""
        raise NotImplementedError

    @abstractmethod
    def _load_references(self) -> list[Reference]:
        """Load and return references."""
        raise NotImplementedError


DATASETS: Registry[AbstractDataset] = Registry("dataset", AbstractDataset)
"""Registry of dataset implementations."""
