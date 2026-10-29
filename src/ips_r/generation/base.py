"""Abstract generator: produces the final output for a query from its context."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from ips_r.components.base import AbstractComponent
from ips_r.core.exceptions import ComponentError
from ips_r.core.registry import Registry
from ips_r.core.types import Generation, Query, ScoredDocument


class AbstractGenerator(AbstractComponent):
    """Generates a :class:`~ips_r.core.types.Generation` conditioned on context.

    Contract: ``Generation.query_id`` equals the id of the input query.
    """

    component_type: ClassVar[str] = "generator"

    def generate(self, query: Query, context: Sequence[ScoredDocument]) -> Generation:
        """Generate an output for ``query``.

        Raises:
            ComponentError: If generation fails or the output is for a different query.
        """
        output = self._invoke(
            "generate",
            lambda: self._generate(query, context),
            query_id=query.id,
            context_size=len(context),
        )
        if output.query_id != query.id:
            raise ComponentError(
                self.name,
                f"generation query_id {output.query_id!r} does not match query {query.id!r}",
            )
        return output

    def generate_batch(
        self, queries: Sequence[Query], contexts: Sequence[Sequence[ScoredDocument]]
    ) -> list[Generation]:
        """Generate for several queries. Override for batched implementations.

        Raises:
            ComponentError: If ``queries`` and ``contexts`` differ in length.
        """
        if len(queries) != len(contexts):
            raise ComponentError(
                self.name, f"got {len(queries)} queries but {len(contexts)} contexts"
            )
        return [self.generate(q, c) for q, c in zip(queries, contexts, strict=True)]

    @abstractmethod
    def _generate(self, query: Query, context: Sequence[ScoredDocument]) -> Generation:
        """Implementation hook for :meth:`generate`."""
        raise NotImplementedError


GENERATORS: Registry[AbstractGenerator] = Registry("generator", AbstractGenerator)
"""Registry of generator implementations."""
