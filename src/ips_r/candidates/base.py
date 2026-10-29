"""Abstract candidate generator: proposes candidates from a query and its context."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from ips_r.components.base import AbstractComponent
from ips_r.core.registry import Registry
from ips_r.core.types import Candidate, Query, ScoredDocument


class AbstractCandidateGenerator(AbstractComponent):
    """Produces a set of candidates for downstream decision making."""

    component_type: ClassVar[str] = "candidate_generator"

    def generate(self, query: Query, context: Sequence[ScoredDocument]) -> list[Candidate]:
        """Generate candidates for ``query`` given retrieved ``context``.

        Raises:
            ComponentError: If generation fails.
        """
        candidates = self._invoke(
            "generate",
            lambda: self._generate(query, context),
            query_id=query.id,
            context_size=len(context),
        )
        self._logger.debug("candidates.generated", query_id=query.id, count=len(candidates))
        return candidates

    @abstractmethod
    def _generate(self, query: Query, context: Sequence[ScoredDocument]) -> list[Candidate]:
        """Implementation hook for :meth:`generate`."""
        raise NotImplementedError


CANDIDATE_GENERATORS: Registry[AbstractCandidateGenerator] = Registry(
    "candidate_generator", AbstractCandidateGenerator
)
"""Registry of candidate-generator implementations."""
