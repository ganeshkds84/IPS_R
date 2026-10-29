"""Abstract decision policy: chooses an action given a query and candidates."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from ips_r.components.base import AbstractComponent
from ips_r.core.exceptions import ComponentError
from ips_r.core.registry import Registry
from ips_r.core.types import Candidate, Decision, Query


class AbstractDecisionPolicy(AbstractComponent):
    """Maps a query and its candidates to a :class:`~ips_r.core.types.Decision`.

    Contract: every id in ``Decision.selected_ids`` refers to one of the given candidates.
    """

    component_type: ClassVar[str] = "decision_policy"

    def decide(self, query: Query, candidates: Sequence[Candidate]) -> Decision:
        """Return the policy's decision for ``query``.

        Raises:
            ComponentError: If the decision fails or references unknown candidates.
        """
        decision = self._invoke(
            "decide",
            lambda: self._decide(query, candidates),
            query_id=query.id,
            num_candidates=len(candidates),
        )
        known = {candidate.id for candidate in candidates}
        unknown = [cid for cid in decision.selected_ids if cid not in known]
        if unknown:
            raise ComponentError(self.name, f"decision selected unknown candidate ids {unknown}")
        self._logger.debug(
            "policy.decided",
            query_id=query.id,
            action=decision.action,
            confidence=decision.confidence,
        )
        return decision

    @abstractmethod
    def _decide(self, query: Query, candidates: Sequence[Candidate]) -> Decision:
        """Implementation hook for :meth:`decide`."""
        raise NotImplementedError


DECISION_POLICIES: Registry[AbstractDecisionPolicy] = Registry(
    "decision_policy", AbstractDecisionPolicy
)
"""Registry of decision-policy implementations."""
