"""Abstract evaluator: scores predictions against references."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from ips_r.components.base import AbstractComponent
from ips_r.core.exceptions import ComponentError
from ips_r.core.registry import Registry
from ips_r.core.types import EvaluationResult, Generation, Reference


class AbstractEvaluator(AbstractComponent):
    """Computes metrics for aligned predictions and references.

    Contract: ``predictions[i]`` and ``references[i]`` refer to the same query.
    """

    component_type: ClassVar[str] = "evaluator"

    def evaluate(
        self, predictions: Sequence[Generation], references: Sequence[Reference]
    ) -> EvaluationResult:
        """Evaluate ``predictions`` against ``references``.

        Raises:
            ComponentError: If the inputs are empty, misaligned, or evaluation fails.
        """
        if not predictions:
            raise ComponentError(self.name, "cannot evaluate an empty prediction set")
        if len(predictions) != len(references):
            raise ComponentError(
                self.name,
                f"got {len(predictions)} predictions but {len(references)} references",
            )
        misaligned = [
            (p.query_id, r.query_id)
            for p, r in zip(predictions, references, strict=True)
            if p.query_id != r.query_id
        ]
        if misaligned:
            raise ComponentError(
                self.name, f"misaligned prediction/reference ids: {misaligned[:5]}"
            )
        result = self._invoke(
            "evaluate",
            lambda: self._evaluate(predictions, references),
            num_examples=len(predictions),
        )
        self._logger.info("evaluation.completed", metrics=dict(result.metrics))
        return result

    @abstractmethod
    def _evaluate(
        self, predictions: Sequence[Generation], references: Sequence[Reference]
    ) -> EvaluationResult:
        """Implementation hook for :meth:`evaluate`."""
        raise NotImplementedError


EVALUATORS: Registry[AbstractEvaluator] = Registry("evaluator", AbstractEvaluator)
"""Registry of evaluator implementations."""
