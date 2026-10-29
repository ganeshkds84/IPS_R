"""Container for the optional components that make up a pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields

from ips_r.candidates.base import AbstractCandidateGenerator
from ips_r.components.base import AbstractComponent
from ips_r.core.log import get_logger
from ips_r.data.base import AbstractDataset
from ips_r.evaluation.base import AbstractEvaluator
from ips_r.generation.base import AbstractGenerator
from ips_r.policy.base import AbstractDecisionPolicy
from ips_r.reranking.base import AbstractReranker
from ips_r.retrieval.base import AbstractRetriever

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineComponents:
    """The component instances selected by configuration; ``None`` means disabled."""

    dataset: AbstractDataset | None = None
    retriever: AbstractRetriever | None = None
    candidate_generator: AbstractCandidateGenerator | None = None
    decision_policy: AbstractDecisionPolicy | None = None
    reranker: AbstractReranker | None = None
    generator: AbstractGenerator | None = None
    evaluator: AbstractEvaluator | None = None

    def items(self) -> list[tuple[str, AbstractComponent]]:
        """Return ``(key, component)`` pairs for the enabled components, in pipeline order."""
        pairs: list[tuple[str, AbstractComponent]] = []
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                pairs.append((f.name, value))
        return pairs

    def configured(self) -> dict[str, str]:
        """Return a ``{key: component name}`` summary of the enabled components."""
        return {key: component.name for key, component in self.items()}

    def setup_all(self) -> None:
        """Set up every enabled component, tearing down already-started ones on failure.

        Raises:
            ComponentError: If any component fails to set up.
        """
        started: list[AbstractComponent] = []
        try:
            for _, component in self.items():
                component.setup()
                started.append(component)
        except Exception:
            logger.error("pipeline.components.setup_failed", started=[c.name for c in started])
            for component in reversed(started):
                try:
                    component.teardown()
                except Exception as exc:  # pragma: no cover - best-effort cleanup
                    logger.warning("pipeline.components.cleanup_failed", error=repr(exc))
            raise

    def teardown_all(self) -> None:
        """Tear down every enabled component in reverse order, continuing past failures.

        Raises:
            ComponentError: The first teardown error, after all components were attempted.
        """
        first_error: Exception | None = None
        for _, component in reversed(self.items()):
            try:
                component.teardown()
            except Exception as exc:
                logger.error("pipeline.components.teardown_failed", component=component.name)
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
