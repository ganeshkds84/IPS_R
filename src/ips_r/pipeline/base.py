"""Abstract pipeline: orchestrates components into an experiment."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from ips_r.components.base import AbstractComponent
from ips_r.core.registry import Registry
from ips_r.pipeline.components import PipelineComponents


class AbstractPipeline(AbstractComponent):
    """Owns a :class:`PipelineComponents` bundle and defines how they interact.

    ``setup()`` / ``teardown()`` cascade to the components. Subclasses that override
    :meth:`_setup` or :meth:`_teardown` must call ``super()``.

    Args:
        components: Configured component instances.
        config: Pipeline-specific options.
        name: Instance name.
    """

    component_type: ClassVar[str] = "pipeline"

    def __init__(
        self,
        components: PipelineComponents,
        config: Mapping[str, Any] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(config, name=name)
        self._components = components

    @property
    def components(self) -> PipelineComponents:
        """The components orchestrated by this pipeline."""
        return self._components

    def _setup(self) -> None:
        self._components.setup_all()

    def _teardown(self) -> None:
        self._components.teardown_all()

    def run(self) -> dict[str, Any]:
        """Execute the pipeline end-to-end.

        Returns:
            A JSON-serialisable summary (metrics, counts, artefact paths, ...).

        Raises:
            ComponentError: If execution fails.
        """
        summary = self._invoke("run", self._run, components=self._components.configured())
        self._logger.info("pipeline.completed", keys=sorted(summary))
        return summary

    @abstractmethod
    def _run(self) -> dict[str, Any]:
        """Implementation hook for :meth:`run`."""
        raise NotImplementedError


PIPELINES: Registry[AbstractPipeline] = Registry("pipeline", AbstractPipeline)
"""Registry of pipeline implementations."""
