"""Instantiate components and pipelines from configuration via the registries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from omegaconf import DictConfig, OmegaConf

from ips_r.candidates.base import CANDIDATE_GENERATORS
from ips_r.config.schema import COMPONENT_KEYS
from ips_r.core.exceptions import ConfigurationError, PipelineError, RegistryError
from ips_r.core.log import get_logger
from ips_r.core.registry import Registry
from ips_r.data.base import DATASETS
from ips_r.data.chunking.base import CHUNKERS
from ips_r.evaluation.base import EVALUATORS
from ips_r.generation.base import GENERATORS
from ips_r.knowledge.embedding import EMBEDDING_MODELS
from ips_r.knowledge.indexing import HYBRID_INDEXERS, SPARSE_INDEXERS, VECTOR_STORES
from ips_r.pipeline.base import PIPELINES, AbstractPipeline
from ips_r.pipeline.components import PipelineComponents
from ips_r.policy.base import DECISION_POLICIES
from ips_r.reranking.base import RERANKERS
from ips_r.retrieval.base import RETRIEVERS

logger = get_logger(__name__)

COMPONENT_REGISTRIES: Final[Mapping[str, Registry[Any]]] = {
    "dataset": DATASETS,
    "chunker": CHUNKERS,
    "retriever": RETRIEVERS,
    "candidate_generator": CANDIDATE_GENERATORS,
    "decision_policy": DECISION_POLICIES,
    "reranker": RERANKERS,
    "generator": GENERATORS,
    "evaluator": EVALUATORS,
    "pipeline": PIPELINES,
    "embedding": EMBEDDING_MODELS,
    "vector_store": VECTOR_STORES,
    "sparse_indexer": SPARSE_INDEXERS,
    "hybrid_indexer": HYBRID_INDEXERS,
}
"""All registries by kind (pipeline components are built from ``COMPONENT_KEYS``)."""


def parse_selection(kind: str, node: Any) -> tuple[str | None, dict[str, Any]]:
    """Extract ``(name, params)`` from a component config node.

    Raises:
        ConfigurationError: If the node is malformed.
    """
    if node is None:
        return None, {}
    if isinstance(node, DictConfig):
        container = OmegaConf.to_container(node, resolve=True)
        node = container
    if not isinstance(node, Mapping):
        raise ConfigurationError(f"{kind} config must be a mapping, got {type(node).__name__}")
    name = node.get("name")
    if name is not None and not isinstance(name, str):
        raise ConfigurationError(f"{kind}.name must be a string or null, got {name!r}")
    params = node.get("params") or {}
    if not isinstance(params, Mapping):
        raise ConfigurationError(f"{kind}.params must be a mapping, got {type(params).__name__}")
    return name, {str(key): value for key, value in params.items()}


def build_component[T](registry: Registry[T], node: Any) -> T | None:
    """Instantiate the component selected by ``node`` (``{name, params}``).

    Returns:
        The component, or ``None`` if ``node.name`` is null (component disabled).

    Raises:
        ConfigurationError: If the node is malformed or names an unregistered component.
        PipelineError: If construction fails.
    """
    name, params = parse_selection(registry.kind, node)
    if name is None:
        logger.debug("component.disabled", kind=registry.kind)
        return None
    if name not in registry:
        available = ", ".join(registry.names()) or "<none registered>"
        raise ConfigurationError(f"Unknown {registry.kind} '{name}'. Available: {available}")
    try:
        component = registry.create(name, params, name=name)
    except RegistryError as exc:
        raise PipelineError(str(exc)) from exc
    logger.info("component.built", kind=registry.kind, name=name)
    return component


def build_pipeline_components(cfg: DictConfig | Mapping[str, Any]) -> PipelineComponents:
    """Instantiate every component configured in ``cfg``.

    Raises:
        ConfigurationError: If any component selection is invalid.
        PipelineError: If construction fails.
    """
    built = {
        key: build_component(COMPONENT_REGISTRIES[key], cfg.get(key)) for key in COMPONENT_KEYS
    }
    return PipelineComponents(**built)


def build_pipeline(
    cfg: DictConfig | Mapping[str, Any], components: PipelineComponents
) -> AbstractPipeline | None:
    """Instantiate the pipeline selected by ``cfg.pipeline``.

    Returns:
        The pipeline, or ``None`` if no pipeline is configured.

    Raises:
        ConfigurationError: If the selection is invalid.
        PipelineError: If construction fails.
    """
    name, params = parse_selection("pipeline", cfg.get("pipeline"))
    if name is None:
        logger.debug("pipeline.disabled")
        return None
    if name not in PIPELINES:
        available = ", ".join(PIPELINES.names()) or "<none registered>"
        raise ConfigurationError(f"Unknown pipeline '{name}'. Available: {available}")
    try:
        pipeline = PIPELINES.create(name, components, params, name=name)
    except RegistryError as exc:
        raise PipelineError(str(exc)) from exc
    logger.info("pipeline.built", name=name, components=components.configured())
    return pipeline
