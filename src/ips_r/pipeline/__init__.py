"""Pipeline interface and config-driven component assembly."""

from ips_r.pipeline.base import PIPELINES, AbstractPipeline
from ips_r.pipeline.builder import (
    COMPONENT_REGISTRIES,
    build_component,
    build_pipeline,
    build_pipeline_components,
)
from ips_r.pipeline.components import PipelineComponents

__all__ = [
    "COMPONENT_REGISTRIES",
    "PIPELINES",
    "AbstractPipeline",
    "PipelineComponents",
    "build_component",
    "build_pipeline",
    "build_pipeline_components",
]
