"""Integration tests: config -> components -> pipeline -> tracked run directory."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from omegaconf import DictConfig

from ips_r.core.exceptions import ConfigurationError, PipelineError
from ips_r.experiment import execute_run, run_context
from ips_r.pipeline import COMPONENT_REGISTRIES, build_pipeline_components
from tests.fakes import (
    FakeDataset,
    FakeEvaluator,
    FakeGenerator,
    FakePipeline,
    FakeRetriever,
)

MakeConfig = Callable[..., DictConfig]

FAKES = {
    "dataset": FakeDataset,
    "retriever": FakeRetriever,
    "generator": FakeGenerator,
    "evaluator": FakeEvaluator,
    "pipeline": FakePipeline,
}


@pytest.fixture
def registered_fakes() -> Iterator[list[str]]:
    """Register fake implementations under the name ``fake`` for the duration of a test."""
    for kind, cls in FAKES.items():
        COMPONENT_REGISTRIES[kind].add("fake", cls)
    overrides = [f"{kind}.name=fake" for kind in FAKES]
    yield overrides
    for kind in FAKES:
        COMPONENT_REGISTRIES[kind].remove("fake")


def _metadata(run_dir: Path) -> dict[str, object]:
    data: dict[str, object] = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    return data


def test_dry_run_with_no_components(make_config: MakeConfig) -> None:
    cfg = make_config(["experiment=example"])
    with run_context(cfg) as ctx:
        summary = execute_run(ctx, dry_run=True)
    assert summary == {"status": "dry_run", "components": {}}
    assert ctx.output_dir.parent.name == "example"
    assert (ctx.output_dir / "config.yaml").is_file()
    assert (ctx.output_dir / "run.log").is_file()
    metadata = _metadata(ctx.output_dir)
    assert metadata["status"] == "completed"
    seed = metadata["seed"]
    assert isinstance(seed, dict)
    assert seed["seed"] == 1234
    assert {"random", "numpy"} <= set(seed["libraries"])


def test_no_pipeline_configured(make_config: MakeConfig) -> None:
    with run_context(make_config()) as ctx:
        assert execute_run(ctx)["status"] == "no_pipeline"


def test_full_pipeline_with_fakes(make_config: MakeConfig, registered_fakes: list[str]) -> None:
    cfg = make_config(registered_fakes)
    with run_context(cfg) as ctx:
        summary = execute_run(ctx)
    assert summary["status"] == "completed"
    assert summary["results"]["metrics"] == {"accuracy": 1.0}
    assert (ctx.output_dir / "results.json").is_file()


def test_failed_run_is_recorded(make_config: MakeConfig, tmp_path: Path) -> None:
    out = tmp_path / "custom-run"
    cfg = make_config([f"run.output_dir='{out.as_posix()}'"])
    with pytest.raises(RuntimeError), run_context(cfg):
        raise RuntimeError("boom")
    metadata = _metadata(out)
    assert metadata["status"] == "failed"
    assert "boom" in str(metadata["error"])


def test_unknown_component_name(make_config: MakeConfig) -> None:
    with pytest.raises(ConfigurationError, match="Unknown retriever 'nope'"):
        build_pipeline_components(make_config(["retriever.name=nope"]))


class ExplodingRetriever(FakeRetriever):
    def __init__(self, config: object = None, *, name: str | None = None) -> None:
        raise RuntimeError("no GPU available")


def test_construction_failure_is_wrapped(make_config: MakeConfig) -> None:
    COMPONENT_REGISTRIES["retriever"].add("exploding", ExplodingRetriever)
    try:
        with pytest.raises(PipelineError, match="no GPU available"):
            build_pipeline_components(make_config(["retriever.name=exploding"]))
    finally:
        COMPONENT_REGISTRIES["retriever"].remove("exploding")


def test_malformed_component_node() -> None:
    with pytest.raises(ConfigurationError, match="params must be a mapping"):
        build_pipeline_components({"retriever": {"name": "x", "params": [1]}})
