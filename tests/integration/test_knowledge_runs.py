"""Integration tests: knowledge builds in tracked runs, index-component wiring and the CLI."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from omegaconf import DictConfig
from typer.testing import CliRunner

from ips_r.cli import app
from ips_r.core.exceptions import ConfigurationError, PipelineError
from ips_r.experiment import build_index_components, execute_knowledge, run_context
from ips_r.knowledge import (
    EMBEDDING_MODELS,
    HYBRID_INDEXERS,
    SPARSE_INDEXERS,
    VECTOR_STORES,
    IndexComponents,
    load_manifest,
    verify_artifacts,
)
from tests import data_factories as factories
from tests.knowledge_fakes import HashEmbedding, MemorySparseIndexer, MemoryVectorStore, SumHybrid

MakeConfig = Callable[..., DictConfig]
runner = CliRunner()


FAKE_REGISTRIES = (
    (EMBEDDING_MODELS, HashEmbedding),
    (VECTOR_STORES, MemoryVectorStore),
    (SPARSE_INDEXERS, MemorySparseIndexer),
    (HYBRID_INDEXERS, SumHybrid),
)


@pytest.fixture
def fake_index_components() -> Iterator[None]:
    """Register the in-memory doubles under the name ``fake``."""
    for registry, cls in FAKE_REGISTRIES:
        registry.add("fake", cls)  # type: ignore[arg-type]  # heterogeneous registry tuple
    yield
    for registry, _ in FAKE_REGISTRIES:
        registry.remove("fake")


def _squad_overrides(tmp_path: Path) -> list[str]:
    source = factories.squad_file(tmp_path / "src")
    return ["dataset=squad", f"dataset.params.path='{source.as_posix()}'"]


def test_execute_knowledge_in_tracked_run(make_config: MakeConfig, tmp_path: Path) -> None:
    cfg = make_config(
        [*_squad_overrides(tmp_path), "chunker=sentence", "knowledge.output_subdir=kb"]
    )
    with run_context(cfg) as ctx:
        summary = execute_knowledge(ctx)
    assert summary["status"] == "completed"
    assert summary["documents"] == 2 and summary["documents_dropped"] == 1
    assert summary["stages"]["chunking"] == "completed"
    assert verify_artifacts(ctx.output_dir / "kb") == []
    results = json.loads((ctx.output_dir / "results.json").read_text(encoding="utf-8"))
    assert results["artifact_hash"] == summary["artifact_hash"]


def test_execute_knowledge_with_index_components(
    make_config: MakeConfig, tmp_path: Path, fake_index_components: None
) -> None:
    overrides = [
        *_squad_overrides(tmp_path),
        "embedding.name=fake",
        "index.vector_store.name=fake",
        "index.sparse.name=fake",
        "index.hybrid.name=fake",
    ]
    with run_context(make_config(overrides)) as ctx:
        summary = execute_knowledge(ctx)
    assert summary["stages"]["embedding"] == "completed"
    manifest = load_manifest(ctx.output_dir / "knowledge")
    assert [i["kind"] for i in manifest["indexes"]] == ["vector", "sparse", "hybrid"]


def test_execute_knowledge_without_hybrid_manages_index_components(
    make_config: MakeConfig, tmp_path: Path, fake_index_components: None
) -> None:
    overrides = [*_squad_overrides(tmp_path), "index.sparse.name=fake"]
    with run_context(make_config(overrides)) as ctx:
        summary = execute_knowledge(ctx)
    assert summary["stages"]["sparse_index"] == "completed"
    assert summary["stages"]["vector_store"].startswith("skipped")


def test_execute_knowledge_requires_dataset(make_config: MakeConfig) -> None:
    with (
        pytest.raises(ConfigurationError, match="requires a dataset"),
        run_context(make_config()) as ctx,
    ):
        execute_knowledge(ctx)


def test_build_index_components_defaults_to_none(make_config: MakeConfig) -> None:
    assert build_index_components(make_config()) == IndexComponents()


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        (["index.hybrid.name=fake"], ConfigurationError, "requires index.vector_store"),
        (
            ["index.vector_store.name=fake", "index.sparse.name=fake", "index.hybrid.name=nope"],
            ConfigurationError,
            "Unknown hybrid_indexer 'nope'",
        ),
        (
            [
                "index.vector_store.name=fake",
                "index.sparse.name=fake",
                "index.hybrid.name=fake",
                "+index.hybrid.params.dense_weight=heavy",
            ],
            PipelineError,
            "Failed to construct",
        ),
    ],
)
def test_build_index_components_errors(
    make_config: MakeConfig,
    fake_index_components: None,
    overrides: list[str],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        build_index_components(make_config(overrides))


def test_cli_build_and_verify(tmp_path: Path) -> None:
    root = f"paths.root='{tmp_path.as_posix()}'"
    result = runner.invoke(app, ["build-knowledge", root, *_squad_overrides(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Knowledge artifacts" in result.stdout and "[completed]" in result.stdout
    (run_dir,) = (tmp_path / "outputs" / "default").iterdir()
    artifacts = run_dir / "knowledge"

    verified = runner.invoke(app, ["verify-artifacts", str(artifacts)])
    assert verified.exit_code == 0 and "OK:" in verified.stdout

    (artifacts / "jsonl" / "chunks.jsonl").write_text("tampered\n", encoding="utf-8")
    tampered = runner.invoke(app, ["verify-artifacts", str(artifacts)])
    assert tampered.exit_code == 1
    assert "checksum mismatch" in tampered.output


def test_cli_reports_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["build-knowledge", f"paths.root='{tmp_path.as_posix()}'"])
    assert result.exit_code == 1 and "requires a dataset" in result.output
    missing = runner.invoke(app, ["verify-artifacts", str(tmp_path / "nowhere")])
    assert missing.exit_code == 1 and "Cannot read manifest" in missing.output
