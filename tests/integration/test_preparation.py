"""Integration tests: preparation service, tracked runs and the ``ips-r prepare`` CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from omegaconf import DictConfig
from typer.testing import CliRunner

from ips_r.cli import app
from ips_r.config.schema import PreparationConfig
from ips_r.core.exceptions import ConfigurationError, DataValidationError
from ips_r.data import CHUNKERS, DATASETS, Chunk, Document, Question, prepare_dataset
from ips_r.data.preparation import CHUNKS_FILE, DOCUMENTS_FILE, QUESTIONS_FILE
from ips_r.experiment import execute_preparation, execute_run, run_context
from tests import data_factories as factories

MakeConfig = Callable[..., DictConfig]
runner = CliRunner()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_dataset_writes_all_artifacts(tmp_path: Path) -> None:
    dataset = DATASETS.create("squad", {"path": str(factories.squad_file(tmp_path))}, name="squad")
    chunker = CHUNKERS.create(
        "sentence", {"sentences_per_chunk": 1, "sentence_overlap": 0}, name="sentence"
    )
    out = tmp_path / "prepared"
    result = prepare_dataset(dataset, out, chunker=chunker)

    assert result.dropped_documents == 1  # whitespace-only paragraph
    assert result.dropped_questions == 0
    assert result.statistics.num_documents == 2
    assert result.statistics.num_chunks == 3
    assert result.metadata.num_documents == 2
    assert result.metadata.content_hash is not None and result.metadata.content_hash.startswith(
        "sha256:"
    )
    assert result.metadata.extra["chunk_strategy"] == "sentence"
    assert not result.report.is_valid
    assert result.report.counts_by_code["empty_text"] == 1

    documents = [Document.from_dict(r) for r in _read_jsonl(out / DOCUMENTS_FILE)]
    chunks = [Chunk.from_dict(r) for r in _read_jsonl(out / CHUNKS_FILE)]
    questions = [Question.from_dict(r) for r in _read_jsonl(out / QUESTIONS_FILE)]
    assert [d.id for d in documents] == ["Super_Bowl_50#0", "article-1#0"]
    by_id = {d.id: d for d in documents}
    assert all(c.text == by_id[c.document_id].text[c.start : c.end] for c in chunks)
    assert len(questions) == 4
    for name in (
        "metadata",
        "validation_report",
        "validation_report_md",
        "statistics",
        "statistics_md",
    ):
        assert result.artifacts[name].is_file()
    summary = result.summary()
    assert summary["num_chunks"] == 3
    assert summary["validation"]["errors"] == result.report.num_errors

    again = prepare_dataset(
        DATASETS.create("squad", {"path": str(factories.squad_file(tmp_path))}, name="squad"),
        tmp_path / "again",
        chunker=chunker,
    )
    assert again.metadata.content_hash == result.metadata.content_hash


def test_prepare_dataset_options(tmp_path: Path) -> None:
    path = factories.squad_file(tmp_path)
    settings = PreparationConfig(drop_invalid=False, write_documents=False, write_questions=False)
    result = prepare_dataset(
        DATASETS.create("squad", {"path": str(path)}), tmp_path / "out", settings=settings
    )
    assert result.dropped_documents == 0
    assert result.statistics.num_documents == 3
    assert result.statistics.num_chunks == 0
    assert "documents" not in result.artifacts
    assert not (tmp_path / "out" / DOCUMENTS_FILE).exists()

    strict = PreparationConfig(fail_on_error=True)
    with pytest.raises(DataValidationError, match="failed validation"):
        prepare_dataset(
            DATASETS.create("squad", {"path": str(path)}), tmp_path / "strict", settings=strict
        )
    assert (tmp_path / "strict" / "validation_report.md").is_file()


def test_prepare_drops_invalid_questions(tmp_path: Path) -> None:
    path = factories.write_lines(
        tmp_path / "q.jsonl", ['{"id": "a", "question": " "}', '{"id": "b", "question": "ok?"}']
    )
    factories.paper_pdf(tmp_path / "pdfs" / "paper.pdf")
    dataset = DATASETS.create(
        "academic_pdf", {"directory": str(tmp_path / "pdfs"), "questions_path": str(path)}
    )
    result = prepare_dataset(dataset, tmp_path / "out")
    assert result.dropped_questions == 1
    assert result.statistics.num_pages == 2


def test_execute_preparation_in_tracked_run(make_config: MakeConfig, tmp_path: Path) -> None:
    source = factories.hotpot_file(tmp_path / "src")
    cfg = make_config(
        [
            "dataset=hotpotqa",
            f"dataset.params.path='{source.as_posix()}'",
            "chunker=token",
            "chunker.params.chunk_size=4",
            "chunker.params.chunk_overlap=1",
            "preparation.output_subdir=prepared",
        ]
    )
    with run_context(cfg) as ctx:
        summary = execute_preparation(ctx)
    assert summary["status"] == "completed"
    assert summary["num_documents"] == 2
    assert summary["num_chunks"] > 2
    assert (ctx.output_dir / "prepared" / "chunks.jsonl").is_file()
    results = json.loads((ctx.output_dir / "results.json").read_text(encoding="utf-8"))
    assert results["content_hash"] == summary["content_hash"]


def test_execute_preparation_requires_dataset(make_config: MakeConfig) -> None:
    with (
        pytest.raises(ConfigurationError, match="requires a dataset"),
        run_context(make_config()) as ctx,
    ):
        execute_preparation(ctx)


def test_dry_run_builds_dataset_and_chunker(make_config: MakeConfig, tmp_path: Path) -> None:
    source = factories.squad_file(tmp_path / "src")
    cfg = make_config(
        [
            "dataset=squad",
            f"dataset.params.path='{source.as_posix()}'",
            "chunker=recursive_character",
        ]
    )
    with run_context(cfg) as ctx:
        summary = execute_run(ctx, dry_run=True)
    assert summary["components"] == {"dataset": "squad", "chunker": "recursive_character"}


@pytest.mark.parametrize(
    "dataset", ["squad", "hotpotqa", "natural_questions", "ms_marco", "academic_pdf"]
)
@pytest.mark.parametrize(
    "chunker", ["none", "recursive_character", "sentence", "token", "sliding_window"]
)
def test_every_dataset_and_chunker_config_composes(
    make_config: MakeConfig, dataset: str, chunker: str
) -> None:
    cfg = make_config([f"dataset={dataset}", f"chunker={chunker}"])
    assert cfg.dataset.name == dataset
    assert cfg.chunker.name == (None if chunker == "none" else chunker)
    if chunker != "none":
        CHUNKERS.create(chunker, dict(cfg.chunker.params))


def test_semantic_chunker_fails_dry_run(tmp_path: Path) -> None:
    root = tmp_path.as_posix()
    result = runner.invoke(app, ["run", "--dry-run", f"paths.root='{root}'", "chunker=semantic"])
    assert result.exit_code == 1
    assert "placeholder" in result.output


def test_cli_prepare(tmp_path: Path) -> None:
    files = factories.msmarco_files(tmp_path / "msm")
    overrides = [f"dataset.params.{key}='{path.as_posix()}'" for key, path in files.items()]
    result = runner.invoke(
        app,
        [
            "prepare",
            f"paths.root='{tmp_path.as_posix()}'",
            "dataset=ms_marco",
            "chunker=sliding_window",
            *overrides,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Prepared dataset: ms_marco" in result.stdout
    (run_dir,) = (tmp_path / "outputs" / "default").iterdir()
    statistics = json.loads((run_dir / "dataset" / "statistics.json").read_text(encoding="utf-8"))
    assert statistics["num_documents"] == 4
    assert statistics["duplicate_documents"] == 1


def test_cli_prepare_reports_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["prepare", f"paths.root='{tmp_path.as_posix()}'"])
    assert result.exit_code == 1
    assert "requires a dataset" in result.output
