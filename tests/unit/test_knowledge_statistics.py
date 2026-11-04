"""Tests for knowledge statistics and processing reports (Part I)."""

from __future__ import annotations

import json
from typing import Any

from ips_r.config.schema import StatisticsConfig
from ips_r.knowledge import KnowledgeProcessor, KnowledgeStatisticsAccumulator, ProcessingReport
from ips_r.knowledge.statistics import statistics_markdown
from tests.knowledge_factories import paper


def _stats(settings: StatisticsConfig | None = None) -> dict[str, Any]:
    processor = KnowledgeProcessor()
    accumulator = KnowledgeStatisticsAccumulator("memory", settings)
    for doc_id in ("a", "b"):
        accumulator.add(processor.process(paper(doc_id)))
    return accumulator.result()


def test_statistics_cover_every_object_kind() -> None:
    stats = _stats()
    documents = stats["documents"]
    assert documents["count"] == 2 and documents["pages"] == 4
    assert set(documents["per_document"]) >= {"sections", "chunks", "citations", "references"}
    sections, chunks = stats["sections"], stats["chunks"]
    assert sections["appendix"] == 4
    assert set(sections["by_level"]) == {"1", "2"}
    assert chunks["count"] == sum(chunks["by_kind"].values())
    assert chunks["duplicates"] > 0  # the second paper repeats the first
    assert 0 < chunks["duplicate_ratio"] < 1
    citations, references = stats["citations"], stats["references"]
    assert set(citations["by_style"]) == {"numeric", "author_year"}
    assert 0 < citations["resolution_rate"] <= 1
    assert citations["most_cited"][0][1] >= citations["most_cited"][-1][1]
    assert references["count"] == 4 and references["with_doi"] == 2
    assert references["by_year"] == {"2019": 2, "2020": 2}
    tokens = stats["tokens"]
    assert tokens["total"] > tokens["vocabulary_size"] > 0
    json.dumps(stats)  # JSON-serialisable


def test_statistics_options() -> None:
    folded = _stats(StatisticsConfig(lowercase=True))
    cased = _stats(StatisticsConfig(lowercase=False))
    assert folded["tokens"]["vocabulary_size"] < cased["tokens"]["vocabulary_size"]
    assert len(_stats(StatisticsConfig(top_k_terms=1))["citations"]["most_cited"]) == 1


def test_empty_statistics_have_zero_ratios() -> None:
    stats = KnowledgeStatisticsAccumulator("empty").result()
    assert stats["chunks"]["duplicate_ratio"] == 0.0
    assert stats["citations"]["resolution_rate"] == 0.0
    assert stats["tokens"]["type_token_ratio"] == 0.0
    assert "# Knowledge statistics: empty" in statistics_markdown(stats)


def test_statistics_markdown() -> None:
    markdown = statistics_markdown(_stats())
    assert markdown.startswith("# Knowledge statistics: memory")
    for heading in ("## Chunks by kind", "## Sections by level", "## Objects per document"):
        assert heading in markdown
    assert "| Documents | 2 |" in markdown
    assert markdown.endswith("\n")


def test_processing_report_serialisation() -> None:
    report = ProcessingReport(dataset="memory", documents_seen=3, documents_processed=2)
    report.add_time("chunking", 0.12345)
    report.add_time("chunking", 0.1)
    report.stages = {"chunking": "completed", "embedding": "skipped"}
    assert report.to_dict()["stage_seconds"] == {"chunking": 0.223}
    markdown = report.to_markdown()
    assert "| chunking | completed | 0.223 |" in markdown
    assert "| embedding | skipped | 0.000 |" in markdown
    assert "No overlap issues." in markdown and "## Failures" not in markdown


def test_processing_report_lists_issues_and_failures() -> None:
    report = ProcessingReport(dataset="memory")
    report.chunk_validation = {"issue_counts": {"gap": 2}, "duplicates": 5}
    report.failures.append({"document_id": "d1", "error": "ValueError: boom"})
    markdown = report.to_markdown()
    assert "- `gap`: 2" in markdown and "- duplicates: 5" in markdown
    assert "- `d1`: ValueError: boom" in markdown
