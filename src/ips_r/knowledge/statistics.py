"""Knowledge statistics and processing reports (JSON + Markdown).

:class:`KnowledgeStatisticsAccumulator` streams over :class:`KnowledgeDocument` objects
and reports:

- **documents**: count, pages, objects per document;
- **sections**: count, depth distribution, appendix sections, length (characters of
  running text);
- **chunks**: count by kind, characters and tokens, chunks per document, duplicates;
- **citations**: count, styles, resolution rate, most-cited references;
- **references**: count, with year / DOI / first author, year distribution;
- **tokens**: total tokens, vocabulary size, type/token ratio.

:class:`ProcessingReport` describes a pipeline run: per-stage timings, document
outcomes, chunk validation, normalisation effect and the embedding/index stages.
"""

from __future__ import annotations

from array import array
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ips_r.config.schema import StatisticsConfig
from ips_r.data.statistics import LengthStats
from ips_r.data.text import compile_pattern
from ips_r.knowledge.models import KnowledgeDocument


def _length(values: list[int]) -> dict[str, Any]:
    return asdict(LengthStats.from_values(array("q", values)))


def _ratio(part: int, whole: int) -> float:
    return round(part / whole, 6) if whole else 0.0


class KnowledgeStatisticsAccumulator:
    """Incrementally computes knowledge statistics.

    Args:
        dataset: Dataset name.
        settings: Token pattern, case folding and top-k size.
    """

    def __init__(self, dataset: str, settings: StatisticsConfig | None = None) -> None:
        self._dataset = dataset
        self._s = settings or StatisticsConfig()
        self._word = compile_pattern(self._s.word_pattern)
        self._docs = 0
        self._pages = 0
        self._per_doc: dict[str, list[int]] = {}
        self._section_depth: Counter[int] = Counter()
        self._appendix = 0
        self._chunk_kinds: Counter[str] = Counter()
        self._chunk_chars: list[int] = []
        self._chunk_tokens: list[int] = []
        self._duplicates = 0
        self._citation_styles: Counter[str] = Counter()
        self._citations_resolved = 0
        self._cited: Counter[str] = Counter()
        self._xref_kinds: Counter[str] = Counter()
        self._xref_resolved = 0
        self._ref_years: Counter[int] = Counter()
        self._ref_with = Counter[str]()
        self._references = 0
        self._vocab: Counter[str] = Counter()

    def add(self, document: KnowledgeDocument) -> None:
        """Account for one knowledge document."""
        self._docs += 1
        self._pages += document.num_pages or 0
        for name, items in document.objects().items():
            self._per_doc.setdefault(name, []).append(len(items))
        for section in document.sections[1:]:
            self._section_depth[section.level] += 1
            self._appendix += section.is_appendix
        for chunk in document.chunks:
            self._chunk_kinds[chunk.kind.value] += 1
            self._chunk_chars.append(len(chunk.text))
            words = self._word.findall(chunk.text)
            self._chunk_tokens.append(len(words))
            self._duplicates += chunk.duplicate_of is not None
        for paragraph in document.paragraphs:
            words = self._word.findall(paragraph.text)
            self._vocab.update([w.lower() for w in words] if self._s.lowercase else words)
        for citation in document.citations:
            self._citation_styles[citation.style] += 1
            self._citations_resolved += citation.resolved
            self._cited.update(citation.target_ids)
        for xref in document.cross_references:
            self._xref_kinds[xref.target_kind] += 1
            self._xref_resolved += xref.target_id is not None
        for reference in document.references:
            self._references += 1
            if reference.year is not None:
                self._ref_years[reference.year] += 1
                self._ref_with["year"] += 1
            self._ref_with["doi"] += reference.doi is not None
            self._ref_with["first_author"] += reference.first_author is not None
            self._ref_with["marker"] += reference.marker is not None

    def result(self) -> dict[str, Any]:
        """Return the statistics as a JSON-serialisable dict."""
        citations = sum(self._citation_styles.values())
        xrefs = sum(self._xref_kinds.values())
        total_tokens = sum(self._vocab.values())
        top = sorted(self._cited.items(), key=lambda item: (-item[1], item[0]))[
            : self._s.top_k_terms
        ]
        return {
            "dataset": self._dataset,
            "documents": {
                "count": self._docs,
                "pages": self._pages,
                "per_document": {
                    name: _length(values) for name, values in sorted(self._per_doc.items())
                },
            },
            "sections": {
                "count": sum(self._section_depth.values()),
                "by_level": {str(k): v for k, v in sorted(self._section_depth.items())},
                "appendix": self._appendix,
            },
            "chunks": {
                "count": len(self._chunk_chars),
                "by_kind": dict(sorted(self._chunk_kinds.items())),
                "chars": _length(self._chunk_chars),
                "tokens": _length(self._chunk_tokens),
                "duplicates": self._duplicates,
                "duplicate_ratio": _ratio(self._duplicates, len(self._chunk_chars)),
            },
            "citations": {
                "count": citations,
                "by_style": dict(sorted(self._citation_styles.items())),
                "resolved": self._citations_resolved,
                "resolution_rate": _ratio(self._citations_resolved, citations),
                "most_cited": [[ref, count] for ref, count in top],
            },
            "cross_references": {
                "count": xrefs,
                "by_kind": dict(sorted(self._xref_kinds.items())),
                "resolved": self._xref_resolved,
                "resolution_rate": _ratio(self._xref_resolved, xrefs),
            },
            "references": {
                "count": self._references,
                "with_year": self._ref_with["year"],
                "with_doi": self._ref_with["doi"],
                "with_first_author": self._ref_with["first_author"],
                "with_marker": self._ref_with["marker"],
                "by_year": {str(k): v for k, v in sorted(self._ref_years.items())},
            },
            "tokens": {
                "total": total_tokens,
                "vocabulary_size": len(self._vocab),
                "type_token_ratio": _ratio(len(self._vocab), total_tokens),
            },
        }


def statistics_markdown(stats: Mapping[str, Any]) -> str:
    """Render :meth:`KnowledgeStatisticsAccumulator.result` as Markdown."""
    d, s, c, ci, x, r, t = (
        stats["documents"],
        stats["sections"],
        stats["chunks"],
        stats["citations"],
        stats["cross_references"],
        stats["references"],
        stats["tokens"],
    )
    lines = [
        f"# Knowledge statistics: {stats['dataset']}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Documents | {d['count']} |",
        f"| Pages | {d['pages']} |",
        f"| Sections | {s['count']} (appendix: {s['appendix']}) |",
        f"| Chunks | {c['count']} (duplicates: {c['duplicates']}) |",
        f"| Mean chunk chars / tokens | {c['chars']['mean']:.1f} / {c['tokens']['mean']:.1f} |",
        f"| Citations | {ci['count']} (resolved {ci['resolution_rate']:.1%}) |",
        f"| Cross references | {x['count']} (resolved {x['resolution_rate']:.1%}) |",
        f"| References | {r['count']} (with year {r['with_year']}, with DOI {r['with_doi']}) |",
        f"| Tokens / vocabulary | {t['total']} / {t['vocabulary_size']} |",
        "",
        "## Chunks by kind",
        "",
        *[f"- `{kind}`: {count}" for kind, count in c["by_kind"].items()],
        "",
        "## Sections by level",
        "",
        *[f"- level {level}: {count}" for level, count in s["by_level"].items()],
        "",
        "## Objects per document (mean)",
        "",
        "| Object | Mean | Max |",
        "| --- | ---: | ---: |",
        *[
            f"| {name} | {values['mean']:.1f} | {values['max']} |"
            for name, values in d["per_document"].items()
        ],
    ]
    return "\n".join(lines) + "\n"


@dataclass(slots=True)
class ProcessingReport:
    """Outcome of a knowledge pipeline run."""

    dataset: str
    documents_seen: int = 0
    documents_processed: int = 0
    documents_dropped: int = 0
    documents_failed: int = 0
    questions_written: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    chars_in: int = 0
    chars_out: int = 0
    chunk_validation: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, str] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    def add_time(self, stage: str, seconds: float) -> None:
        """Accumulate wall time for ``stage``."""
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + seconds

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (timings rounded to ms)."""
        data = asdict(self)
        data["stage_seconds"] = {k: round(v, 3) for k, v in self.stage_seconds.items()}
        return data

    def to_markdown(self) -> str:
        """Render a human-readable summary."""
        lines = [
            f"# Processing report: {self.dataset}",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Documents seen | {self.documents_seen} |",
            f"| Processed | {self.documents_processed} |",
            f"| Dropped (validation) | {self.documents_dropped} |",
            f"| Failed | {self.documents_failed} |",
            f"| Questions written | {self.questions_written} |",
            f"| Characters in / out | {self.chars_in} / {self.chars_out} |",
            "",
            "## Stages",
            "",
            "| Stage | Status | Seconds |",
            "| --- | --- | ---: |",
        ]
        for stage, status in self.stages.items():
            lines.append(f"| {stage} | {status} | {self.stage_seconds.get(stage, 0.0):.3f} |")
        counts = self.chunk_validation.get("issue_counts", {})
        lines += ["", "## Chunk validation", ""]
        lines += [f"- `{k}`: {v}" for k, v in counts.items()] or ["No overlap issues."]
        lines.append(f"- duplicates: {self.chunk_validation.get('duplicates', 0)}")
        if self.failures:
            lines += ["", "## Failures", ""]
            lines += [f"- `{f['document_id']}`: {f['error']}" for f in self.failures]
        return "\n".join(lines) + "\n"
