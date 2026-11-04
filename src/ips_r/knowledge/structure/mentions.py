"""In-text citations and cross references: detection in running text and resolution.

Citations are numeric (``[1, 3-5]``) or author-year (``(Smith et al., 2020)``) and
resolve to reference entries; cross references (``Figure 2``, ``Table 1``,
``Section 3.1``, ``Eq. (4)``, ``Appendix A``) resolve to figures, tables, sections,
equations and appendix sections.
"""

from __future__ import annotations

import re

from ips_r.config.schema import StructureConfig
from ips_r.data.models import DocumentElement
from ips_r.knowledge.models import Citation, CrossReference, SourceSpan
from ips_r.knowledge.normalization import TrackedText
from ips_r.knowledge.structure.patterns import (
    APPENDIX_LETTER,
    CROSS_KIND,
    MAX_CITATION_RANGE,
    RANGE_BOUNDS,
    SURNAME,
    YEAR,
)
from ips_r.knowledge.structure.results import AnalysisState, Mention


class MentionResolver:
    """Finds citation / cross-reference mentions and resolves them at the end of a document.

    Args:
        settings: Detection switches and patterns.

    Raises:
        re.error: If a configured pattern is invalid.
    """

    def __init__(self, settings: StructureConfig) -> None:
        self._s = settings
        self._numeric = re.compile(settings.numeric_citation_pattern)
        self._author_year = re.compile(settings.author_year_citation_pattern)
        self._cross = re.compile(settings.cross_reference_pattern)

    def find(
        self,
        b: AnalysisState,
        tracked: TrackedText,
        element: DocumentElement | None,
        index: int | None,
        *,
        host_id: str,
        section_id: str,
        order: int,
        skip_leading_label: bool = False,
    ) -> None:
        """Record the mentions inside ``tracked`` (a caption skips its own leading label)."""

        def span(match: re.Match[str]) -> SourceSpan:
            start, end = tracked.source_span(match.start(), match.end())
            return SourceSpan(
                b.document.id,
                start,
                end,
                page=element.page if element else None,
                element_index=index,
            )

        text = tracked.text
        if self._s.detect_citations:
            for style, pattern in (("numeric", self._numeric), ("author_year", self._author_year)):
                for match in pattern.finditer(text):
                    b.mentions.append(
                        Mention(host_id, section_id, order, match, span(match), True, style)
                    )
        if self._s.detect_cross_references:
            for match in self._cross.finditer(text):
                if skip_leading_label and match.start() == 0:
                    continue
                b.mentions.append(Mention(host_id, section_id, order, match, span(match), False))

    def resolve(self, b: AnalysisState) -> None:
        """Turn every recorded mention into a :class:`Citation` or :class:`CrossReference`."""
        r = b.result
        by_marker = {ref.marker: ref.id for ref in r.references if ref.marker}
        by_author_year = {
            (ref.first_author.lower(), ref.year): ref.id
            for ref in reversed(r.references)
            if ref.first_author and ref.year
        }
        targets: dict[str, dict[str, str]] = {
            "figure": {f.number: f.id for f in r.figures if f.number},
            "table": {t.number: t.id for t in r.tables if t.number},
            "section": {s.number: s.id for s in r.sections if s.number and not s.is_appendix},
            "equation": {e.number: e.id for e in r.equations if e.number},
            "appendix": {s.number or "": s.id for s in r.sections if s.is_appendix},
        }
        for appendix in (s for s in r.sections if s.is_appendix and s.title):
            letter = APPENDIX_LETTER.match(appendix.title or "")
            if letter:
                targets["appendix"].setdefault(letter.group(1).upper(), appendix.id)
        for mention in b.mentions:
            if mention.is_citation:
                keys, ids = self._citation_targets(mention, by_marker, by_author_year)
                r.citations.append(
                    Citation(
                        id=b.new_id("c"),
                        document_id=b.document.id,
                        order=mention.order,
                        section_id=mention.section_id,
                        host_id=mention.host_id,
                        marker=mention.match.group(0),
                        style=mention.style,
                        target_ids=ids,
                        keys=keys,
                        source=mention.span,
                    )
                )
            else:
                word, number = mention.match.group(1), mention.match.group(2)
                kind = next(k for prefix, k in CROSS_KIND if word.lower().startswith(prefix))
                r.cross_references.append(
                    CrossReference(
                        id=b.new_id("x"),
                        document_id=b.document.id,
                        order=mention.order,
                        section_id=mention.section_id,
                        host_id=mention.host_id,
                        label=mention.match.group(0),
                        target_kind=kind,
                        target_number=number,
                        target_id=targets[kind].get(number),
                        source=mention.span,
                    )
                )

    @staticmethod
    def _citation_targets(
        mention: Mention, by_marker: dict[str, str], by_author_year: dict[tuple[str, int], str]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Expand a citation marker into keys and resolved reference ids."""
        body = mention.match.group(1)
        keys: list[str] = []
        if mention.style == "numeric":
            for part in re.split(r"[,;]", body):
                bounds = re.split(r"\s*[-\u2013]\s*", part.strip())
                if (
                    len(bounds) == RANGE_BOUNDS
                    and all(x.isdigit() for x in bounds)
                    and int(bounds[0]) <= int(bounds[1]) <= int(bounds[0]) + MAX_CITATION_RANGE
                ):
                    keys.extend(str(n) for n in range(int(bounds[0]), int(bounds[1]) + 1))
                elif part.strip().isdigit():
                    keys.append(part.strip())
            ids = [by_marker[k] for k in keys if k in by_marker]
        else:
            ids = []
            for part in body.split(";"):
                surname = SURNAME.search(part)
                year = YEAR.search(part)
                if surname and year:
                    keys.append(f"{surname.group(1)} {year.group(1)}")
                    ref = by_author_year.get((surname.group(1).lower(), int(year.group(1))))
                    if ref:
                        ids.append(ref)
        return tuple(dict.fromkeys(keys)), tuple(dict.fromkeys(ids))
