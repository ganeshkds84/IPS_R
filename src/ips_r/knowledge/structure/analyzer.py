"""Document walk: canonical document → section tree, knowledge objects and text units."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from ips_r.config.schema import ChunkProcessingConfig, StructureConfig
from ips_r.core.exceptions import ConfigurationError
from ips_r.data.models import Document, DocumentElement, ElementType
from ips_r.knowledge.models import ChunkKind, Section, SourceSpan
from ips_r.knowledge.normalization import TextNormalizer, TrackedText
from ips_r.knowledge.structure.builders import ObjectBuilder
from ips_r.knowledge.structure.mentions import MentionResolver
from ips_r.knowledge.structure.patterns import TABLE_LABEL
from ips_r.knowledge.structure.results import (
    FLOW_SEPARATOR,
    AnalysisState,
    StructureResult,
    TextUnit,
)

_Handler = Callable[[AnalysisState, Section, TrackedText, DocumentElement, int], None]


class StructuralAnalyzer:
    """Builds knowledge objects and text units from a canonical document.

    Args:
        normalizer: Text normaliser (also provides original-offset maps).
        structure: Detection patterns.
        chunking: Which content becomes separate (atomic) units.

    Raises:
        ConfigurationError: If a configured pattern is invalid.
    """

    def __init__(
        self,
        normalizer: TextNormalizer,
        structure: StructureConfig | None = None,
        chunking: ChunkProcessingConfig | None = None,
    ) -> None:
        self._normalizer = normalizer
        self._s = structure or StructureConfig()
        self._c = chunking or ChunkProcessingConfig()
        try:
            self._mentions = MentionResolver(self._s)
            section_number = re.compile(self._s.section_number_pattern)
            appendix = re.compile(self._s.appendix_heading_pattern, re.IGNORECASE)
            self._paragraph_split = re.compile(self._s.paragraph_separator_pattern)
        except re.error as exc:
            raise ConfigurationError(f"Invalid processing.structure pattern: {exc}") from exc
        self._objects = ObjectBuilder(
            normalizer, self._c, self._mentions, section_number=section_number, appendix=appendix
        )

    def analyze(self, document: Document) -> StructureResult:
        """Run the analysis on one document."""
        b = AnalysisState(document)
        root = Section(
            id=b.new_id("s"),
            document_id=document.id,
            order=0,
            level=0,
            title=document.title,
            source=SourceSpan(
                document.id,
                0,
                len(document.text),
                page=1 if document.num_pages else None,
                page_end=document.num_pages,
            ),
        )
        b.result.sections.append(root)
        if document.elements:
            self._walk_elements(b, root)
        else:
            self._walk_plain(b, root)
        self._emit_flow_units(b, has_headings=len(b.result.sections) > 1)
        self._mentions.resolve(b)
        b.result.units.sort(key=lambda unit: unit.order)
        return b.result

    def _walk_plain(self, b: AnalysisState, root: Section) -> None:
        text = b.document.text
        position = 0
        for boundary in [*(m.start() for m in self._paragraph_split.finditer(text)), len(text)]:
            tracked = self._normalizer.normalize(text[position:boundary], position)
            if tracked.text:
                self._objects.add_paragraph(b, root, tracked, None, None, role="body")
            position = boundary

    def _handlers(self) -> dict[ElementType, _Handler]:
        """Builders for element types that need no walk state."""
        add = self._objects

        def paragraph(role: str) -> _Handler:
            return lambda b, s, t, e, i: add.add_paragraph(b, s, t, e, i, role=role)

        return {
            ElementType.TITLE: paragraph("title"),
            ElementType.AUTHORS: paragraph("authors"),
            ElementType.ABSTRACT: paragraph("abstract"),
            ElementType.PARAGRAPH: paragraph("body"),
            ElementType.EQUATION: add.add_equation,
            ElementType.REFERENCE: add.add_reference,
            ElementType.FOOTNOTE: add.add_footnote,
        }

    def _walk_elements(self, b: AnalysisState, root: Section) -> None:
        elements = b.document.elements
        pairs = self._pair_table_captions(elements)
        captions_used = set(pairs.values())
        stack: list[Section] = [root]
        appendix = False
        after_references = False
        handlers = self._handlers()
        for index, element in enumerate(elements):
            tracked = self._normalizer.normalize(element.text, element.start)
            if not tracked.text:
                continue
            if element.type is ElementType.HEADING:
                section = self._objects.add_section(
                    b,
                    stack,
                    tracked,
                    element,
                    index,
                    appendix=appendix,
                    after_references=after_references,
                )
                appendix = section.is_appendix
                after_references = after_references or (section.title or "").lower().strip(
                    " .:"
                ) in {"references", "bibliography"}
                continue
            current = stack[-1]
            if element.type is ElementType.TABLE:
                caption_index = next((c for t, c in pairs.items() if t == index), None)
                caption = elements[caption_index] if caption_index is not None else None
                self._objects.add_table(
                    b,
                    current,
                    tracked,
                    element,
                    index,
                    caption=caption,
                    caption_index=caption_index,
                )
            elif element.type is ElementType.CAPTION:
                if index not in captions_used:
                    self._objects.add_caption(b, current, tracked, element, index)
            else:
                handlers[element.type](b, current, tracked, element, index)

    @staticmethod
    def _pair_table_captions(elements: Sequence[DocumentElement]) -> dict[int, int]:
        """Map table element index → index of its ``Table N`` caption (adjacent, same page)."""
        pairs: dict[int, int] = {}
        used: set[int] = set()
        for index, element in enumerate(elements):
            if element.type is not ElementType.TABLE:
                continue
            for candidate in (index - 1, index + 1):
                if 0 <= candidate < len(elements) and candidate not in used:
                    other = elements[candidate]
                    if (
                        other.type is ElementType.CAPTION
                        and other.page == element.page
                        and TABLE_LABEL.match(other.text)
                    ):
                        pairs[index] = candidate
                        used.add(candidate)
                        break
        return pairs

    def _emit_flow_units(self, b: AnalysisState, *, has_headings: bool) -> None:
        """Turn per-section running text into units (or one document unit if not section-aware)."""
        root_id = b.result.sections[0].id
        if not self._c.section_aware:
            ordered = sorted(b.flows, key=lambda sid: b.flow_order[sid])
            parts = [part for sid in ordered for part in b.flows[sid]]
            if parts:
                b.result.units.append(
                    TextUnit(
                        root_id,
                        ChunkKind.TEXT,
                        root_id,
                        0,
                        TrackedText.concat(parts, FLOW_SEPARATOR),
                    )
                )
            return
        for section_id, parts in b.flows.items():
            kind = (
                ChunkKind.FRONT_MATTER if section_id == root_id and has_headings else ChunkKind.TEXT
            )
            if kind is ChunkKind.FRONT_MATTER and not self._c.include_front_matter:
                continue
            b.result.units.append(
                TextUnit(
                    section_id,
                    kind,
                    section_id,
                    b.flow_order[section_id],
                    TrackedText.concat(parts, FLOW_SEPARATOR),
                )
            )
