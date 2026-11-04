"""Knowledge-object construction: one method per element kind.

Every builder appends its object to the :class:`AnalysisState` result, registers the
text it contributes (running-text flow or a standalone :class:`TextUnit`) and records
the citation / cross-reference mentions it contains.
"""

from __future__ import annotations

import re

from ips_r.config.schema import ChunkProcessingConfig
from ips_r.data.models import DocumentElement
from ips_r.knowledge.models import (
    ChunkKind,
    Equation,
    Figure,
    Footnote,
    Paragraph,
    Reference,
    Section,
    Table,
)
from ips_r.knowledge.normalization import TextNormalizer, TrackedText
from ips_r.knowledge.structure.mentions import MentionResolver
from ips_r.knowledge.structure.patterns import (
    DOI,
    EQUATION_NUMBER,
    FIGURE_LABEL,
    FOOTNOTE_MARKER,
    REFERENCE_MARKER,
    TABLE_LABEL,
    YEAR,
    first_author,
)
from ips_r.knowledge.structure.results import AnalysisState, TextUnit


class ObjectBuilder:
    """Builds sections and content objects for :class:`StructuralAnalyzer`.

    Args:
        normalizer: Text normaliser.
        chunking: Which content becomes separate (atomic) units.
        mentions: Citation / cross-reference detector.
        section_number: Compiled ``processing.structure.section_number_pattern``.
        appendix: Compiled ``processing.structure.appendix_heading_pattern``.
    """

    def __init__(
        self,
        normalizer: TextNormalizer,
        chunking: ChunkProcessingConfig,
        mentions: MentionResolver,
        *,
        section_number: re.Pattern[str],
        appendix: re.Pattern[str],
    ) -> None:
        self._normalizer = normalizer
        self._c = chunking
        self._mentions = mentions
        self._section_number = section_number
        self._appendix = appendix

    def add_section(
        self,
        b: AnalysisState,
        stack: list[Section],
        tracked: TrackedText,
        element: DocumentElement,
        index: int,
        *,
        appendix: bool,
        after_references: bool,
    ) -> Section:
        """Open a section for a heading, closing deeper or equal-level ones on ``stack``."""
        title = tracked.text
        match = self._section_number.match(title)
        number = match.group(1) if match else None
        letter_numbered = number is not None and number[0].isalpha()
        if letter_numbered and not (appendix or after_references):
            number, letter_numbered = None, False  # "A Study of ..." is not numbered "A"
        level = number.count(".") + 1 if number is not None else element.level or 1
        is_appendix = (
            bool(self._appendix.match(title))
            or (appendix and (level > 1 or letter_numbered))
            or (after_references and letter_numbered)
        )
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1]
        section = Section(
            id=b.new_id("s"),
            document_id=b.document.id,
            order=b.next_order(),
            level=level,
            title=title,
            number=number,
            parent_id=parent.id,
            path=(*parent.path, title),
            is_appendix=is_appendix,
            source=b.span(tracked, element, index),
        )
        stack.append(section)
        b.result.sections.append(section)
        return section

    def add_paragraph(
        self,
        b: AnalysisState,
        section: Section,
        tracked: TrackedText,
        element: DocumentElement | None,
        index: int | None,
        *,
        role: str,
    ) -> None:
        """Add a paragraph (title, authors, abstract or body text) to the section flow."""
        paragraph = Paragraph(
            id=b.new_id("p"),
            document_id=b.document.id,
            order=b.next_order(),
            section_id=section.id,
            text=tracked.text,
            role=role,
            source=b.span(tracked, element, index),
        )
        b.result.paragraphs.append(paragraph)
        self.add_to_flow(b, section, tracked, paragraph.order)
        if role not in ("title", "authors"):
            self._mentions.find(
                b,
                tracked,
                element,
                index,
                host_id=paragraph.id,
                section_id=section.id,
                order=paragraph.order,
            )

    @staticmethod
    def add_to_flow(b: AnalysisState, section: Section, tracked: TrackedText, order: int) -> None:
        """Append ``tracked`` to the running text of ``section``."""
        b.flows.setdefault(section.id, []).append(tracked)
        b.flow_order.setdefault(section.id, section.order if section.level else order)

    def add_caption(
        self,
        b: AnalysisState,
        section: Section,
        tracked: TrackedText,
        element: DocumentElement,
        index: int,
    ) -> None:
        """Add an unpaired caption as a figure (or a caption-only ``Table N``)."""
        match = FIGURE_LABEL.match(tracked.text)
        table_match = TABLE_LABEL.match(tracked.text)
        span = b.span(tracked, element, index)
        order = b.next_order()
        host_id: str
        if table_match and not match:
            table = Table(
                id=b.new_id("t"),
                document_id=b.document.id,
                order=order,
                section_id=section.id,
                text=tracked.text,
                caption=tracked.text,
                label=f"Table {table_match.group(2)}",
                number=table_match.group(2),
                source=span,
                caption_source=span,
            )
            b.result.tables.append(table)
            host_id = table.id
            self._caption_unit(
                b, section, tracked, parent_id=table.id, kind=ChunkKind.TABLE, order=order
            )
        else:
            figure = Figure(
                id=b.new_id("f"),
                document_id=b.document.id,
                order=order,
                section_id=section.id,
                caption=tracked.text,
                label=f"Figure {match.group(2)}" if match else None,
                number=match.group(2) if match else None,
                source=span,
            )
            b.result.figures.append(figure)
            host_id = figure.id
            self._caption_unit(
                b, section, tracked, parent_id=figure.id, kind=ChunkKind.FIGURE, order=order
            )
        self._mentions.find(
            b,
            tracked,
            element,
            index,
            host_id=host_id,
            section_id=section.id,
            order=order,
            skip_leading_label=True,
        )

    def _caption_unit(
        self,
        b: AnalysisState,
        section: Section,
        tracked: TrackedText,
        *,
        parent_id: str,
        kind: ChunkKind,
        order: int,
    ) -> None:
        if self._c.caption_aware:
            b.result.units.append(
                TextUnit(parent_id, kind, section.id, order, tracked, atomic=True)
            )
        else:
            self.add_to_flow(b, section, tracked, order)

    def add_table(
        self,
        b: AnalysisState,
        section: Section,
        tracked: TrackedText,
        element: DocumentElement,
        index: int,
        *,
        caption: DocumentElement | None,
        caption_index: int | None,
    ) -> None:
        """Add a table with its paired caption (one atomic unit when table-aware)."""
        caption_tracked = (
            self._normalizer.normalize(caption.text, caption.start) if caption is not None else None
        )
        label_match = TABLE_LABEL.match(caption_tracked.text) if caption_tracked else None
        rows = tuple(
            tuple(self._normalizer.normalize_text(cell) for cell in row) for row in element.rows
        )
        table = Table(
            id=b.new_id("t"),
            document_id=b.document.id,
            order=b.next_order(),
            section_id=section.id,
            text=tracked.text,
            rows=rows,
            caption=caption_tracked.text if caption_tracked and caption_tracked.text else None,
            label=f"Table {label_match.group(2)}" if label_match else None,
            number=label_match.group(2) if label_match else None,
            source=b.span(tracked, element, index),
            caption_source=(
                b.span(caption_tracked, caption, caption_index)
                if caption_tracked and caption_tracked.text
                else None
            ),
        )
        b.result.tables.append(table)
        parts = [p for p in (caption_tracked, tracked) if p is not None and p.text]
        unit_text = TrackedText.concat(parts, "\n")
        if self._c.table_aware:
            row_lines = tuple(line for line in tracked.text.split("\n") if line.strip())
            header = "\n".join(
                ([caption_tracked.text] if caption_tracked and caption_tracked.text else [])
                + list(row_lines[:1])
            )
            b.result.units.append(
                TextUnit(
                    table.id,
                    ChunkKind.TABLE,
                    section.id,
                    table.order,
                    unit_text,
                    atomic=True,
                    table_rows=row_lines[1:],
                    header=header,
                )
            )
        else:
            self.add_to_flow(b, section, unit_text, table.order)
        if caption_tracked and caption_tracked.text and caption is not None:
            self._mentions.find(
                b,
                caption_tracked,
                caption,
                caption_index,
                host_id=table.id,
                section_id=section.id,
                order=table.order,
                skip_leading_label=True,
            )

    def add_equation(
        self,
        b: AnalysisState,
        section: Section,
        tracked: TrackedText,
        element: DocumentElement,
        index: int,
    ) -> None:
        """Add a display equation (its ``(n)`` number is kept) to the section flow."""
        match = EQUATION_NUMBER.search(tracked.text)
        equation = Equation(
            id=b.new_id("e"),
            document_id=b.document.id,
            order=b.next_order(),
            section_id=section.id,
            text=tracked.text,
            number=match.group(1) if match else None,
            source=b.span(tracked, element, index),
        )
        b.result.equations.append(equation)
        self.add_to_flow(b, section, tracked, equation.order)

    def add_reference(
        self,
        b: AnalysisState,
        section: Section,
        tracked: TrackedText,
        element: DocumentElement,
        index: int,
    ) -> None:
        """Add a bibliography entry with its marker, first author, year and DOI."""
        text = tracked.text
        marker_match = REFERENCE_MARKER.match(text)
        marker = (marker_match.group(1) or marker_match.group(2)) if marker_match else None
        body = text[marker_match.end() :] if marker_match else text
        surname = first_author(body)
        year = YEAR.search(body)
        doi = DOI.search(body)
        reference = Reference(
            id=b.new_id("r"),
            document_id=b.document.id,
            order=b.next_order(),
            section_id=section.id,
            text=text,
            marker=marker,
            first_author=surname,
            year=int(year.group(1)) if year else None,
            doi=doi.group(1).rstrip(".)]") if doi else None,
            source=b.span(tracked, element, index),
        )
        b.result.references.append(reference)
        if self._c.include_references:
            b.result.units.append(
                TextUnit(
                    reference.id,
                    ChunkKind.REFERENCE,
                    section.id,
                    reference.order,
                    tracked,
                    atomic=True,
                )
            )

    def add_footnote(
        self,
        b: AnalysisState,
        section: Section,
        tracked: TrackedText,
        element: DocumentElement,
        index: int,
    ) -> None:
        """Add a footnote (an atomic unit when ``include_footnotes``)."""
        match = FOOTNOTE_MARKER.match(tracked.text)
        footnote = Footnote(
            id=b.new_id("n"),
            document_id=b.document.id,
            order=b.next_order(),
            section_id=section.id,
            text=tracked.text,
            marker=match.group(1) if match else None,
            source=b.span(tracked, element, index),
        )
        b.result.footnotes.append(footnote)
        if self._c.include_footnotes:
            b.result.units.append(
                TextUnit(
                    footnote.id,
                    ChunkKind.FOOTNOTE,
                    section.id,
                    footnote.order,
                    tracked,
                    atomic=True,
                )
            )
        self._mentions.find(
            b,
            tracked,
            element,
            index,
            host_id=footnote.id,
            section_id=section.id,
            order=footnote.order,
        )
