"""Structural knowledge objects (sections, paragraphs, figures, tables, ...).

The classes cover sections, paragraphs, figures, tables, equations, references,
citations, cross references and footnotes. Every object carries a deterministic ``id``, ``document_id``, ``order`` (reading
order), the enclosing ``section_id`` and a :class:`~ips_r.knowledge.primitives.SourceSpan`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ips_r.knowledge.primitives import SourceSpan, check, nonempty


@dataclass(frozen=True, slots=True)
class Section:
    """A node of the section hierarchy. Level 0 is the document root."""

    id: str
    document_id: str
    order: int
    level: int
    title: str | None = None
    number: str | None = None
    parent_id: str | None = None
    path: tuple[str, ...] = ()
    is_appendix: bool = False
    source: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty("Section", "id", self.id)
        check(self.level >= 0 and self.order >= 0, "Section.level and order must be >= 0")
        check((self.level == 0) == (self.parent_id is None), "only the root section has no parent")
        object.__setattr__(self, "path", tuple(self.path))


@dataclass(frozen=True, slots=True)
class Paragraph:
    """A block of running text (``role``: body, abstract, title, authors, front_matter)."""

    id: str
    document_id: str
    order: int
    section_id: str
    text: str
    role: str = "body"
    source: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty("Paragraph", "id", self.id)
        nonempty("Paragraph", "text", self.text)


@dataclass(frozen=True, slots=True)
class Figure:
    """A figure, represented by its caption."""

    id: str
    document_id: str
    order: int
    section_id: str
    caption: str
    label: str | None = None
    number: str | None = None
    source: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty("Figure", "id", self.id)
        nonempty("Figure", "caption", self.caption)


@dataclass(frozen=True, slots=True)
class Table:
    """A table with optional caption; ``rows`` is empty when only the caption was found."""

    id: str
    document_id: str
    order: int
    section_id: str
    text: str
    rows: tuple[tuple[str, ...], ...] = ()
    caption: str | None = None
    label: str | None = None
    number: str | None = None
    source: SourceSpan | None = None
    caption_source: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty("Table", "id", self.id)
        nonempty("Table", "text", self.text)
        object.__setattr__(self, "rows", tuple(tuple(r) for r in self.rows))


@dataclass(frozen=True, slots=True)
class Equation:
    """A display equation with its optional number (``(3)`` → ``"3"``)."""

    id: str
    document_id: str
    order: int
    section_id: str
    text: str
    number: str | None = None
    source: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty("Equation", "id", self.id)
        nonempty("Equation", "text", self.text)


@dataclass(frozen=True, slots=True)
class Reference:
    """A bibliography entry with lightly parsed fields."""

    id: str
    document_id: str
    order: int
    section_id: str
    text: str
    marker: str | None = None
    first_author: str | None = None
    year: int | None = None
    doi: str | None = None
    source: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty("Reference", "id", self.id)
        nonempty("Reference", "text", self.text)


@dataclass(frozen=True, slots=True)
class Citation:
    """An in-text citation (``[1, 3]`` or ``(Smith et al., 2020)``) and its resolved targets."""

    id: str
    document_id: str
    order: int
    section_id: str
    host_id: str
    marker: str
    style: str
    target_ids: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    source: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty("Citation", "id", self.id)
        nonempty("Citation", "marker", self.marker)
        check(self.style in ("numeric", "author_year"), f"unknown citation style {self.style!r}")
        object.__setattr__(self, "target_ids", tuple(self.target_ids))
        object.__setattr__(self, "keys", tuple(self.keys))

    @property
    def resolved(self) -> bool:
        """Whether at least one target reference was found."""
        return bool(self.target_ids)


@dataclass(frozen=True, slots=True)
class CrossReference:
    """An in-text reference to a figure, table, section, equation or appendix."""

    id: str
    document_id: str
    order: int
    section_id: str
    host_id: str
    label: str
    target_kind: str
    target_number: str
    target_id: str | None = None
    source: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty("CrossReference", "id", self.id)
        nonempty("CrossReference", "label", self.label)


@dataclass(frozen=True, slots=True)
class Footnote:
    """A footnote with its marker (``1``, ``*``, ``†``)."""

    id: str
    document_id: str
    order: int
    section_id: str
    text: str
    marker: str | None = None
    source: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty("Footnote", "id", self.id)
        nonempty("Footnote", "text", self.text)
