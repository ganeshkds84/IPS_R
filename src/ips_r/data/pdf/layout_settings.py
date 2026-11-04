"""Settings of the rule-based layout analysis and a strict settings-from-config helper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

from ips_r.core.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class LayoutSettings:
    """Thresholds and patterns of the layout analysis (all lengths in PDF points)."""

    line_tolerance: float = 2.0
    column_gap: float = 12.0
    detect_columns: bool = True
    heading_size_ratio: float = 1.15
    title_size_ratio: float = 1.3
    title_search_lines: int = 15
    max_heading_words: int = 15
    max_numbered_heading_words: int = 8
    paragraph_gap_ratio: float = 1.4
    drop_page_numbers: bool = True
    page_number_pattern: str = r"^(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?$"
    caption_pattern: str = r"^(?:fig(?:ure)?|tab(?:le)?)\.?\s*\d+[.:]"
    numbered_heading_pattern: str = r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+(?=[A-Z])"
    reference_marker_pattern: str = r"^(?:\[\d+\]|\d+\.\s)"
    abstract_prefix_pattern: str = r"^abstract\s*[:.\u2014\u2013-]\s*"
    keywords_prefix_pattern: str = r"^(?:keywords|key words|index terms)\s*[:.\u2014\u2013-]"
    reference_headings: tuple[str, ...] = ("references", "bibliography", "works cited")
    detect_equations: bool = True
    equation_symbols: str = (
        "=+<>~^_|\u2212\u00d7\u00f7\u00b1\u2211\u220f\u222b\u221a\u2264\u2265\u2248\u2260\u221e\u2202\u2207\u2208\u2209\u2282\u2286\u222a\u2229\u2192\u2190\u2194\u03b1\u03b2\u03b3\u03b4\u03b5\u03b8\u03bb\u03bc\u03c0\u03c3\u03c6\u03c8\u03c9\u03a3\u03a0\u03a9\u0394"
    )
    equation_symbol_ratio: float = 0.12
    equation_number_pattern: str = r"\(\d+(?:\.\d+)?[a-z]?\)\s*$"
    max_equation_words: int = 20
    detect_footnotes: bool = True
    footnote_zone: float = 0.8
    footnote_size_ratio: float = 0.95
    footnote_marker_pattern: str = r"^(?:\d{1,2}|[*\u2020\u2021\u00a7\u00b6])\s*\S"
    known_headings: tuple[str, ...] = (
        "abstract",
        "introduction",
        "related work",
        "background",
        "method",
        "methods",
        "methodology",
        "experiments",
        "experimental setup",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "limitations",
        "acknowledgments",
        "acknowledgements",
        "appendix",
        "references",
        "bibliography",
    )
    affiliation_markers: tuple[str, ...] = (
        "@",
        "university",
        "institute",
        "department",
        "school",
        "laboratory",
        "college",
        "faculty",
        "inc.",
        "ltd",
        "corporation",
        "research",
        "centre",
        "center",
    )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> LayoutSettings:
        """Build settings from a config mapping, rejecting unknown keys.

        Raises:
            ConfigurationError: On unknown keys or values of the wrong type.
        """
        return settings_from_mapping(cls, mapping, "parser.layout")


def settings_from_mapping[S](cls: type[S], mapping: Mapping[str, Any] | None, where: str) -> S:
    """Instantiate a settings dataclass from a mapping, coercing lists to tuples."""
    values = dict(mapping or {})
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigurationError(f"Unknown {where} settings: {unknown}")
    defaults = cls()
    for key, value in values.items():
        default = getattr(defaults, key)
        if isinstance(default, tuple):
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise ConfigurationError(f"{where}.{key} must be a list, got {value!r}")
            values[key] = tuple(value)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigurationError(f"{where}.{key} must be a boolean, got {value!r}")
        elif isinstance(default, int | float):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigurationError(f"{where}.{key} must be a number, got {value!r}")
            values[key] = type(default)(value)
        elif not isinstance(value, type(default)):
            raise ConfigurationError(
                f"{where}.{key} must be {type(default).__name__}, got {value!r}"
            )
    return replace(defaults, **values)  # type: ignore[type-var]
