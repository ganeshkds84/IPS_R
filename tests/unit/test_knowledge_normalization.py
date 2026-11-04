"""Tests for offset-preserving text normalisation (Part A)."""

from __future__ import annotations

import re

import pytest

from ips_r.config.schema import NormalizationConfig
from ips_r.core.exceptions import ConfigurationError
from ips_r.knowledge.normalization import TextNormalizer, TrackedText

LIGATURE_FI = chr(0xFB01)
LEFT_QUOTE, RIGHT_QUOTE = chr(0x201C), chr(0x201D)
EN_DASH, MINUS = chr(0x2013), chr(0x2212)
NBSP, ZWSP, SOFT_HYPHEN = chr(0xA0), chr(0x200B), chr(0xAD)
E_ACUTE, COMBINING_ACUTE = chr(0xE9), chr(0x301)


def _norm(text: str, **settings: object) -> str:
    return TextNormalizer(NormalizationConfig(**settings)).normalize_text(text)  # type: ignore[arg-type]


def test_default_pipeline_cleans_typical_pdf_text() -> None:
    raw = (
        f"  The {LIGATURE_FI}rst {LEFT_QUOTE}result{RIGHT_QUOTE}{ZWSP} is 3{EN_DASH}5 "
        f"{MINUS}1,\x07 repre-\nsentation{NBSP}and  co{SOFT_HYPHEN}operation.\r\n\n\n\nNext  "
    )
    assert _norm(raw) == 'The first "result" is 3-5 -1, representation and cooperation.\n\nNext'


@pytest.mark.parametrize(
    ("settings", "text", "expected"),
    [
        ({"unicode_form": "NFKC"}, f"x{chr(0xB2)} {chr(0xFF21)}", "x2 A"),
        ({"unicode_form": "NFD"}, E_ACUTE, f"e{COMBINING_ACUTE}"),
        ({"unicode_form": "NFC"}, f"e{COMBINING_ACUTE}", E_ACUTE),
        ({"unicode_form": "none", "expand_ligatures": False}, LIGATURE_FI, LIGATURE_FI),
        ({"ligatures": {chr(0xE6): "ae"}}, f"{chr(0xE6)}on", "aeon"),
        ({"normalize_quotes": False}, f"{LEFT_QUOTE}q{RIGHT_QUOTE}", f"{LEFT_QUOTE}q{RIGHT_QUOTE}"),
        ({"normalize_hyphens": False}, f"a{EN_DASH}b", f"a{EN_DASH}b"),
        ({"dash_replacement": " - "}, f"a{EN_DASH}b", "a - b"),
        ({"dehyphenate": False}, "repre-\nsentation", "repre-\nsentation"),
        ({"lowercase": True}, "Hello WORLD", "hello world"),
        ({"lowercase": True}, f"{chr(0x130)}STANBUL", f"i{chr(0x307)}stanbul"),
        ({"preserve_punctuation": False}, "a, b; (c)!", "a b c"),
        ({"preserve_newlines": False}, "a\n\nb\nc", "a b c"),
        ({"normalize_whitespace": False}, " a  b ", " a  b "),
        ({"remove_control_chars": False}, "a\x07b", "a\x07b"),
        ({"remove_zero_width": False}, f"a{ZWSP}b", f"a{ZWSP}b"),
    ],
)
def test_each_step_is_configurable(settings: dict[str, object], text: str, expected: str) -> None:
    assert _norm(text, **settings) == expected


def test_offsets_map_back_to_the_original_text() -> None:
    raw = f"Intro: the {LIGATURE_FI}eld of  repre-\nsentation"
    tracked = TextNormalizer().normalize(raw, offset=100)
    assert tracked.text == "Intro: the field of representation"
    for word in ("field", "representation", "Intro"):
        start = tracked.text.index(word)
        source_start, source_end = tracked.source_span(start, start + len(word))
        original = raw[source_start - 100 : source_end - 100]
        assert TextNormalizer().normalize_text(original) == word


def test_offsets_survive_length_changing_lowercase() -> None:
    raw = f"A {chr(0x130)}B"
    tracked = TextNormalizer(NormalizationConfig(lowercase=True)).normalize(raw)
    assert tracked.text == f"a i{chr(0x307)}b"
    assert tracked.source_span(len(tracked.text) - 1, len(tracked.text)) == (3, 4)


def test_tracked_text_primitives() -> None:
    tracked = TrackedText.original("abcdef", offset=10)
    assert len(tracked) == 6
    assert tracked.source_span(1, 3) == (11, 13)
    sliced = tracked.slice(2, 4)
    assert sliced.text == "cd"
    assert sliced.source_span(0, 2) == (12, 14)
    expanded = tracked.sub(re.compile("c"), "XYZ")
    assert expanded.text == "abXYZdef"
    assert expanded.source_span(2, 5) == (12, 13)
    inserted = tracked.sub(re.compile(r"(?=d)"), "-")
    assert inserted.text == "abc-def"
    assert tracked.sub(re.compile("zz"), "q") is tracked
    assert tracked.sub(re.compile("c"), "c") is tracked
    at_end = tracked.sub(re.compile("$"), "!")
    assert at_end.text == "abcdef!"
    assert at_end.source_span(6, 7) == (16, 16)
    joined = TrackedText.concat([TrackedText.original("ab"), TrackedText.original("cd", 5)], " ")
    assert joined.text == "ab cd"
    assert joined.source_span(3, 5) == (5, 7)
    leading_empty = TrackedText.concat(
        [TrackedText.original(""), TrackedText.original("x", 3)], "|"
    )
    assert leading_empty.text == "|x"
    empty_end = TrackedText.original("").sub(re.compile("$"), "!")
    assert empty_end.source_span(0, 1) == (0, 0)
    with pytest.raises(ValueError):
        tracked.source_span(3, 3)
    with pytest.raises(ValueError):
        TrackedText("ab", tracked.slice(0, 1)._starts, tracked.slice(0, 1)._ends)


def test_fingerprint_tracks_settings() -> None:
    assert TextNormalizer().fingerprint == TextNormalizer(NormalizationConfig()).fingerprint
    assert (
        TextNormalizer().fingerprint
        != TextNormalizer(NormalizationConfig(lowercase=True)).fingerprint
    )
    assert TextNormalizer().settings.unicode_form == "NFC"


def test_invalid_settings() -> None:
    with pytest.raises(ConfigurationError, match="unicode_form"):
        TextNormalizer(NormalizationConfig(unicode_form="NFX"))
    with pytest.raises(ConfigurationError, match="ligatures"):
        TextNormalizer(NormalizationConfig(ligatures={"": "x"}))
