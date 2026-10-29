"""Tests for text utilities and language identification."""

from __future__ import annotations

import pytest

from ips_r.data.language import UNDETERMINED, detect_language
from ips_r.data.text import (
    count_tokens,
    encoding_problems,
    normalize_whitespace,
    sentence_spans,
    stable_hash,
    strip_span,
    token_spans,
    tokenize,
)


def _sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


def test_sentence_spans_respect_abbreviations_and_initials() -> None:
    text = (
        "Dr. Smith et al. proposed it, e.g. in 2020. J. R. Tolkien wrote books! Did he?\n\nNew para"
    )
    assert _sentences(text) == [
        "Dr. Smith et al. proposed it, e.g. in 2020.",
        "J. R. Tolkien wrote books!",
        "Did he?",
        "New para",
    ]


def test_sentence_spans_edge_cases() -> None:
    assert sentence_spans("") == []
    assert sentence_spans("   ") == []
    assert _sentences('He said "stop." Then left.') == ['He said "stop."', "Then left."]
    assert _sentences(
        "Version 2.5 is out. Done",
    ) == ["Version 2.5 is out.", "Done"]
    custom = [s for s, _ in sentence_spans("Foo bar. Baz.", abbreviations=["bar."])]
    assert custom == [0]


def test_tokenisation() -> None:
    assert tokenize("Hello, World!") == ["Hello", ",", "World", "!"]
    assert tokenize("Hello World", r"\w+", lowercase=True) == ["hello", "world"]
    assert token_spans("ab cd") == [(0, 2), (3, 5)]
    assert count_tokens("a b, c") == 4


def test_strip_and_normalise() -> None:
    assert strip_span("  ab  ", 0, 6) == (2, 4)
    assert strip_span("    ", 0, 4) is None
    assert normalize_whitespace(" a \n\t b ") == "a b"


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("bad � char", "U+FFFD"),
        ("lone \ud800 surrogate", "surrogate"),
        ("bell \x07 char", "control"),
        ("cafÃ©", "mojibake"),
    ],
)
def test_encoding_problems(text: str, fragment: str) -> None:
    assert any(fragment in problem for problem in encoding_problems(text))


def test_clean_text_has_no_encoding_problems() -> None:
    assert encoding_problems("Ünïcødé text\nwith\ttabs") == []


def test_stable_hash_is_deterministic_and_separates_parts() -> None:
    assert stable_hash(["a", "b"]) == stable_hash(["a", "b"])
    assert stable_hash(["ab"]) != stable_hash(["a", "b"])
    assert len(stable_hash(["x"], digest_size=8)) == 16


def test_detect_language() -> None:
    english = "The model is trained on the data and we evaluate it on a set of the queries. " * 3
    german = (
        "Der Hund und die Katze sind in dem Haus, das nicht auf der Straße ist und es wird. " * 3
    )
    assert detect_language(english) == "en"
    assert detect_language(german) == "de"
    assert detect_language("too short") == UNDETERMINED
    assert detect_language("xyzzy " * 40) == UNDETERMINED
    tie = {"aa": frozenset({"foo"}), "bb": frozenset({"foo"})}
    assert detect_language("foo " * 30, profiles=tie) == UNDETERMINED
