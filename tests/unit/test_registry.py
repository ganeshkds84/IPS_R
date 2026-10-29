"""Tests for the component registry."""

from __future__ import annotations

import pytest

from ips_r.core.exceptions import RegistryError
from ips_r.core.registry import Registry
from ips_r.retrieval import AbstractRetriever
from tests.fakes import FakeReranker, FakeRetriever


@pytest.fixture
def registry() -> Registry[AbstractRetriever]:
    return Registry("retriever", AbstractRetriever)


def test_register_get_create(registry: Registry[AbstractRetriever]) -> None:
    registry.register("fake")(FakeRetriever)
    assert "fake" in registry
    assert registry.get("fake") is FakeRetriever
    assert len(registry) == 1
    assert list(registry) == ["fake"]
    instance = registry.create("fake", {"a": 1}, name="x")
    assert isinstance(instance, FakeRetriever)
    assert instance.config["a"] == 1
    assert "fake" in repr(registry)


def test_duplicate_requires_overwrite(registry: Registry[AbstractRetriever]) -> None:
    registry.add("fake", FakeRetriever)
    with pytest.raises(RegistryError, match="already registered"):
        registry.add("fake", FakeRetriever)
    registry.add("fake", FakeRetriever, overwrite=True)


def test_rejects_invalid_classes(registry: Registry[AbstractRetriever]) -> None:
    with pytest.raises(RegistryError, match="abstract"):
        registry.add("abstract", AbstractRetriever)  # type: ignore[type-abstract]
    with pytest.raises(RegistryError, match="not a subclass"):
        registry.add("wrong", FakeReranker)  # type: ignore[arg-type]
    with pytest.raises(RegistryError, match="non-empty"):
        registry.add(" ", FakeRetriever)


def test_unknown_and_remove(registry: Registry[AbstractRetriever]) -> None:
    with pytest.raises(RegistryError, match="Unknown retriever"):
        registry.get("missing")
    with pytest.raises(RegistryError):
        registry.remove("missing")
    registry.add("fake", FakeRetriever)
    registry.remove("fake")
    assert "fake" not in registry


def test_create_wraps_constructor_errors(registry: Registry[AbstractRetriever]) -> None:
    registry.add("fake", FakeRetriever)
    with pytest.raises(RegistryError, match="Failed to construct"):
        registry.create("fake", "not-a-mapping")
