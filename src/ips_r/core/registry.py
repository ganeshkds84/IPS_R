"""Name-based component registry.

Implementations register themselves under a short name so that Hydra configs
can select them (``retriever=my_retriever``) without import-path coupling::

    @RETRIEVERS.register("my_retriever")
    class MyRetriever(AbstractRetriever): ...
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from typing import Any

from ips_r.core.exceptions import RegistryError
from ips_r.core.log import get_logger

logger = get_logger(__name__)


class Registry[T]:
    """A typed mapping from names to concrete subclasses of a base class.

    Args:
        kind: Human-readable component kind, used in messages (e.g. ``"retriever"``).
        base: Base class every registered class must subclass.
    """

    def __init__(self, kind: str, base: type) -> None:
        self._kind = kind
        self._base = base
        self._entries: dict[str, type[T]] = {}

    @property
    def kind(self) -> str:
        """Component kind handled by this registry."""
        return self._kind

    def register(self, name: str, *, overwrite: bool = False) -> Callable[[type[T]], type[T]]:
        """Return a class decorator registering the decorated class under ``name``."""

        def decorator(cls: type[T]) -> type[T]:
            self.add(name, cls, overwrite=overwrite)
            return cls

        return decorator

    def add(self, name: str, cls: type[T], *, overwrite: bool = False) -> None:
        """Register ``cls`` under ``name``.

        Raises:
            RegistryError: If the name is empty or taken, or ``cls`` is abstract or of
                the wrong type.
        """
        if not name or not name.strip():
            raise RegistryError(f"{self._kind} name must be a non-empty string")
        if not (inspect.isclass(cls) and issubclass(cls, self._base)):
            raise RegistryError(
                f"Cannot register {cls!r} as {self._kind}: not a subclass of {self._base.__name__}"
            )
        if inspect.isabstract(cls):
            raise RegistryError(
                f"Cannot register abstract class {cls.__name__} as {self._kind} '{name}'"
            )
        if name in self._entries and not overwrite:
            raise RegistryError(
                f"{self._kind} '{name}' is already registered to {self._entries[name].__name__}"
            )
        self._entries[name] = cls
        logger.debug("registry.added", kind=self._kind, name=name, cls=cls.__qualname__)

    def remove(self, name: str) -> None:
        """Unregister ``name``.

        Raises:
            RegistryError: If ``name`` is not registered.
        """
        if name not in self._entries:
            raise RegistryError(f"{self._kind} '{name}' is not registered")
        del self._entries[name]

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name``.

        Raises:
            RegistryError: If ``name`` is not registered.
        """
        try:
            return self._entries[name]
        except KeyError:
            available = ", ".join(self.names()) or "<none>"
            raise RegistryError(f"Unknown {self._kind} '{name}'. Available: {available}") from None

    def create(self, key: str, /, *args: Any, **kwargs: Any) -> T:
        """Instantiate the class registered under ``key`` with the given arguments.

        ``key`` is positional-only so that ``name=...`` can be forwarded to the constructor.

        Raises:
            RegistryError: If ``key`` is unknown or construction fails.
        """
        cls = self.get(key)
        try:
            instance = cls(*args, **kwargs)
        except Exception as exc:
            raise RegistryError(f"Failed to construct {self._kind} '{key}': {exc}") from exc
        logger.debug("registry.created", kind=self._kind, name=key)
        return instance

    def names(self) -> tuple[str, ...]:
        """Return registered names in sorted order."""
        return tuple(sorted(self._entries))

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry(kind={self._kind!r}, entries={list(self.names())})"
