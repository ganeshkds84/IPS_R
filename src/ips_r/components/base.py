"""Common lifecycle, configuration, logging and error handling for components.

Every interface (retriever, reranker, ...) follows the *template method*
pattern: a public, non-abstract method validates inputs, logs, times and
guards the call, then delegates to a protected abstract hook (``_retrieve``,
``_rerank``, ...) that implementations override. Implementations therefore
never need to repeat boilerplate and always surface failures as
:class:`~ips_r.core.exceptions.ComponentError`.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType, TracebackType
from typing import Any, ClassVar, Self

from ips_r.core.exceptions import ComponentError, ConfigurationError, IPSRError
from ips_r.core.log import get_logger

_MISSING: Any = object()


class AbstractComponent(abc.ABC):  # noqa: B024 - lifecycle hooks are optional by design
    """Base class providing lifecycle, configuration access and guarded execution.

    Lifecycle: ``setup()`` acquires resources (models, indices, ...), ``teardown()``
    releases them. Both are idempotent, and components may be used as context
    managers. Operations call ``setup()`` lazily if it has not run yet.

    Args:
        config: Implementation-specific options (the ``params`` node of the config).
        name: Instance name used in logs and errors. Defaults to the class name.
    """

    component_type: ClassVar[str] = "component"

    def __init__(self, config: Mapping[str, Any] | None = None, *, name: str | None = None) -> None:
        resolved_name = name or type(self).__name__
        if config is not None and not isinstance(config, Mapping):
            raise ComponentError(
                resolved_name, f"config must be a mapping, got {type(config).__name__}"
            )
        self._name: str = resolved_name
        self._config: dict[str, Any] = dict(config or {})
        self._ready: bool = False
        self._logger = get_logger(type(self).__module__).bind(
            component=self._name, component_type=self.component_type
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Instance name."""
        return self._name

    @property
    def config(self) -> Mapping[str, Any]:
        """Read-only view of the component configuration."""
        return MappingProxyType(self._config)

    @property
    def is_ready(self) -> bool:
        """Whether :meth:`setup` has completed and :meth:`teardown` has not run since."""
        return self._ready

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #
    def param(self, key: str, default: Any = _MISSING) -> Any:
        """Return config value ``key``.

        Raises:
            ConfigurationError: If ``key`` is missing and no default is given.
        """
        if key in self._config:
            return self._config[key]
        if default is _MISSING:
            raise ConfigurationError(
                f"{self.component_type} '{self._name}' requires config parameter '{key}'"
            )
        return default

    def _param_error(self, key: str, expected: str, value: Any) -> ConfigurationError:
        """Build the error raised when parameter ``key`` has an invalid value."""
        return ConfigurationError(
            f"{self.component_type} '{self._name}' parameter '{key}' must be {expected}, "
            f"got {value!r}"
        )

    def int_param(self, key: str, default: Any = _MISSING, *, minimum: int | None = None) -> int:
        """Return parameter ``key`` as an integer, optionally bounded below.

        Raises:
            ConfigurationError: If the value is missing, not an integer or below ``minimum``.
        """
        value = self.param(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._param_error(key, "an integer", value)
        if minimum is not None and value < minimum:
            raise self._param_error(key, f">= {minimum}", value)
        return value

    def optional_int_param(self, key: str, *, minimum: int | None = None) -> int | None:
        """Return parameter ``key`` as an integer, or ``None`` when unset or null.

        Raises:
            ConfigurationError: If the value is set but invalid.
        """
        if self.param(key, None) is None:
            return None
        return self.int_param(key, minimum=minimum)

    def float_param(
        self,
        key: str,
        default: Any = _MISSING,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        """Return parameter ``key`` as a float within ``[minimum, maximum]``.

        Raises:
            ConfigurationError: If the value is missing, not numeric or out of range.
        """
        value = self.param(key, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._param_error(key, "a number", value)
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            raise self._param_error(key, f"in [{minimum}, {maximum}]", value)
        return float(value)

    def bool_param(self, key: str, default: Any = _MISSING) -> bool:
        """Return parameter ``key`` as a boolean.

        Raises:
            ConfigurationError: If the value is missing or not a boolean.
        """
        value = self.param(key, default)
        if not isinstance(value, bool):
            raise self._param_error(key, "a boolean", value)
        return value

    def str_param(self, key: str, default: Any = _MISSING) -> str:
        """Return parameter ``key`` as a non-empty string.

        Raises:
            ConfigurationError: If the value is missing, not a string or empty.
        """
        value = self.param(key, default)
        if not isinstance(value, str) or not value:
            raise self._param_error(key, "a non-empty string", value)
        return value

    def optional_str_param(self, key: str) -> str | None:
        """Return parameter ``key`` as a string, or ``None`` when unset or null.

        Raises:
            ConfigurationError: If the value is set but not a non-empty string.
        """
        if self.param(key, None) is None:
            return None
        return self.str_param(key)

    def str_list_param(self, key: str, default: Any = _MISSING) -> tuple[str, ...]:
        """Return parameter ``key`` as a tuple of strings.

        Raises:
            ConfigurationError: If the value is missing or not a sequence of strings.
        """
        value = self.param(key, default)
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise self._param_error(key, "a list of strings", value)
        if not all(isinstance(item, str) for item in value):
            raise self._param_error(key, "a list of strings", value)
        return tuple(value)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        """Acquire resources. Idempotent.

        Raises:
            ComponentError: If :meth:`_setup` fails.
        """
        if self._ready:
            return
        self._logger.debug("component.setup.start")
        self._guarded("setup", self._setup)
        self._ready = True
        self._logger.info("component.ready")

    def teardown(self) -> None:
        """Release resources. Idempotent; the component is marked not-ready even on failure.

        Raises:
            ComponentError: If :meth:`_teardown` fails.
        """
        if not self._ready:
            return
        try:
            self._guarded("teardown", self._teardown)
        finally:
            self._ready = False
        self._logger.debug("component.teardown.done")

    def _setup(self) -> None:  # noqa: B027 - intentionally optional hook
        """Hook for resource acquisition. Override in subclasses; default is a no-op."""

    def _teardown(self) -> None:  # noqa: B027 - intentionally optional hook
        """Hook for resource release. Override in subclasses; default is a no-op."""

    def __enter__(self) -> Self:
        self.setup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.teardown()

    # ------------------------------------------------------------------ #
    # Guarded execution
    # ------------------------------------------------------------------ #
    def _guarded[R](self, operation: str, func: Callable[[], R]) -> R:
        """Run ``func``, logging failures and wrapping unexpected errors.

        :class:`IPSRError` subclasses propagate unchanged; any other exception is
        wrapped in :class:`ComponentError` with the original chained as ``__cause__``.
        """
        try:
            return func()
        except IPSRError as exc:
            self._logger.error("component.operation.failed", operation=operation, error=str(exc))
            raise
        except Exception as exc:
            self._logger.error(
                "component.operation.failed",
                operation=operation,
                error=repr(exc),
                exc_info=True,
            )
            raise ComponentError(self._name, f"{operation} failed: {exc}") from exc

    def _invoke[R](self, operation: str, func: Callable[[], R], **log_fields: Any) -> R:
        """Run an interface operation with lazy setup, timing and error handling."""
        if not self._ready:
            self.setup()
        self._logger.debug("component.operation.start", operation=operation, **log_fields)
        started = time.perf_counter()
        result = self._guarded(operation, func)
        self._logger.debug(
            "component.operation.end",
            operation=operation,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return result

    def _guarded_iter[R](
        self, operation: str, factory: Callable[[], Iterable[R]], *, limit: int | None = None
    ) -> Iterator[R]:
        """Lazily iterate ``factory()`` with lazy setup and error wrapping on every step.

        Args:
            operation: Operation name used in logs and errors.
            factory: Returns the iterable to consume (e.g. an implementation generator).
            limit: Stop after this many items (``None`` = unlimited).

        Raises:
            ComponentError: If creating or advancing the iterator fails unexpectedly.
        """
        if not self._ready:
            self.setup()
        iterator = self._guarded(operation, lambda: iter(factory()))
        count = 0
        try:
            while limit is None or count < limit:
                item = self._guarded(operation, lambda: next(iterator, _MISSING))
                if item is _MISSING:
                    break
                count += 1
                yield item
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            self._logger.debug("component.iteration.end", operation=operation, count=count)

    def _require_positive(self, field: str, value: int) -> None:
        """Validate that an integer argument is strictly positive.

        Raises:
            ComponentError: If ``value`` is not a positive integer.
        """
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ComponentError(self._name, f"{field} must be a positive integer, got {value!r}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r}, ready={self._ready})"
