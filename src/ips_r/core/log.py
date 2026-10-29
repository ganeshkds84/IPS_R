"""Structured logging built on :mod:`structlog` and the standard library.

Records emitted through structlog *and* through stdlib :mod:`logging` (including
third-party libraries) share a single processor chain: the console gets either
human-friendly or JSON output, and an optional log file always receives JSON
lines suitable for later analysis.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Final

import structlog
from structlog.typing import Processor

from ips_r.core.exceptions import ConfigurationError

VALID_LEVELS: Final[frozenset[str]] = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})
VALID_FORMATS: Final[frozenset[str]] = frozenset({"console", "json"})

_HANDLER_MARKER: Final = "_ips_r_handler"
_NOISY_LOGGERS: Final = ("urllib3", "filelock", "fsspec", "matplotlib", "PIL", "asyncio")


def _shared_processors() -> list[Processor]:
    """Return processors applied to both structlog and foreign (stdlib) records."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]


def _supports_color() -> bool:
    """Return whether ANSI colours should be used on stderr."""
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _formatter(renderer: Processor, *, structured_tracebacks: bool) -> logging.Formatter:
    """Build a stdlib formatter that renders records through structlog processors."""
    processors: list[Processor] = [structlog.stdlib.ProcessorFormatter.remove_processors_meta]
    if structured_tracebacks:
        processors.append(structlog.processors.dict_tracebacks)
    processors.append(renderer)
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=processors,
    )


def _remove_handlers(*, only_files: bool = False) -> None:
    """Detach handlers previously installed by :func:`configure_logging`.

    Handlers installed by other parties (e.g. pytest's capture handler) are left alone.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if not getattr(handler, _HANDLER_MARKER, False):
            continue
        if only_files and not isinstance(handler, logging.FileHandler):
            continue
        root.removeHandler(handler)
        handler.close()


def configure_logging(
    level: str = "INFO",
    fmt: str = "console",
    log_file: str | Path | None = None,
    *,
    capture_warnings: bool = True,
) -> None:
    """Configure process-wide structured logging.

    Safe to call repeatedly; each call replaces the handlers installed by the
    previous one.

    Args:
        level: Minimum level name (``DEBUG``, ``INFO``, ...). Case-insensitive.
        fmt: Console format, ``"console"`` (pretty) or ``"json"``.
        log_file: Optional path of a JSON-lines log file. Parent directories are created.
        capture_warnings: Route :mod:`warnings` through the logging system.

    Raises:
        ConfigurationError: If ``level`` or ``fmt`` is invalid or the log file cannot be opened.
    """
    level_name = str(level).upper()
    if level_name not in VALID_LEVELS:
        raise ConfigurationError(
            f"Invalid log level {level!r}; expected one of {sorted(VALID_LEVELS)}"
        )
    if fmt not in VALID_FORMATS:
        raise ConfigurationError(
            f"Invalid log format {fmt!r}; expected one of {sorted(VALID_FORMATS)}"
        )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_shared_processors(),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=False,
    )

    handlers: list[logging.Handler] = []
    console_renderer: Processor = (
        structlog.dev.ConsoleRenderer(colors=_supports_color())
        if fmt == "console"
        else structlog.processors.JSONRenderer()
    )
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(_formatter(console_renderer, structured_tracebacks=fmt == "json"))
    handlers.append(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(f"Cannot open log file {path}: {exc}") from exc
        file_handler.setFormatter(
            _formatter(structlog.processors.JSONRenderer(), structured_tracebacks=True)
        )
        handlers.append(file_handler)

    _remove_handlers()
    root = logging.getLogger()
    for handler in handlers:
        setattr(handler, _HANDLER_MARKER, True)
        root.addHandler(handler)
    root.setLevel(level_name)

    quiet_level = max(logging.WARNING, logging.getLevelNamesMapping()[level_name])
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(quiet_level)
    logging.captureWarnings(capture_warnings)

    get_logger(__name__).debug(
        "logging.configured",
        level=level_name,
        format=fmt,
        log_file=str(log_file) if log_file is not None else None,
    )


def remove_file_handlers() -> None:
    """Close and detach log-file handlers installed by :func:`configure_logging`."""
    _remove_handlers(only_files=True)


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a structured logger.

    Args:
        name: Logger name, conventionally ``__name__``.
        **initial_values: Key/value pairs bound to every event from this logger.

    Returns:
        A structlog bound logger. It is lazily configured, so module-level loggers
        created before :func:`configure_logging` pick up the final configuration.
    """
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name, **initial_values)
    return logger


def bind_context(**values: Any) -> None:
    """Bind key/value pairs to every subsequent log event in the current context."""
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    """Remove all context-local values bound via :func:`bind_context`."""
    structlog.contextvars.clear_contextvars()
