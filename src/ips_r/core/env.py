"""Environment-variable and ``.env`` handling.

All project-specific variables use the ``IPS_R_`` prefix. Variables are read
from the process environment; :func:`load_env_file` can populate it from a
``.env`` file first (existing variables win unless ``override=True``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from ips_r.core.exceptions import EnvironmentConfigError
from ips_r.core.log import get_logger

logger = get_logger(__name__)

ENV_PREFIX: Final = "IPS_R_"
_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_FALSY: Final = frozenset({"0", "false", "no", "off"})


def load_env_file(
    path: str | Path | None = None,
    *,
    override: bool = False,
    required: bool = False,
) -> Path | None:
    """Load variables from a ``.env`` file into :data:`os.environ`.

    Args:
        path: File to load. Defaults to ``.env`` in the current working directory.
        override: Whether values in the file replace variables already set.
        required: Raise if the file does not exist instead of silently skipping it.

    Returns:
        The path that was loaded, or ``None`` if no file was found.

    Raises:
        EnvironmentConfigError: If the file is required but missing, or unreadable.
    """
    candidate = Path(path) if path is not None else Path.cwd() / ".env"
    if not candidate.is_file():
        if required:
            raise EnvironmentConfigError(f"Environment file not found: {candidate}")
        logger.debug("env_file.absent", path=str(candidate))
        return None
    try:
        load_dotenv(candidate, override=override)
    except (OSError, UnicodeDecodeError) as exc:
        raise EnvironmentConfigError(f"Cannot read environment file {candidate}: {exc}") from exc
    logger.debug("env_file.loaded", path=str(candidate), override=override)
    return candidate


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Read an environment variable.

    Empty strings are treated as unset.

    Args:
        name: Variable name.
        default: Value returned when the variable is unset.
        required: Raise instead of returning ``default`` when unset.

    Returns:
        The variable's value, or ``default``.

    Raises:
        EnvironmentConfigError: If ``required`` and the variable is unset.
    """
    value = os.environ.get(name)
    if value is None or value == "":
        if required:
            raise EnvironmentConfigError(f"Required environment variable {name} is not set")
        return default
    return value


def get_env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (``1/0``, ``true/false``, ``yes/no``, ``on/off``).

    Raises:
        EnvironmentConfigError: If the value is not a recognised boolean literal.
    """
    raw = get_env(name)
    if raw is None:
        return default
    normalised = raw.strip().lower()
    if normalised in _TRUTHY:
        return True
    if normalised in _FALSY:
        return False
    raise EnvironmentConfigError(f"Environment variable {name}={raw!r} is not a boolean")


def get_env_int(name: str, default: int | None = None) -> int | None:
    """Read an integer environment variable.

    Raises:
        EnvironmentConfigError: If the value cannot be parsed as an integer.
    """
    raw = get_env(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise EnvironmentConfigError(
            f"Environment variable {name}={raw!r} is not an integer"
        ) from exc


def get_env_path(name: str) -> Path | None:
    """Read a filesystem path from an environment variable, expanding ``~``."""
    raw = get_env(name)
    return Path(raw).expanduser() if raw is not None else None


@dataclass(frozen=True, slots=True)
class Settings:
    """Snapshot of the ``IPS_R_*`` environment variables.

    These values seed defaults *before* Hydra configuration is composed (e.g. the
    log level used by the CLI itself). Hydra configs read the same variables via
    ``${oc.env:...}`` so both views stay consistent.
    """

    env: str = "development"
    log_level: str = "INFO"
    log_format: str = "console"
    seed: int | None = None
    root: Path | None = None
    config_dir: Path | None = None
    data_dir: Path | None = None
    cache_dir: Path | None = None
    models_dir: Path | None = None
    output_dir: Path | None = None


def load_settings() -> Settings:
    """Build :class:`Settings` from the current process environment.

    Raises:
        EnvironmentConfigError: If a variable is present but malformed.
    """
    settings = Settings(
        env=get_env(f"{ENV_PREFIX}ENV") or "development",
        log_level=(get_env(f"{ENV_PREFIX}LOG_LEVEL") or "INFO").upper(),
        log_format=get_env(f"{ENV_PREFIX}LOG_FORMAT") or "console",
        seed=get_env_int(f"{ENV_PREFIX}SEED"),
        root=get_env_path(f"{ENV_PREFIX}ROOT"),
        config_dir=get_env_path(f"{ENV_PREFIX}CONFIG_DIR"),
        data_dir=get_env_path(f"{ENV_PREFIX}DATA_DIR"),
        cache_dir=get_env_path(f"{ENV_PREFIX}CACHE_DIR"),
        models_dir=get_env_path(f"{ENV_PREFIX}MODELS_DIR"),
        output_dir=get_env_path(f"{ENV_PREFIX}OUTPUT_DIR"),
    )
    logger.debug("settings.loaded", env=settings.env, log_level=settings.log_level)
    return settings
