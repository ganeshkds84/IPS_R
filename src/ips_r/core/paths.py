"""Project-root discovery and canonical directory layout."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from ips_r.core.env import get_env_path
from ips_r.core.exceptions import ConfigurationError
from ips_r.core.log import get_logger

logger = get_logger(__name__)

CONFIG_DIRNAME: Final = "configs"
PRIMARY_CONFIG: Final = "config.yaml"
_PATH_FIELDS: Final = ("data", "cache", "models", "outputs", "experiments")
_WRITABLE_FIELDS: Final = ("data", "cache", "models", "outputs")


def _is_project_root(path: Path) -> bool:
    """Return whether ``path`` contains the primary Hydra config."""
    return (path / CONFIG_DIRNAME / PRIMARY_CONFIG).is_file()


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the project root directory.

    Resolution order:

    1. ``IPS_R_ROOT`` environment variable.
    2. The first of ``start`` (default: CWD) or its parents containing ``configs/config.yaml``.
    3. The source checkout this package was imported from (editable installs).
    4. ``start`` itself, with a warning.

    Raises:
        ConfigurationError: If ``IPS_R_ROOT`` points to a non-existent directory.
    """
    env_root = get_env_path("IPS_R_ROOT")
    if env_root is not None:
        if not env_root.is_dir():
            raise ConfigurationError(f"IPS_R_ROOT={env_root} is not a directory")
        return env_root.resolve()

    origin = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for candidate in (origin, *origin.parents):
        if _is_project_root(candidate):
            return candidate

    # src/ips_r/core/paths.py -> parents[3] is the repository root.
    package_root = Path(__file__).resolve().parents[3]
    if _is_project_root(package_root):
        return package_root

    logger.warning("project_root.fallback", path=str(origin))
    return origin


def find_config_dir(root: Path | None = None) -> Path:
    """Locate the Hydra configuration directory.

    Uses ``IPS_R_CONFIG_DIR`` if set, otherwise ``<root>/configs``.

    Raises:
        ConfigurationError: If the directory does not contain ``config.yaml``.
    """
    env_dir = get_env_path("IPS_R_CONFIG_DIR")
    config_dir = env_dir if env_dir is not None else (root or find_project_root()) / CONFIG_DIRNAME
    if not (config_dir / PRIMARY_CONFIG).is_file():
        raise ConfigurationError(
            f"No {PRIMARY_CONFIG} found in config directory {config_dir}. "
            "Set IPS_R_CONFIG_DIR or run from inside the project."
        )
    return config_dir.resolve()


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Absolute locations of the project's working directories."""

    root: Path
    data: Path
    cache: Path
    models: Path
    outputs: Path
    experiments: Path

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ProjectPaths:
        """Build from a mapping such as the ``paths`` config node.

        Relative entries are resolved against ``root``.

        Raises:
            ConfigurationError: If a required key is missing.
        """
        missing = [key for key in ("root", *_PATH_FIELDS) if mapping.get(key) in (None, "")]
        if missing:
            raise ConfigurationError(f"paths config is missing keys: {missing}")
        root = Path(str(mapping["root"])).expanduser().resolve()

        def _resolve(key: str) -> Path:
            path = Path(str(mapping[key])).expanduser()
            return (path if path.is_absolute() else root / path).resolve()

        return cls(
            root=root,
            data=_resolve("data"),
            cache=_resolve("cache"),
            models=_resolve("models"),
            outputs=_resolve("outputs"),
            experiments=_resolve("experiments"),
        )

    def ensure(self) -> None:
        """Create the writable directories (data, cache, models, outputs) if missing.

        Raises:
            ConfigurationError: If a directory cannot be created.
        """
        for name in _WRITABLE_FIELDS:
            path: Path = getattr(self, name)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigurationError(f"Cannot create {name} directory {path}: {exc}") from exc
        logger.debug("paths.ensured", **self.as_dict())

    def as_dict(self) -> dict[str, str]:
        """Return the paths as plain strings (for logging and serialisation)."""
        return {key: str(value) for key, value in asdict(self).items()}
