"""IPS-R research codebase.

Sprint 1 provides infrastructure (configuration, logging, environment handling,
seed management, run tracking, CLI) and the abstract interfaces that research
components implement in later sprints.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("ips-r")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
