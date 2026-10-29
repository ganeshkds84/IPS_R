"""Exception hierarchy for IPS-R.

Every error raised deliberately by the package derives from :class:`IPSRError`
so that entrypoints can distinguish expected failures (bad config, missing
component, ...) from genuine bugs.
"""

from __future__ import annotations


class IPSRError(Exception):
    """Base class for all IPS-R errors."""


class ConfigurationError(IPSRError):
    """Raised when configuration cannot be loaded, composed or validated."""


class EnvironmentConfigError(IPSRError):
    """Raised when an environment variable or ``.env`` file is missing or malformed."""


class SeedError(IPSRError):
    """Raised when a random seed is invalid or seeding fails."""


class RegistryError(IPSRError):
    """Raised on invalid component registration or lookup."""


class PipelineError(IPSRError):
    """Raised when a pipeline cannot be assembled or executed."""


class ComponentError(IPSRError):
    """Raised when a research component fails or violates its interface contract.

    Attributes:
        component: Name of the component that raised the error.
    """

    def __init__(self, component: str, message: str) -> None:
        super().__init__(f"[{component}] {message}")
        self.component = component


class ComponentNotReadyError(ComponentError):
    """Raised when a component is used in a state that does not permit the operation."""
