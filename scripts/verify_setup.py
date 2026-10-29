"""Verify that the development environment is correctly set up.

Checks the Python version, config composition, seeding reproducibility and
directory permissions. Exits non-zero on the first failure.

Usage:
    uv run python scripts/verify_setup.py [hydra overrides...]
"""

from __future__ import annotations

import random
import sys

from ips_r.config import as_app_config, load_config
from ips_r.core.env import load_env_file
from ips_r.core.exceptions import IPSRError
from ips_r.core.log import configure_logging, get_logger
from ips_r.core.paths import ProjectPaths
from ips_r.core.seed import seed_everything

logger = get_logger("verify_setup")

REQUIRED_PYTHON = (3, 12)


def check_python() -> None:
    """Fail if the interpreter is not the required major.minor version."""
    if sys.version_info[:2] != REQUIRED_PYTHON:
        raise RuntimeError(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} required, "
            f"found {sys.version.split()[0]}"
        )
    logger.info("check.python.ok", version=sys.version.split()[0])


def check_seeding(seed: int) -> None:
    """Fail if re-seeding does not reproduce the same random draws."""
    seed_everything(seed)
    first = random.random()
    seed_everything(seed)
    if random.random() != first:
        raise RuntimeError("Seeding is not reproducible")
    logger.info("check.seed.ok", seed=seed)


def main(argv: list[str]) -> int:
    """Run all checks. Returns a process exit code."""
    configure_logging("INFO")
    try:
        load_env_file()
        check_python()
        cfg = load_config(argv)
        app_cfg = as_app_config(cfg)
        logger.info("check.config.ok", experiment=app_cfg.experiment.name)
        paths = ProjectPaths.from_mapping(cfg.paths)
        paths.ensure()
        logger.info("check.paths.ok", **paths.as_dict())
        check_seeding(app_cfg.seed.value)
    except (IPSRError, RuntimeError) as exc:
        logger.error("verify_setup.failed", error=str(exc))
        return 1
    logger.info("verify_setup.passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
