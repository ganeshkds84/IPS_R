# Common developer tasks. Requires GNU make and uv (on Windows: Git Bash or WSL).
.DEFAULT_GOAL := help
SHELL := bash

UV       ?= uv
IMAGE    ?= ips-r
ARGS     ?=

.PHONY: help install lock update format lint typecheck test cov check pre-commit \
	    verify run dry-run docker-build docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install Python 3.12, all dependencies and git hooks
	$(UV) python install 3.12
	$(UV) sync
	@if [ -d .git ]; then $(UV) run pre-commit install; else echo "skip pre-commit install (not a git repo)"; fi

lock: ## Re-resolve uv.lock
	$(UV) lock

update: ## Upgrade all dependencies within constraints
	$(UV) lock --upgrade
	$(UV) sync

format: ## Auto-format and auto-fix lint issues
	$(UV) run ruff check --fix .
	$(UV) run black .

lint: ## Lint (ruff) and check formatting (black)
	$(UV) run ruff check .
	$(UV) run black --check .

typecheck: ## Static type checking (mypy --strict)
	$(UV) run mypy

test: ## Run the test suite
	$(UV) run pytest

cov: ## Run tests with coverage report (HTML in htmlcov/)
	$(UV) run pytest --cov --cov-report=term --cov-report=html

check: lint typecheck cov ## Everything CI runs

pre-commit: ## Run all pre-commit hooks on all files
	$(UV) run pre-commit run --all-files

verify: ## Verify the local environment
	$(UV) run python scripts/verify_setup.py

run: ## Run an experiment, e.g. make run ARGS="experiment=example seed.value=7"
	$(UV) run ips-r run $(ARGS)

dry-run: ## Validate config/components without executing
	$(UV) run ips-r run --dry-run $(ARGS)

docker-build: ## Build the Docker image
	docker build -t $(IMAGE) .

docker-run: ## Run the CLI in Docker, e.g. make docker-run ARGS="run --dry-run"
	docker compose run --rm ips-r $(ARGS)

clean: ## Remove caches and build artefacts (keeps data/models/outputs)
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
