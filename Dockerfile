# syntax=docker/dockerfile:1.7
# Multi-stage build: resolve dependencies with uv, ship a slim non-root runtime.
#   docker build -t ips-r .
#   docker run --rm ips-r info

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.1

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Dependencies first for layer caching.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG UID=1000
RUN useradd --create-home --uid "${UID}" ipsr

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IPS_R_ROOT=/app \
    IPS_R_CONFIG_DIR=/app/configs \
    IPS_R_LOG_FORMAT=json

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs
RUN mkdir -p data cache models outputs experiments \
    && chown -R ipsr:ipsr /app

USER ipsr

VOLUME ["/app/data", "/app/cache", "/app/models", "/app/outputs"]

ENTRYPOINT ["ips-r"]
CMD ["--help"]
