# Backend image for the API. Not wired into docker-compose.yml yet: the backend
# runs locally under uv for fast iteration, and compose provides only Postgres
# and Redis.

# --- builder ------------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve in their own layer so application edits don't invalidate
# the (slow) dependency install.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY atlas ./atlas
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- runtime ------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root: nothing in this image needs to write outside /app.
RUN groupadd --system atlas && useradd --system --gid atlas --create-home atlas

WORKDIR /app

COPY --from=builder --chown=atlas:atlas /app/.venv /app/.venv
COPY --from=builder --chown=atlas:atlas /app/atlas /app/atlas

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER atlas
EXPOSE 8000

# Via the module entrypoint so loop selection stays in one place. On Linux this
# resolves to loop="auto", so uvloop is used.
CMD ["python", "-m", "atlas", "--host", "0.0.0.0", "--port", "8000"]
