FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY src/ ./src/
COPY config/ ./config/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENTRYPOINT [ "python", "-m" ]
CMD ["src.pipeline.weekly_pipeline", "--env", "dev"]
