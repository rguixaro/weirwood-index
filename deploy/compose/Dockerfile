FROM ghcr.io/astral-sh/uv:0.11.12 AS uv

FROM python:3.12.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    WEIRWOOD_INDEX_PATH=/app/runtime/index \
    WEIRWOOD_MODEL_CACHE=/app/runtime/models \
    WEIRWOOD_VERIFY_SOURCE=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
COPY src ./src

RUN uv sync --frozen --no-dev --extra api \
    && useradd --create-home --uid 10001 weirwood \
    && mkdir -p /app/runtime \
    && chown -R weirwood:weirwood /app

USER weirwood

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "weirwood_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
