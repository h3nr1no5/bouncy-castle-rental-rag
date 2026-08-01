FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY . .

RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils \
    && rm -rf /var/lib/apt/lists/* \
    && uv sync --frozen --no-dev \
    && rm -rf /root/.cache/uv \
    && find /app/.venv -name '*.pyi' -delete \
    && find /app/.venv -name '*.so' ! -path '*/*.libs/*' -exec strip --strip-unneeded {} + \
    && find /app/.venv -name '__pycache__' -type d -prune -exec rm -rf {} +

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HF_HOME=/app/.cache \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN useradd --create-home --uid 1000 --user-group app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app app.py pyproject.toml ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app ui/ ./ui/
COPY --chown=app:app data/ ./data/
COPY --chown=app:app tuned_params.json ./
COPY --chown=app:app docker-entrypoint.sh ./

RUN mkdir -p /app/db /app/.cache \
    && chown -R app:app /app \
    && chmod +x docker-entrypoint.sh \
    && python -c "from src.ingest import build_indexes; build_indexes()" \
    && chown -R app:app /app/db /app/.cache

USER app

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=5)"]
