FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY . .

RUN uv sync --frozen --no-dev \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        --index-strategy first-index \
    && rm -rf /root/.cache/uv \
    && find /app/.venv -name '*.pyi' -delete \
    && rm -rf \
        /app/.venv/lib/python3.11/site-packages/torch/include \
    && find /app/.venv/lib/python3.11/site-packages/torch/bin -type f ! -name 'torch_shm_manager' -delete

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN useradd --create-home --uid 1000 --user-group app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app app.py pyproject.toml ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app ui/ ./ui/
COPY --chown=app:app data/ ./data/
COPY --chown=app:app docker-entrypoint.sh ./

RUN mkdir -p /app/db \
    && chown -R app:app /app \
    && chmod +x docker-entrypoint.sh

USER app

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=5)"]
