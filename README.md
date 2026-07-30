# Bouncy Castle FAQ – RAG Agent

RAG system over a bouncy castle rental FAQ, using hybrid search (BM25 + FAISS) with Groq (primary) and OpenAI (fallback).

## Quick start

### Prerequisites

- **Python 3.12+** with `uv` installed
- **Docker Desktop** — required to run Postgres locally for logging RAG interactions

### Setup

```bash
cp .env.example .env
# Fill in your API keys in .env
# GROQ_API_KEY is required for LLM calls; OPENAI_API_KEY is the fallback

uv sync
```

### Start Postgres (required for logging)

**Docker Desktop must be running** before starting Postgres:

```bash
docker compose up -d postgres
```

This starts a Postgres 16 container on port 5432 with the connection string configured in `.env.example`.

### Run the notebook

```bash
uv run jupyter notebook exploration.ipynb
```

> The notebook demonstrates all completed tasks (1–8). Sections 6–7 need API keys for LLM calls. Section 8 needs Postgres running (via Docker Desktop).
