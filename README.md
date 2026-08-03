# Bouncy Castle FAQ – RAG Agent

RAG system over a bouncy castle rental FAQ, using hybrid search (BM25 + FAISS) with Groq (primary) and OpenAI (fallback).

## Quick start

Clone the repo, then follow the path below to get the full stack running in under two minutes.

### Prerequisites

- **Docker Desktop** — required to run the full stack (app, Postgres, Grafana)
- **API keys** — `GROQ_API_KEY` (required for LLM calls) and `OPENAI_API_KEY` (optional fallback)

### Setup

```bash
cp .env.example .env
# Fill in your API keys in .env
# GROQ_API_KEY is required; OPENAI_API_KEY is the optional fallback
```

### Run

```bash
docker compose up
```

This single command starts the full stack from the repo's [`docker-compose.yaml`](docker-compose.yaml) (discovered by default — no `-f` flag needed):

- **App** on http://localhost:8000
- **Postgres** on `:5432`
- **Grafana** on `:3000`

Open the UI at **http://localhost:8000**.

For Postgres-only local dev (e.g. to log RAG interactions while running the app via `uv`):

```bash
docker compose up -d postgres
```

To tear the whole stack down:

```bash
docker compose down
```

### Run the tests

```bash
uv sync
uv run pytest
```

## Project structure

```
├── app.py                 # FastAPI chat backend (serves ui/)
├── src/                   # Core application code (ingest, search, rag, llm, db)
├── ui/                    # Static chat UI
├── data/                  # FAQ CSV + evaluation datasets
├── tests/                 # Test suite
├── db/                    # Search indexes (bm25, faiss, ingest docs)
├── docker-compose.yaml    # Full stack: app + Postgres + Grafana
└── README.md
```

## Documentation

- **Plan** — [`_docs/plan.md`](_docs/plan.md)
- **Backlog / tasks** — [`_docs/tasks.md`](_docs/tasks.md)
- **Contributor commands & rules** — [`agents.md`](agents.md)

## Deployment & architecture

The quick-start above is all you need to run locally. For the longer-form deployment, architecture, and monitoring detail, see the dedicated docs:

- **Deployment (Render Blueprint, secrets, index pre-baking, DB fallback, data migration)** — [`_docs/deployment.md`](_docs/deployment.md)
- **Architecture (hybrid search / RRF re-ranking, LLM provider fallback)** — [`_docs/architecture.md`](_docs/architecture.md)
- **Monitoring (Grafana runbook, 9-panel dashboard, useful commands)** — [`_docs/monitoring.md`](_docs/monitoring.md)

The in-repo files remain the source of truth: [`render.yaml`](render.yaml), the [`Dockerfile`](Dockerfile), [`grafana/`](grafana/), [`src/search.py`](src/search.py), [`src/llm.py`](src/llm.py).