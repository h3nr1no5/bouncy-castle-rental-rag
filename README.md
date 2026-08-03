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

The quick-start above is all you need to run locally. For the longer-form deployment and architecture detail:

- **Render deployment** — see [`render.yaml`](render.yaml) and the [`Dockerfile`](Dockerfile)
- **Grafana monitoring** — see [`grafana/`](grafana/) (provisioned dashboards, datasource, seed script)
- **Hybrid search / RRF re-ranking** — see [`src/search.py`](src/search.py)
- **LLM provider (Groq primary, OpenAI fallback)** — see [`src/llm.py`](src/llm.py)

> These topics are being consolidated into dedicated docs (`_docs/deployment.md`, `_docs/architecture.md`, `_docs/monitoring.md`) under issue #68; the in-repo files above are the source of truth until then.