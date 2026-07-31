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

### Ground truth data (optional)

A pre-generated ground truth dataset is available at `data/ground_truth.csv` (205 query–document pairs generated from the 41 FAQs).

To regenerate it yourself:

```bash
uv run python generate_ground_truth.py
```

Requires `OPENAI_API_KEY` in `.env`. Optional — the RAG pipeline works without it.

### Chat UI (FastAPI)

Run the FastAPI chat backend, which serves the static UI in `ui/`:

```bash
docker compose up -d postgres  # Postgres for feedback logging (optional)
uv run uvicorn app:app --reload --port 8000
```

Then open http://localhost:8000.

- `POST /api/chat` runs `answer_question()` and returns the answer with metadata and an `interaction_id`
- `POST /api/feedback` persists thumbs up/down for an `interaction_id`
- `GET /health` returns 200
- CORS is enabled so the API can be called from a website

Requires the search indexes (`db/bm25_index.pkl`, `db/faiss_index.bin`, `db/ingest_docs.json`); build them with `build_indexes()` in `src/ingest.py` if they are missing. Chat also needs `GROQ_API_KEY` (and optionally `OPENAI_API_KEY`) in `.env`.
