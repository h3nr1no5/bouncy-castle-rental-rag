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

### Grafana monitoring dashboard

A Grafana dashboard visualises the RAG interactions logged in Postgres (`rag_logs`). All config is provisioned from files in `grafana/` — no manual setup in the Grafana UI.

Start Postgres, seed some sample rows, then start Grafana:

```bash
docker compose up -d postgres                       # Postgres for rag_logs
uv run python grafana/seed_demo.py                  # optional: seed demo rows
docker run -d --name grafana-issue12 \
  --network llm-zoomcamp-rag_default \
  -p 3000:3000 \
  -v $(pwd)/grafana/provisioning:/etc/grafana/provisioning \
  -v $(pwd)/grafana/dashboard.json:/var/lib/grafana/dashboards/dashboard.json \
  -e GF_SECURITY_ADMIN_USER=admin \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana:10.4.3
```

Then open http://localhost:3000 and log in with `admin`/`admin`.

The dashboard is provisioned with 6 panels over `rag_logs`:

1. **Recent conversations** — table of the last 5 interactions (question, answer, feedback, model, created_at)
2. **Feedback distribution** — pie chart of `up` / `down` / missing feedback
3. **Average latency over time** — AVG of `metadata->>'latency'`
4. **Token usage over time** — SUM of `metadata->'tokens'->>'total'`
5. **Estimated cost over time** — SUM of `metadata->>'cost'`
6. **Model usage** — bar chart of counts by `metadata->>'model'`

Useful commands:

- Stop/remove the container: `docker stop grafana-issue12 && docker rm grafana-issue12`
- Re-apply provisioning file changes: Grafana reloads provisioned dashboards automatically (every 10s); datasource changes need a container restart
- Delete all demo rows: `psql postgres://postgres:postgres@localhost:5432/rag_logs -c "DELETE FROM rag_logs"`

### Deploy to Render (app + Postgres + Grafana)

The whole stack deploys to Render from a single [`render.yaml`](render.yaml) Blueprint: the app web service (builds the root `Dockerfile`), a second web service running Grafana (builds [`grafana/Dockerfile`](grafana/Dockerfile)), and a managed Postgres from [Neon](https://neon.tech) (free tier). There is no Docker Compose on Render — the image is fully env-configured.

**Deployed app:** `https://<your-app>.onrender.com` (auto-assigned when you create the Blueprint).
**Deployed Grafana:** `https://<your-grafana>.onrender.com` (admin/`GF_SECURITY_ADMIN_PASSWORD`).

**One-click deploy:** push this repo to GitHub, open the Render Dashboard → **New → Blueprint**, select the repo, and Render creates both services from `render.yaml`.

The Blueprint uses `sync: false` secrets, so the following env vars are **not** committed — fill them in on the Render service dashboard after the first deploy:

| Service | Env var | Value |
|---|---|---|
| App | `GROQ_API_KEY` | your Groq API key |
| App | `DATABASE_URL` | your Neon connection string, e.g. `postgres://user:password@ep-xxx.us-east-2.aws.neon.tech:5432/neondb?sslmode=require` |
| Grafana | `GF_SECURITY_ADMIN_PASSWORD` | Grafana admin password |
| Grafana | `POSTGRES_HOST` | Neon host, e.g. `ep-xxx.us-east-2.aws.neon.tech` |
| Grafana | `POSTGRES_USER` | Neon database user |
| Grafana | `POSTGRES_PASSWORD` | Neon database password |

Non-secret values come from the Blueprint itself: the app runs on `PORT=8000`, Grafana on `PORT=3000`/`GF_SERVER_HTTP_PORT=3000`, `POSTGRES_DB=neondb`, `POSTGRES_PORT=5432`, `POSTGRES_SSLMODE=require`, and the Grafana datasource reads every connection field from these `POSTGRES_*` env vars (Grafana `$VAR` interpolation in `grafana/provisioning/datasources/postgres.yml`). `OPENAI_API_KEY` is deliberately **not** set in the cloud — chat uses Groq only.

**Indexes are pre-baked at build time.** `render.yaml` has no `/app/db` volume, so the `Dockerfile` runs `build_indexes()` during the image build (warming the fastembed ONNX model into `/app/.cache`). Deploys and restarts start in seconds with no runtime model download. The `docker-entrypoint.sh` rebuild fallback still runs if the index files are ever missing.

**Database fallback:** when Neon is unreachable (free-tier scale-to-zero, network blip), the app keeps serving chat — feedback is simply not persisted and `interaction_id` is omitted. `rag_logs` is auto-created on first successful connect (`init_db`). See Task 14.

**Migrating existing local data to Neon** (verify with row-count and `created_at` range parity after):

```bash
pg_dump "$DATABASE_URL" --no-owner --no-privileges --data-only -t rag_logs > rag_logs.sql
psql "$DATABASE_URL_CLOUD" -f rag_logs.sql
psql "$DATABASE_URL" -c "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM rag_logs"
psql "$DATABASE_URL_CLOUD" -c "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM rag_logs"
```

`DATABASE_URL` is your local Postgres and `DATABASE_URL_CLOUD` your Neon URL (both already in `.env`).
