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
docker build -f grafana/Dockerfile -t grafana-rag . # image with the DATABASE_URL entrypoint wrapper
docker run -d --name grafana-issue28 \
  --network llm-zoomcamp-rag_default \
  -p 3000:3000 \
  -v $(pwd)/grafana/provisioning:/etc/grafana/provisioning \
  -v $(pwd)/grafana/dashboard.json:/var/lib/grafana/dashboards/dashboard.json \
  -e GF_SECURITY_ADMIN_USER=admin \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  -e DATABASE_URL=postgres://postgres:postgres@postgres:5432/rag_logs?sslmode=disable \
  grafana-rag
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

- Stop/remove the container: `docker stop grafana-issue28 && docker rm grafana-issue28`
- Re-apply provisioning file changes: Grafana reloads provisioned dashboards automatically (every 10s); datasource changes need a container restart
- Delete all demo rows: `psql postgres://postgres:postgres@localhost:5432/rag_logs -c "DELETE FROM rag_logs"`

### Deploy to Render (app + Postgres + Grafana)

The whole stack deploys to Render from a single [`render.yaml`](render.yaml) Blueprint: the app web service (builds the root `Dockerfile`), a second web service running Grafana (builds [`grafana/Dockerfile`](grafana/Dockerfile)), and a managed Postgres from [Neon](https://neon.tech) (free tier). There is no Docker Compose on Render — the image is fully env-configured.

**Deployed app:** `https://<your-app>.onrender.com` (auto-assigned when you create the Blueprint).
**Deployed Grafana:** `https://<your-grafana>.onrender.com` (admin/`GF_SECURITY_ADMIN_PASSWORD`).

#### Step-by-step deploy

1. **Push this repo to GitHub** (public or private — either works).
2. **Sign up / log in to Render.** Prefer the GitHub/Google OAuth signup: signups are much faster than email verification, which requires clicking a link in your inbox before you can create anything.
3. **Add a payment method (required, even for the free tier).** Render now gates *all* resource creation (Blueprints and services alike) behind an account-level card requirement on new accounts — even `plan: free` instances hit a hard "Add Card" modal and API calls return `HTTP 402 Payment Required` until a card is on file. The card is charged $1 temporarily and refunded; no recurring charge on free instances. This gate applies regardless of how you signed up (email or OAuth) and is not mentioned in Render's free-tier docs. Go to **Account → Billing** → **Add Card** before trying to deploy.
4. **Create the Blueprint:** Render Dashboard → **New → Blueprint**.
   - If your repo is listed under the GitHub tab, select it.
   - If it isn't (e.g. the repo belongs to a different GitHub account than the one Render is connected to, which shows a "We weren't able to load your deployment credentials" error), switch to the **Public Git Repository** tab and paste the repo URL (e.g. `https://github.com/<you>/bouncy-castle-rental-rag`).
5. **On the Blueprint page:** Render parses `render.yaml`, shows both services (`bouncy-castle-rag`, `bouncy-castle-rag-grafana`), and offers to associate already-existing services with the Blueprint. Give the Blueprint a name (e.g. `bouncy-castle-rag`) and click **Deploy Blueprint**.
6. **Fill in the `sync: false` secrets.** After the first deploy, Render shows the unset secret env vars; paste the values below and redeploy (or set them via the service's **Environment** tab). If you created the app service first via the API, associate it with the Blueprint and the env vars sync over.

The Blueprint uses `sync: false` secrets, so the following env vars are **not** committed — fill them in on the Render service dashboard after the first deploy:

| Service | Env var | Value |
|---|---|---|
| App | `GROQ_API_KEY` | your Groq API key |
| App | `DATABASE_URL` | your Neon connection string, e.g. `postgres://user:password@ep-xxx.us-east-2.aws.neon.tech:5432/neondb?sslmode=require` |
| Grafana | `GF_SECURITY_ADMIN_PASSWORD` | Grafana admin password |
| Grafana | `DATABASE_URL` | the same Neon connection string as the App, e.g. `postgres://user:password@ep-xxx.us-east-2.aws.neon.tech:5432/neondb?sslmode=require` |

Non-secret values come from the Blueprint itself: the app runs on `PORT=8000`, Grafana on `PORT=3000`/`GF_SERVER_HTTP_PORT=3000`. The Grafana entrypoint wrapper (`grafana/docker-entrypoint.sh`) parses the service's `DATABASE_URL` into `POSTGRES_HOST`/`POSTGRES_PORT`/`POSTGRES_DB`/`POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_SSLMODE` before Grafana starts, and the datasource reads each field from those via Grafana `$VAR` interpolation in `grafana/provisioning/datasources/postgres.yml`. `OPENAI_API_KEY` is deliberately **not** set in the cloud — chat uses Groq only.

#### LLM provider: Groq primary, OpenAI fallback

`ask_llm()` (`src/llm.py`) resolves the provider from the two API keys it finds in the environment:

- If `GROQ_API_KEY` is **set**, it always tries **Groq first** (with built-in rate-limit enforcement).
- If that Groq call **fails**, it **falls back to OpenAI** — but only if `OPENAI_API_KEY` is also set.
- If Groq fails and `OPENAI_API_KEY` is **not** set, the request raises an error (no fallback).

Only one key is *required*, but they play different roles:

| Secret set in Render | Behavior |
|---|---|
| `GROQ_API_KEY` only | Uses **Groq only**. No fallback — if Groq errors or hits a rate limit, the request fails. |
| `OPENAI_API_KEY` only | Uses **OpenAI only** (the Groq path is skipped, since `groq_key` is unset). |
| Both | **Groq primary**, with **OpenAI as the automatic fallback** on Groq failure. |

On Render, only `GROQ_API_KEY` is configured (`OPENAI_API_KEY` is not in `render.yaml`), so the deployed app uses **Groq only** and has no fallback. If you set only `OPENAI_API_KEY` instead, the app uses OpenAI only. Setting **both** gives you the resilience of an automatic fallback.

**Verify the deploy:** `GET https://<your-app>.onrender.com/health` → 200; the UI loads at the root; Grafana renders all 6 panels at `https://<your-grafana>.onrender.com` (datasource fully env-driven via `DATABASE_URL`). Redeploys restart in seconds with cached build layers — confirm the deploy log shows the `build_indexes()` step as cached and no HF Hub download at runtime.

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
