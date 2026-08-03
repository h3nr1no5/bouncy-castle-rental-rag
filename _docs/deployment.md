# Deployment (Render Blueprint)

This document covers deploying the full stack (app + Postgres + Grafana) to
Render from the single [`render.yaml`](../render.yaml) Blueprint. The source
of truth for everything below is `render.yaml`, the
[`Dockerfile`](../Dockerfile), [`docker-entrypoint.sh`](../docker-entrypoint.sh),
the `grafana/` provisioning files, and the live-deployment records in closed
issues #23 (Task 18), #24 (Task 19) and #28. There is no Docker Compose on
Render — the image is fully env-configured and port-portable.

## The Blueprint: two free-tier web services

[`render.yaml`](../render.yaml) defines two `type: web` services, both
`region: oregon` and `plan: free`:

- **`bouncy-castle-rag`** — the app service. Builds the root
  [`Dockerfile`](../Dockerfile) (`dockerfilePath: ./Dockerfile`), runs on
  `PORT=8000`, and declares `healthCheckPath: /health`.
- **`bouncy-castle-rag-grafana`** — the Grafana service. Builds
  [`grafana/Dockerfile`](../grafana/Dockerfile), runs on `PORT=3000`, and sets
  `GF_SECURITY_ADMIN_USER=admin`.

Non-secret values come from the Blueprint itself:

| Service | Env var | Value |
|---|---|---|
| App | `PORT` | `8000` |
| App | `healthCheckPath` | `/health` |
| Grafana | `PORT` | `3000` |
| Grafana | `GF_SECURITY_ADMIN_USER` | `admin` |

`OPENAI_API_KEY` is deliberately **not** set in the cloud — the deployed chat
uses Groq only (see [`_docs/architecture.md`](architecture.md)).

## The four `sync: false` secrets

Four env vars use `sync: false`, so their values are **never committed** — they
are filled in on the Render dashboard after the first deploy:

| Service | Env var | Value |
|---|---|---|
| App | `GROQ_API_KEY` | your Groq API key |
| App | `DATABASE_URL` | Neon connection string, e.g. `postgres://user:password@ep-xxx.us-east-2.aws.neon.tech:5432/neondb?sslmode=require` |
| Grafana | `GF_SECURITY_ADMIN_PASSWORD` | Grafana admin password |
| Grafana | `DATABASE_URL` | the same Neon connection string as the app |

## Grafana datasource: one `DATABASE_URL`, split into `POSTGRES_*`

Grafana's Postgres datasource plugin does not accept a full connection string
in provisioning — its `url` field is strictly `host:port`, with database, user,
password and sslmode as separate fields. The single `DATABASE_URL` is therefore
split into the `POSTGRES_*` vars at container start:

- [`grafana/docker-entrypoint.sh`](../grafana/docker-entrypoint.sh) parses
  `DATABASE_URL` (scheme `postgres://`/`postgresql://`, stripping everything
  before `://` and splitting credentials from host/port/db/query at the last
  `@`) into `POSTGRES_HOST`, `POSTGRES_PORT` (default `5432`), `POSTGRES_DB`,
  `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_SSLMODE` (default
  `require`, or taken from the URL query string), `export`s them, then
  `exec /run.sh`.
- [`grafana/provisioning/datasources/postgres.yml`](../grafana/provisioning/datasources/postgres.yml)
  consumes them via Grafana `$VAR` interpolation:
  `url: $POSTGRES_HOST:$POSTGRES_PORT`, `database: $POSTGRES_DB`,
  `user: $POSTGRES_USER`, password via `secureJsonData.password: $POSTGRES_PASSWORD`,
  `sslmode: $POSTGRES_SSLMODE`.

Closed issue #28 replaced the previously separate `POSTGRES_HOST` /
`POSTGRES_USER` / `POSTGRES_PASSWORD` secrets (plus the static
`POSTGRES_DB=neondb`, `POSTGRES_PORT=5432`, `POSTGRES_SSLMODE=require`) with
this single-`DATABASE_URL` flow, so each service has only one DB-related secret
to fill in on Render.

## Step-by-step deploy walkthrough

These steps come from the live deploy recorded in closed issue #24 (QA report)
and the pre-trim README (issue #14):

1. **Push this repo to GitHub** (public or private — either works).
2. **Sign up / log in to Render.** OAuth signup (GitHub/Google) is faster than
   email verification, which requires clicking a link in your inbox.
3. **Add a payment method (required even for the free tier).** Observed during
   the live deploy (#24): Render gates *all* resource creation on new accounts
   behind an account-level card requirement — both email and GitHub-OAuth
   signups hit a hard "Add Card" modal, and API calls return `HTTP 402 Payment
   Required` until a card is on file. The card is charged $1 temporarily and
   refunded; there is no recurring charge on free instances. This gate is not
   mentioned in Render's free-tier docs. Go to **Account → Billing → Add Card**
   before trying to deploy.
4. **Create the Blueprint:** Render Dashboard → **New → Blueprint**.
   - If the repo is listed under the GitHub tab, select it.
   - If it isn't (e.g. the repo belongs to a different GitHub account than the
     one Render is connected to, which shows a "We weren't able to load your
     deployment credentials" error), switch to the **Public Git Repository**
     tab and paste the repo URL (e.g. `https://github.com/<you>/bouncy-castle-rental-rag`).
5. **On the Blueprint page** Render parses `render.yaml`, shows both services
   (`bouncy-castle-rag`, `bouncy-castle-rag-grafana`), and offers to associate
   already-existing services with the Blueprint. Give the Blueprint a name
   (e.g. `bouncy-castle-rag`) and click **Deploy Blueprint**.
6. **Fill in the `sync: false` secrets** after the first deploy. Render shows
   the unset secret env vars; paste the values from the table above and
   redeploy (or set them via each service's **Environment** tab). If a service
   was created first via the API, associate it with the Blueprint so the env
   vars sync over.

The live instances from the #24 record were the app at
`https://bouncy-castle-rag.onrender.com` and Grafana at
`https://bouncy-castle-rag-grafana.onrender.com`; a fresh Blueprint gets its
own auto-assigned URLs (`https://<your-app>.onrender.com` etc.). #24 also noted
that the dashboard/API required the owner `ownerID` during the live deploy.

## Index pre-baking at build time

`render.yaml` has no `/app/db` volume, so the search indexes are pre-baked into
the image at build time:

- The [`Dockerfile`](../Dockerfile) runs `build_indexes()` in a build step
  (before `USER app`), building `db/bm25_index.pkl`, `db/faiss_index.bin` and
  `db/ingest_docs.json` and warming the fastembed ONNX cache under
  `HF_HOME=/app/.cache` / `FASTEMBED_CACHE_PATH=/app/.cache/fastembed`.
- `.dockerignore` excludes `db/`, so the build step is what actually creates
  the index files in the image.

At runtime, [`docker-entrypoint.sh`](../docker-entrypoint.sh) keeps a rebuild
fallback: if any of `db/bm25_index.pkl`, `db/faiss_index.bin` or
`db/ingest_docs.json` is missing it calls `build_indexes()` before starting
uvicorn (`exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"`). With
pre-baked indexes this branch is skipped, so cold starts take seconds and no HF
Hub download happens at runtime — verified in #24: all build layers cached
including the `build_indexes()` step, and uvicorn was up ~1s after container
start.

## Database fallback

The app's logging degrades gracefully when Postgres is unreachable (Neon
free-tier scale-to-zero, network blip):

- `_log_interaction()` in `app.py` returns `None` when `DATABASE_URL` is unset
  or any connection/logging error occurs, so chat still works.
- Feedback is not persisted and the `interaction_id` in the `/api/chat`
  response is omitted (`null`).
- `rag_logs` is auto-created on the first successful connect (`init_db()`
  runs `CREATE TABLE IF NOT EXISTS rag_logs`).

## One-time data migration (local Postgres → Neon)

Existing local `rag_logs` rows can be migrated with `pg_dump` / `psql`, then
verified by row-count and `created_at` range parity. This is how the 48 local
rows were moved to Neon in #23 (source and target matched, Neon = PostgreSQL
16.14):

```bash
pg_dump "$DATABASE_URL" --no-owner --no-privileges --data-only -t rag_logs > rag_logs.sql
psql "$DATABASE_URL_CLOUD" -f rag_logs.sql
psql "$DATABASE_URL" -c "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM rag_logs"
psql "$DATABASE_URL_CLOUD" -c "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM rag_logs"
```

`DATABASE_URL` is the local Postgres and `DATABASE_URL_CLOUD` the Neon URL
(both in `.env.example`). Compare the two `COUNT(*)` / `MIN(created_at)` /
`MAX(created_at)` results before cutting over.

## Verify the deploy (live steps from #24)

1. `GET https://<your-app>.onrender.com/health` → `200`; the static UI loads at
   the root.
2. A real chat round-trip works with Groq only (`provider: groq`, no
   `OPENAI_API_KEY` on Render), and `up`/`down` feedback persists to Neon
   `rag_logs`.
3. Trigger a redeploy: cold start is seconds with cached build layers — the
   deploy log shows the `build_indexes()` step as cached and no HF Hub /
   fastembed download at runtime.
4. Grafana at `https://<your-grafana>.onrender.com` renders all 6 panels
   against the deployed data, with the datasource fully env-driven (see
   [`_docs/monitoring.md`](monitoring.md) for the panel list).

## Source of truth

- [`render.yaml`](../render.yaml) — the Blueprint (services, secrets, defaults)
- [`Dockerfile`](../Dockerfile) — image build incl. index pre-baking
- [`docker-entrypoint.sh`](../docker-entrypoint.sh) — index rebuild fallback
- [`grafana/docker-entrypoint.sh`](../grafana/docker-entrypoint.sh) — `DATABASE_URL` parsing
- [`grafana/provisioning/datasources/postgres.yml`](../grafana/provisioning/datasources/postgres.yml) — datasource env flow
