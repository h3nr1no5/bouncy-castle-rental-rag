# Monitoring (Grafana runbook)

This document covers the Grafana monitoring dashboard that visualises the RAG
interactions logged to Postgres (`rag_logs`). Everything is provisioned from
files in [`grafana/`](../grafana/) — no manual setup in the Grafana UI. The
source of truth is the `grafana/` directory (dashboard, provisioning, seed
script) and the task-12 description in `_docs/tasks.md`.

## What is provisioned

- [`grafana/provisioning/datasources/postgres.yml`](../grafana/provisioning/datasources/postgres.yml)
  — provisions the **PostgreSQL** datasource (name `PostgreSQL`, uid `postgres`,
  `type: postgres`, `isDefault: true`, `postgresVersion: 1600`). The
  url/database/user/password/sslmode are read from the environment via Grafana
  `$VAR` interpolation (`$POSTGRES_HOST` / `$POSTGRES_PORT` / `$POSTGRES_DB` /
  `$POSTGRES_USER` / `$POSTGRES_PASSWORD` / `$POSTGRES_SSLMODE`) — see
  [`_docs/deployment.md`](deployment.md) for how those come from a single
  `DATABASE_URL`.
- [`grafana/provisioning/dashboards/dashboards.yml`](../grafana/provisioning/dashboards/dashboards.yml)
  — a file provider that loads dashboards from `/var/lib/grafana/dashboards`
  (every 10s), with `allowUiUpdates: false` so the dashboard stays file-owned.
- [`grafana/dashboard.json`](../grafana/dashboard.json) — the **RAG Monitoring**
  dashboard (uid `rag-monitoring`, refresh 30s) with the 6 panels below.
- [`grafana/Dockerfile`](../grafana/Dockerfile) — the Grafana image
  (`FROM grafana/grafana:10.4.3`) that copies in the provisioning files, the
  dashboard, and the `DATABASE_URL` entrypoint wrapper
  ([`grafana/docker-entrypoint.sh`](../grafana/docker-entrypoint.sh)).

## Starting Grafana

### Option 1 — the `grafana` service in `docker-compose.yaml`

The full stack starts with `docker compose up` from the repo root; the
`grafana` service maps port `3000`, reads admin credentials from `.env`
(`${GRAFANA_ADMIN_USER:-admin}` / `${GRAFANA_ADMIN_PASSWORD:-admin}`), sets
`DATABASE_URL` to the local `postgres` host, and mounts `grafana/provisioning`
and `grafana/dashboard.json` read-only. It depends on `postgres`.

### Option 2 — ad-hoc `docker run` (task 12)

To verify Grafana alone, build and run the image with the provisioning files
mounted (from the repo root):

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

(`--network llm-zoomcamp-rag_default` is the compose default network for this
repo directory — adjust it if the project directory name changes.)

## Logging in

Open http://localhost:3000 (or the deployed Grafana URL) and log in with
**`admin`** / **`GF_SECURITY_ADMIN_PASSWORD`** (compose default `admin` from
`.env` unless `GRAFANA_ADMIN_PASSWORD` is set).

## The 6 dashboard panels

Every panel reads only from the `rag_logs` table, extracts metrics from the
JSONB `metadata` column, and filters `created_at` with `$__timeFrom()` /
`$__timeTo()`:

1. **Recent conversations** — table of the last 5 interactions: `question`,
   `answer`, `feedback`, `metadata->>'model'`, `created_at` (ordered
   `created_at DESC LIMIT 5`).
2. **Feedback distribution** — pie chart counting `feedback`, with missing
   feedback coalesced to `none` (`COALESCE(feedback, 'none')`).
3. **Average latency over time** — timeseries of `AVG((metadata->>'latency')::numeric)`,
   bucketed hourly.
4. **Token usage over time** — timeseries of `SUM((metadata->'tokens'->>'total')::numeric)`,
   bucketed hourly.
5. **Estimated cost over time** — timeseries of `SUM((metadata->>'cost')::numeric)`,
   bucketed hourly (currency USD).
6. **Model usage** — bar chart counting rows by `metadata->>'model'`.

Panels render an empty state without errors when `rag_logs` has no rows.

## Seeding demo data

[`grafana/seed_demo.py`](../grafana/seed_demo.py) inserts 10 deterministic demo
rows (mixed `up`/`down`/no feedback, Groq + OpenAI models, latency / token /
cost values) so the panels can be verified. It first deletes any previous rows
for the demo questions, so re-running stays deterministic:

```bash
uv run python grafana/seed_demo.py
```

Requires Postgres running and `DATABASE_URL` set (compose sets it; locally it
comes from `.env`).

## Useful commands

- Stop/remove the ad-hoc container: `docker stop grafana-issue28 && docker rm grafana-issue28`
- Re-apply provisioning changes: dashboards reload automatically (the file
  provider polls every 10s); **datasource changes require a container
  restart** (e.g. `docker compose restart grafana` or re-`docker run`).
- Delete all demo rows:
  `psql postgres://postgres:postgres@localhost:5432/rag_logs -c "DELETE FROM rag_logs"`
  (or re-run `grafana/seed_demo.py`, which removes its own demo rows first).

## Source of truth

- [`grafana/dashboard.json`](../grafana/dashboard.json) — the dashboard (6 panels)
- [`grafana/provisioning/datasources/postgres.yml`](../grafana/provisioning/datasources/postgres.yml) — datasource
- [`grafana/provisioning/dashboards/dashboards.yml`](../grafana/provisioning/dashboards/dashboards.yml) — dashboard provider
- [`grafana/seed_demo.py`](../grafana/seed_demo.py) — demo-data seeding
- [`grafana/Dockerfile`](../grafana/Dockerfile) — the Grafana image
