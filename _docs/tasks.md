# Task Backlog

## 1. Project scaffold with passing test

Goal: Set up an empty project skeleton that installs and passes a smoke test.

Description: Create the directory structure, a minimal `pyproject.toml` (or `requirements.txt`) with key dependencies, a placeholder `src/` package, and a trivial test (e.g. `assert True`) that can be run with `pytest`. This gives the team a working starting point before any real logic is written.

---

## 2. Load and inspect FAQ CSV

### Goal

Provide a `load_faqs()` function in `src/faqs.py` that reads `data/faq.csv`, strips whitespace from values, and returns structured data so downstream ingestion (tasks 3–4) can consume it without worrying about file format or path resolution.

### Acceptance criteria

- [x] `src/faqs.py` exports `load_faqs()` returning `list[dict[str, str]]` with keys `Category`, `Question`, `Answer`
- [x] CSV cell values have leading/trailing whitespace stripped
- [x] A `python -m src.faqs` entry point prints `Rows: 42` and the column names
- [x] `tests/test_faqs.py` verifies:
  - the CSV is non-empty (42 rows)
  - every row has all three non-empty string keys
  - `load_faqs()` raises `FileNotFoundError` with a clear message when the CSV is missing
- [x] The CSV path defaults to `<project_root>/data/faq.csv` and is overridable via an optional `path` argument

### Out of scope

- Data cleaning beyond whitespace stripping (e.g. normalising category names, splitting multi-value fields)
- Any index building, embeddings, or BM25 (task 4)
- Any data exploration notebooks or visualisations

### Constraints

- Only stdlib — use `csv` and `pathlib`, no pandas or new dependencies
- Tests go in `tests/test_faqs.py`
- Path resolution via `pathlib.Path(__file__).resolve().parents[1] / "data" / "faq.csv"` as the default

---

## 3. dlt ingestion pipeline

Goal: Set up a dlt pipeline to ingest the FAQ CSV into a normalized, queryable dataset.

Description: Write a dlt pipeline definition that reads the FAQ CSV via `dlt.sources.filesystem` or a custom `@dlt.source`, applies any needed transformations (rename columns, parse dates, validate non-nulls), and loads the data into Postgres. The pipeline should be idempotent — re-running it upserts or replaces without duplicating rows. Add a test that verifies the pipeline runs end-to-end and the destination contains the expected rows.

---

## 4. Build hybrid search index (ingestion)

Goal: Create an ingestion script that builds a BM25 index + FAISS vector index from the FAQ data.

Description: Write `src/ingest.py` that reads FAQ entries via `load_faqs()`, computes TF-IDF/BM25 tokens with `rank_bm25`, generates embeddings with `all-MiniLM-L6-v2` (via `fastembed`, ONNX Runtime), stores vectors in a FAISS index, and saves both indexes to disk so they don't rebuild on every run.

---

## 5. Implement hybrid retrieval with RRF

Goal: Given a user question, return the top-k FAQ entries using hybrid search + reciprocal rank fusion.

Description: Write `src/search.py` that loads the persisted BM25 and FAISS indexes, runs both retrievers against a query, and merges results via RRF. The module should expose a `search(query, k=5)` callable returning ranked FAQ entries with scores.

---

## 5.1. Jupyter notebook for closed tasks

### Goal

An `exploration.ipynb` notebook at the project root that demonstrates every function from the completed tasks (1–5) with explanatory markdown, so anyone reading it can understand the full ingest–search pipeline without running the test suite.

### Acceptance criteria

- [x] `exploration.ipynb` exists at the project root
- [x] Notebook cells are preceded by markdown explaining what each cell does and why
- [x] Section 1 — Setup: imports from `src.faqs`, `src.pipeline`, `src.ingest`, `src.search`; create a `db/` working directory
- [x] Section 2 — Load FAQ Data: calls `load_faqs()`, prints row count (42), displays 3 sample rows, shows column-level missing-value stats
- [x] Section 3 — dlt Pipeline: runs `run_pipeline()` into `db/test_faq.duckdb`, queries the loaded table with `duckdb`, re-runs the pipeline and confirms the row count stays the same (idempotency)
- [x] Section 4 — Build Indexes: calls `build_indexes(force=True)` with all paths under `db/`, inspects the BM25 pickle count, FAISS index dimensions (ntotal, d), and docs JSON structure
- [x] Section 5 — Hybrid Search: calls `search()` with at least three different queries (e.g. "booking", "cost", "safety"), prints ranked results with RRF scores, demonstrates `k=1` vs `k=5` and an empty-query edge case
- [x] Section 6 — Cleanup (optional): skipped — `db/` is persistent; remove manually if desired
- [x] `pyproject.toml` has `ipykernel` and `notebook` in dev dependencies
- [x] Running all cells top-to-bottom produces no errors

### Out of scope

- Any open tasks (6–15) — the notebook only covers completed work
- UI or visualisation beyond printed tables and markdown
- Embedding or re-implementing logic that already lives in `src/`

### Constraints

- Notebook must import from `src.*` modules — no duplicated logic
- Artifacts (indexes, DuckDB) live in `db/` (gitignored)
- Follow the same coding conventions as the rest of the project (no pandas, minimal new deps)

---

## 6. LLM client with Groq + OpenAI fallback

Goal: Create a thin LLM client that calls Groq by default and falls back to OpenAI on failure.

Description: Write `src/llm.py` that wraps the `groq` and `openai` Python SDKs. It should accept a system prompt and user message, attempt Groq first, catch errors (rate limits, timeouts, auth failures), and retry with OpenAI. Return the response text plus metadata (model used, latency, token counts).

---

## 6.1. Update exploration notebook with LLM client

### Goal

The `exploration.ipynb` notebook demonstrates every closed task from 1–5; this task adds a section for the Task 6 LLM client (`src/llm.py`) so the notebook covers all completed work.

### Acceptance criteria

- [x] A new top-level markdown heading "6. LLM Client" exists between "5. Hybrid Search" and the existing "6. Cleanup" sections
- [x] The former "6. Cleanup" section is renumbered to "7. Cleanup"
- [x] `ask_llm` is imported from `src.llm`
- [x] A code cell attempts a real `ask_llm()` call with a FAQ-related system prompt and question; if API keys are missing it prints a graceful message instead of crashing
- [x] Returned metadata (provider, model, latency, token counts) is displayed
- [x] Rate-limit constants (`GROQ_RPM_LIMIT`, `GROQ_RPD_LIMIT`) are shown and the `_enforce_groq_rate_limits()` mechanism is explained in markdown
- [x] All cells run top-to-bottom without errors
- [x] No duplicated logic — everything comes from `src.*` imports
- [x] Only `src/` and standard-library/test dependencies are used; no new dependencies added

### Out of scope

- Implementing `src/rag.py` (Task 7) — the notebook only covers closed tasks
- Mock-based demonstrations — real API calls with graceful key-gated fallback

### Constraints

- Notebook must import from `src.*` modules — no inlined logic
- API-key‑dependent cells must use `os.environ.get(...)` and fail gracefully
- Follow the same coding conventions as the rest of the project (no pandas)

---

## 7. RAG orchestration pipeline

Goal: Wire retrieval + LLM call into a single `answer_question()` function.

Description: Write `src/rag.py` that takes a user question, calls `search()` for context, builds a prompt with the retrieved FAQ entries, calls the LLM client, and returns a structured result: answer text, retrieved contexts, model used, latency, and token usage.

---

## 7.1. Update exploration notebook with RAG pipeline

### Goal

The `exploration.ipynb` notebook demonstrates every closed task from 1–6; this task adds a section for Task 7's `answer_question()` (`src/rag.py`) so the notebook covers the full RAG pipeline end-to-end.

### Acceptance criteria

- [x] A new top-level markdown heading "7. RAG Pipeline" exists between "6. LLM Client" and the existing "8. Cleanup" sections
- [x] The former "7. Cleanup" section is renumbered to "8. Cleanup"
- [x] `answer_question` is imported from `src.rag`
- [x] A code cell calls `answer_question()` with a sample question (e.g. "What's your cancellation policy?") using the same `db/` indexes built in section 4; if API keys are missing it prints a graceful message instead of crashing
- [x] Returned metadata (answer, contexts, model, provider, latency, tokens) is displayed
- [x] A code cell demonstrates the empty-context edge case (search returns no results)
- [x] All cells run top-to-bottom without errors
- [x] No duplicated logic — everything comes from `src.*` imports

### Out of scope

- Any open tasks beyond 7 (8–15) — the notebook only covers completed work
- Mock-based demonstrations — real calls with graceful key-gated fallback

### Constraints

- Notebook must import from `src.*` modules — no inlined logic
- API-key‑dependent cells must use `os.environ.get(...)` and fail gracefully
- Follow the same coding conventions as the rest of the project (no pandas)
- Sequential run assumed: sections 1–5 must have already built the indexes in `db/`

## 8. Postgres logging layer

Goal: Log every RAG interaction to a Postgres database.

Description: Write `src/db.py` that manages a Postgres connection (via `asyncpg` or `psycopg2`), creates a table for logs (question, answer, feedback, metadata, timestamp), and exposes `log_interaction()` and `update_feedback()` functions.

## 8.1. Update exploration notebook with Postgres logging layer

### Goal

The `exploration.ipynb` notebook demonstrates every closed task from 1–8; this task adds a section for Task 8's Postgres logging layer (`src/db.py`) so the notebook covers all completed work.

### Acceptance criteria

- [x] A new top-level markdown heading "8. Postgres Logging Layer" exists between "7. RAG Pipeline" and the existing "9. Cleanup" sections
- [x] The former "9. Cleanup" section is renumbered to "10. Cleanup"
- [x] `init_db`, `log_interaction`, `update_feedback`, `get_connection`, and `CREATE_TABLE_SQL` are imported from `src.db`
- [x] The `rag_logs` table schema (DDL) is printed for inspection
- [x] A code cell attempts a real Postgres connection: calls `init_db()` to create the table, `log_interaction()` to log a sample RAG interaction, and `update_feedback()` to set feedback; if `DATABASE_URL` is missing or the connection fails, a graceful message is printed instead of crashing
- [x] All cells run top-to-bottom without errors
- [x] No duplicated logic — everything comes from `src.*` imports

### Out of scope

- Any open tasks beyond 8 (9–15) — the notebook only covers completed work
- Mock-based demonstrations — real Postgres connection with graceful key-gated fallback

### Constraints

- Notebook must import from `src.*` modules — no inlined logic
- Database-dependent cells must use `os.environ.get("DATABASE_URL")` and fail gracefully
- Follow the same coding conventions as the rest of the project (no pandas)
- Sequential run assumed: sections 1–7 must have already built the indexes in `db/`

---

## 8.2. Wire Postgres logging into the RAG pipeline

### Goal

Currently `log_interaction()` is only called with hardcoded demo data in the exploration notebook. This task wires it into `answer_question()` in `src/rag.py` so every real RAG interaction is logged to Postgres automatically.

### Acceptance criteria

- [x] `answer_question()` in `src/rag.py` calls `log_interaction()` after every successful LLM call, passing the real question, answer, and metadata (provider, model, tokens, latency, cost)
- [x] Logging is graceful: if `DATABASE_URL` is not set or the connection fails, the function still returns the answer without crashing
- [x] `init_db()` is called on first use (lazily) to ensure the table exists
- [x] The notebook's section 8 demo still works unchanged
- [x] All existing tests (`uv run pytest`) still pass

### Out of scope

- Wiring `update_feedback()` into any API or UI — that remains a manual/library function
- Connection pooling or async — a simple short-lived connection per call is fine

---

## 9. Evaluation: retrieval metrics

Goal: Compute hit rate and MRR against a small ground-truth question set.

Description: Create an evaluation script that loads a hand-crafted set of test questions with known relevant FAQ IDs, runs `search()` for each, and reports hit rate@k and MRR@k. Save results for later comparison.

---

## 9.1. Generate ground truth questions from FAQs

### Goal

Replace the manually written `data/ground_truth.json` with a script that uses an LLM to generate at least 5 natural-language query variants per FAQ entry using structured output, so evaluation has meaningful coverage (~210+ entries). Switch to CSV output (question, document_id) and use document IDs throughout the retrieval pipeline.

### Acceptance criteria

- [x] `generate_ground_truth.py` exists at project root and is runnable via `uv run python generate_ground_truth.py`
- [x] It loads all 42 FAQs from `data/faq.csv` via `src.faqs.load_faqs()`
- [x] Uses OpenAI **structured output** (`client.responses.parse()` with Pydantic `Questions` model) — not `ask_llm()`
- [x] Model: `gpt-5.4-mini`, with pricing displayed after run ($0.75/M input, $4.50/M output)
- [x] Retry logic with exponential backoff on API errors
- [x] Processes FAQs in parallel using `ThreadPoolExecutor` + `tqdm` progress bar
- [x] Output is written to `data/ground_truth.csv` with columns `question,document_id` with ≥210 entries
- [x] All original 20 manual entries are replaced (files are overwritten entirely)
- [x] Re-running the script overwrites cleanly (no duplicates accumulate)
- [x] `src/ingest.py` adds a deterministic `id` field to each FAQ document in search index
- [x] `src/search.py` includes `id` in returned result dicts
- [x] `src/evaluate.py` refactored to match ground truth by `document_id` instead of question text
- [x] `pyproject.toml` has `tqdm` and `pydantic` added to dependencies
- [x] `uv run pytest` still passes (tests updated for ID-based matching)

### Out of scope

- Cross-mapping queries to multiple relevant documents (each maps to exactly one FAQ)
- Any UI or CLI flags beyond the defaults — the script runs with zero arguments
- Any changes to the LLM fallback logic in `src.llm`

### Constraints

- Uses `src.faqs.load_faqs()` for FAQ loading; LLM call is OpenAI SDK directly (not `ask_llm()`)
- One LLM call per FAQ (not one giant call) — structured output returns a typed list of variants
- Output format is CSV: `question,document_id` (not the old JSON schema)
- Each FAQ must have a stable `id` derived from its index/position in the FAQ list
- API errors are retried with exponential backoff; persistent failures produce a clear terminal message

---

## 9.2. Update exploration notebook with evaluation metrics

### Goal

The `exploration.ipynb` notebook demonstrates every closed task from 1–9; this task adds a section for Tasks 9–9.1's evaluation module (`src.evaluate`) so the notebook covers the full retrieval evaluation pipeline.

### Acceptance criteria

- [x] A new top-level markdown heading "9. Evaluation" exists between "8. Postgres Logging Layer" and the existing "10. Cleanup" sections
- [x] The former "9. Cleanup" section is renumbered to "10. Cleanup"
- [x] `load_ground_truth`, `compute_hit_rate`, `compute_mrr`, and `evaluate_retrieval` are imported from `src.evaluate`
- [x] A code cell loads the ground truth CSV and prints the entry count and a few sample rows
- [x] A code cell demonstrates `compute_hit_rate()` and `compute_mrr()` on a small set of known results
- [x] A code cell calls `evaluate_retrieval()` with the `db/` indexes and displays hit rate@k and MRR@k with per-query details
- [x] A markdown cell notes that ground truth can be regenerated via `uv run python generate_ground_truth.py`
- [x] All cells run top-to-bottom without errors
- [x] No duplicated logic — everything comes from `src.*` imports

### Out of scope

- Running LLM-as-a-judge evaluation (Task 10)
- Re-generating ground truth within the notebook (handled by `generate_ground_truth.py` standalone)
- Any changes to the evaluation logic itself

### Constraints

- Notebook must import from `src.*` modules — no inlined logic
- Sequential run assumed: sections 1–5 must have already built the indexes in `db/`
- API-key-dependent cells must use `os.environ.get(...)` and fail gracefully
- Follow the same coding conventions as the rest of the project (no pandas)

---

## 10.1. Update exploration notebook with LLM-as-a-judge

### Goal

The `exploration.ipynb` notebook demonstrates every closed task from 1–10; this task adds a section for Task 10's LLM-as-a-judge relevance scoring (`src/evaluate_llm.py`) and renumbers the existing keyword search section so the notebook reflects the correct task numbering.

### Acceptance criteria

- [ ] `src/evaluate_llm.py` exists with `evaluate_relevance()` that:
  - Loads ground truth questions from `data/ground_truth.csv`
  - Runs each through `answer_question()` for real RAG answers
  - Sends each (question, answer, context) to an LLM judge via `ask_llm()` for composite scoring (relevance 1-5, faithfulness 1-5)
  - Default judge model: `gpt-5.4-mini` (OpenAI), falls back to Groq
  - Accepts `sample=` for limiting queries; graceful skip when no API key set
  - Returns aggregate report with mean scores, distribution, per-query details
- [ ] A new `## 10. LLM-as-a-judge Relevance Scoring` section exists in the notebook between sections 9 and 11
- [ ] Section 10 demonstrates `evaluate_relevance()` and compares relevance/faithfulness scores with retrieval metrics (hit rate/MRR) from section 9
- [ ] `## 10. Keyword Search Evaluation` is renumbered to `## 11. Keyword Search Evaluation`
- [ ] All cells run top-to-bottom without errors
- [ ] No duplicated logic — everything comes from `src.*` imports
- [ ] API-key-dependent cells use `os.environ.get(...)` and fail gracefully
- [ ] `tests/test_evaluate_llm.py` exists with tests for report structure (no mocking, graceful skip when no key)

### Out of scope

- Changes to the retrieval evaluation logic or ground truth data
- Any UI, dashboard, or visualisation beyond printed tables

### Constraints

- Notebook must import from `src.*` modules — no inlined logic
- Judge calls use `ask_llm()` (respects Groq rate limits, fallback chain)
- Sequential run assumed: sections 1–5 must have already built indexes in `db/`
- No new dependencies

---

## 10. Evaluation: LLM-as-a-judge relevance scoring

Goal: Score answer quality by asking an LLM to rate relevance.

Description: Write a script that takes the test questions, runs them through the full RAG pipeline, then sends each (question, answer, context) triplet to an LLM judge prompt and records a relevance score (e.g. 1-5). Aggregate results into a report.

---

## 11. Chat UI

### Goal

Build a FastAPI chat backend with a self-contained static frontend (chat input, answer display, thumbs up/down feedback) that can later be embedded in a website.

### Acceptance criteria

- [ ] `app.py` is a FastAPI app exposing `POST /api/chat` and `POST /api/feedback`
  - `POST /api/chat` runs `answer_question()` and returns answer, contexts, model, provider, latency, cost, tokens, `interaction_id`
  - `POST /api/feedback` persists thumbs up/down via `update_feedback()`; unknown id → 404; invalid feedback → 422
- [ ] `answer_question()` in `src/rag.py` returns the Postgres `interaction_id` so feedback can be persisted (`None` when logging is skipped or fails)
- [ ] `GET /health` returns 200
- [ ] CORS enabled so the API can be called from a website
- [ ] Static UI in `ui/` with no build step: text input at the bottom, conversation history above, thumbs up/down on each assistant response, metadata (model, latency, tokens, cost, contexts) in an expandable section
- [ ] Thumbs up/down persist via `/api/feedback`; disabled with a hint when `interaction_id` is null
- [ ] Failed LLM calls show a graceful error message in the UI
- [ ] `uv run pytest` passes (new `tests/test_api.py`, updated `tests/test_rag.py`)

### Out of scope

- Monitoring dashboard (handled in task 12)
- Multi-turn conversation memory — history is display-only

### Constraints

- FastAPI + static HTML/JS/CSS (no Streamlit)
- New dependencies: `fastapi`, `uvicorn` (runtime), `httpx` (dev)

---

## 12. Grafana monitoring dashboard

Goal: A Grafana dashboard provisioned from files in `grafana/` that visualises the RAG interactions logged to Postgres (`rag_logs`), with at least 6 panels covering recent conversations, feedback, latency, token usage, estimated cost, and model usage — and whose correctness is verified in a real browser through Chrome DevTools MCP.

### Acceptance criteria

Config (checkable by inspecting files + running pytest):

- [ ] `grafana/dashboard.json` exists, is valid JSON, and is loaded by Grafana via a provisioning dashboards provider with no manual import
- [ ] `grafana/provisioning/datasources/postgres.yml` provisions a Postgres datasource: name "PostgreSQL", type `postgres`, url `postgres:5432`, database `rag_logs`, `isDefault: true`
- [ ] `grafana/provisioning/dashboards/dashboards.yml` provisions the dashboard from `grafana/dashboard.json` with `allowUiUpdates` disabled
- [ ] The dashboard contains at least 6 panels: (1) Recent conversations — table of last 5 rows: question, answer, feedback, model, created_at; (2) Feedback distribution — pie chart counting `up`, `down`, and missing feedback; (3) Average latency over time — timeseries of AVG(`metadata->>'latency'`); (4) Token usage over time — timeseries of SUM(`metadata->'tokens'->>'total'`); (5) Estimated cost over time — timeseries of SUM(`metadata->>'cost'`); (6) Model usage — bar chart counting by `metadata->>'model'`
- [ ] Every panel reads only from the existing `rag_logs` table, extracts metrics from the JSONB `metadata` column, and filters `created_at` with `$__timeFrom()` / `$__timeTo()`
- [ ] Panels render an empty state without errors when `rag_logs` has no rows
- [ ] `tests/test_grafana.py` passes under `uv run pytest`: asserts the dashboard JSON is valid, all 6 required panel titles exist, every panel targets `rag_logs` via the Postgres datasource, and the provisioning YAMLs parse with the correct settings

Browser verification (via Chrome DevTools MCP):

- [ ] With Postgres running, `rag_logs` seeded with known sample rows, and Grafana started from the provisioning files, an agent using the Chrome DevTools MCP tools can:
  - [ ] navigate to http://localhost:3000 and log in with admin credentials
  - [ ] confirm the Postgres datasource shows as connected (green)
  - [ ] open the dashboard and verify each of the 6 panels renders data matching the seeded rows (seeded questions appear in the table; up/down split matches; latency/tokens/cost series are non-empty; model bar shows the seeded models)
  - [ ] capture a screenshot of the rendered dashboard as evidence

### Out of scope

- Logging/feedback implementation — done in Task 8 (#8)
- Chat UI — Task 11 (#11)
- Adding a `grafana` service to `docker-compose.yml` and a `Dockerfile` — Task 13 (#13). Grafana is started ad-hoc (`docker run`) only for verification; compose is not modified here
- Alerts, notifications, SSO, or auth beyond Grafana defaults
- Schema changes to `rag_logs` (no new columns — panels must use the existing JSONB `metadata`)
- Deployment and README — Tasks 13–14 (#13, #14)

### Constraints

- Files confined to `grafana/` (`dashboard.json`, `provisioning/datasources/postgres.yml`, `provisioning/dashboards/dashboards.yml`, `seed_demo.py`) and `tests/test_grafana.py`
- Provisioning via Grafana provisioning files only — no `init.py` HTTP-API script
- `feedback` is TEXT (`up`/`down`/NULL); `metadata` is JSONB with keys `provider`, `model`, `tokens.{prompt,completion,total}`, `latency`, `cost` (see `src/db.py`, `src/rag.py`)
- No new Python dependencies; `uv run pytest` must stay green
- Do not modify `docker-compose.yml`, `app.py`, or `src/` (shipped by Tasks 8/11)

### Test plan

1. `docker compose up postgres` (existing compose)
2. Seed `rag_logs` with ~10 deterministic rows (mixed feedback, models, latency/cost/tokens, recent `created_at`) via `uv run python grafana/seed_demo.py`
3. Start Grafana ad-hoc: `docker run -p 3000:3000` mounting `grafana/provisioning` and `grafana/dashboard.json`
4. `uv run pytest tests/test_grafana.py` (config assertions)
5. Agent executes the browser-verification ACs with Chrome DevTools MCP (navigate → login → datasource check → per-panel check → screenshot)

---

## 13. Docker Compose packaging

Goal: Package the app, Postgres, and Grafana so everything starts with `docker compose up`, with a container that is portable across hosts (`$PORT`, env-only config, graceful DB fallback) and small enough for free-tier clouds (target ~512MB).

Description: Write a `Dockerfile` for the FastAPI app (served via uvicorn), a `docker-compose.yaml` that defines `app`, `postgres`, and `grafana` services with proper environment variables, volume mounts, and network config. Add a `.env.example` file.

### Acceptance criteria

Dockerfile (config assertions in `tests/test_docker.py`, no Docker daemon required):

- [x] Multi-stage `Dockerfile` builds with `uv sync --frozen --no-dev` so the runtime image contains only the compiled `.venv` and app code, not the package manager or build caches
- [x] Embeddings run on ONNX Runtime via `fastembed` (no torch/sentence-transformers/CUDA in the image) — this is the dominant size lever (~800MB+ of torch/scipy/transformers removed) and brings the image near the ~512MB free-tier target
- [x] The builder stage strips `*.so` debug symbols and deletes `*.pyi` stubs / `__pycache__` from `.venv` so the runtime layer carries only the compiled libraries
- [x] Runtime image is `python:3.11-slim`-based, runs as a non-root `app` user (uid 1000), and sets `PYTHONUNBUFFERED=1` / `PYTHONDONTWRITEBYTECODE=1`
- [x] App serves via `uvicorn app:app` on `0.0.0.0` on port `$PORT` (default `8000`) so the same image runs on any host that injects `PORT`
- [x] Indexes are built at container startup when `db/` files are missing: `docker-entrypoint.sh` calls `src.ingest.build_indexes()` if any of `bm25_index.pkl` / `faiss_index.bin` / `ingest_docs.json` is absent, then `exec`s uvicorn
- [x] `HEALTHCHECK` hits `GET /health` using stdlib `urllib` (no curl in slim) with a long `--start-period` for first-run model download
- [x] `.dockerignore` excludes `.env`, `db/`, `.venv`, `.git`, tests, and docs so no secrets or local indexes leak into the image

docker-compose.yaml:

- [x] `docker-compose.yaml` (renamed from `docker-compose.yml`) defines `app`, `postgres`, and `grafana` services on a shared network
- [x] `app`: builds from `.`, exposes `${APP_PORT:-8000}:8000`, gets `GROQ_API_KEY` / `OPENAI_API_KEY` from `.env`, sets `DATABASE_URL` to the `postgres` host, mounts a named `appdb` volume on `/app/db` for persisted indexes, `depends_on: [postgres]`, `restart: unless-stopped`
- [x] `postgres`: `postgres:16` with `POSTGRES_USER/PASSWORD/DATABASE=rag_logs` matching the Grafana datasource, `pgdata` volume, port `5432`
- [x] `grafana`: mounts `grafana/provisioning` → `/etc/grafana/provisioning` and `grafana/dashboard.json` → `/var/lib/grafana/dashboards/dashboard.json` read-only, admin credentials via `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD`, `depends_on: [postgres]`, port `3000`
- [x] Named volumes declared (`pgdata`, `appdb`); `.env.example` lists `GROQ_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`, `GRAFANA_ADMIN_USER`, `GRAFANA_ADMIN_PASSWORD`

Verification:

- [x] `tests/test_docker.py` passes under `uv run pytest` (config-level assertions on the Dockerfile, compose YAML, `.env.example`, and entrypoint)
- [x] `docker compose config` parses cleanly; `docker compose up --build -d` starts all three services; `/health` returns 200, a chat round-trip works, and Grafana loads at `:3000`
- [x] Image size reported via `docker images` and noted against the ~512MB free-tier target

Image size: `llm-zoomcamp-rag-app` = 618MB (down from 1.93GB with the torch stack, which was itself down from 8.8GB with default CUDA torch). The torch stack was replaced with `fastembed` (ONNX Runtime), dropping `torch` (~527MB), `scipy` (~121MB), `transformers` (~113MB), `sympy` (~80MB), `scikit-learn` (~59MB), `networkx`, and `sentence-transformers`. The builder stage additionally strips `*.so` debug symbols (excluding bundled `*.libs`), removes `*.pyi` stubs and `__pycache__`, and the runtime keeps `UV_LINK_MODE=copy` so no caches cross stages. The remaining ~618MB is dominated by hard dependencies: base image + system libs, `numpy` (~69MB), `duckdb` (~52MB), `onnxruntime` (~51MB), `faiss` (~31MB), `pillow` (~25MB), `openai` (~20MB) — with this embedding stack the floor is ~600MB, within ~20% of the ~512MB target.

### Out of scope

- Individual service implementation
- CI/CD pipelines or deploy tooling (Render blueprint, Fly.io config, etc.) — just a portable image + compose
- README documentation — Task 14 (#14)

### Constraints

- Use Docker Compose
- Replace `sentence-transformers` + `torch` with `fastembed` (ONNX Runtime) — the app runs on CPU and only needs 384-dim text embeddings from `all-MiniLM-L6-v2`; the swap removes ~1GB of torch/scipy/transformers and drops the pytorch-cpu index pin from `pyproject.toml`
  - Multi-stage build with uv: runtime gets only the compiled `.venv` + app code; `--no-cache-dir`, `rm -rf /root/.cache`, `find ... -delete` for `*.pyi` / `__pycache__`, and `strip --strip-unneeded` on `*.so` (excluding bundled `*.libs` to avoid breaking numpy's page-aligned OpenBLAS) so caches and debug symbols never cross stages
  - `python:3.11-slim` base; keep `faiss-cpu`, single model (`all-MiniLM-L6-v2`), stdlib urllib healthcheck
- `app.py` and `src/` are only touched where the embedding backend swaps (`src/ingest.py`, `src/search.py`); DB logging and index paths are already graceful and portable
- Rename `docker-compose.yml` → `docker-compose.yaml` (compose auto-detects both names)

---

## 14. README with setup and run instructions

Goal: Write a clear README so a new developer can clone and run the project in under two minutes.

Description: Document prerequisites (Docker, API keys), setup steps (`cp .env.example .env`, fill in keys), and run instructions (`docker compose up`). Include a quick-start section, project structure overview, and links to the plan and task backlog.

---

## 15. Error handling and polish pass

Goal: Add sensible error handling, logging, and defaults across the codebase.

Description: Wrap API calls and file operations in try/except blocks with user-friendly fallback messaging. Add structured logging (`import logging`) to key modules. Set sensible defaults for config values (model names, chunk sizes, port numbers) so the project runs out of the box with minimal configuration.

---

## 16. Keyword search evaluation with tunable field weights

### Goal

Add DuckDB-backed keyword search (SQL ILIKE) to the evaluation pipeline alongside the existing hybrid BM25+FAISS search, with tunable per-field weights.

### Acceptance criteria

- [x] `keyword_search(query, k, db_path, docs_path, field_weights)` added to `src/search.py`
  - Runs SQL ILIKE on DuckDB `faq.faq_resource` across `Question`, `Answer`, `Category`
  - `field_weights` controls per-column scoring (default: `{"Question": 2.0, "Answer": 1.0}`)
  - Returns same result format as `search()`
- [x] `evaluate_retrieval()` in `src/evaluate.py` accepts a pluggable `search_fn=` parameter
  - Defaults to hybrid `search()` for backward compatibility
  - Extra kwargs forwarded to the search function
- [x] Notebook cell evaluates keyword search with `evaluate_retrieval(search_fn=keyword_search, db_path=..., docs_path=...)`
- [x] Notebook cell compares hybrid vs keyword hit rate@k and MRR@k side by side
- [x] Notebook cell demonstrates tuning `field_weights` and its effect on results
- [x] All cells run top-to-bottom without errors

### Out of scope

- Replacing or removing the existing hybrid search
- Running DuckDB without the dlt pipeline having been executed first
- Changes to the ground truth or metric computation logic

### Constraints

- Keyword search is DuckDB-only (no Postgres)
- Sequential run assumed: sections 1–3 must have already loaded DuckDB in `db/`
- No new dependencies

---

## 17. Retrieval hyperparameter tuning

### Goal

Parameterise and sweep the three knobs that control retrieval quality — RRF K, k (top‑k cutoff), and BM25 (k1, b) — and report which settings maximise hit rate@k and MRR@k for hybrid search, and how sensitive each metric is to the chosen values.

### Acceptance criteria

- [x] `RRF_K` made a parameter of `search()` (default `60`) instead of a module‑level constant
- [x] A notebook cell or script sweeps `RRF_K ∈ [10, 30, 60, 100]` and prints hit rate@k + MRR@k for each
- [x] A notebook cell or script sweeps `k ∈ [1, 3, 5, 10]` for hybrid and keyword search, printing a table of metric vs. k
- [x] A notebook cell or script sweeps BM25 `k1 ∈ [0.5, 1.0, 1.5, 2.0]` and `b ∈ [0.5, 0.75, 1.0]` and prints the best‑performing (k1, b) combination with its metrics
- [x] Rebuilds the BM25 index each time k1/b changes (or accepts the parameters at query time if the library supports it)
- [x] Uses `evaluate_retrieval(search_fn=…)` for all sweeps
- [x] All cells run top‑to‑bottom without errors

### Out of scope

- Changes to the `keyword_search()` function
- Changes to the evaluation metric computation
- GUI or interactive controls — plain printed tables
- Persisting tuned defaults to config or environment

### Constraints

- Only `rank_bm25` BM25Okapi parameters — no forking or reimplementing BM25
- Sweeps run inside the existing notebook (no separate script)
- Sequential run assumed: sections 1–5 must have already built indexes and DuckDB in `db/`
- No new dependencies

---

## 9.3. Category weight tuning for retrieval

### Goal

Add a `cat_weight` parameter to hybrid search and sweep category field weight for keyword search, so retrieval can be tuned to prefer or deprioritise category‑term overlap independent of the main BM25+FAISS signal.

### Acceptance criteria

- [x] `search()` in `src/search.py` accepts `cat_weight=0` parameter
  - Tokenizes the query and compares term overlap with each document's `category` field
  - Adds the category‑match score as a third entry point in the RRF fusion (alongside BM25 and FAISS ranks)
  - Default `0` means no category boost (backward compatible)
- [x] A notebook cell sweeps `cat_weight ∈ [0, 0.5, 1, 2, 3]` for hybrid search and prints hit rate@k + MRR@k for each
- [x] A notebook cell sweeps keyword-search Category field weight `∈ [0, 0.5, 1, 2, 3]` with `Question=1, Answer=1` fixed and prints metrics
- [x] Both sweeps use `evaluate_retrieval(search_fn=…)`
- [x] All cells run top‑to‑bottom without errors

### Out of scope

- Changes to the index‑building process (category is extracted at query time from the stored docs metadata)
- Changes to the evaluation metric computation
- GUI or interactive controls

### Constraints

- Category overlap is computed as a simple ratio: matching terms / total distinct query terms
- Default `cat_weight=0` must not change existing hybrid search behaviour
- Sequential run assumed: sections 1–5 must have already built indexes and DuckDB in `db/`
- No new dependencies
