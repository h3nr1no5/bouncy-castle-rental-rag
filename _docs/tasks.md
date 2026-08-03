# Task Backlog

## 1. Project scaffold with passing test

### Goal

Set up an empty project skeleton that installs and passes a smoke test: a
directory structure, a minimal `pyproject.toml` (or `requirements.txt`) with key
dependencies, a placeholder `src/` package, and a trivial test (e.g.
`assert True`) runnable with `pytest`.

---

## 2. Load and inspect FAQ CSV

### Goal

Provide a `load_faqs()` function in `src/faqs.py` that reads `data/faq.csv`,
strips whitespace from values, and returns structured data for downstream
ingestion (tasks 3–4).

### Acceptance criteria

- [x] `src/faqs.py` exports `load_faqs()` returning `list[dict[str, str]]` with keys `Category`, `Question`, `Answer`
- [x] CSV cell values have leading/trailing whitespace stripped
- [x] `python -m src.faqs` prints `Rows: 41` and the column names
- [x] `tests/test_faqs.py` verifies the CSV is non-empty (41 rows), every row has all three non-empty string keys, and a missing CSV raises `FileNotFoundError` with a clear message
- [x] The CSV path defaults to `<project_root>/data/faq.csv` and is overridable via an optional `path` argument

---

## 3. dlt ingestion pipeline

### Goal

Set up an idempotent dlt pipeline that reads the FAQ CSV, applies any needed
transformations, and loads normalized data into Postgres without duplicating
rows on re-run; add a test verifying the end-to-end run and expected rows.

---

## 4. Build hybrid search index (ingestion)

### Goal

Create `src/ingest.py` that reads FAQs via `load_faqs()`, builds a BM25 index
(`rank_bm25`) + FAISS vector index (`all-MiniLM-L6-v2` via `fastembed`), and
saves both to disk so they don't rebuild on every run.

---

## 5. Implement hybrid retrieval with RRF

### Goal

Given a user question, return the top-k FAQ entries via hybrid search +
reciprocal rank fusion. `src/search.py` loads the persisted BM25 and FAISS
indexes, runs both retrievers, and merges results via RRF, exposing
`search(query, k=5)` that returns ranked entries with scores.

---

## 5.1. Jupyter notebook for closed tasks

### Goal

An `exploration.ipynb` notebook at the project root demonstrating every function
from the completed tasks (1–5) with explanatory markdown, so the ingest–search
pipeline is understandable without running the test suite.

### Acceptance criteria

- [x] `exploration.ipynb` exists at the project root, with markdown explaining what each cell does and why
- [x] Section 1 — Setup: imports from `src.faqs`, `src.pipeline`, `src.ingest`, `src.search`; create a `db/` working directory
- [x] Section 2 — Load FAQ Data: calls `load_faqs()`, prints row count (41), displays 3 sample rows, shows missing-value stats
- [x] Section 3 — dlt Pipeline: runs `run_pipeline()` into `db/faq_ingestion.duckdb`, queries with `duckdb`, re-runs and confirms the row count stays the same (idempotency)
- [x] Section 4 — Build Indexes: `build_indexes(force=True)` with paths under `db/`; inspects BM25 count, FAISS dims (ntotal, d), and docs JSON
- [x] Section 5 — Hybrid Search: `search()` with ≥3 queries (e.g. "booking", "cost", "safety"), printed RRF scores, `k=1` vs `k=5`, and an empty-query edge case
- [x] Section 6 — Cleanup (optional): skipped — `db/` is persistent
- [x] `pyproject.toml` has `ipykernel` and `notebook` in dev dependencies
- [x] All cells run top-to-bottom without errors

---

## 6. LLM client with Groq + OpenAI fallback

### Goal

Create `src/llm.py`: a thin client that accepts a system prompt and user message,
attempts Groq first, falls back to OpenAI on failure (rate limits, timeouts,
auth), and returns the response text plus metadata (model, latency, tokens).

---

## 6.1. Update exploration notebook with LLM client

### Goal

Add a section to `exploration.ipynb` for the Task 6 LLM client (`src/llm.py`) so
the notebook covers all completed work.

### Acceptance criteria

- [x] New top-level markdown heading "6. LLM Client" between "5. Hybrid Search" and the existing "6. Cleanup" (former renumbered to "7. Cleanup")
- [x] `ask_llm` imported from `src.llm`
- [x] Code cell attempts a real `ask_llm()` call with a FAQ-related system prompt and question; missing API keys print a graceful message instead of crashing
- [x] Returned metadata (provider, model, latency, token counts) displayed
- [x] Rate-limit constants (`GROQ_RPM_LIMIT`, `GROQ_RPD_LIMIT`) shown and `_enforce_groq_rate_limits()` explained in markdown
- [x] All cells run top-to-bottom without errors; no duplicated logic (`src.*` imports only); no new dependencies

---

## 7. RAG orchestration pipeline

### Goal

Wire retrieval + LLM into a single `answer_question()` in `src/rag.py`: takes a
user question, calls `search()` for context, builds a prompt with the retrieved
entries, calls the LLM client, and returns answer, contexts, model, latency, and
token usage.

---

## 7.1. Update exploration notebook with RAG pipeline

### Goal

Add a section to `exploration.ipynb` for Task 7's `answer_question()` so the
notebook covers the full RAG pipeline end-to-end.

### Acceptance criteria

- [x] New top-level markdown heading "7. RAG Pipeline" between "6. LLM Client" and the existing "8. Cleanup" (former renumbered to "8. Cleanup")
- [x] `answer_question` imported from `src.rag`
- [x] Code cell calls `answer_question()` on a sample question using the `db/` indexes from section 4; missing API keys print a graceful message
- [x] Returned metadata (answer, contexts, model, provider, latency, tokens) displayed
- [x] Code cell demonstrates the empty-context edge case (search returns no results)
- [x] All cells run top-to-bottom without errors; no duplicated logic (`src.*` imports only)

---

## 8. Postgres logging layer

### Goal

Write `src/db.py`: a Postgres connection layer (`asyncpg`/`psycopg2`) with a
`rag_logs` table (question, answer, feedback, metadata, timestamp) and
`log_interaction()` / `update_feedback()` functions.

---

## 8.1. Update exploration notebook with Postgres logging layer

### Goal

Add a section to `exploration.ipynb` for Task 8's Postgres logging layer
(`src/db.py`).

### Acceptance criteria

- [x] New top-level markdown heading "8. Postgres Logging Layer" between "7. RAG Pipeline" and the existing "9. Cleanup" (former renumbered to "10. Cleanup")
- [x] `init_db`, `log_interaction`, `update_feedback`, `get_connection`, and `CREATE_TABLE_SQL` imported from `src.db`
- [x] The `rag_logs` table schema (DDL) is printed for inspection
- [x] Code cell attempts a real Postgres connection: `init_db()`, `log_interaction()`, `update_feedback()`; missing `DATABASE_URL` or connection failure prints a graceful message
- [x] All cells run top-to-bottom without errors; no duplicated logic (`src.*` imports only)

---

## 8.2. Wire Postgres logging into the RAG pipeline

### Goal

Wire `log_interaction()` into `answer_question()` (`src/rag.py`) so every real
RAG interaction is logged to Postgres automatically.

### Acceptance criteria

- [x] `answer_question()` calls `log_interaction()` after each successful LLM call with the real question, answer, and metadata (provider, model, tokens, latency, cost)
- [x] Logging fails gracefully — no crash if `DATABASE_URL` is unset or the connection fails
- [x] `init_db()` runs lazily on first use to ensure the table exists
- [x] The notebook's section 8 demo works unchanged
- [x] `uv run pytest` still passes

---

## 9. Evaluation: retrieval metrics

### Goal

Create an evaluation script that loads a small ground-truth question set with
known relevant FAQ IDs, runs `search()` for each, and reports hit rate@k and
MRR@k, saving results for later comparison.

---

## 9.1. Generate ground truth questions from FAQs

### Goal

Replace the manual `data/ground_truth.json` with a script that uses LLM
structured output to generate ≥5 natural-language query variants per FAQ
(~210+ entries) into `data/ground_truth.csv` (`question,document_id`), and use
document IDs throughout the retrieval pipeline.

### Acceptance criteria

- [x] `generate_ground_truth.py` exists at project root, runnable via `uv run python generate_ground_truth.py`
- [x] Loads all 41 FAQs from `data/faq.csv` via `src.faqs.load_faqs()`
- [x] Uses OpenAI structured output (`client.responses.parse()` with Pydantic `Questions`), not `ask_llm()`
- [x] Model `gpt-5.4-mini`, pricing displayed after run ($0.75/M input, $4.50/M output)
- [x] Retry logic with exponential backoff; parallel processing via `ThreadPoolExecutor` + `tqdm`
- [x] Writes `data/ground_truth.csv` with columns `question,document_id` and ≥210 entries; re-running overwrites cleanly (no duplicates)
- [x] `src/ingest.py` adds a deterministic `id` to each FAQ document; `src/search.py` includes `id` in returned result dicts
- [x] `src/evaluate.py` matches ground truth by `document_id` instead of question text
- [x] `pyproject.toml` has `tqdm` and `pydantic` added to dependencies
- [x] `uv run pytest` still passes (tests updated for ID-based matching)

---

## 9.2. Update exploration notebook with evaluation metrics

### Goal

Add a section to `exploration.ipynb` for the evaluation module (`src.evaluate`)
so the notebook covers the full retrieval evaluation pipeline.

### Acceptance criteria

- [x] New top-level markdown heading "9. Evaluation" between "8. Postgres Logging Layer" and the existing "10. Cleanup" (former renumbered to "10. Cleanup")
- [x] `load_ground_truth`, `compute_hit_rate`, `compute_mrr`, and `evaluate_retrieval` imported from `src.evaluate`
- [x] Code cell loads the ground truth CSV and prints the entry count and a few sample rows
- [x] Code cell demonstrates `compute_hit_rate()` and `compute_mrr()` on a small set of known results
- [x] Code cell calls `evaluate_retrieval()` with the `db/` indexes, showing hit rate@k, MRR@k, and per-query details
- [x] Markdown cell notes ground truth can be regenerated via `uv run python generate_ground_truth.py`
- [x] All cells run top-to-bottom without errors; no duplicated logic (`src.*` imports only)

---

## 10.1. Update exploration notebook with LLM-as-a-judge

### Goal

Add a section to `exploration.ipynb` for LLM-as-a-judge relevance scoring
(`src/evaluate_llm.py`) and renumber the keyword search section so numbering
matches the tasks.

### Acceptance criteria

- [x] `src/evaluate_llm.py` with `evaluate_relevance()`: loads ground truth, runs each question through `answer_question()`, LLM-judges each (question, answer, context) for relevance + faithfulness (1-5), default judge `gpt-5.4-mini` (OpenAI, Groq fallback), `sample=` limit, graceful skip without API key, returns aggregate report
- [x] New `## LLM-as-a-judge Relevance Scoring` section in the notebook demonstrating `evaluate_relevance()` and comparing relevance/faithfulness scores with hit rate/MRR from section 9
- [x] Keyword Search Evaluation section renumbered to `## 10. Keyword Search Evaluation`
- [x] All cells run top-to-bottom without errors; no duplicated logic (`src.*` imports only)
- [x] API-key-dependent cells use `os.environ.get(...)` and fail gracefully
- [x] `tests/test_evaluate_llm.py` covers report structure (no mocking, graceful skip without key)

---

## 10. Evaluation: LLM-as-a-judge relevance scoring

### Goal

Score answer quality by running test questions through the full RAG pipeline,
sending each (question, answer, context) to an LLM judge for a relevance score
(1-5), and aggregating results into a report.

---

## 11. Chat UI

### Goal

Build a FastAPI chat backend with a self-contained static frontend (chat input,
answer display, thumbs up/down feedback) that can later be embedded in a
website.

### Acceptance criteria

- [x] `app.py` is a FastAPI app exposing `POST /api/chat` and `POST /api/feedback`
  - `POST /api/chat` runs `answer_question()` and returns answer, contexts, model, provider, latency, cost, tokens, `interaction_id`
  - `POST /api/feedback` persists thumbs up/down via `update_feedback()`; unknown id → 404; invalid feedback → 422
- [x] `answer_question()` returns the Postgres `interaction_id` so feedback can be persisted (`None` when logging is skipped or fails)
- [x] `GET /health` returns 200
- [x] CORS enabled so the API can be called from a website
- [x] Static UI in `ui/` with no build step: text input at the bottom, conversation history above, thumbs up/down on each assistant response, expandable metadata section
- [x] Thumbs up/down persist via `/api/feedback`; disabled with a hint when `interaction_id` is null
- [x] Failed LLM calls show a graceful error message in the UI
- [x] `uv run pytest` passes (new `tests/test_api.py`, updated `tests/test_rag.py`)

---

## 12. Grafana monitoring dashboard

### Goal

A Grafana dashboard provisioned from `grafana/` that visualises the RAG
interactions logged to Postgres (`rag_logs`) with at least 6 panels, verified in
a real browser through Chrome DevTools MCP.

### Acceptance criteria

Config (file inspection + `uv run pytest`):

- [x] `grafana/dashboard.json` exists, is valid JSON, and is loaded via a provisioning dashboards provider with no manual import
- [x] `grafana/provisioning/datasources/postgres.yml`: name "PostgreSQL", type `postgres`, url `postgres:5432`, database `rag_logs`, `isDefault: true`
- [x] `grafana/provisioning/dashboards/dashboards.yml` provisions the dashboard with `allowUiUpdates` disabled
- [x] ≥6 panels: (1) Recent conversations table (last 5 rows); (2) Feedback distribution pie chart (`up`/`down`/missing); (3) Average latency over time; (4) Token usage over time; (5) Estimated cost over time; (6) Model usage bar chart
- [x] All panels read only from `rag_logs`, extract metrics from the JSONB `metadata` column, and filter `created_at` with `$__timeFrom()` / `$__timeTo()`
- [x] Panels render an empty state without errors when `rag_logs` has no rows
- [x] `tests/test_grafana.py` passes: valid dashboard JSON, all 6 required panel titles, every panel targets `rag_logs` via the Postgres datasource, provisioning YAMLs parse with the correct settings

Browser verification (Chrome DevTools MCP):

- [x] With Postgres running, `rag_logs` seeded with known sample rows, and Grafana started from the provisioning files: navigate to `:3000`, log in with admin credentials, confirm the datasource shows as connected (green), verify all 6 panels render data matching the seeded rows, and capture a screenshot as evidence

---

## 13. Docker Compose packaging

### Goal

Package the app, Postgres, and Grafana so everything starts with `docker compose
up`, with a portable (env-only, `$PORT`), small (~512MB target) container.

### Acceptance criteria

Dockerfile (config assertions in `tests/test_docker.py`, no Docker daemon required):

- [x] Multi-stage build with `uv sync --frozen --no-dev`; runtime image has only the compiled `.venv` + app code
- [x] Embeddings run on ONNX Runtime via `fastembed` — no torch/sentence-transformers/CUDA (dominant size lever, ~800MB removed)
- [x] Builder stage strips `*.so` debug symbols and deletes `*.pyi` stubs / `__pycache__` from `.venv`
- [x] Runtime `python:3.11-slim`, non-root `app` user (uid 1000), `PYTHONUNBUFFERED=1` / `PYTHONDONTWRITEBYTECODE=1`
- [x] App serves via `uvicorn app:app` on `0.0.0.0` port `$PORT` (default 8000)
- [x] Entrypoint builds indexes at startup when `db/` files are missing, then `exec`s uvicorn
- [x] `HEALTHCHECK` hits `GET /health` via stdlib `urllib` with a long `--start-period` for first-run model download
- [x] `.dockerignore` excludes `.env`, `db/`, `.venv`, `.git`, tests, and docs

docker-compose.yaml:

- [x] `docker-compose.yaml` defines `app`, `postgres`, and `grafana` on a shared network
- [x] `app`: builds from `.`, `${APP_PORT:-8000}:8000`, keys from `.env`, `DATABASE_URL` → `postgres` host, `appdb` volume on `/app/db`, `depends_on: [postgres]`, `restart: unless-stopped`
- [x] `postgres`: `postgres:16` with env matching the Grafana datasource, `pgdata` volume, port 5432
- [x] `grafana`: mounts `grafana/provisioning` + `dashboard.json` read-only, admin creds via env, `depends_on: [postgres]`, port 3000
- [x] Named volumes (`pgdata`, `appdb`) declared; `.env.example` lists all required vars

Verification:

- [x] `tests/test_docker.py` passes under `uv run pytest`
- [x] `docker compose config` parses; `docker compose up --build -d` starts all services; `/health` 200, a chat round-trip works, Grafana loads at `:3000`
- [x] Image size ~618MB (down from 1.93GB with the torch stack), within ~20% of the ~512MB free-tier target

---

## 14. README with setup and run instructions

### Goal

Write a clear README so a new developer can clone and run the project in under
two minutes: prerequisites (Docker, API keys), setup (`cp .env.example .env`,
fill in keys), run instructions (`docker compose up`), a quick-start section,
project structure overview, and links to the plan and task backlog.

---

## 15. Error handling and polish pass

### Goal

Add sensible error handling, logging, and defaults across the codebase:
try/except with user-friendly fallback messaging, structured logging
(`import logging`) in key modules, and sensible config defaults (model names,
chunk sizes, port numbers) so the project runs out of the box.

---

## 16. Keyword search evaluation with tunable field weights

### Goal

Add DuckDB-backed keyword search (SQL ILIKE) to the evaluation pipeline
alongside the existing hybrid BM25+FAISS search, with tunable per-field weights.

### Acceptance criteria

- [x] `keyword_search(query, k, db_path, docs_path, field_weights)` in `src/search.py`: SQL ILIKE on DuckDB across `Question`, `Answer`, `Category`; `field_weights` per-column scoring (default `{"Question": 2.0, "Answer": 1.0}`); same result format as `search()`
- [x] `evaluate_retrieval()` accepts a pluggable `search_fn=` parameter (defaults to hybrid `search()`), forwarding extra kwargs
- [x] Notebook cell evaluates keyword search via `evaluate_retrieval(search_fn=keyword_search, db_path=..., docs_path=...)`
- [x] Notebook compares hybrid vs keyword hit rate@k and MRR@k side by side
- [x] Notebook demonstrates tuning `field_weights` and its effect on results
- [x] All cells run top-to-bottom without errors

---

## 17. Retrieval hyperparameter tuning

### Goal

Parameterise and sweep the three knobs that control retrieval quality — RRF K,
top-k cutoff, and BM25 (k1, b) — and report which settings maximise hit rate@k
and MRR@k for hybrid search.

### Acceptance criteria

- [x] `RRF_K` is a parameter of `search()` (default 60), not a module-level constant
- [x] Sweep `RRF_K ∈ [10, 30, 60, 100]`, printing hit rate@k + MRR@k for each
- [x] Sweep `k ∈ [1, 3, 5, 10]` for hybrid and keyword search, printing a table of metric vs. k
- [x] Sweep BM25 `k1 ∈ [0.5, 1.0, 1.5, 2.0]` and `b ∈ [0.5, 0.75, 1.0]`, reporting the best (k1, b) with its metrics
- [x] Rebuilds the BM25 index when k1/b changes (or accepts params at query time if supported)
- [x] All sweeps use `evaluate_retrieval(search_fn=…)`
- [x] All cells run top-to-bottom without errors

---

## 9.3. Category weight tuning for retrieval

### Goal

Add a `cat_weight` parameter to hybrid search and sweep the keyword-search
Category field weight, so retrieval can prefer or deprioritise category-term
overlap independent of the main BM25+FAISS signal.

### Acceptance criteria

- [x] `search()` accepts `cat_weight=0`: tokenizes the query, compares term overlap with each doc's `category`, and adds the match as a third entry in RRF fusion (default 0 = backward compatible)
- [x] Notebook sweeps `cat_weight ∈ [0, 0.5, 1, 2, 3]` for hybrid search, printing hit rate@k + MRR@k
- [x] Notebook sweeps keyword-search Category weight `∈ [0, 0.5, 1, 2, 3]` with `Question=1, Answer=1` fixed
- [x] Both sweeps use `evaluate_retrieval(search_fn=…)`
- [x] All cells run top-to-bottom without errors

---

## 18. Deploy to Render (app + Postgres + Grafana)

### Goal

Deploy the RAG app, a managed Postgres (Neon free tier), and Grafana monitoring
to Render so the service runs behind a public URL, reusing the env-only,
port-portable image from Task 13 (no Docker Compose on Render).

### Acceptance criteria

- [x] Render web service deploys the existing `Dockerfile`; `/health` returns 200 at the deployed `$PORT`
- [x] A real chat round-trip works against the deployed URL; feedback persists to the Neon `rag_logs` table (auto-created by `init_db()`)
- [x] Indexes pre-baked into the image at build time; redeploys skip the rebuild (cold start in seconds, no runtime HF Hub download)
- [x] `DATABASE_URL` on Render holds the Neon URL; local `.env` keeps `DATABASE_URL` → local Postgres and `DATABASE_URL_CLOUD` → Neon
- [x] Grafana (2nd Render web service from `grafana/Dockerfile`) renders all 6 panels, with datasource url/user/password/database/sslmode read from env (Grafana `$VAR` interpolation) instead of hardcoded
- [x] Existing `rag_logs` data migrated to Neon via `pg_dump`/`psql` (one-time), verified by row-count and `created_at` range parity
- [x] `uv run pytest` still green; no functional changes to retrieval/eval code
- [x] README updated with the deployed URL, env var setup, and DB-fallback behaviour

---

## 19. Document re-ranking

### Goal

Re-order the retrieved documents by Reciprocal Rank Fusion (RRF) as taught in
the course, so the most relevant FAQ entry surfaces within the top-k; evaluate
whether reranking improves hit rate@k and MRR@k over the non-reranked baseline
and use the better approach.

### Acceptance criteria

- [x] RRF reranking per the lesson: run vector (FAISS) and keyword (BM25) searches separately, compute `compute_rrf(rank, k) = 1 / (k + rank)` per result set, sum per document, sort descending
- [x] Reranking applied at the end of the pipeline — `search()` returns the reranked top-k
- [x] Reranked (hybrid + RRF) results evaluated against `data/ground_truth.csv` via `evaluate_retrieval(search_fn=…)`, reporting hit rate@k and MRR@k
- [x] At least one non-reranked baseline evaluated for comparison; the side-by-side shows whether RRF improves both metrics
- [x] The better approach is the production default (confirm `tuned_params.json` `rrf_k`)
- [x] Notebook cells demonstrate the reranking scores and the before/after metrics
- [x] All cells run top-to-bottom without errors; `uv run pytest` still passes

---

## 20. User query rewriting

### Goal

Reformulate the user's raw question into a better search query with an LLM
before retrieval, so hybrid search surfaces more relevant FAQ entries; evaluate
the rewritten-query pipeline against the raw-query baseline (hit rate@k /
MRR@k) and make the better approach the production default.

### Acceptance criteria

- [x] LLM-based query rewrite using `ask_llm()` (Groq + OpenAI fallback) that expands abbreviations/synonyms, adds domain vocabulary, and normalises phrasing
- [x] Retrieval runs on the rewritten query via `search()`; the raw question is still sent to the LLM for the final answer
- [x] Rewriting degrades safely — falls back to the original question if the rewrite fails or returns empty
- [x] Both pipelines evaluated against `data/ground_truth.csv` via `evaluate_retrieval(search_fn=…)`
- [x] Side-by-side comparison shows whether rewriting improves both metrics over the raw-query baseline
- [x] The better approach is the production default
- [x] Rewrite step wired into `answer_question()` (`src/rag.py`) so the chat UI uses it
- [x] Notebook cells demonstrate the rewritten queries and the before/after metrics
- [x] All cells run top-to-bottom without errors; `uv run pytest` still passes

---

## 21. History-aware multi-turn query rewriting

### Goal

Add `rewrite_query_with_history()` to `src/rag.py` that reformulates the user's
follow-up question using recent conversation history (default 4 turns), so
follow-up queries surface more relevant FAQ entries; evaluate against raw
follow-ups and make it the production default.

### Acceptance criteria

- [x] `rewrite_query_with_history(question, history, history_turns=4)` in `src/rag.py`, using `ask_llm()` (Groq + OpenAI fallback)
- [x] Degrades safely: empty/missing history → single-turn rewrite; rewrite failure/empty → raw question
- [x] Wired into `answer_question()` via `history_rewrite_enabled` / `history_turns`
- [x] Enabled by default (`tuned_params.json`: `history_rewrite_enabled: true`)
- [x] Notebook section 15 demonstrates rewritten queries and before/after metrics
- [x] `uv run pytest` still passes

---

## 22. Conversation history in answers + server-side chat sessions

### Goal

Thread conversation history into answer generation and persist chat sessions
server-side, so follow-up questions get context-aware answers.

### Acceptance criteria

- [x] `answer_question()` accepts `history=` and includes it in the answer-generation prompt
- [x] `ChatRequest` carries `history` / `session_id`; `GET /api/chat/{session_id}/history` returns stored turns
- [x] Sessions persisted via `src/db.get_session_history`; UI keeps a session id in localStorage
- [x] Notebook section 16 demonstrates history vs no-history answer quality
- [x] `uv run pytest` still passes

---

## 23. Multi-turn ground truth generation

### Goal

Generate an LLM-written multi-turn ground-truth set so history-aware rewriting
and answer quality can be evaluated.

### Acceptance criteria

- [x] `generate_ground_truth_multi_turn.py` runnable via `uv run python generate_ground_truth_multi_turn.py`
- [x] OpenAI structured output, 4-attempt exponential-backoff retry, `ThreadPoolExecutor` + `tqdm`
- [x] Writes `data/ground_truth_multi_turn_generated.csv` with columns `conversation_id, prior_user_turns, follow_up_question, document_id`
- [x] Re-running overwrites cleanly (no rows accumulate)
- [x] `load_multi_turn_ground_truth()` in `evaluate_multi_turn_rewrite.py` consumes it unchanged

---

## 24. Multi-turn retrieval & answer-quality evaluation

### Goal

Evaluate multi-turn retrieval (rewritten vs raw follow-up) and answer quality
(history vs no-history) against the multi-turn ground truth.

### Acceptance criteria

- [x] `evaluate_multi_turn_rewrite.py` reports hit rate@k, MRR@k, Recall@k, Precision@k, nDCG@k for raw vs rewritten; supports `--k`, `--k-sweep`, `--limit`, and per-FAQ/category breakdowns
- [x] `evaluate_multi_turn_answer.py` LLM-judges coherence + relevance (1-5) for history vs no-history arms (identical retrieval), with delta and verdict
- [x] `tests/test_evaluate_multi_turn_rewrite.py` and `tests/test_evaluate_multi_turn_answer.py` pass
- [x] Notebook section 16 demonstrates the multi-turn answer-quality comparison

---

## 25. Tuned hyperparameters as production defaults

### Goal

Persist the best retrieval/rewrite hyperparameters from the notebook and load
them at runtime, so tuned settings are the production defaults.

### Acceptance criteria

- [x] `src/config.py` `load_tuned_params()` reads `tuned_params.json` with fallback defaults (k, rrf_k, cat_weight, bm25_k1, bm25_b, rewrite flags, history_turns)
- [x] `src/search.py` and `src/rag.py` read defaults from tuned params
- [x] `tuned_params.json` is written from the notebook (section 17 "Applying the tuned parameters")
- [x] `tests/test_config.py` passes
