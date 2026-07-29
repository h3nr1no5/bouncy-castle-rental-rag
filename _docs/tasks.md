# Task Backlog

## 1. Project scaffold with passing test

Goal: Set up an empty project skeleton that installs and passes a smoke test.

Description: Create the directory structure, a minimal `pyproject.toml` (or `requirements.txt`) with key dependencies, a placeholder `src/` package, and a trivial test (e.g. `assert True`) that can be run with `pytest`. This gives the team a working starting point before any real logic is written.

---

## 2. Load and inspect FAQ CSV

Goal: Read the FAQ CSV file and display basic stats (row count, columns, missing values).

Description: Write a script that loads the CSV from `data/`, prints column names and row count, and exposes a `load_faqs()` function the ingestion module can call. Add a test that verifies the CSV is non-empty and has expected columns.

---

## 3. dlt ingestion pipeline

Goal: Set up a dlt pipeline to ingest the FAQ CSV into a normalized, queryable dataset.

Description: Write a dlt pipeline definition that reads the FAQ CSV via `dlt.sources.filesystem` or a custom `@dlt.source`, applies any needed transformations (rename columns, parse dates, validate non-nulls), and loads the data into Postgres. The pipeline should be idempotent — re-running it upserts or replaces without duplicating rows. Add a test that verifies the pipeline runs end-to-end and the destination contains the expected rows.

---

## 4. Build hybrid search index (ingestion)

Goal: Create an ingestion script that builds a BM25 index + FAISS vector index from the FAQ data.

Description: Write `src/ingest.py` that reads FAQ entries via `load_faqs()`, computes TF-IDF/BM25 tokens with `rank_bm25`, generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`, stores vectors in a FAISS index, and saves both indexes to disk so they don't rebuild on every run.

---

## 5. Implement hybrid retrieval with RRF

Goal: Given a user question, return the top-k FAQ entries using hybrid search + reciprocal rank fusion.

Description: Write `src/search.py` that loads the persisted BM25 and FAISS indexes, runs both retrievers against a query, and merges results via RRF. The module should expose a `search(query, k=5)` callable returning ranked FAQ entries with scores.

---

## 6. LLM client with Groq + OpenAI fallback

Goal: Create a thin LLM client that calls Groq by default and falls back to OpenAI on failure.

Description: Write `src/llm.py` that wraps the `groq` and `openai` Python SDKs. It should accept a system prompt and user message, attempt Groq first, catch errors (rate limits, timeouts, auth failures), and retry with OpenAI. Return the response text plus metadata (model used, latency, token counts).

---

## 7. RAG orchestration pipeline

Goal: Wire retrieval + LLM call into a single `answer_question()` function.

Description: Write `src/rag.py` that takes a user question, calls `search()` for context, builds a prompt with the retrieved FAQ entries, calls the LLM client, and returns a structured result: answer text, retrieved contexts, model used, latency, and token usage.

---

## 8. Postgres logging layer

Goal: Log every RAG interaction to a Postgres database.

Description: Write `src/db.py` that manages a Postgres connection (via `asyncpg` or `psycopg2`), creates a table for logs (question, answer, feedback, metadata, timestamp), and exposes `log_interaction()` and `update_feedback()` functions.

---

## 9. Evaluation: retrieval metrics

Goal: Compute hit rate and MRR against a small ground-truth question set.

Description: Create an evaluation script that loads a hand-crafted set of test questions with known relevant FAQ IDs, runs `search()` for each, and reports hit rate@k and MRR@k. Save results for later comparison.

---

## 10. Evaluation: LLM-as-a-judge relevance scoring

Goal: Score answer quality by asking an LLM to rate relevance.

Description: Write a script that takes the test questions, runs them through the full RAG pipeline, then sends each (question, answer, context) triplet to an LLM judge prompt and records a relevance score (e.g. 1-5). Aggregate results into a report.

---

## 11. Streamlit chat UI

Goal: Build a Streamlit app with a chat input, answer display, and thumbs up/down feedback.

Description: Write `app.py` that provides a simple chat interface: text input at the bottom, conversation history above, and a thumbs-up / thumbs-down button on each assistant response. Display metadata (model, latency, tokens) in an expandable section.

---

## 12. Grafana monitoring dashboard

Goal: Create a Grafana dashboard with at least 5 charts showing recent conversations, feedback, latency, and cost.

Description: Provision a Grafana dashboard config (JSON or YAML) that connects to the Postgres logs table and visualises at least 5 panels: (1) recent interactions table, (2) feedback distribution pie chart, (3) average latency over time, (4) token usage over time, (5) estimated cost over time, plus any additional panels that add insight (e.g. model usage breakdown, queries per time period, or a heatmap of busy hours). Include in `grafana/dashboard.json`.

---

## 13. Docker Compose packaging

Goal: Package the app, Postgres, and Grafana so everything starts with `docker compose up`.

Description: Write a `Dockerfile` for the Streamlit app, a `docker-compose.yaml` that defines `app`, `postgres`, and `grafana` services with proper environment variables, volume mounts, and network config. Add a `.env.example` file.

---

## 14. README with setup and run instructions

Goal: Write a clear README so a new developer can clone and run the project in under two minutes.

Description: Document prerequisites (Docker, API keys), setup steps (`cp .env.example .env`, fill in keys), and run instructions (`docker compose up`). Include a quick-start section, project structure overview, and links to the plan and task backlog.

---

## 15. Error handling and polish pass

Goal: Add sensible error handling, logging, and defaults across the codebase.

Description: Wrap API calls and file operations in try/except blocks with user-friendly fallback messaging. Add structured logging (`import logging`) to key modules. Set sensible defaults for config values (model names, chunk sizes, port numbers) so the project runs out of the box with minimal configuration.
