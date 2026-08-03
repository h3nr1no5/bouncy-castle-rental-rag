# Architecture

This document describes the retrieval and generation architecture as
implemented in [`src/search.py`](../src/search.py) and
[`src/llm.py`](../src/llm.py). The files are the source of truth; this doc is a
guided reading of them.

## Hybrid search with Reciprocal Rank Fusion (RRF)

`search()` in [`src/search.py`](../src/search.py) implements hybrid retrieval:
BM25 keyword search and FAISS vector search are run independently, and their
rankings are fused with **Reciprocal Rank Fusion (RRF)**.

The pipeline, given a query `q`:

1. **BM25** — the query is tokenized (`re.findall(r"\w+", q.lower())`) and
   scored with `BM25Okapi.get_scores()`; the top-`k` document ids are kept.
2. **FAISS** — the query is embedded with fastembed (`all-MiniLM-L6-v2`,
   384-dim), L2-normalized, and searched against an `IndexFlatIP` index; the
   top-`k` ids are kept.
3. **RRF fusion** — each retriever's ranked list contributes
   `1 / (rank + rrf_k)` per document, summed across retrievers:

   ```text
   rrf_score(doc) = sum over retrievers of 1 / (rank(doc) + rrf_k)
   ```

4. **Optional category boost** — when `cat_weight` is non-zero, documents
   whose `category` shares terms with the query get an extra
   `cat_weight / (rank + rrf_k)` entry in the fusion (a third scoring source on
   top of BM25 and FAISS).
5. **Ranking** — documents are sorted by fused score descending and the top-`k`
   are returned with `id`, `category`, `question`, `answer`, `text`, and
   `score` (rounded to 4 decimals).

Defaults come from `tuned_params.json` via `load_tuned_params()`:
`k=5`, `rrf_k=1`, `cat_weight=0`, `bm25_k1=2.0`, `bm25_b=0.75`. The tuned
`rrf_k=1` was selected from the `rrf_k` sweep in `exploration.ipynb` — the
sweep measured MRR@5 ≈ 0.835 at `rrf_k=1`, declining as `rrf_k` grows. Against
the non-reranked baselines in the notebook, hybrid search reaches hit rate@5 ≈
0.956 and MRR@5 ≈ 0.835 vs. keyword-only 0.756 / 0.582. `cat_weight` defaults
to `0`, so the category boost is off unless explicitly enabled.

A separate `keyword_search()` (DuckDB SQL `ILIKE` with tunable per-field
weights) exists in the same file for the evaluation pipeline; it is not part of
the production chat path.

## LLM provider resolution (Groq primary, OpenAI fallback)

`ask_llm()` in [`src/llm.py`](../src/llm.py) resolves the provider from the two
API keys in the environment:

- If `GROQ_API_KEY` is set, Groq is always tried **first**, with built-in
  rate-limit enforcement.
- If that Groq call **fails**, it falls back to **OpenAI** — but only if
  `OPENAI_API_KEY` is also set.
- If Groq fails and `OPENAI_API_KEY` is **not** set, the request raises an
  error (no fallback).

Default models: Groq `llama-3.3-70b-versatile`, OpenAI `gpt-5.4-mini`. Both
clients use a 30s timeout and 2 retries. The result includes `response`,
`model`, `provider`, `latency`, `cost` (from per-model pricing tables) and
`tokens {prompt, completion, total}`.

| Secret set in the environment | Behavior |
|---|---|
| `GROQ_API_KEY` only | **Groq only**. No fallback — if Groq errors or hits a rate limit, the request fails. |
| `OPENAI_API_KEY` only | **OpenAI only** (the Groq path is skipped, since `groq_key` is unset). |
| Both | **Groq primary**, with **OpenAI as the automatic fallback** on Groq failure. |
| Neither | `ask_llm()` raises `ValueError` immediately. |

On Render only `GROQ_API_KEY` is configured (`OPENAI_API_KEY` is not in
`render.yaml`), so the deployed app uses **Groq only** — see
[`_docs/deployment.md`](deployment.md).

### Groq rate-limit enforcement

`_enforce_groq_rate_limits()` in `src/llm.py` tracks successful Groq call
timestamps in two in-process deques and enforces:

- `GROQ_RPM_LIMIT = 25` — requests per minute; when reached, the process
  sleeps until the oldest timestamp leaves the 60-second window.
- `GROQ_RPD_LIMIT = 900` — requests per day; when reached, the call raises
  `RuntimeError` (no automatic retry).

## Source of truth

- [`src/search.py`](../src/search.py) — hybrid search, RRF, `cat_weight`
- [`src/llm.py`](../src/llm.py) — provider resolution, rate limits, pricing
- [`tuned_params.json`](../tuned_params.json) — tuned retrieval defaults
