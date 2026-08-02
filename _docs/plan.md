# Bouncy Castle Rental FAQ RAG Agent – Project Plan

## Agreed Scope

- **Domain**: Bouncy Castle rental FAQ
- **Data**: Existing FAQ CSV file
- **Retrieval**: Hybrid (keyword + vector search)
- **Application flow**: Single-turn RAG only  
  (retrieve context → build prompt → call LLM)
- **LLM**: Groq as primary, OpenAI as fallback
- **Interface**: Streamlit chat UI
- **Evaluation**:  
  - Retrieval metrics (hit rate, MRR)  
  - LLM-as-a-judge relevance scoring  
  - Small set of manual test questions
- **Feedback & Monitoring**: Postgres + Grafana dashboard (thumbs up/down feedback)
- **Packaging**: Docker Compose + cloud-deployable

Inspired by: [fitness-assistant](https://github.com/alexeygrigorev/fitness-assistant)

---

## High-level Plan

### 1. Data & Knowledge Base
- Load the FAQ CSV
- Build a hybrid search index (keyword/TF-IDF + embeddings)
- Index should be easy to rebuild on startup

### 2. Core RAG Flow
- Retrieve relevant FAQ entries for a user question
- Construct a clear prompt with the retrieved context
- Call Groq (fallback to OpenAI on failure)
- Return the answer together with useful metadata (tokens, latency, model used, etc.)

### 3. Evaluation
- Create a small ground-truth question set from the FAQ
- Measure retrieval quality (hit rate / MRR)
- Run LLM-as-a-judge to score answer relevance
- Keep a handful of manual test questions for quick sanity checks

### 4. Interface
- Streamlit chat-style UI
- Question input + answer display
- Thumbs up / thumbs down feedback buttons

### 5. Logging, Feedback & Monitoring
- Log every interaction (question, answer, feedback, metadata) to Postgres
- Grafana dashboard showing:
  - Recent conversations
  - Feedback distribution
  - Relevance scores
  - Latency / token usage / estimated cost

### 6. Packaging & Deployment
- Docker Compose setup (Streamlit app + Postgres + Grafana)
- Environment variables for API keys and configuration
- Designed so the same stack (or a simplified version) can be deployed to cloud platforms (Render, Railway, Fly.io, AWS, etc.)

### 7. Polish
- Clear README with setup & run instructions
- Basic error handling and logging
- Sensible defaults so the project is easy to start

### 8. Best-Practice Enhancements

- **Document re-ranking**
  - Fetch a wider candidate set (e.g. `k=20`) via hybrid search, re-score, keep top-5
  - Preferred path: LLM reranker via `ask_llm` (no new dependency); cross-encoder only with explicit approval (adds a dependency, no torch in Dockerfile)
  - Insert either inside `search()` or as a `rerank()` wrapper in the `rag.py` path
  - Gate: rerun `evaluate_retrieval(k=5)` on `data/ground_truth.csv` — no regression (ideally improvement) on hit-rate@5 / MRR@5

- **User query rewriting**
  - Add a `rewrite_query()` step at the top of `answer_question()` in `src/rag.py`, before `search()`
  - Preferred path: one cheap Groq call via `ask_llm` to produce a clean, self-contained search query (no new dependency)
  - Optional: multi-query expansion (2–3 variants) reusing the existing RRF fusion in `src/search.py`
  - Gate: rerun `evaluate_retrieval()` on ground truth with a rewritten-query `search_fn`; also sanity-check a few conversational paraphrases

---

## Suggested Project Structure (high-level)

```
bouncy-castle-rag/
├── data/                  # FAQ CSV + evaluation datasets
├── src/                   # Core application code
│   ├── ingest.py
│   ├── search.py          # Hybrid retrieval
│   ├── rag.py             # RAG orchestration
│   ├── llm.py             # Groq + OpenAI clients
│   └── db.py              # Postgres logging
├── app.py                 # Streamlit UI
├── evaluation/            # Notebooks or scripts for eval
├── grafana/               # Dashboard config
├── docker-compose.yaml
├── Dockerfile
├── .env.example
└── README.md
```

---

## Next Steps (once coding begins)
1. Inspect the FAQ CSV and decide exact fields to index
2. Implement hybrid search
3. Build the RAG pipeline
4. Add Streamlit UI
5. Add logging + Grafana
6. Write evaluation scripts
7. Dockerize everything
