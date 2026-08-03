# Bouncy Castle Rental FAQ RAG Agent – Project Plan

## Agreed Scope

- **Domain**: Bouncy Castle rental FAQ
- **Data**: Existing FAQ CSV file
- **Ingestion**: Automated with dlt into a BM25 + FAISS hybrid index
- **Retrieval**: Hybrid (keyword + vector search) fused with Reciprocal Rank Fusion (RRF), plus LLM query rewriting (single-turn and history-aware)
- **Application flow**: Conversational RAG
  (rewrite query → retrieve context → build prompt → call LLM), with conversation history threaded into the answer and sessions persisted server-side
- **LLM**: Groq as primary, OpenAI as fallback
- **Interface**: FastAPI chat API (`app.py`) serving a static web UI (`ui/`)
- **Evaluation**:  
  - Retrieval metrics (hit rate@k, MRR@k, precision, recall, nDCG)  
  - LLM-as-a-judge relevance scoring  
  - Single-turn and multi-turn ground truth sets  
  - Tuned hyperparameters written to `tuned_params.json`
- **Feedback & Monitoring**: Postgres + Grafana dashboard (thumbs up/down feedback)
- **Packaging**: Docker Compose locally, deployed to Render (app + Postgres + Grafana)

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
- Create ground-truth question sets from the FAQ (single-turn and multi-turn)
- Measure retrieval quality (hit rate@k / MRR@k / precision / recall / nDCG)
- Run LLM-as-a-judge to score answer relevance
- Tune hyperparameters (RRF k, field weights, rewriting) in the notebook and persist the best settings
- Keep a handful of manual test questions for quick sanity checks

### 4. Interface
- FastAPI chat API serving a static web chat UI
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
- Docker Compose setup (app + Postgres + Grafana)
- Environment variables for API keys and configuration
- Designed so the same stack (or a simplified version) can be deployed to cloud platforms (Render, Railway, Fly.io, AWS, etc.)

### 7. Polish
- Clear README with setup & run instructions
- Basic error handling and logging
- Sensible defaults so the project is easy to start

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
├── app.py                 # FastAPI chat backend (serves ui/)
├── ui/                    # Static chat UI
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
