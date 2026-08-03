# Problem Description

## The problem

Bouncy castle rental companies answer the same customer questions over and over — deposits, booking lead times, payment methods, delivery, safety rules, weather policy, and so on. Customers expect instant answers outside business hours, but a small rental business cannot staff a 24/7 support desk, and making customers dig through a long FAQ page is slow and error-prone.

This project solves that by building a **RAG (retrieval-augmented generation) agent** that answers customer questions about bouncy castle rentals directly from the company's FAQ, in a chat interface, grounded in the source FAQ entries.

## What the project does

A RAG system over a bouncy castle rental FAQ (`data/faq.csv`, 41 entries):

- **Ingestion** — FAQ rows are ingested into a knowledge base as a BM25 + FAISS hybrid index (`src/ingest.py`), automated with dlt.
- **Retrieval** — hybrid search (BM25 + FAISS) fused with Reciprocal Rank Fusion (RRF), plus LLM query rewriting so the search query better matches the FAQ.
- **Generation** — retrieved FAQ entries are injected into the prompt and sent to an LLM (Groq primary, OpenAI fallback). The answer is grounded in the FAQ and says so explicitly when the FAQ does not cover the question.
- **Conversational memory** — follow-up questions work: the latest question is rewritten into a standalone query using conversation history, history is threaded into the answer prompt, and chat sessions persist server-side.
- **Interface** — a FastAPI chat API (`app.py`) plus a static web UI (`ui/`).
- **Evaluation** — retrieval metrics (hit rate@k, MRR@k, precision, recall, nDCG), LLM-as-a-judge relevance scoring, and multi-turn ground truth, explored in `exploration.ipynb` with the best settings written to `tuned_params.json`.
- **Feedback & monitoring** — every interaction (question, answer, tokens, latency, cost, model) is logged to Postgres; thumbs up/down feedback is collected and visualised in a Grafana dashboard.
- **Packaging & deployment** — Docker Compose for local development, one-click deploy to Render (app + Postgres + Grafana).

## Evaluation criteria

The project is scored against the rubric in [`_docs/project-evaluation.md`](project-evaluation.md).
