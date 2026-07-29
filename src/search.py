import json
import pathlib
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingest import (
    DEFAULT_BM25_PATH,
    DEFAULT_FAISS_PATH,
    DEFAULT_DOCS_PATH,
    MODEL_NAME,
    _normalize,
    _tokenize,
)

RRF_K = 60

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _load_indexes(bm25_path=None, faiss_path=None, docs_path=None):
    if bm25_path is None:
        bm25_path = DEFAULT_BM25_PATH
    if faiss_path is None:
        faiss_path = DEFAULT_FAISS_PATH
    if docs_path is None:
        docs_path = DEFAULT_DOCS_PATH

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    index = faiss.read_index(str(faiss_path))
    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)

    return bm25, index, docs


def search(query, k=5, bm25_path=None, faiss_path=None, docs_path=None):
    bm25, index, docs = _load_indexes(bm25_path, faiss_path, docs_path)

    tokenized_query = _tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_k = np.argsort(bm25_scores)[::-1][:k]

    model = _get_model()
    query_emb = model.encode([query], show_progress_bar=False)
    query_emb = _normalize(np.array(query_emb, dtype=np.float32))
    faiss_scores, faiss_top_k = index.search(query_emb, k)
    faiss_top_k = faiss_top_k[0]

    rrf_scores = {}
    for rank, idx in enumerate(bm25_top_k):
        rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (rank + RRF_K)
    for rank, idx in enumerate(faiss_top_k):
        idx = int(idx)
        if idx != -1:
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + RRF_K)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked[:k]:
        doc = docs[idx]
        results.append({
            "category": doc["category"],
            "question": doc["question"],
            "answer": doc["answer"],
            "text": doc["text"],
            "score": round(score, 4),
        })

    return results
