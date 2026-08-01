import json
import pathlib
import pickle

import duckdb
import faiss
import numpy as np
from fastembed import TextEmbedding

from src.config import load_tuned_params
from src.ingest import (
    DEFAULT_BM25_PATH,
    DEFAULT_FAISS_PATH,
    DEFAULT_DOCS_PATH,
    MODEL_NAME,
    _normalize,
    _tokenize,
)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)
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


def search(query, k=None, rrf_k=None, cat_weight=None, bm25_path=None, faiss_path=None, docs_path=None):
    params = load_tuned_params()
    if k is None:
        k = params["k"]
    if rrf_k is None:
        rrf_k = params["rrf_k"]
    if cat_weight is None:
        cat_weight = params["cat_weight"]

    bm25, index, docs = _load_indexes(bm25_path, faiss_path, docs_path)

    tokenized_query = _tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_k = np.argsort(bm25_scores)[::-1][:k]

    model = _get_model()
    query_emb = np.array(list(model.embed([query])), dtype=np.float32)
    query_emb = _normalize(query_emb)
    faiss_scores, faiss_top_k = index.search(query_emb, k)
    faiss_top_k = faiss_top_k[0]

    rrf_scores = {}
    for rank, idx in enumerate(bm25_top_k):
        rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (rank + rrf_k)
    for rank, idx in enumerate(faiss_top_k):
        idx = int(idx)
        if idx != -1:
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + rrf_k)

    if cat_weight:
        cat_top_k = []
        for idx in range(len(docs)):
            cat_terms = set(_tokenize(docs[idx]["category"]))
            matches = sum(1 for t in tokenized_query if t in cat_terms)
            if matches > 0:
                cat_top_k.append((idx, matches))
        cat_top_k.sort(key=lambda x: x[1], reverse=True)
        for rank, (idx, _) in enumerate(cat_top_k[:k]):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + cat_weight / (rank + rrf_k)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked[:k]:
        doc = docs[idx]
        results.append({
            "id": doc["id"],
            "category": doc["category"],
            "question": doc["question"],
            "answer": doc["answer"],
            "text": doc["text"],
            "score": round(score, 4),
        })

    return results


def keyword_search(query, k=5, db_path=None, docs_path=None, field_weights=None):
    if field_weights is None:
        field_weights = {"Question": 2.0, "Answer": 1.0}

    if docs_path is None:
        docs_path = DEFAULT_DOCS_PATH

    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)

    terms = [t for t in query.lower().split() if t]
    if not terms:
        return []

    con = duckdb.connect(str(db_path))

    per_term_conditions = []
    for term in terms:
        safe = term.replace("'", "''")
        for field in field_weights:
            per_term_conditions.append(f'"{field}" ILIKE \'%{safe}%\'')
    where_clause = " OR ".join(per_term_conditions)

    rows = con.sql(f"""
        SELECT "Category", "Question", "Answer"
        FROM "faq"."faq_resource"
        WHERE {where_clause}
    """).fetchall()
    con.close()

    doc_map = {}
    for doc in docs:
        doc_map[(doc["category"], doc["question"], doc["answer"])] = doc

    scored = []
    for category, question, answer in rows:
        key = (category, question, answer)
        doc = doc_map.get(key)
        if doc is None:
            continue

        score = 0.0
        field_values = {"Category": category, "Question": question, "Answer": answer}
        for term in terms:
            for field, weight in field_weights.items():
                if term in field_values[field].lower():
                    score += weight

        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for doc, score in scored[:k]:
        results.append({
            "id": doc["id"],
            "category": doc["category"],
            "question": doc["question"],
            "answer": doc["answer"],
            "text": doc["text"],
            "score": round(score, 4),
        })

    return results
