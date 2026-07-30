import json
import pathlib
import pickle
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.faqs import load_faqs

DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
DEFAULT_BM25_PATH = DEFAULT_DATA_DIR / "bm25_index.pkl"
DEFAULT_FAISS_PATH = DEFAULT_DATA_DIR / "faiss_index.bin"
DEFAULT_DOCS_PATH = DEFAULT_DATA_DIR / "ingest_docs.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _combine_text(row):
    return f"{row['Category']}: {row['Question']} {row['Answer']}"


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def build_indexes(faqs=None, bm25_path=None, faiss_path=None, docs_path=None, force=False):
    if faqs is None:
        faqs = load_faqs()
    if bm25_path is None:
        bm25_path = DEFAULT_BM25_PATH
    if faiss_path is None:
        faiss_path = DEFAULT_FAISS_PATH
    if docs_path is None:
        docs_path = DEFAULT_DOCS_PATH

    paths_exist = bm25_path.exists() and faiss_path.exists() and docs_path.exists()
    if paths_exist and not force:
        return {"bm25_path": str(bm25_path), "faiss_path": str(faiss_path), "docs_path": str(docs_path)}

    docs = [_combine_text(row) for row in faqs]
    tokenized = [_tokenize(d) for d in docs]

    bm25 = BM25Okapi(tokenized)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(docs, show_progress_bar=False)
    embeddings = _normalize(np.array(embeddings, dtype=np.float32))

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    docs_data = [
        {"id": f"faq_{i}", "text": docs[i], "category": row["Category"], "question": row["Question"], "answer": row["Answer"]}
        for i, row in enumerate(faqs)
    ]

    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    faiss.write_index(index, str(faiss_path))
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, ensure_ascii=False, indent=2)

    return {"bm25_path": str(bm25_path), "faiss_path": str(faiss_path), "docs_path": str(docs_path)}
