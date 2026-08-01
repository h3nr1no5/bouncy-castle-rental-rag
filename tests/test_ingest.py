import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from src.ingest import _combine_text, _normalize, _tokenize, build_indexes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNED_PARAMS_FILE = PROJECT_ROOT / "tuned_params.json"

SAMPLE_FAQS = [
    {"Category": "A", "Question": "q1", "Answer": "a1"},
    {"Category": "B", "Question": "q2", "Answer": "a2"},
    {"Category": "C", "Question": "q3", "Answer": "a3"},
]


def test_combine_text():
    row = {"Category": "Booking", "Question": "How to book?", "Answer": "Call us."}
    assert _combine_text(row) == "Booking: How to book? Call us."


def test_tokenize():
    assert _tokenize("Hello World! It's 1.") == ["hello", "world", "it", "s", "1"]


def test_tokenize_empty():
    assert _tokenize("") == []


def test_normalize_produces_unit_vectors():
    vecs = np.array([[3.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    normalized = _normalize(vecs)
    assert normalized.shape == (2, 2), f"Expected (2, 2), got {normalized.shape}"
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0), f"Expected unit norms, got {norms}"


def test_build_indexes_creates_files(tmp_path):
    out = build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")  # fmt: skip
    assert out["bm25_path"].endswith("bm25.pkl")
    assert out["faiss_path"].endswith("faiss.bin")
    assert out["docs_path"].endswith("docs.json")
    assert (tmp_path / "bm25.pkl").exists()
    assert (tmp_path / "faiss.bin").exists()
    assert (tmp_path / "docs.json").exists()


def test_build_indexes_skips_when_exists(tmp_path):
    bm25_p = tmp_path / "bm25.pkl"
    faiss_p = tmp_path / "faiss.bin"
    docs_p = tmp_path / "docs.json"
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=bm25_p, faiss_path=faiss_p, docs_path=docs_p)
    mtime_before = faiss_p.stat().st_mtime_ns
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=bm25_p, faiss_path=faiss_p, docs_path=docs_p)
    assert faiss_p.stat().st_mtime_ns == mtime_before


def test_build_indexes_force_rebuild(tmp_path):
    bm25_p = tmp_path / "bm25.pkl"
    faiss_p = tmp_path / "faiss.bin"
    docs_p = tmp_path / "docs.json"
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=bm25_p, faiss_path=faiss_p, docs_path=docs_p)
    mtime_before = faiss_p.stat().st_mtime_ns
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=bm25_p, faiss_path=faiss_p, docs_path=docs_p, force=True)
    assert faiss_p.stat().st_mtime_ns > mtime_before


def test_faiss_index_has_correct_shape(tmp_path):
    out = build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")  # fmt: skip
    index = faiss.read_index(out["faiss_path"])
    assert index.ntotal == len(SAMPLE_FAQS)
    assert index.d == 384


def test_bm25_can_be_loaded_and_queried(tmp_path):
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")  # fmt: skip
    with open(tmp_path / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    assert isinstance(bm25, BM25Okapi)
    scores = bm25.get_scores(_tokenize("q1"))
    assert scores[0] > 0


def test_docs_json_contains_original_data(tmp_path):
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")  # fmt: skip
    with open(tmp_path / "docs.json") as f:
        docs = json.load(f)
    assert len(docs) == 3
    assert docs[0]["question"] == "q1"
    assert docs[0]["answer"] == "a1"
    assert docs[0]["category"] == "A"


def test_faiss_embedding_similarity(tmp_path):
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")  # fmt: skip
    index = faiss.read_index(str(tmp_path / "faiss.bin"))
    vecs = index.reconstruct_n(0, index.ntotal)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_indexes_uses_tuned_k1_b_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.DEFAULT_TUNED_PARAMS_PATH", TUNED_PARAMS_FILE)
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")  # fmt: skip
    with open(tmp_path / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    assert bm25.k1 == 2.0
    assert bm25.b == 0.75


def test_build_indexes_falls_back_to_legacy_k1_when_config_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.DEFAULT_TUNED_PARAMS_PATH", tmp_path / "missing.json")
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")  # fmt: skip
    with open(tmp_path / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    assert bm25.k1 == 1.5
    assert bm25.b == 0.75


def test_build_indexes_explicit_k1_b_override_config(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.DEFAULT_TUNED_PARAMS_PATH", TUNED_PARAMS_FILE)
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json", k1=0.5, b=0.5)  # fmt: skip
    with open(tmp_path / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    assert bm25.k1 == 0.5
    assert bm25.b == 0.5
