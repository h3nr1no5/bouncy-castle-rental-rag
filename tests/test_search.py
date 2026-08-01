import json
import pickle

import numpy as np
import pytest

from src.ingest import build_indexes
from src.search import _load_indexes, search

SAMPLE_FAQS = [
    {"Category": "Pricing", "Question": "How much does a castle cost?", "Answer": "Prices start at $100."},
    {"Category": "Booking", "Question": "How do I book?", "Answer": "Call us or visit our website."},
    {"Category": "Safety", "Question": "Is it safe?", "Answer": "Yes, all equipment is inspected regularly."},
]

def test_search_returns_correct_number_of_results(tmp_path):
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    results = search("booking", k=2, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    assert len(results) == 2


def test_search_returns_expected_keys(tmp_path):
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    results = search("booking", k=2, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    for r in results:
        assert "id" in r
        assert "category" in r
        assert "question" in r
        assert "answer" in r
        assert "text" in r
        assert "score" in r


def test_search_results_sorted_by_score_descending(tmp_path):
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    results = search("booking", k=3, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_with_k_larger_than_dataset(tmp_path):
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    results = search("booking", k=10, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    assert len(results) == 3


def test_load_indexes_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_indexes(bm25_path=tmp_path / "nonexistent.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")


def test_search_with_empty_query(tmp_path):
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    results = search("", k=3, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    assert len(results) >= 0
    for r in results:
        assert "score" in r


def test_search_uses_config_k_default(tmp_path, monkeypatch):
    config = tmp_path / "tuned.json"
    config.write_text(json.dumps({"k": 1}))
    monkeypatch.setattr("src.config.DEFAULT_TUNED_PARAMS_PATH", config)
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    results = search("booking", bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    assert len(results) == 1


def test_search_explicit_k_overrides_config_default(tmp_path, monkeypatch):
    config = tmp_path / "tuned.json"
    config.write_text(json.dumps({"k": 1}))
    monkeypatch.setattr("src.config.DEFAULT_TUNED_PARAMS_PATH", config)
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    results = search("booking", k=3, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    assert len(results) == 3


def test_search_falls_back_to_legacy_k_when_config_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.DEFAULT_TUNED_PARAMS_PATH", tmp_path / "missing.json")
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    results = search("booking", bm25_path=tmp_path / "bm25.pkl", faiss_path=tmp_path / "faiss.bin", docs_path=tmp_path / "docs.json")
    assert len(results) == 3
