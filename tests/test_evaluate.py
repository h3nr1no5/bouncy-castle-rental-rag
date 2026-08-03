import csv
import pytest

from src.evaluate import (
    compute_hit_rate,
    compute_mrr,
    compute_ndcg,
    compute_precision,
    compute_recall,
    evaluate_retrieval,
    load_ground_truth,
    DEFAULT_GROUND_TRUTH_PATH,
)
from src.ingest import build_indexes

SAMPLE_FAQS = [
    {"Category": "Pricing", "Question": "How much does a castle cost?", "Answer": "Prices start at $100."},
    {"Category": "Booking", "Question": "How far in advance should I book?", "Answer": "Book as early as possible."},
    {"Category": "Safety", "Question": "Is adult supervision required?", "Answer": "Yes, always."},
]


def test_compute_hit_rate_finds_relevant():
    results = [{"id": "faq_0", "question": "What is the cancellation policy?"}]
    assert compute_hit_rate(results, ["faq_0"]) == 1.0


def test_compute_hit_rate_misses():
    results = [{"id": "faq_0", "question": "How much does it cost?"}]
    assert compute_hit_rate(results, ["faq_1"]) == 0.0


def test_compute_hit_rate_at_k():
    results = [
        {"id": "faq_0", "question": "How much does a castle cost?"},
        {"id": "faq_1", "question": "How far in advance should I book?"},
    ]
    assert compute_hit_rate(results, ["faq_1"], k=1) == 0.0
    assert compute_hit_rate(results, ["faq_1"], k=2) == 1.0


def test_compute_hit_rate_empty_results():
    assert compute_hit_rate([], ["faq_0"]) == 0.0


def test_compute_hit_rate_multiple_relevant_ids():
    results = [{"id": "faq_0", "question": "How much does a castle cost?"}]
    assert compute_hit_rate(results, ["faq_0", "faq_1"]) == 1.0


def test_compute_mrr_finds_at_rank_1():
    results = [
        {"id": "faq_0", "question": "How much does a castle cost?"},
        {"id": "faq_1", "question": "How far in advance should I book?"},
    ]
    assert compute_mrr(results, ["faq_0"]) == 1.0


def test_compute_mrr_finds_at_rank_2():
    results = [
        {"id": "faq_0", "question": "How much does a castle cost?"},
        {"id": "faq_1", "question": "How far in advance should I book?"},
    ]
    assert compute_mrr(results, ["faq_1"]) == 0.5


def test_compute_mrr_finds_at_rank_3():
    results = [
        {"id": "faq_0", "question": "A"},
        {"id": "faq_1", "question": "B"},
        {"id": "faq_2", "question": "C"},
    ]
    assert compute_mrr(results, ["faq_2"]) == pytest.approx(1.0 / 3)


def test_compute_mrr_misses():
    results = [{"id": "faq_0", "question": "How much does a castle cost?"}]
    assert compute_mrr(results, ["faq_1"]) == 0.0


def test_compute_mrr_at_k():
    results = [
        {"id": "faq_0", "question": "A"},
        {"id": "faq_1", "question": "B"},
        {"id": "faq_2", "question": "C"},
    ]
    assert compute_mrr(results, ["faq_2"], k=2) == 0.0
    assert compute_mrr(results, ["faq_2"], k=3) == pytest.approx(1.0 / 3)


def test_compute_mrr_empty_results():
    assert compute_mrr([], ["faq_0"]) == 0.0


def test_compute_mrr_best_rank_used():
    results = [
        {"id": "faq_0", "question": "A"},
        {"id": "faq_1", "question": "B"},
    ]
    assert compute_mrr(results, ["faq_1", "faq_0"]) == 1.0


# ---------------------------------------------------------------------------
# compute_recall
# ---------------------------------------------------------------------------

def test_compute_recall_all_relevant_retrieved():
    results = [{"id": "faq_0"}, {"id": "faq_1"}]
    assert compute_recall(results, ["faq_0", "faq_1"]) == 1.0


def test_compute_recall_partial():
    results = [{"id": "faq_0"}, {"id": "faq_1"}]
    assert compute_recall(results, ["faq_0", "faq_1", "faq_2"]) == pytest.approx(2.0 / 3)


def test_compute_recall_miss():
    results = [{"id": "faq_0"}]
    assert compute_recall(results, ["faq_1"]) == 0.0


def test_compute_recall_at_k_truncates():
    results = [{"id": "faq_0"}, {"id": "faq_1"}]
    assert compute_recall(results, ["faq_0", "faq_1"], k=1) == pytest.approx(0.5)
    assert compute_recall(results, ["faq_0", "faq_1"], k=3) == 1.0


def test_compute_recall_empty_results():
    assert compute_recall([], ["faq_0"]) == 0.0


def test_compute_recall_empty_relevant_ids():
    # Division by zero must be avoided.
    assert compute_recall([{"id": "faq_0"}], []) == 0.0


# ---------------------------------------------------------------------------
# compute_precision
# ---------------------------------------------------------------------------

def test_compute_precision_all_relevant():
    results = [{"id": "faq_0"}, {"id": "faq_1"}]
    assert compute_precision(results, ["faq_0", "faq_1"]) == 1.0


def test_compute_precision_partial():
    results = [{"id": "faq_0"}, {"id": "faq_1"}]
    assert compute_precision(results, ["faq_0"]) == pytest.approx(0.5)


def test_compute_precision_miss():
    results = [{"id": "faq_0"}, {"id": "faq_1"}]
    assert compute_precision(results, ["faq_5"]) == 0.0


def test_compute_precision_at_k_truncates():
    results = [{"id": "faq_0"}, {"id": "faq_1"}, {"id": "faq_2"}]
    assert compute_precision(results, ["faq_1"], k=1) == 0.0
    assert compute_precision(results, ["faq_1"], k=2) == pytest.approx(0.5)


def test_compute_precision_empty_results():
    assert compute_precision([], ["faq_0"]) == 0.0


# ---------------------------------------------------------------------------
# compute_ndcg
# ---------------------------------------------------------------------------

def test_compute_ndcg_perfect_ranking():
    results = [{"id": "faq_0"}, {"id": "faq_1"}]
    assert compute_ndcg(results, ["faq_0", "faq_1"]) == pytest.approx(1.0)


def test_compute_ndcg_miss():
    results = [{"id": "faq_0"}, {"id": "faq_1"}]
    assert compute_ndcg(results, ["faq_5"]) == 0.0


def test_compute_ndcg_non_perfect_ranking():
    # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    results = [{"id": "faq_0"}, {"id": "faq_1"}, {"id": "faq_2"}]
    assert compute_ndcg(results, ["faq_0", "faq_2"]) == pytest.approx(1.5 / 1.6309297535714573)


def test_compute_ndcg_at_k_truncates():
    # faq_1 at rank 2 (within k=2); faq_2 at rank 3 is cut off by k=2.
    results = [{"id": "faq_0"}, {"id": "faq_1"}, {"id": "faq_2"}]
    assert compute_ndcg(results, ["faq_1", "faq_2"], k=1) == 0.0
    dcg = 1.0 / 1.5849625007211563  # faq_1 at rank 2
    idcg = 1.0 + 1.0 / 1.5849625007211563  # two relevant, min(k=2, 2)
    assert compute_ndcg(results, ["faq_1", "faq_2"], k=2) == pytest.approx(dcg / idcg)


def test_compute_ndcg_empty_results():
    assert compute_ndcg([], ["faq_0"]) == 0.0


def test_compute_ndcg_empty_relevant_ids():
    assert compute_ndcg([{"id": "faq_0"}], []) == 0.0


def test_evaluate_retrieval_against_sample_data(tmp_path):
    bm25_path = tmp_path / "bm25.pkl"
    faiss_path = tmp_path / "faiss.bin"
    docs_path = tmp_path / "docs.json"
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=bm25_path, faiss_path=faiss_path, docs_path=docs_path)

    ground_truth = [
        {"question": "How much does a castle cost?", "document_id": "faq_0"},
        {"question": "How do I book?", "document_id": "faq_1"},
        {"question": "Is adult supervision required?", "document_id": "faq_2"},
    ]

    report = evaluate_retrieval(ground_truth=ground_truth, k=3, bm25_path=bm25_path, faiss_path=faiss_path, docs_path=docs_path)

    assert report["k"] == 3
    assert report["n"] == 3
    assert 0 <= report["hit_rate"] <= 1
    assert 0 <= report["mrr"] <= 1
    assert 0 <= report["recall"] <= 1
    assert 0 <= report["precision"] <= 1
    assert 0 <= report["ndcg"] <= 1
    assert len(report["details"]) == 3
    for detail in report["details"]:
        assert "question" in detail
        assert "hit" in detail
        assert "mrr" in detail
        assert "recall" in detail
        assert "precision" in detail
        assert "ndcg" in detail
        assert "document_id" in detail
        assert "retrieved_ids" in detail


def test_load_ground_truth_from_csv(tmp_path):
    gt_path = tmp_path / "test_gt.csv"
    with open(gt_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "document_id"])
        writer.writerow(["test?", "faq_0"])

    loaded = load_ground_truth(gt_path)
    assert loaded == [{"question": "test?", "document_id": "faq_0"}]


def test_load_ground_truth_missing_file():
    with pytest.raises(FileNotFoundError):
        load_ground_truth("/nonexistent/path.csv")


# ---------------------------------------------------------------------------
# evaluate_retrieval: multi-relevant-id items
# ---------------------------------------------------------------------------

def test_evaluate_retrieval_multi_relevant_ids():
    ground_truth = [
        {"question": "q1", "document_ids": ["faq_0", "faq_1"]},
        {"question": "q2", "document_ids": ["faq_5", "faq_6"]},
    ]

    def fake_search(query, k=5, **kwargs):
        if query == "q1":
            return [{"id": "faq_3"}, {"id": "faq_0"}]  # faq_0 at rank 2
        return [{"id": "faq_9"}]

    report = evaluate_retrieval(ground_truth=ground_truth, k=5, search_fn=fake_search)

    # Hit if ANY relevant id is retrieved; MRR from the best rank among them.
    assert report["details"][0]["hit"] is True
    assert report["details"][0]["mrr"] == pytest.approx(0.5)
    assert report["details"][1]["hit"] is False
    assert report["details"][1]["mrr"] == 0.0
    assert report["hit_rate"] == pytest.approx(0.5)
    assert report["mrr"] == pytest.approx(0.25)
    assert report["n"] == 2


def test_evaluate_retrieval_document_id_as_list():
    ground_truth = [{"question": "q", "document_id": ["faq_0", "faq_1"]}]

    def fake_search(query, k=5, **kwargs):
        return [{"id": "faq_1"}, {"id": "faq_0"}]

    report = evaluate_retrieval(ground_truth=ground_truth, k=5, search_fn=fake_search)
    assert report["details"][0]["hit"] is True
    assert report["details"][0]["mrr"] == pytest.approx(1.0)
    assert report["details"][0]["document_id"] == ["faq_0", "faq_1"]


def test_evaluate_retrieval_best_rank_among_relevant_ids():
    ground_truth = [{"question": "q", "document_ids": ["faq_2", "faq_0"]}]

    def fake_search(query, k=5, **kwargs):
        return [{"id": "faq_1"}, {"id": "faq_2"}, {"id": "faq_0"}]

    report = evaluate_retrieval(ground_truth=ground_truth, k=5, search_fn=fake_search)
    # faq_2 at rank 2 is the best rank among the two relevant ids.
    assert report["details"][0]["hit"] is True
    assert report["details"][0]["mrr"] == pytest.approx(0.5)


def test_evaluate_retrieval_single_document_id_regression():
    # Single document_id items must produce results identical to the previous
    # [item["document_id"]] behaviour.
    ground_truth = [
        {"question": "q1", "document_id": "faq_0"},
        {"question": "q2", "document_id": "faq_5"},
    ]

    def fake_search(query, k=5, **kwargs):
        if query == "q1":
            return [{"id": "faq_0"}, {"id": "faq_1"}]
        return [{"id": "faq_9"}]

    report = evaluate_retrieval(ground_truth=ground_truth, k=5, search_fn=fake_search)
    assert report["details"][0]["hit"] is True
    assert report["details"][0]["mrr"] == pytest.approx(1.0)
    assert report["details"][1]["hit"] is False
    assert report["details"][1]["mrr"] == 0.0
    assert report["hit_rate"] == pytest.approx(0.5)
    assert report["mrr"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# New metrics: evaluate_retrieval integration
# ---------------------------------------------------------------------------

def test_evaluate_retrieval_zero_results_reports_zero_metrics():
    ground_truth = [
        {"question": "q1", "document_id": "faq_0"},
        {"question": "q2", "document_id": "faq_1"},
    ]

    def no_results(query, k=5, **kwargs):
        return []

    report = evaluate_retrieval(ground_truth=ground_truth, k=5, search_fn=no_results)
    for detail in report["details"]:
        assert detail["hit"] is False
        assert detail["mrr"] == 0.0
        assert detail["recall"] == 0.0
        assert detail["precision"] == 0.0
        assert detail["ndcg"] == 0.0
    assert report["hit_rate"] == 0.0
    assert report["mrr"] == 0.0
    assert report["recall"] == 0.0
    assert report["precision"] == 0.0
    assert report["ndcg"] == 0.0


def test_evaluate_retrieval_metrics_use_full_relevant_id_set():
    ground_truth = [
        {"question": "q1", "document_ids": ["faq_0", "faq_1"]},
        {"question": "q2", "document_ids": ["faq_5", "faq_6"]},
    ]

    def fake_search(query, k=5, **kwargs):
        if query == "q1":
            return [{"id": "faq_3"}, {"id": "faq_0"}]  # only one of two relevant
        return [{"id": "faq_9"}]

    report = evaluate_retrieval(ground_truth=ground_truth, k=5, search_fn=fake_search)

    d0 = report["details"][0]
    # recall = 1 relevant retrieved / 2 relevant in set
    assert d0["recall"] == pytest.approx(0.5)
    # precision = 1 relevant / 2 retrieved
    assert d0["precision"] == pytest.approx(0.5)
    # DCG = 1/log2(3), IDCG = 1/log2(2) + 1/log2(3) (detail values rounded to 4dp)
    expected_ndcg = (1.0 / 1.5849625007211563) / 1.6309297535714573
    assert d0["ndcg"] == pytest.approx(expected_ndcg, abs=5e-5)

    d1 = report["details"][1]
    assert d1["recall"] == 0.0
    assert d1["precision"] == 0.0
    assert d1["ndcg"] == 0.0

    assert report["recall"] == pytest.approx(0.25)
    assert report["precision"] == pytest.approx(0.25)


def test_evaluate_retrieval_document_id_as_list_metrics():
    ground_truth = [{"question": "q", "document_id": ["faq_0", "faq_1"]}]

    def fake_search(query, k=5, **kwargs):
        return [{"id": "faq_1"}, {"id": "faq_0"}]

    report = evaluate_retrieval(ground_truth=ground_truth, k=5, search_fn=fake_search)
    detail = report["details"][0]
    assert detail["hit"] is True
    assert detail["mrr"] == pytest.approx(1.0)
    assert detail["recall"] == pytest.approx(1.0)
    assert detail["precision"] == pytest.approx(1.0)
    assert detail["ndcg"] == pytest.approx(1.0)


def test_evaluate_retrieval_large_k_beyond_corpus(tmp_path):
    # k larger than the corpus (41 FAQs in production, 3 here) must not error;
    # search() returns at most the corpus size and all metrics stay well-defined.
    bm25_path = tmp_path / "bm25.pkl"
    faiss_path = tmp_path / "faiss.bin"
    docs_path = tmp_path / "docs.json"
    build_indexes(faqs=SAMPLE_FAQS, bm25_path=bm25_path, faiss_path=faiss_path, docs_path=docs_path)

    ground_truth = [
        {"question": "How much does a castle cost?", "document_id": "faq_0"},
        {"question": "How do I book?", "document_id": "faq_1"},
        {"question": "Is adult supervision required?", "document_id": "faq_2"},
    ]

    report = evaluate_retrieval(
        ground_truth=ground_truth, k=100,
        bm25_path=bm25_path, faiss_path=faiss_path, docs_path=docs_path,
    )
    assert report["k"] == 100
    assert report["n"] == 3
    for detail in report["details"]:
        assert len(detail["retrieved_ids"]) <= len(SAMPLE_FAQS)
        assert 0 <= detail["recall"] <= 1
        assert 0 <= detail["precision"] <= 1
        assert 0 <= detail["ndcg"] <= 1
