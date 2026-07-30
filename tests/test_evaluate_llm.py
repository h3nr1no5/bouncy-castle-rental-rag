import os
import pytest

from src.evaluate_llm import _parse_scores, evaluate_relevance


def test_parse_scores_valid_json():
    text = '{"relevance": 4, "faithfulness": 5, "explanation": "Good answer"}'
    result = _parse_scores(text)
    assert result["relevance"] == 4
    assert result["faithfulness"] == 5
    assert result["explanation"] == "Good answer"


def test_parse_scores_with_extra_text():
    text = 'Here are my scores:\n{"relevance": 3, "faithfulness": 2, "explanation": "Missing details"}'
    result = _parse_scores(text)
    assert result["relevance"] == 3
    assert result["faithfulness"] == 2


def test_parse_scores_invalid():
    result = _parse_scores("not json at all")
    assert result["relevance"] is None
    assert result["faithfulness"] is None


def test_parse_scores_empty():
    result = _parse_scores("")
    assert result["relevance"] is None
    assert result["faithfulness"] is None


def test_evaluate_relevance_report_structure(tmp_path):
    has_keys = bool(os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    report = evaluate_relevance(sample=1)

    assert "mean_relevance" in report
    assert "mean_faithfulness" in report
    assert "n" in report
    assert "n_valid" in report
    assert "distribution" in report
    assert "details" in report

    if has_keys and report["n_valid"] > 0:
        assert report["mean_relevance"] is not None
        assert report["mean_faithfulness"] is not None
        assert 1 <= report["mean_relevance"] <= 5
        assert 1 <= report["mean_faithfulness"] <= 5
        assert report["distribution"]["relevance"] != {}
        assert report["distribution"]["faithfulness"] != {}
        detail = report["details"][0]
        assert not detail["error"]
        assert "question" in detail
        assert "document_id" in detail
        assert "answer" in detail
        assert "relevance" in detail
        assert "faithfulness" in detail
        assert "judge_provider" in detail
        assert "judge_model" in detail
        assert "rag_provider" in detail
        assert "rag_model" in detail
