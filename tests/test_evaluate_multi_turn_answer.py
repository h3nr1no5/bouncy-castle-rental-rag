import os
import pathlib

import pytest

import evaluate_multi_turn_answer as eva


def _write_gt(tmp_path, rows, header=None):
    """Write a ground-truth CSV and return its path."""
    if header is None:
        header = ["conversation_id", "prior_user_turns", "follow_up_question", "document_id"]
    path = tmp_path / "ground_truth_multi_turn.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")
    return path


def _arm(coherence=None, relevance=None, error=False, error_message=None,
         rag_cost=0.0, judge_cost=0.0, rag_latency=0.0, judge_latency=0.0,
         rag_tokens=None, judge_tokens=None):
    return {
        "error": error,
        "error_message": error_message,
        "coherence": coherence,
        "relevance": relevance,
        "rag_cost": rag_cost,
        "judge_cost": judge_cost,
        "rag_latency": rag_latency,
        "judge_latency": judge_latency,
        "rag_tokens": rag_tokens or {"total": 0},
        "judge_tokens": judge_tokens or {"total": 0},
    }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestLoader:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eva.load_multi_turn_ground_truth(tmp_path / "missing.csv")

    def test_default_path_loads_repo_ground_truth(self):
        path = eva.DEFAULT_GROUND_TRUTH_PATH
        assert path.exists()
        loaded = eva.load_multi_turn_ground_truth()
        assert loaded["items"]
        item = loaded["items"][0]
        assert "question" in item
        assert "document_id" in item
        assert "history" in item

    def test_empty_file_header_only(self, tmp_path):
        path = _write_gt(tmp_path, [])
        loaded = eva.load_multi_turn_ground_truth(path)
        assert loaded["items"] == []
        assert loaded["skipped"] == 0
        assert loaded["no_history"] == 0

    def test_valid_rows_parsed(self, tmp_path):
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "How much for a weekend?", "faq_36"],
            [2, "What does the price include?;Any fees?", "Hidden fees?", "faq_35"],
        ])
        loaded = eva.load_multi_turn_ground_truth(path)
        assert len(loaded["items"]) == 2
        assert loaded["skipped"] == 0
        assert loaded["no_history"] == 0

        first = loaded["items"][0]
        assert first["question"] == "How much for a weekend?"
        assert first["document_id"] == "faq_36"
        assert first["history"] == [{"role": "user", "content": "Do you rent bouncy castles?"}]

        second = loaded["items"][1]
        assert second["history"] == [
            {"role": "user", "content": "What does the price include?"},
            {"role": "user", "content": "Any fees?"},
        ]

    def test_blank_follow_up_question_skipped(self, tmp_path):
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "", "faq_36"],
            [2, "Do you rent bouncy castles?", "How much?", "faq_35"],
        ])
        loaded = eva.load_multi_turn_ground_truth(path)
        assert len(loaded["items"]) == 1
        assert loaded["skipped"] == 1

    def test_missing_document_id_skipped(self, tmp_path):
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "How much?", ""],
            [2, "Do you rent bouncy castles?", "How much?", "faq_35"],
        ])
        loaded = eva.load_multi_turn_ground_truth(path)
        assert len(loaded["items"]) == 1
        assert loaded["skipped"] == 1

    def test_missing_required_column_skipped(self, tmp_path):
        # Header omits document_id entirely.
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "How much?"],
        ], header=["conversation_id", "prior_user_turns", "follow_up_question"])
        loaded = eva.load_multi_turn_ground_truth(path)
        assert loaded["items"] == []
        assert loaded["skipped"] == 1

    def test_empty_prior_user_turns_counts_no_history(self, tmp_path):
        path = _write_gt(tmp_path, [
            [1, "", "How much for a weekend?", "faq_36"],
            [2, "Do you rent bouncy castles?", "How much?", "faq_35"],
        ])
        loaded = eva.load_multi_turn_ground_truth(path)
        assert len(loaded["items"]) == 2
        assert loaded["no_history"] == 1
        assert loaded["items"][0]["history"] == []


# ---------------------------------------------------------------------------
# Score parsing (reuses _parse_scores from src/evaluate_llm)
# ---------------------------------------------------------------------------

class TestScoreParsing:
    def test_parse_scores_judge_output(self):
        parsed = eva._parse_scores(
            '{"coherence": 4, "relevance": 5, "explanation": "Good answer"}'
        )
        assert parsed["coherence"] == 4
        assert parsed["relevance"] == 5

    def test_parse_scores_with_extra_text(self):
        parsed = eva._parse_scores(
            'Here are my scores:\n{"coherence": 3, "relevance": 2, "explanation": "ok"}'
        )
        assert parsed["coherence"] == 3
        assert parsed["relevance"] == 2

    def test_parse_scores_invalid(self):
        parsed = eva._parse_scores("not json")
        assert parsed.get("coherence") is None
        assert parsed.get("relevance") is None


# ---------------------------------------------------------------------------
# Judge message
# ---------------------------------------------------------------------------

class TestJudgeMessage:
    def test_includes_history_question_context_answer(self):
        msg = eva.build_judge_user_message(
            question="How much for a weekend?",
            history=[{"role": "user", "content": "Do you rent bouncy castles?"}],
            contexts=[{"category": "Pricing", "question": "Cost?", "answer": "$100."}],
            answer="It costs $100.",
        )
        assert "Do you rent bouncy castles?" in msg
        assert "How much for a weekend?" in msg
        assert "Pricing" in msg
        assert "$100." in msg
        assert "It costs $100." in msg

    def test_empty_history_placeholder(self):
        msg = eva.build_judge_user_message(
            question="How much?",
            history=[],
            contexts=[],
            answer="No idea.",
        )
        assert "(no prior conversation)" in msg
        assert "No relevant FAQ entries found" in msg


# ---------------------------------------------------------------------------
# Aggregation / verdict
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_aggregates_mixed_valid_and_error(self):
        details = [
            {"without_history": _arm(4, 5), "with_history": _arm(5, 5)},
            {"without_history": _arm(error=True, error_message="boom"),
             "with_history": _arm(3, 4)},
        ]
        without = eva.compute_aggregates(details, "without_history")
        with_ = eva.compute_aggregates(details, "with_history")

        assert without["n"] == 2
        assert without["n_valid"] == 1
        assert without["mean_coherence"] == 4.0
        assert without["mean_relevance"] == 5.0

        assert with_["n"] == 2
        assert with_["n_valid"] == 2
        assert with_["mean_coherence"] == 4.0
        assert with_["mean_relevance"] == 4.5

    def test_aggregate_all_errors_means_none(self):
        details = [
            {"without_history": _arm(error=True), "with_history": _arm(error=True)},
        ]
        without = eva.compute_aggregates(details, "without_history")
        assert without["n"] == 1
        assert without["n_valid"] == 0
        assert without["mean_coherence"] is None
        assert without["mean_relevance"] is None

    def test_aggregate_empty(self):
        agg = eva.compute_aggregates([], "without_history")
        assert agg["n"] == 0
        assert agg["n_valid"] == 0
        assert agg["mean_coherence"] is None

    def test_compute_delta(self):
        without = {"n_valid": 1, "mean_coherence": 3.0, "mean_relevance": 3.0}
        with_ = {"n_valid": 1, "mean_coherence": 4.0, "mean_relevance": 3.5}
        delta_c, delta_r = eva.compute_delta(without, with_)
        assert delta_c == 1.0
        assert delta_r == 0.5

    def test_compute_delta_none_when_no_valid(self):
        without = {"n_valid": 0, "mean_coherence": None, "mean_relevance": None}
        with_ = {"n_valid": 1, "mean_coherence": 4.0, "mean_relevance": 3.5}
        assert eva.compute_delta(without, with_) == (None, None)


class TestVerdict:
    def test_improves(self):
        assert eva.compute_verdict(0.4) == "improves follow-up answer quality"

    def test_degrades(self):
        assert eva.compute_verdict(-0.2) == "degrades follow-up answer quality"

    def test_unchanged(self):
        assert eva.compute_verdict(0.0) == "leaves follow-up answer quality unchanged"

    def test_insufficient(self):
        assert eva.compute_verdict(None) == (
            "leaves follow-up answer quality unchanged (insufficient valid scores to determine)"
        )


# ---------------------------------------------------------------------------
# run_evaluation / main
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    def test_records_errors_without_api_keys(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        items = [{
            "conversation_id": "1",
            "question": "How much for a weekend?",
            "document_id": "faq_36",
            "history": [{"role": "user", "content": "Do you rent bouncy castles?"}],
        }]
        # Point at non-existent index files so retrieval fails before any LLM call.
        report = eva.run_evaluation(
            items,
            k=5,
            bm25_path=tmp_path / "bm25.pkl",
            faiss_path=tmp_path / "faiss.bin",
            docs_path=tmp_path / "docs.json",
        )

        assert report["n"] == 1
        assert report["without_history"]["n"] == 1
        assert report["with_history"]["n"] == 1
        assert report["without_history"]["n_valid"] == 0
        assert report["with_history"]["n_valid"] == 0
        assert report["without_history"]["mean_coherence"] is None
        assert report["with_history"]["mean_coherence"] is None
        assert report["delta_coherence"] is None
        assert report["verdict"] == (
            "leaves follow-up answer quality unchanged (insufficient valid scores to determine)"
        )

        detail = report["details"][0]
        assert detail["without_history"]["error"] is True
        assert detail["with_history"]["error"] is True
        assert "RAG pipeline failed" in detail["without_history"]["error_message"]

    def test_empty_ground_truth_no_division_by_zero(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        path = _write_gt(tmp_path, [])
        assert eva.main(ground_truth_path=str(path)) == 0


class TestMain:
    def test_skips_without_api_keys(self, monkeypatch, capsys):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert eva.main() == 0
        out = capsys.readouterr().out
        assert "No API keys found" in out

    def test_missing_file_returns_nonzero(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        missing = tmp_path / "does_not_exist.csv"
        assert eva.main(ground_truth_path=str(missing)) == 1
        err = capsys.readouterr().err
        assert "Ground truth file not found" in err

    def test_empty_file_reports_zero_items(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        path = _write_gt(tmp_path, [])
        assert eva.main(ground_truth_path=str(path)) == 0
        out = capsys.readouterr().out
        assert "Loaded 0 follow-up questions" in out