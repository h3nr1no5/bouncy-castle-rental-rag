"""Unit tests for evaluate_multi_turn_rewrite.py wiring.

These tests are API-free: they exercise the ``--ground-truth`` CLI parsing, the
loader's malformed-row skipping, the zero-valid-rows non-zero exit, and the
positional (index-based) history resolution for duplicate follow-up questions.
No network or API keys are required.
"""

import os
import subprocess
import sys

import pytest

import evaluate_multi_turn_rewrite as evr


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


def _no_key_env():
    env = dict(os.environ)
    env.pop("GROQ_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    return env


# ---------------------------------------------------------------------------
# --ground-truth CLI parsing
# ---------------------------------------------------------------------------

class TestCliArg:
    def test_ground_truth_flag_parses(self, tmp_path):
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "How much for a weekend?", "faq_36"],
        ])
        result = subprocess.run(
            [sys.executable, "evaluate_multi_turn_rewrite.py", "--ground-truth", str(path)],
            capture_output=True,
            text=True,
            env=_no_key_env(),
        )
        # An unrecognised flag would make argparse exit non-zero; with no API
        # keys the script prints the friendly message and exits 0.
        assert result.returncode == 0
        assert "No API keys found" in result.stdout


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestLoader:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            evr.load_multi_turn_ground_truth(tmp_path / "missing.csv")

    def test_default_path_loads_repo_ground_truth(self):
        loaded = evr.load_multi_turn_ground_truth()
        assert len(loaded["items"]) == 12
        assert loaded["skipped"] == 0
        item = loaded["items"][0]
        assert item["question"] == "How much for a weekend?"
        assert item["document_id"] == "faq_36"
        assert item["history"] == [{"role": "user", "content": "Do you rent bouncy castles?"}]

    def test_valid_rows_parsed(self, tmp_path):
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "How much for a weekend?", "faq_36"],
            [2, "What does the price include?;Any fees?", "Hidden fees?", "faq_35"],
        ])
        loaded = evr.load_multi_turn_ground_truth(path)
        assert len(loaded["items"]) == 2
        assert loaded["skipped"] == 0
        assert loaded["items"][1]["history"] == [
            {"role": "user", "content": "What does the price include?"},
            {"role": "user", "content": "Any fees?"},
        ]

    def test_blank_follow_up_question_skipped(self, tmp_path):
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "", "faq_36"],
            [2, "Do you rent bouncy castles?", "How much?", "faq_35"],
        ])
        loaded = evr.load_multi_turn_ground_truth(path)
        assert len(loaded["items"]) == 1
        assert loaded["skipped"] == 1

    def test_blank_document_id_skipped(self, tmp_path):
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "How much?", ""],
            [2, "Do you rent bouncy castles?", "How much?", "faq_35"],
        ])
        loaded = evr.load_multi_turn_ground_truth(path)
        assert len(loaded["items"]) == 1
        assert loaded["skipped"] == 1

    def test_missing_required_column_skipped(self, tmp_path):
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "How much?"],
        ], header=["conversation_id", "prior_user_turns", "follow_up_question"])
        loaded = evr.load_multi_turn_ground_truth(path)
        assert loaded["items"] == []
        assert loaded["skipped"] == 1

    def test_empty_prior_user_turns_counts_no_history(self, tmp_path):
        path = _write_gt(tmp_path, [
            [1, "", "How much for a weekend?", "faq_36"],
        ])
        loaded = evr.load_multi_turn_ground_truth(path)
        assert len(loaded["items"]) == 1
        assert loaded["no_history"] == 1
        assert loaded["items"][0]["history"] == []


# ---------------------------------------------------------------------------
# main() error handling
# ---------------------------------------------------------------------------

class TestMain:
    def test_skips_without_api_keys(self, monkeypatch, capsys):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert evr.main() == 0
        out = capsys.readouterr().out
        assert "No API keys found" in out

    def test_missing_file_returns_nonzero(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        missing = tmp_path / "does_not_exist.csv"
        assert evr.main(ground_truth_path=str(missing)) == 1
        err = capsys.readouterr().err
        assert "Ground truth file not found" in err
        assert str(missing) in err

    def test_zero_valid_rows_returns_nonzero(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Single-turn ground_truth.csv passed by mistake: no required columns.
        path = tmp_path / "ground_truth.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("question,document_id\n")
            f.write("What payment options do you accept?,faq_2\n")
        assert evr.main(ground_truth_path=str(path)) == 1
        err = capsys.readouterr().err
        assert "no valid multi-turn rows" in err
        assert str(path) in err
        assert "prior_user_turns" in err
        assert "follow_up_question" in err
        assert "document_id" in err

    def test_prints_loaded_path_and_row_count(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "How much for a weekend?", "faq_36"],
            [2, "Do you rent bouncy castles?", "", "faq_35"],  # skipped
        ])
        # API-free stubs: no real retrieval or LLM calls.
        monkeypatch.setattr(evr, "search", lambda q, k=5, **kw: [])
        monkeypatch.setattr(evr, "rewrite_query_with_history", lambda q, h: q)
        assert evr.main(ground_truth_path=str(path)) == 0
        out = capsys.readouterr().out
        assert f"Loaded 1 queries from {path}" in out
        assert "Skipped 1 malformed rows" in out


# ---------------------------------------------------------------------------
# Positional history resolution for duplicate follow-up questions
# ---------------------------------------------------------------------------

class TestHistoryResolution:
    def test_duplicate_follow_up_histories_resolved_positionally(self, monkeypatch):
        gt = [
            {"question": "How much space do I need?", "document_id": "faq_8",
             "history": [{"role": "user", "content": "history-A"}]},
            {"question": "How much space do I need?", "document_id": "faq_8",
             "history": [{"role": "user", "content": "history-B"}]},
        ]
        calls = []
        monkeypatch.setattr(evr, "rewrite_query_with_history",
                            lambda q, h: calls.append((q, h)) or "rewritten")
        monkeypatch.setattr(evr, "search", lambda q, k=5, **kw: [])

        evr.search_multi.ground_truth = gt
        evr.search_multi.cursor = 0
        evr.search_multi("How much space do I need?", k=5)
        evr.search_multi("How much space do I need?", k=5)

        assert [c[1] for c in calls] == [gt[0]["history"], gt[1]["history"]]

    def test_history_lookup_works_across_both_arms(self, monkeypatch):
        gt = [
            {"question": "How much space do I need?", "document_id": "faq_8",
             "history": [{"role": "user", "content": "h1"}]},
            {"question": "How much space do I need?", "document_id": "faq_8",
             "history": [{"role": "user", "content": "h2"}]},
        ]
        calls = []
        monkeypatch.setattr(evr, "rewrite_query_with_history",
                            lambda q, h: calls.append(h) or "rewritten")
        monkeypatch.setattr(evr, "search", lambda q, k=5, **kw: [])

        evr.search_multi.ground_truth = gt
        evr.search_multi.cursor = 0
        # Raw arm (search_raw) does not touch the history cursor.
        evr.evaluate_retrieval(ground_truth=gt, k=5, search_fn=evr.search_raw)
        # Multi-turn arm resolves each item's own history positionally.
        evr.evaluate_retrieval(ground_truth=gt, k=5, search_fn=evr.search_multi)

        assert calls == [gt[0]["history"], gt[1]["history"]]