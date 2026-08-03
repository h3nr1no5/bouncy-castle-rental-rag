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
from src.faqs import load_faqs


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
        assert len(loaded["items"]) == 205
        assert loaded["skipped"] == 0
        assert loaded["no_history"] == 0
        item = loaded["items"][0]
        assert item["question"] == "How soon should I get this set up?"
        assert item["document_id"] == "faq_0"
        assert item["history"] == [
            {"role": "user", "content": "We're planning my son's birthday party and want to lock in a date."},
            {"role": "user", "content": "It's probably going to be on a Saturday in June."},
        ]

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


# ---------------------------------------------------------------------------
# --k / --k-sweep / --limit CLI parsing
# ---------------------------------------------------------------------------

class TestCliKAndLimit:
    def _run(self, tmp_path, *flags):
        path = _write_gt(tmp_path, [
            [1, "Do you rent bouncy castles?", "How much for a weekend?", "faq_36"],
        ])
        return subprocess.run(
            [sys.executable, "evaluate_multi_turn_rewrite.py",
             "--ground-truth", str(path), *flags],
            capture_output=True,
            text=True,
            env=_no_key_env(),
        )

    def test_k_flag_parses(self, tmp_path):
        result = self._run(tmp_path, "--k", "5")
        assert result.returncode == 0
        assert "No API keys found" in result.stdout

    def test_k_zero_rejected(self, tmp_path):
        result = self._run(tmp_path, "--k", "0")
        assert result.returncode != 0
        assert "positive integer" in result.stderr

    def test_k_negative_rejected(self, tmp_path):
        result = self._run(tmp_path, "--k", "-1")
        assert result.returncode != 0
        assert "positive integer" in result.stderr

    def test_k_non_integer_rejected(self, tmp_path):
        result = self._run(tmp_path, "--k", "abc")
        assert result.returncode != 0
        assert "invalid" in result.stderr

    def test_k_sweep_parses(self, tmp_path):
        result = self._run(tmp_path, "--k-sweep", "1,3,5,10")
        assert result.returncode == 0
        assert "No API keys found" in result.stdout

    def test_k_sweep_single_value_parses(self, tmp_path):
        result = self._run(tmp_path, "--k-sweep", "5")
        assert result.returncode == 0
        assert "No API keys found" in result.stdout

    def test_k_sweep_bad_value_rejected(self, tmp_path):
        result = self._run(tmp_path, "--k-sweep", "1,0,5")
        assert result.returncode != 0
        assert "positive integer" in result.stderr

    def test_k_and_k_sweep_mutually_exclusive(self, tmp_path):
        result = self._run(tmp_path, "--k", "5", "--k-sweep", "1,3")
        assert result.returncode != 0
        assert "not allowed with argument" in result.stderr

    def test_limit_parses(self, tmp_path):
        result = self._run(tmp_path, "--limit", "10")
        assert result.returncode == 0
        assert "No API keys found" in result.stdout

    def test_limit_zero_rejected(self, tmp_path):
        result = self._run(tmp_path, "--limit", "0")
        assert result.returncode != 0
        assert "positive integer" in result.stderr

    def test_limit_negative_rejected(self, tmp_path):
        result = self._run(tmp_path, "--limit", "-3")
        assert result.returncode != 0
        assert "positive integer" in result.stderr

    def test_limit_non_integer_rejected(self, tmp_path):
        result = self._run(tmp_path, "--limit", "abc")
        assert result.returncode != 0
        assert "invalid" in result.stderr


# ---------------------------------------------------------------------------
# main() output: new metrics, --limit, breakdowns, --k-sweep
# ---------------------------------------------------------------------------

class TestMainOutput:
    def _stub(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(evr, "search", lambda q, k=5, **kw: [])
        monkeypatch.setattr(evr, "rewrite_query_with_history", lambda q, h: q)

    def _gt(self, tmp_path, n=3):
        rows = [[i, "Do you rent bouncy castles?", f"How much? {i}", f"faq_{i}"]
                for i in range(n)]
        return _write_gt(tmp_path, rows)

    def test_prints_new_metrics_for_both_arms(self, tmp_path, monkeypatch, capsys):
        self._stub(monkeypatch)
        path = self._gt(tmp_path)
        assert evr.main(ground_truth_path=str(path), k=5) == 0
        out = capsys.readouterr().out
        # Each arm prints the three new metric lines.
        assert out.count("Recall@5") >= 2
        assert out.count("Precision@5") >= 2
        assert out.count("nDCG@5") >= 2
        # The side-by-side comparison table includes the new metric rows.
        assert "Recall@5" in out
        assert "Precision@5" in out
        assert "nDCG@5" in out

    def test_limit_slices_rows(self, tmp_path, monkeypatch, capsys):
        self._stub(monkeypatch)
        path = self._gt(tmp_path, n=3)
        assert evr.main(ground_truth_path=str(path), limit=2) == 0
        out = capsys.readouterr().out
        assert "Loaded 2 queries" in out

    def test_limit_larger_than_rows_evaluates_all(self, tmp_path, monkeypatch, capsys):
        self._stub(monkeypatch)
        path = self._gt(tmp_path, n=3)
        assert evr.main(ground_truth_path=str(path), limit=100) == 0
        out = capsys.readouterr().out
        assert "Loaded 3 queries" in out

    def test_breakdowns_printed_for_single_k(self, tmp_path, monkeypatch, capsys):
        self._stub(monkeypatch)
        path = self._gt(tmp_path)
        assert evr.main(ground_truth_path=str(path), k=5) == 0
        out = capsys.readouterr().out
        assert "Per-FAQ breakdown" in out
        assert "Per-Category breakdown" in out

    def test_breakdowns_not_printed_for_sweep(self, tmp_path, monkeypatch, capsys):
        self._stub(monkeypatch)
        path = self._gt(tmp_path)
        assert evr.main(ground_truth_path=str(path), k_sweep=[1, 3]) == 0
        out = capsys.readouterr().out
        assert "Per-FAQ breakdown" not in out
        assert "Per-Category breakdown" not in out

    def test_sweep_single_value_matches_k_metric_blocks(self, tmp_path, monkeypatch, capsys):
        self._stub(monkeypatch)
        path = self._gt(tmp_path)

        def metric_lines(out):
            return [l for l in out.splitlines()
                    if l.strip().startswith(("Hit Rate@", "MRR@", "Recall@",
                                             "Precision@", "nDCG@"))]

        evr.main(ground_truth_path=str(path), k=5)
        out_k = capsys.readouterr().out
        evr.main(ground_truth_path=str(path), k_sweep=[5])
        out_sweep = capsys.readouterr().out
        # The per-arm metric blocks for k=5 are identical between --k 5 and
        # --k-sweep 5 (breakdowns are only printed for single-k runs).
        assert metric_lines(out_k) == metric_lines(out_sweep)

    def test_sweep_resets_cursor_before_each_multi_arm(self, tmp_path, monkeypatch, capsys):
        self._stub(monkeypatch)
        path = self._gt(tmp_path)
        real = evr.evaluate_retrieval
        seen = []

        def spy(ground_truth=None, k=5, search_fn=None, **kwargs):
            if search_fn is evr.search_multi:
                assert evr.search_multi.cursor == 0
            seen.append(k)
            return real(ground_truth=ground_truth, k=k, search_fn=search_fn, **kwargs)

        monkeypatch.setattr(evr, "evaluate_retrieval", spy)
        assert evr.main(ground_truth_path=str(path), k_sweep=[1, 3]) == 0
        # raw@1, multi@1, raw@3, multi@3
        assert seen == [1, 1, 3, 3]

    def test_generated_csv_prints_all_41_faqs(self, monkeypatch, capsys):
        self._stub(monkeypatch)
        assert evr.main(
            ground_truth_path="data/ground_truth_multi_turn_generated.csv", k=5
        ) == 0
        out = capsys.readouterr().out
        assert "Loaded 205 queries" in out
        for i in range(41):
            assert f"faq_{i}" in out


# ---------------------------------------------------------------------------
# Per-FAQ / per-category breakdown helpers
# ---------------------------------------------------------------------------

class TestBreakdowns:
    def test_generated_set_groups_into_all_41_faqs(self):
        loaded = evr.load_multi_turn_ground_truth(
            "data/ground_truth_multi_turn_generated.csv"
        )
        assert len(loaded["items"]) == 205
        details = [
            {"document_id": item["document_id"], "hit": False, "mrr": 0.0,
             "recall": 0.0, "precision": 0.0, "ndcg": 0.0}
            for item in loaded["items"]
        ]
        groups = evr._group_by_document_ids(details)
        assert set(groups.keys()) == {f"faq_{i}" for i in range(41)}

    def test_load_faq_categories_maps_by_row_index(self):
        cats = evr._load_faq_categories()
        faqs = load_faqs()
        assert cats["faq_0"] == faqs[0]["Category"]
        assert cats["faq_40"] == faqs[40]["Category"]

    def test_group_by_document_ids_multi_id_counts(self):
        details = [
            {"document_id": ["faq_0", "faq_1"], "hit": True, "mrr": 1.0,
             "recall": 1.0, "precision": 1.0, "ndcg": 1.0},
            {"document_id": "faq_0", "hit": False, "mrr": 0.0,
             "recall": 0.0, "precision": 0.0, "ndcg": 0.0},
        ]
        groups = evr._group_by_document_ids(details)
        assert len(groups["faq_0"]) == 2
        assert len(groups["faq_1"]) == 1

    def test_per_category_unknown_bucket(self, capsys):
        report = {
            "details": [
                {"document_id": "faq_0", "hit": True, "mrr": 1.0,
                 "recall": 1.0, "precision": 1.0, "ndcg": 1.0},
                {"document_id": "not_a_faq", "hit": False, "mrr": 0.0,
                 "recall": 0.0, "precision": 0.0, "ndcg": 0.0},
            ]
        }
        cats = evr._load_faq_categories()
        evr._print_per_category_table(report, "Raw", cats)
        out = capsys.readouterr().out
        assert "(unknown)" in out
        assert "Per-Category breakdown (Raw)" in out