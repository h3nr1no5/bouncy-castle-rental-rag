from unittest.mock import patch

import pytest

from src.rag import answer_question, rewrite_query, rewrite_query_with_history, _format_history_for_prompt

SAMPLE_CONTEXTS = [
    {
        "category": "Pricing",
        "question": "How much does a castle cost?",
        "answer": "Prices start at $100.",
        "text": "Pricing: How much does a castle cost? Prices start at $100.",
        "score": 0.85,
    },
    {
        "category": "Booking",
        "question": "How do I book?",
        "answer": "Call us or visit our website.",
        "text": "Booking: How do I book? Call us or visit our website.",
        "score": 0.72,
    },
]

LLM_RESULT = {
    "response": "Prices start at $100. You can book by calling us or visiting our website.",
    "model": "llama-3.3-70b-versatile",
    "provider": "groq",
    "latency": 0.5,
    "cost": 0.000076,
    "tokens": {"prompt": 100, "completion": 20, "total": 120},
}

REWRITE_LLM_RESULT = {
    "response": "bouncy castle rental price cost",
    "model": "llama-3.3-70b-versatile",
    "provider": "groq",
    "latency": 0.3,
    "cost": 0.000050,
    "tokens": {"prompt": 50, "completion": 10, "total": 60},
}


class TestRewriteQuery:
    def test_rewrite_success_returns_rewritten_query(self):
        with patch("src.rag.ask_llm", return_value=REWRITE_LLM_RESULT) as mock_llm:
            result = rewrite_query("How much does a castle cost?")
        
        assert result == "bouncy castle rental price cost"
        mock_llm.assert_called_once()
        _, kwargs = mock_llm.call_args
        assert "How much does a castle cost?" in kwargs["user_message"]
        assert kwargs["groq_model"] is None
        assert kwargs["openai_model"] is None

    def test_rewrite_forwards_model_kwargs(self):
        with patch("src.rag.ask_llm", return_value=REWRITE_LLM_RESULT) as mock_llm:
            rewrite_query("cost", groq_model="mixtral-8x7b-32768", openai_model="gpt-5.4-mini")
        
        mock_llm.assert_called_once()
        _, kwargs = mock_llm.call_args
        assert kwargs["groq_model"] == "mixtral-8x7b-32768"
        assert kwargs["openai_model"] == "gpt-5.4-mini"

    def test_rewrite_failure_falls_back_to_original(self):
        with patch("src.rag.ask_llm", side_effect=RuntimeError("Both providers failed")):
            result = rewrite_query("How much does a castle cost?")
        
        assert result == "How much does a castle cost?"

    def test_rewrite_value_error_falls_back_to_original(self):
        with patch("src.rag.ask_llm", side_effect=ValueError("No API key")):
            result = rewrite_query("How much does a castle cost?")
        
        assert result == "How much does a castle cost?"

    def test_rewrite_empty_response_falls_back_to_original(self):
        empty_result = {**REWRITE_LLM_RESULT, "response": ""}
        with patch("src.rag.ask_llm", return_value=empty_result):
            result = rewrite_query("How much does a castle cost?")
        
        assert result == "How much does a castle cost?"

    def test_rewrite_whitespace_response_falls_back_to_original(self):
        whitespace_result = {**REWRITE_LLM_RESULT, "response": "   \n\t  "}
        with patch("src.rag.ask_llm", return_value=whitespace_result):
            result = rewrite_query("How much does a castle cost?")
        
        assert result == "How much does a castle cost?"

    def test_rewrite_none_input_returns_none(self):
        result = rewrite_query(None)
        assert result is None

    def test_rewrite_empty_string_returns_empty_string(self):
        result = rewrite_query("")
        assert result == ""

    def test_rewrite_whitespace_string_returns_whitespace_string(self):
        result = rewrite_query("   ")
        assert result == "   "

    def test_rewrite_returns_string(self):
        with patch("src.rag.ask_llm", return_value=REWRITE_LLM_RESULT):
            result = rewrite_query("cost")
        
        assert isinstance(result, str)

    def test_rewrite_none_llm_result_falls_back_to_original(self):
        """When ask_llm returns None, fall back to original question."""
        with patch("src.rag.ask_llm", return_value=None):
            result = rewrite_query("How much does a castle cost?")
        
        assert result == "How much does a castle cost?"

    def test_rewrite_none_response_falls_back_to_original(self):
        """When ask_llm returns a dict with None response, fall back to original question."""
        with patch("src.rag.ask_llm", return_value={"response": None, "model": "test", "provider": "test", "latency": 0.1, "cost": 0.0, "tokens": {"prompt": 1, "completion": 1, "total": 2}}):
            result = rewrite_query("How much does a castle cost?")
        
        assert result == "How much does a castle cost?"

    def test_rewrite_non_string_response_falls_back_to_original(self):
        """When ask_llm returns a dict with non-string response, fall back to original question."""
        with patch("src.rag.ask_llm", return_value={"response": 123, "model": "test", "provider": "test", "latency": 0.1, "cost": 0.0, "tokens": {"prompt": 1, "completion": 1, "total": 2}}):
            result = rewrite_query("How much does a castle cost?")
        
        assert result == "How much does a castle cost?"

    def test_rewrite_attribute_error_falls_back_to_original(self):
        """When ask_llm raises AttributeError (e.g., result is None), fall back to original question."""
        with patch("src.rag.ask_llm", side_effect=AttributeError("'NoneType' object has no attribute 'get'")):
            result = rewrite_query("How much does a castle cost?")
        
        assert result == "How much does a castle cost?"


class TestAnswerQuestion:
    def test_returns_expected_structure(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
        ):
            result = answer_question("How much does a castle cost?")

        assert "answer" in result
        assert "contexts" in result
        assert "model" in result
        assert "provider" in result
        assert "latency" in result
        assert "tokens" in result
        assert result["answer"] == LLM_RESULT["response"]
        assert result["contexts"] == SAMPLE_CONTEXTS
        assert result["model"] == "llama-3.3-70b-versatile"
        assert result["provider"] == "groq"

    def test_passes_question_to_search(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question("booking")

        mock_search.assert_called_once()
        args, kwargs = mock_search.call_args
        assert args[0] == "booking"

    def test_passes_k_to_search(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
        ):
            answer_question("booking", k=3)

        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        assert kwargs.get("k") == 3

    def test_does_not_pass_k_when_default(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
        ):
            answer_question("booking")

        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        assert kwargs.get("k") is None

    def test_injects_contexts_into_prompt(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question("cost")

        mock_llm.assert_called_once()
        _, kwargs = mock_llm.call_args
        system_prompt = kwargs["system_prompt"]
        assert "Prices start at $100" in system_prompt
        assert "Call us or visit our website" in system_prompt
        assert "How much does a castle cost?" in system_prompt

    def test_handles_empty_search_results(self):
        with (
            patch("src.rag.search", return_value=[]),
            patch("src.rag.ask_llm", return_value=LLM_RESULT) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            result = answer_question("unknown topic")

        assert result["contexts"] == []
        mock_llm.assert_called_once()
        _, kwargs = mock_llm.call_args
        assert "No relevant FAQ entries found" in kwargs["system_prompt"]

    def test_propagates_llm_errors(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", side_effect=RuntimeError("LLM failed")),
        ):
            with pytest.raises(RuntimeError, match="LLM failed"):
                answer_question("cost")

    def test_forwards_search_path_kwargs(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
        ):
            answer_question(
                "cost",
                bm25_path="/tmp/bm25.pkl",
                faiss_path="/tmp/faiss.bin",
                docs_path="/tmp/docs.json",
            )

        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        assert kwargs["bm25_path"] == "/tmp/bm25.pkl"
        assert kwargs["faiss_path"] == "/tmp/faiss.bin"
        assert kwargs["docs_path"] == "/tmp/docs.json"

    def test_forwards_llm_model_kwargs(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question("cost", groq_model="mixtral-8x7b-32768", openai_model="gpt-5.4-mini")

        mock_llm.assert_called_once()
        _, kwargs = mock_llm.call_args
        assert kwargs["groq_model"] == "mixtral-8x7b-32768"
        assert kwargs["openai_model"] == "gpt-5.4-mini"

    def test_does_not_log_interaction(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
            patch("src.db.log_interaction") as mock_log,
        ):
            answer_question("cost")

        mock_log.assert_not_called()

    def test_rewrite_enabled_false_uses_raw_question_for_search(self):
        """When rewrite_enabled=False, the raw question is passed to search()."""
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question("booking")

        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "booking"

    def test_rewrite_enabled_true_uses_rewritten_query_for_search(self):
        """When rewrite_enabled=True, the rewritten query is passed to search()."""
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", side_effect=[REWRITE_LLM_RESULT, LLM_RESULT]) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": True, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question("cost")

        # search() should be called with the rewritten query
        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "bouncy castle rental price cost"
        
        # ask_llm() should be called twice: once for rewrite, once for answer
        assert mock_llm.call_count == 2
        # First call is for rewrite
        _, kwargs1 = mock_llm.call_args_list[0]
        assert "cost" in kwargs1["user_message"]
        # Second call is for answer with raw question
        _, kwargs2 = mock_llm.call_args_list[1]
        assert kwargs2["user_message"] == "cost"

    def test_rewrite_failure_falls_back_to_raw_question(self):
        """When rewrite fails, the raw question is used for search()."""
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", side_effect=[RuntimeError("LLM failed"), LLM_RESULT]),
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": True, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question("cost")

        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "cost"  # Falls back to raw question

    def test_rewrite_empty_falls_back_to_raw_question(self):
        """When rewrite returns empty, the raw question is used for search()."""
        empty_rewrite = {**REWRITE_LLM_RESULT, "response": ""}
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", side_effect=[empty_rewrite, LLM_RESULT]),
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": True, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question("cost")

        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "cost"  # Falls back to raw question

    def test_rewrite_does_not_increase_k(self):
        """Rewriting does not increase the number of FAQ entries sent to the LLM prompt beyond k."""
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", side_effect=[REWRITE_LLM_RESULT, LLM_RESULT]),
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": True, "k": 3, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question("cost", k=3)

        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        assert kwargs.get("k") == 3

    def test_answer_question_multi_turn_rewrite_passes_rewritten_query_to_search(self):
        """With history_rewrite_enabled=True and non-empty history, search() receives the rewritten query and ask_llm receives raw question for final answer."""
        MULTI_TURN_REWRITE_RESULT = {
            "response": "bouncy castle weekend rental price cost",
            "model": "llama-3.3-70b-versatile",
            "provider": "groq",
            "latency": 0.3,
            "cost": 0.000050,
            "tokens": {"prompt": 50, "completion": 10, "total": 60},
        }
        history = [{"role": "user", "content": "Do you rent bouncy castles?"}]
        
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", side_effect=[MULTI_TURN_REWRITE_RESULT, LLM_RESULT]) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"history_rewrite_enabled": True, "history_turns": 4, "rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question(
                question="how much for a weekend?",
                history=history,
                groq_model=None,
                openai_model=None,
            )

        # search() should be called with the rewritten query from multi-turn rewrite
        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "bouncy castle weekend rental price cost"
        
        # ask_llm() should be called twice: once for multi-turn rewrite, once for final answer
        assert mock_llm.call_count == 2
        
        # First call: multi-turn rewrite with history
        _, kwargs1 = mock_llm.call_args_list[0]
        assert "how much for a weekend?" in kwargs1["user_message"]
        assert "Do you rent bouncy castles?" in kwargs1["user_message"]
        
        # Second call: final answer with RAW question as user_message
        _, kwargs2 = mock_llm.call_args_list[1]
        assert kwargs2["user_message"] == "how much for a weekend?"

    def test_answer_question_empty_history_uses_single_turn(self):
        """With history=None or history=[], rewrite_enabled=False: search gets raw question; ask_llm called once (no rewrite call)."""
        # Test history=None
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=LLM_RESULT) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question(question="cost", history=None)

        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "cost"
        assert mock_llm.call_count == 1

        # Test history=[]
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=LLM_RESULT) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question(question="cost", history=[])

        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "cost"
        assert mock_llm.call_count == 1

    def test_answer_question_history_truncated_to_history_turns(self):
        """With history_turns=2 and 4 prior turns (8 messages), search receives rewritten query using last 2 messages; ask_llm called twice."""
        MULTI_TURN_REWRITE_RESULT = {
            "response": "bouncy castle weekend rental price cost",
            "model": "llama-3.3-70b-versatile",
            "provider": "groq",
            "latency": 0.3,
            "cost": 0.000050,
            "tokens": {"prompt": 50, "completion": 10, "total": 60},
        }
        # 4 prior turns (8 messages), but history_turns=2 means only last 2 MESSAGES should be used
        history = [
            {"role": "user", "content": "Do you rent bouncy castles?"},
            {"role": "assistant", "content": "Yes, we do."},
            {"role": "user", "content": "What sizes are available?"},
            {"role": "assistant", "content": "Small, medium, large."},
            {"role": "user", "content": "How much for a day?"},
            {"role": "assistant", "content": "$100 per day."},
            {"role": "user", "content": "Do you deliver?"},
            {"role": "assistant", "content": "Yes, within 50 miles."},
        ]

        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", side_effect=[MULTI_TURN_REWRITE_RESULT, LLM_RESULT]) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"history_rewrite_enabled": True, "history_turns": 2, "rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question(
                question="how much for a weekend?",
                history=history,
                groq_model=None,
                openai_model=None,
            )

        # search() should be called with the rewritten query
        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "bouncy castle weekend rental price cost"

        # ask_llm() should be called twice: once for multi-turn rewrite, once for final answer
        assert mock_llm.call_count == 2

        # First call: multi-turn rewrite with history (should only include last 2 messages = 1 turn)
        _, kwargs1 = mock_llm.call_args_list[0]
        assert "how much for a weekend?" in kwargs1["user_message"]
        # Should include the last 2 messages: "Do you deliver?" / "Yes, within 50 miles."
        assert "Do you deliver?" in kwargs1["user_message"]
        assert "Yes, within 50 miles." in kwargs1["user_message"]
        # Should NOT include earlier messages
        assert "Do you rent bouncy castles?" not in kwargs1["user_message"]
        assert "What sizes are available?" not in kwargs1["user_message"]
        assert "How much for a day?" not in kwargs1["user_message"]

        # Second call: final answer with RAW question as user_message
        _, kwargs2 = mock_llm.call_args_list[1]
        assert kwargs2["user_message"] == "how much for a weekend?"

    def test_answer_question_multi_turn_rewrite_failure_falls_back_to_raw(self):
        """When multi-turn rewrite raises Exception, falls back to single-turn rewrite which also fails, so search receives raw question; no exception propagates."""
        history = [{"role": "user", "content": "Do you rent bouncy castles?"}]

        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", side_effect=[Exception("boom"), Exception("boom"), LLM_RESULT]) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"history_rewrite_enabled": True, "history_turns": 4, "rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question(
                question="how much for a weekend?",
                history=history,
                groq_model=None,
                openai_model=None,
            )

        # search() should be called with the RAW question (fallback after both rewrites fail)
        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "how much for a weekend?"

        # ask_llm() should be called 3 times: multi-turn rewrite fails, single-turn rewrite fails, final answer succeeds
        assert mock_llm.call_count == 3

    def test_answer_question_malformed_history_degraded_safely(self):
        """Malformed history entries (missing role/content, non-string content, unknown role) don't raise; search gets raw question with rewrite_enabled=False."""
        malformed_history = [
            {"role": "user"},  # missing content
            {"content": "hello"},  # missing role
            {"role": "user", "content": 123},  # non-string content
            {"role": "unknown", "content": "test"},  # unknown role
            {"role": "assistant", "content": None},  # None content
            "not a dict",  # not a dict at all
        ]

        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=LLM_RESULT) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question(question="cost", history=malformed_history)

        # Should not raise, search gets raw question
        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "cost"
        assert mock_llm.call_count == 1

    def test_answer_question_history_rewrite_disabled_uses_raw(self):
        """With history_rewrite_enabled=False and non-empty history, search receives raw question; ask_llm called once."""
        history = [{"role": "user", "content": "Do you rent bouncy castles?"}]

        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=LLM_RESULT) as mock_llm,
            patch("src.rag.load_tuned_params", return_value={"history_rewrite_enabled": False, "history_turns": 4, "rewrite_enabled": False, "k": 5, "rrf_k": 1, "cat_weight": 0, "bm25_k1": 1.5, "bm25_b": 0.75}),
        ):
            answer_question(
                question="how much for a weekend?",
                history=history,
                groq_model=None,
                openai_model=None,
            )

        # search() should be called with the RAW question (no rewrite)
        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert args[0] == "how much for a weekend?"

        # ask_llm() should be called only once (no rewrite call)
        assert mock_llm.call_count == 1


class TestAnswerQuestionHistory:
    """History threading into the final answer-generation prompt (issue #53)."""

    DEFAULT_PARAMS = {
        "rewrite_enabled": False,
        "history_rewrite_enabled": False,
        "history_turns": 4,
        "k": 5,
        "rrf_k": 1,
        "cat_weight": 0,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    }

    HISTORY = [
        {"role": "user", "content": "Do you rent bouncy castles?"},
        {"role": "assistant", "content": "Yes, we do."},
    ]

    def _answer(self, question="how much for a weekend?", history=HISTORY, params=None, llm_result=LLM_RESULT):
        params = params or self.DEFAULT_PARAMS
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=llm_result) as mock_llm,
            patch("src.rag.load_tuned_params", return_value=params),
        ):
            result = answer_question(question=question, history=history)
        return result, mock_search, mock_llm

    def test_history_included_in_answer_prompt(self):
        """Non-empty valid history is formatted as User:/Assistant: lines in the answer prompt; raw question stays user_message."""
        _, _, mock_llm = self._answer("how much for a weekend?")

        mock_llm.assert_called_once()
        _, kwargs = mock_llm.call_args
        system_prompt = kwargs["system_prompt"]
        assert "User: Do you rent bouncy castles?" in system_prompt
        assert "Assistant: Yes, we do." in system_prompt
        # FAQ contexts still present alongside history
        assert "Prices start at $100" in system_prompt
        # Raw question remains the user_message
        assert kwargs["user_message"] == "how much for a weekend?"

    def test_history_included_independent_of_rewrite_flags(self):
        """History is included in the answer prompt even when rewrite flags are enabled."""
        params = {
            "rewrite_enabled": True,
            "history_rewrite_enabled": False,
            "history_turns": 4,
            "k": 5,
            "rrf_k": 1,
            "cat_weight": 0,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
        }
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", side_effect=[REWRITE_LLM_RESULT, LLM_RESULT]) as mock_llm,
            patch("src.rag.load_tuned_params", return_value=params),
        ):
            answer_question(question="how much for a weekend?", history=self.HISTORY)

        # Two calls: rewrite + final answer. Final answer must include history.
        assert mock_llm.call_count == 2
        _, kwargs2 = mock_llm.call_args_list[1]
        assert "User: Do you rent bouncy castles?" in kwargs2["system_prompt"]
        assert "Assistant: Yes, we do." in kwargs2["system_prompt"]
        assert kwargs2["user_message"] == "how much for a weekend?"

    def test_history_bounded_to_history_turns(self):
        """History in the answer prompt is truncated to the last history_turns messages."""
        params = dict(self.DEFAULT_PARAMS, history_turns=2)
        long_history = [
            {"role": "user", "content": "Do you rent bouncy castles?"},
            {"role": "assistant", "content": "Yes, we do."},
            {"role": "user", "content": "What sizes are available?"},
            {"role": "assistant", "content": "Small, medium, large."},
            {"role": "user", "content": "How much for a day?"},
            {"role": "assistant", "content": "$100 per day."},
        ]
        _, _, mock_llm = self._answer("how much for a weekend?", history=long_history, params=params)

        _, kwargs = mock_llm.call_args
        system_prompt = kwargs["system_prompt"]
        # Only the last 2 messages are included
        assert "User: How much for a day?" in system_prompt
        assert "Assistant: $100 per day." in system_prompt
        # Earlier messages are truncated, not rejected
        assert "Do you rent bouncy castles?" not in system_prompt
        assert "What sizes are available?" not in system_prompt

    def test_no_history_prompt_identical_to_single_turn(self):
        """With history=None or history=[], the answer prompt has no history section."""
        for empty in (None, []):
            _, _, mock_llm = self._answer("how much for a weekend?", history=empty)
            _, kwargs = mock_llm.call_args
            system_prompt = kwargs["system_prompt"]
            assert "Conversation history" not in system_prompt
            assert "User:" not in system_prompt
            assert "Assistant:" not in system_prompt
            assert kwargs["user_message"] == "how much for a weekend?"

    def test_malformed_history_filtered_and_no_raise(self):
        """Malformed history entries are filtered out and never raise."""
        malformed = [
            {"role": "user"},  # missing content
            {"content": "hello"},  # missing role
            {"role": "user", "content": 123},  # non-string content
            {"role": "unknown", "content": "test"},  # unknown role
            {"role": "assistant", "content": None},  # None content
            "not a dict",  # not a dict at all
        ]
        _, _, mock_llm = self._answer("cost", history=malformed)

        _, kwargs = mock_llm.call_args
        system_prompt = kwargs["system_prompt"]
        assert "Conversation history" not in system_prompt
        assert kwargs["user_message"] == "cost"

    def test_only_user_turns_produces_valid_answer(self):
        """History with only user turns still produces a valid answer with history included."""
        history = [{"role": "user", "content": "Do you rent bouncy castles?"}]
        _, _, mock_llm = self._answer("how much for a weekend?", history=history)

        _, kwargs = mock_llm.call_args
        assert "User: Do you rent bouncy castles?" in kwargs["system_prompt"]
        assert kwargs["user_message"] == "how much for a weekend?"

    def test_only_assistant_turns_produces_valid_answer(self):
        """History with only assistant turns still produces a valid answer with history included."""
        history = [{"role": "assistant", "content": "Yes, we do."}]
        _, _, mock_llm = self._answer("how much for a weekend?", history=history)

        _, kwargs = mock_llm.call_args
        assert "Assistant: Yes, we do." in kwargs["system_prompt"]
        assert kwargs["user_message"] == "how much for a weekend?"

    def test_current_question_not_duplicated_into_history(self):
        """The current question is not duplicated into the history section."""
        question = "how much for a weekend?"
        _, _, mock_llm = self._answer(question)

        _, kwargs = mock_llm.call_args
        system_prompt = kwargs["system_prompt"]
        # The question appears only as the user_message, not in the history section
        assert question not in system_prompt
        assert kwargs["user_message"] == question

    def test_llm_error_propagates_with_history(self):
        """If the final answer ask_llm call fails, the error propagates (no new failure modes)."""
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", side_effect=RuntimeError("LLM failed")),
            patch("src.rag.load_tuned_params", return_value=self.DEFAULT_PARAMS),
        ):
            with pytest.raises(RuntimeError, match="LLM failed"):
                answer_question(question="cost", history=self.HISTORY)

    def test_faq_context_count_not_exceeded(self):
        """The number of FAQ entries in the answer prompt does not exceed k (no inflation by history)."""
        params = dict(self.DEFAULT_PARAMS, k=5)
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS) as mock_search,
            patch("src.rag.ask_llm", return_value=LLM_RESULT) as mock_llm,
            patch("src.rag.load_tuned_params", return_value=params),
        ):
            answer_question(question="cost", history=self.HISTORY, k=5)

        mock_search.assert_called_once()
        _, kwargs = mock_llm.call_args
        system_prompt = kwargs["system_prompt"]
        # The prompt contains exactly the FAQ entries search returned (bounded by k=5)
        assert system_prompt.count("Category:") == len(SAMPLE_CONTEXTS)
        assert system_prompt.count("Category:") <= 5
