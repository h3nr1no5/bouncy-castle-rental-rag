from unittest.mock import patch

import pytest

from src.rag import answer_question

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

    def test_injects_contexts_into_prompt(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT) as mock_llm,
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
        ):
            answer_question("cost", groq_model="mixtral-8x7b-32768", openai_model="gpt-5.4-mini")

        mock_llm.assert_called_once()
        _, kwargs = mock_llm.call_args
        assert kwargs["groq_model"] == "mixtral-8x7b-32768"
        assert kwargs["openai_model"] == "gpt-5.4-mini"

    def test_returns_none_interaction_id_when_logging_skipped(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
            patch("src.rag._log_interaction", return_value=None),
        ):
            result = answer_question("cost")

        assert result["interaction_id"] is None

    def test_returns_interaction_id_from_logging(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
            patch("src.rag._log_interaction", return_value=42),
        ):
            result = answer_question("cost")

        assert result["interaction_id"] == 42

    def test_passes_metadata_to_log_interaction(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
            patch("src.rag._log_interaction") as mock_log,
        ):
            answer_question("cost")

        mock_log.assert_called_once()
        _, kwargs = mock_log.call_args
        assert kwargs["question"] == "cost"
        assert kwargs["answer"] == LLM_RESULT["response"]
        assert kwargs["metadata"]["model"] == LLM_RESULT["model"]
        assert kwargs["metadata"]["provider"] == LLM_RESULT["provider"]
        assert kwargs["metadata"]["tokens"] == LLM_RESULT["tokens"]
        assert kwargs["metadata"]["latency"] == LLM_RESULT["latency"]
        assert kwargs["metadata"]["cost"] == LLM_RESULT["cost"]
