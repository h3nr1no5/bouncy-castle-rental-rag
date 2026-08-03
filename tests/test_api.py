from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

SAMPLE_CONTEXTS = [
    {
        "id": "faq_0",
        "category": "Pricing",
        "question": "How much does a castle cost?",
        "answer": "Prices start at $100.",
        "text": "Pricing: How much does a castle cost? Prices start at $100.",
        "score": 0.85,
    }
]

LLM_RESULT = {
    "response": "Prices start at $100.",
    "model": "llama-3.3-70b-versatile",
    "provider": "groq",
    "latency": 0.5,
    "cost": 0.000076,
    "tokens": {"prompt": 100, "completion": 20, "total": 120},
}


class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestChat:
    def test_returns_answer_and_metadata(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
            patch("app._log_interaction", return_value=7),
        ):
            resp = client.post("/api/chat", json={"question": "How much does a castle cost?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == LLM_RESULT["response"]
        assert data["contexts"] == SAMPLE_CONTEXTS
        assert data["model"] == LLM_RESULT["model"]
        assert data["provider"] == LLM_RESULT["provider"]
        assert data["latency"] == LLM_RESULT["latency"]
        assert data["cost"] == LLM_RESULT["cost"]
        assert data["tokens"] == LLM_RESULT["tokens"]
        assert data["interaction_id"] == 7

    def test_interaction_id_none_when_logging_skipped(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT),
            patch("app._log_interaction", return_value=None),
        ):
            resp = client.post("/api/chat", json={"question": "cost"})

        assert resp.status_code == 200
        assert resp.json()["interaction_id"] is None

    def test_graceful_error_on_llm_failure(self):
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", side_effect=RuntimeError("LLM failed")),
        ):
            resp = client.post("/api/chat", json={"question": "cost"})

        assert resp.status_code == 502
        assert "LLM failed" in resp.json()["detail"]

    def test_chat_with_history_includes_history_in_answer_prompt(self):
        """POST /api/chat with a history body produces an answer whose generation prompt contains the history."""
        history = [
            {"role": "user", "content": "Do you rent bouncy castles?"},
            {"role": "assistant", "content": "Yes, we do."},
        ]
        with (
            patch("src.rag.search", return_value=SAMPLE_CONTEXTS),
            patch("src.rag.ask_llm", return_value=LLM_RESULT) as mock_llm,
            patch("app._log_interaction", return_value=None),
        ):
            resp = client.post(
                "/api/chat",
                json={"question": "how much for a weekend?", "history": history},
            )

        assert resp.status_code == 200
        assert resp.json()["answer"] == LLM_RESULT["response"]

        # The final answer call (last ask_llm call) must include the history.
        # (rewrite_enabled is true in tuned_params.json, so there may be a rewrite call first.)
        _, kwargs = mock_llm.call_args_list[-1]
        assert "User: Do you rent bouncy castles?" in kwargs["system_prompt"]
        assert "Assistant: Yes, we do." in kwargs["system_prompt"]
        # Raw latest question remains the user_message
        assert kwargs["user_message"] == "how much for a weekend?"

    def test_rejects_empty_question(self):
        resp = client.post("/api/chat", json={"question": ""})
        assert resp.status_code == 422

    def test_rejects_missing_question(self):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422


class TestFeedback:
    def test_persists_feedback(self):
        with patch("app.update_feedback") as mock_update:
            resp = client.post("/api/feedback", json={"interaction_id": 7, "feedback": "up"})

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_update.assert_called_once_with(7, "up")

    def test_unknown_id_returns_404(self):
        with patch("app.update_feedback", side_effect=ValueError("No interaction found with id 999")):
            resp = client.post("/api/feedback", json={"interaction_id": 999, "feedback": "up"})

        assert resp.status_code == 404

    def test_invalid_feedback_returns_422(self):
        resp = client.post("/api/feedback", json={"interaction_id": 7, "feedback": "sideways"})
        assert resp.status_code == 422

    def test_rejects_missing_interaction_id(self):
        resp = client.post("/api/feedback", json={"feedback": "up"})
        assert resp.status_code == 422


class TestCors:
    def test_cors_headers_present(self):
        resp = client.get("/health", headers={"Origin": "https://example.com"})
        assert resp.headers.get("access-control-allow-origin") == "*"

    def test_cors_headers_on_api_post(self):
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"
