import os
import time
from unittest.mock import patch

import pytest

from src.llm import (
    GROQ_RPD_LIMIT,
    GROQ_RPM_LIMIT,
    _enforce_groq_rate_limits,
    _groq_timestamps_day,
    _groq_timestamps_minute,
    ask_llm,
)


@pytest.fixture(autouse=True)
def _clean_rate_limiter_state():
    yield
    _groq_timestamps_minute.clear()
    _groq_timestamps_day.clear()


def _make_mock_completion(text, prompt_tokens=10, completion_tokens=20):
    class MockUsage:
        pass

    class MockChoiceMessage:
        pass

    class MockChoice:
        pass

    class MockCompletion:
        pass

    usage = MockUsage()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    msg = MockChoiceMessage()
    msg.content = text

    choice = MockChoice()
    choice.message = msg

    completion = MockCompletion()
    completion.choices = [choice]
    completion.usage = usage

    return completion


class TestMissingEnvVars:
    def test_raises_when_groq_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                ask_llm("sys", "user")

    def test_raises_when_openai_key_missing_and_groq_fails(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "g-key"}, clear=True):
            with patch("src.llm.Groq") as mock_groq:
                mock_groq.side_effect = Exception("Groq down")
                with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                    ask_llm("sys", "user")


class TestGroqSuccess:
    def test_returns_expected_structure(self):
        fake = _make_mock_completion("Hello from Groq")
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "g-key", "OPENAI_API_KEY": "o-key"}, clear=True),
            patch("src.llm.Groq") as mock_groq_cls,
        ):
            mock_groq_cls.return_value.chat.completions.create.return_value = fake
            result = ask_llm("Be helpful", "Say hi")

        assert result["response"] == "Hello from Groq"
        assert result["provider"] == "groq"
        assert result["model"] == "llama-3.3-70b-versatile"
        assert isinstance(result["latency"], float)
        assert result["latency"] >= 0
        assert result["tokens"]["prompt"] == 10
        assert result["tokens"]["completion"] == 20
        assert result["tokens"]["total"] == 30

    def test_uses_custom_groq_model(self):
        fake = _make_mock_completion("custom")
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "g-key"}, clear=True),
            patch("src.llm.Groq") as mock_groq_cls,
        ):
            mock_groq_cls.return_value.chat.completions.create.return_value = fake
            result = ask_llm("sys", "msg", groq_model="mixtral-8x7b-32768")

        assert result["model"] == "mixtral-8x7b-32768"
        assert result["provider"] == "groq"


class TestOpenaiFallback:
    def test_falls_back_when_groq_fails(self):
        groq_fake = _make_mock_completion("should not see")
        openai_fake = _make_mock_completion("Hello from OpenAI")

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "g-key", "OPENAI_API_KEY": "o-key"}, clear=True),
            patch("src.llm.Groq") as mock_groq_cls,
            patch("src.llm._OpenAI") as mock_openai_cls,
        ):
            mock_groq_cls.return_value.chat.completions.create.side_effect = Exception("Groq rate limit")
            mock_openai_cls.return_value.chat.completions.create.return_value = openai_fake
            result = ask_llm("sys", "msg")

        assert result["response"] == "Hello from OpenAI"
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-5.4-mini"

    def test_falls_back_with_custom_openai_model(self):
        openai_fake = _make_mock_completion("custom fallback")
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "g-key", "OPENAI_API_KEY": "o-key"}, clear=True),
            patch("src.llm.Groq") as mock_groq_cls,
            patch("src.llm._OpenAI") as mock_openai_cls,
        ):
            mock_groq_cls.return_value.chat.completions.create.side_effect = Exception("timeout")
            mock_openai_cls.return_value.chat.completions.create.return_value = openai_fake
            result = ask_llm("sys", "msg", openai_model="gpt-5.4-mini")

        assert result["model"] == "gpt-5.4-mini"
        assert result["provider"] == "openai"


class TestBothFail:
    def test_raises_runtime_error(self):
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "g-key", "OPENAI_API_KEY": "o-key"}, clear=True),
            patch("src.llm.Groq") as mock_groq_cls,
            patch("src.llm._OpenAI") as mock_openai_cls,
        ):
            mock_groq_cls.return_value.chat.completions.create.side_effect = Exception("Groq error")
            mock_openai_cls.return_value.chat.completions.create.side_effect = Exception("OpenAI error")
            with pytest.raises(RuntimeError, match="Both Groq and OpenAI failed"):
                ask_llm("sys", "msg")


class TestGroqRateLimiter:
    def test_sleeps_when_rpm_exceeded(self):
        _groq_timestamps_minute.extend([time.time() - 59.5] * GROQ_RPM_LIMIT)
        _groq_timestamps_day.extend([time.time() - 100] * GROQ_RPM_LIMIT)
        with patch("src.llm.time.sleep") as mock_sleep:
            _enforce_groq_rate_limits()
            mock_sleep.assert_called_once()

    def test_raises_when_rpd_exceeded(self):
        _groq_timestamps_minute.extend([time.time() - 100] * GROQ_RPM_LIMIT)
        _groq_timestamps_day.extend([time.time() - 100] * GROQ_RPD_LIMIT)
        with pytest.raises(RuntimeError, match="daily request limit"):
            _enforce_groq_rate_limits()

    def test_allows_request_when_under_limits(self):
        _enforce_groq_rate_limits()
        assert len(_groq_timestamps_minute) == 1
        assert len(_groq_timestamps_day) == 1
