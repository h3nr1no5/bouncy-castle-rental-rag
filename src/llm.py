import os
import time
from collections import deque

from groq import Groq
from openai import OpenAI as _OpenAI

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"

GROQ_RPM_LIMIT = 25
GROQ_RPD_LIMIT = 900

_groq_timestamps_minute = deque()
_groq_timestamps_day = deque()


def _enforce_groq_rate_limits():
    now = time.time()
    minute_ago = now - 60
    day_ago = now - 86400

    while _groq_timestamps_minute and _groq_timestamps_minute[0] < minute_ago:
        _groq_timestamps_minute.popleft()
    while _groq_timestamps_day and _groq_timestamps_day[0] < day_ago:
        _groq_timestamps_day.popleft()

    if len(_groq_timestamps_minute) >= GROQ_RPM_LIMIT:
        sleep_time = _groq_timestamps_minute[0] + 60 - now
        if sleep_time > 0:
            time.sleep(sleep_time)

    if len(_groq_timestamps_day) >= GROQ_RPD_LIMIT:
        raise RuntimeError(
            f"Groq daily request limit ({GROQ_RPD_LIMIT}) reached. "
            "Switch to a paid Groq plan or wait until the window resets."
        )

    _groq_timestamps_minute.append(time.time())
    _groq_timestamps_day.append(time.time())


def ask_llm(
    system_prompt,
    user_message,
    groq_model=None,
    openai_model=None,
):
    if groq_model is None:
        groq_model = DEFAULT_GROQ_MODEL
    if openai_model is None:
        openai_model = DEFAULT_OPENAI_MODEL

    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not groq_key and not openai_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    start = time.time()
    groq_error = None

    if groq_key:
        try:
            _enforce_groq_rate_limits()
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model=groq_model,
                messages=messages,
            )
            return _build_result(response, "groq", groq_model, start)
        except Exception as e:
            groq_error = e

    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    try:
        client = _OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model=openai_model,
            messages=messages,
        )
        return _build_result(response, "openai", openai_model, start)
    except Exception as e:
        if groq_error:
            raise RuntimeError(
                f"Both Groq and OpenAI failed. Groq: {groq_error}. OpenAI: {e}"
            )
        raise


def _build_result(response, provider, model, start):
    latency = time.time() - start
    return {
        "response": response.choices[0].message.content,
        "model": model,
        "provider": provider,
        "latency": round(latency, 3),
        "tokens": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
    }
