import json
import os
import time
import urllib.request

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

GROQ_MODEL = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-5.4-mini"
LLM_TIMEOUT = 30
LLM_MAX_RETRIES = 2


def _fetch(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


def _check_eip():
    try:
        info = _fetch("https://ipinfo.io/json")
        org = info.get("org", "unknown")
        return (
            f"{info.get('city', '?')}, {info.get('region', '?')}, "
            f"{info.get('country', '?')} — {org}"
        )
    except Exception as e:
        return f"could not determine ({e})"


def _check_provider(name, client_call):
    start = time.time()
    try:
        client_call()
        elapsed = round(time.time() - start, 2)
        print(f"[PASS] {name} ({elapsed}s)")
        return True
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        msg = str(e)
        print(f"[FAIL] {name} ({elapsed}s)")
        print(f"       {type(e).__name__}: {msg[:300]}")
        if "403" in msg:
            print(
                "       Hint: egress IP flagged/blocked by Groq — "
                "switch VPN server (prefer a residential/dedicated IP)."
            )
        return False


def _check_groq_api():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("[SKIP] Groq models endpoint (no GROQ_API_KEY)")
        return
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        print(f"[OK]   Groq /v1/models reachable ({len(data.get('data', []))} models)")
    except Exception as e:
        print(f"[FAIL] Groq /v1/models: {e}")


def main():
    load_dotenv()
    print(f"Egress: {_check_eip()}")

    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    print(f"Keys:  GROQ_API_KEY={'set' if groq_key else 'MISSING'}, "
          f"OPENAI_API_KEY={'set' if openai_key else 'MISSING'}")

    messages = [{"role": "user", "content": "hi"}]

    _check_groq_api()

    groq_ok = False
    if groq_key:
        client = Groq(api_key=groq_key, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
        groq_ok = _check_provider(
            f"Groq {GROQ_MODEL}",
            lambda: client.chat.completions.create(model=GROQ_MODEL, messages=messages, max_tokens=5),
        )

    if openai_key:
        client = OpenAI(api_key=openai_key, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
        _check_provider(
            f"OpenAI {OPENAI_MODEL}",
            lambda: client.chat.completions.create(model=OPENAI_MODEL, messages=messages, max_completion_tokens=5),
        )

    print()
    if groq_ok:
        print("Verdict: Groq is reachable — primary provider will be used.")
    else:
        print("Verdict: Groq blocked → ask_llm will fall back to OpenAI.")


if __name__ == "__main__":
    main()
