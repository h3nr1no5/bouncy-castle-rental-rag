"""Unit tests for generate_ground_truth_multi_turn.py.

These tests exercise only pure, API-free logic: the pydantic structured-output
models, conversation-row building, CSV serialization, the retry/backoff loop
(with a fake client), and the no-key early exit. No network or API keys are
required.
"""

import csv
import os

import pytest

import generate_ground_truth_multi_turn as gen


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeParsed:
    def __init__(self, conversations):
        self.conversations = conversations


class _FakeContent:
    def __init__(self, parsed):
        self.parsed = parsed


class _FakeOutputItem:
    def __init__(self, parsed):
        self.content = [_FakeContent(parsed)]


class _FakeResponse:
    def __init__(self, conversations, input_tokens=10, output_tokens=20):
        self.output = [_FakeOutputItem(_FakeParsed(conversations))]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeResponses:
    def __init__(self, fail_times=0, conversations=None):
        self.fail_times = fail_times
        self.conversations = conversations or []
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("boom")
        return _FakeResponse(self.conversations)


class _FakeClient:
    def __init__(self, fail_times=0, conversations=None):
        self.responses = _FakeResponses(fail_times, conversations)


def _sample_conversations():
    return [
        gen.Conversation(
            prior_user_turns=["Do you rent bouncy castles?", "For a weekend?"],
            follow_up_question="How much for a weekend?",
        ),
        gen.Conversation(
            prior_user_turns=["I want one for my son's birthday"],
            follow_up_question="Do I need a deposit for it?",
        ),
    ]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
def test_conversation_model_parses():
    conv = gen.Conversation.model_validate(
        {"prior_user_turns": ["Do you rent castles?"], "follow_up_question": "How much?"}
    )
    assert conv.prior_user_turns == ["Do you rent castles?"]
    assert conv.follow_up_question == "How much?"


def test_conversations_model_holds_list():
    data = {
        "conversations": [
            {"prior_user_turns": ["a"], "follow_up_question": "b"},
            {"prior_user_turns": ["c", "d"], "follow_up_question": "e"},
        ]
    }
    parsed = gen.Conversations.model_validate(data)
    assert len(parsed.conversations) == 2
    assert parsed.conversations[1].prior_user_turns == ["c", "d"]


# ---------------------------------------------------------------------------
# build_rows
# ---------------------------------------------------------------------------
def test_build_rows_schema_and_doc_id():
    rows, next_id = gen.build_rows(_sample_conversations(), faq_index=36)
    assert list(rows[0].keys()) == gen.CSV_COLUMNS
    assert rows[0]["document_id"] == "faq_36"
    assert rows[1]["document_id"] == "faq_36"


def test_build_rows_prior_turns_joined_with_semicolon():
    rows, _ = gen.build_rows(_sample_conversations(), faq_index=0)
    assert rows[0]["prior_user_turns"] == "Do you rent bouncy castles?;For a weekend?"
    assert rows[0]["follow_up_question"] == "How much for a weekend?"


def test_build_rows_unique_incrementing_ids():
    rows, next_id = gen.build_rows(_sample_conversations(), faq_index=0, start_conversation_id=7)
    ids = [r["conversation_id"] for r in rows]
    assert ids == ["7", "8"]
    assert next_id == 9


def test_build_rows_sanitizes_semicolons():
    convs = [
        gen.Conversation(
            prior_user_turns=["Do you rent castles; with slides?"],
            follow_up_question="How much; for a weekend?",
        )
    ]
    rows, _ = gen.build_rows(convs, faq_index=0)
    assert rows[0]["prior_user_turns"] == "Do you rent castles, with slides?"
    assert rows[0]["follow_up_question"] == "How much, for a weekend?"


def test_build_rows_skips_conversation_without_prior_turns():
    convs = [
        gen.Conversation(prior_user_turns=[], follow_up_question="How much?"),
        gen.Conversation(prior_user_turns=["   "], follow_up_question="How much?"),
        gen.Conversation(prior_user_turns=["Do you rent castles?"], follow_up_question="How much?"),
    ]
    rows, _ = gen.build_rows(convs, faq_index=0)
    assert len(rows) == 1
    assert rows[0]["prior_user_turns"] == "Do you rent castles?"


def test_build_rows_skips_conversation_with_blank_follow_up():
    convs = [gen.Conversation(prior_user_turns=["a"], follow_up_question="   ")]
    rows, _ = gen.build_rows(convs, faq_index=0)
    assert rows == []


def test_build_rows_caps_history_at_max_turns():
    convs = [
        gen.Conversation(
            prior_user_turns=[f"turn {i}" for i in range(10)],
            follow_up_question="How much?",
        )
    ]
    rows, _ = gen.build_rows(convs, faq_index=0)
    assert len(rows[0]["prior_user_turns"].split(";")) == gen.MAX_HISTORY_TURNS


# ---------------------------------------------------------------------------
# _generate_for_faq retry/backoff
# ---------------------------------------------------------------------------
def test_generate_for_faq_retries_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(gen.time, "sleep", sleeps.append)
    client = _FakeClient(fail_times=2, conversations=_sample_conversations())
    result = gen._generate_for_faq(client, {"Question": "q", "Answer": "a", "Category": "c"}, index=0)
    assert client.responses.calls == 3
    assert sleeps == [1, 2]  # 2**0, 2**1
    assert len(result["conversations"]) == 2
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20


def test_generate_for_faq_gives_up_after_max_attempts(monkeypatch, capsys):
    sleeps = []
    monkeypatch.setattr(gen.time, "sleep", sleeps.append)
    client = _FakeClient(fail_times=99)
    result = gen._generate_for_faq(client, {"Question": "q", "Answer": "a", "Category": "c"}, index=3)
    assert client.responses.calls == gen.MAX_ATTEMPTS
    assert sleeps == [1, 2, 4]  # 2**0, 2**1, 2**2
    assert result["conversations"] == []
    assert "Failed for FAQ 3" in capsys.readouterr().out


def test_generate_for_faq_retries_on_empty_conversations(monkeypatch):
    sleeps = []
    monkeypatch.setattr(gen.time, "sleep", sleeps.append)
    # First call returns empty list (treated as failure), second succeeds.
    client = _FakeClient(fail_times=1, conversations=_sample_conversations())
    result = gen._generate_for_faq(client, {"Question": "q", "Answer": "a", "Category": "c"}, index=0)
    assert client.responses.calls == 2
    assert sleeps == [1]
    assert len(result["conversations"]) == 2


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------
def test_write_csv_header_and_rows(tmp_path):
    out = tmp_path / "out.csv"
    rows, _ = gen.build_rows(_sample_conversations(), faq_index=36)
    gen.write_csv(out, rows)
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == gen.CSV_COLUMNS
        loaded = list(reader)
    assert len(loaded) == 2
    assert loaded[0]["document_id"] == "faq_36"
    assert loaded[0]["follow_up_question"] == "How much for a weekend?"


def test_write_csv_overwrites(tmp_path):
    out = tmp_path / "out.csv"
    gen.write_csv(out, [])
    gen.write_csv(out, [{"conversation_id": "1", "prior_user_turns": "a", "follow_up_question": "b", "document_id": "faq_0"}])
    with open(out, newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def test_main_exits_without_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(gen, "load_dotenv", lambda: None)
    out = tmp_path / "out.csv"
    monkeypatch.setattr(gen, "OUTPUT_PATH", str(out))
    gen.main()
    assert "OPENAI_API_KEY not set" in capsys.readouterr().out
    assert not out.exists()


def test_main_writes_csv_with_unique_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(gen, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(gen, "OpenAI", lambda api_key: _FakeClient())
    monkeypatch.setattr(gen, "OUTPUT_PATH", str(tmp_path / "out.csv"))

    def fake_generate(client, faq, index):
        return {
            "conversations": _sample_conversations(),
            "input_tokens": 10,
            "output_tokens": 20,
        }

    monkeypatch.setattr(gen, "_generate_for_faq", fake_generate)

    gen.main()

    out = tmp_path / "out.csv"
    assert out.exists()
    with open(out, newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    # 41 FAQs x 2 conversations each
    assert len(loaded) == 2 * len(gen.load_faqs())
    ids = [r["conversation_id"] for r in loaded]
    assert len(ids) == len(set(ids))  # unique
    assert ids == [str(i) for i in range(1, len(loaded) + 1)]
    # every FAQ appears as a document_id
    doc_ids = {r["document_id"] for r in loaded}
    assert doc_ids == {f"faq_{i}" for i in range(len(gen.load_faqs()))}


def test_main_leaves_existing_file_when_all_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(gen, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(gen, "OpenAI", lambda api_key: _FakeClient())
    out = tmp_path / "out.csv"
    out.write_text("conversation_id,prior_user_turns,follow_up_question,document_id\n", encoding="utf-8")
    monkeypatch.setattr(gen, "OUTPUT_PATH", str(out))

    def fake_generate(client, faq, index):
        return {"conversations": [], "input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr(gen, "_generate_for_faq", fake_generate)

    gen.main()

    # Pre-existing file left byte-for-byte unchanged.
    assert out.read_text(encoding="utf-8") == "conversation_id,prior_user_turns,follow_up_question,document_id\n"


# ---------------------------------------------------------------------------
# Drop-in compatibility with load_multi_turn_ground_truth()
# ---------------------------------------------------------------------------
def test_generated_file_loads_through_multi_turn_loader(tmp_path):
    from evaluate_multi_turn_rewrite import load_multi_turn_ground_truth

    out = tmp_path / "out.csv"
    rows, _ = gen.build_rows(_sample_conversations(), faq_index=36)
    gen.write_csv(out, rows)

    loaded = load_multi_turn_ground_truth(path=out)
    assert len(loaded) == 2
    assert loaded[0]["question"] == "How much for a weekend?"
    assert loaded[0]["document_id"] == "faq_36"
    assert loaded[0]["history"] == [
        {"role": "user", "content": "Do you rent bouncy castles?"},
        {"role": "user", "content": "For a weekend?"},
    ]