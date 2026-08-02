import json
from unittest import mock

import dlt
import duckdb
import pytest

from src.collect import collect, fetch_source
from src.extract import content_hash, _parse_qa
from src.merge_bilingual import (
    merge_bilingual,
    normalize_accent,
)
from src.parse import parse_html

HU_FIXTURE = """<html><body>
<h1>Általános szerződési feltételek</h1>
<h2>Foglalás</h2>
<p>A foglalást előleg fizetésével véglegesítheti. Az előleg a teljes bérleti díj 10-50%-a.</p>
<h2>Lemondás</h2>
<p>Esőnapot csak abban az esetben biztosítunk, ha a bérlést nem kezdték meg.</p>
<h2>Biztonsági szabályok</h2>
<p>Felnőtt felügyelete minden esetben kötelező.</p>
</body></html>
"""


# --- normalize_accent ---
def test_normalize_accent_folds_diacritics():
    assert normalize_accent("foglalás") == normalize_accent("foglalas")
    assert normalize_accent("Átvétel") == "atvetel"


# --- parse ---
def test_parse_splits_on_hungarian_headings():
    rows = parse_html(HU_FIXTURE, company="FixtureCo")
    sections = [r["section"] for r in rows]
    assert "Foglalás" in sections
    assert "Lemondás" in sections
    assert all(s == "FixtureCo" for r in rows for s in [r["company"]])
    for r in rows:
        assert r["clause_text"]
        assert r["clause_ref"]


def test_parse_deterministic_clause_ref():
    a = parse_html(HU_FIXTURE, company="X")
    b = parse_html(HU_FIXTURE, company="X")
    assert a == b


# --- extract / _parse_qa ---
def test_parse_qa_accepts_valid_json():
    reply = """{"pairs":[{"question_hu":"Kérdés?","answer_hu":"Válasz.","question_en":"Q?","answer_en":"A."}]}"""
    pairs = _parse_qa(reply)
    assert len(pairs) == 1
    assert pairs[0]["question_en"] == "Q?"


def test_parse_qa_accepts_code_fences():
    reply = """```json\n{"pairs":[{"question_hu":"H","answer_hu":"A","question_en":"Q","answer_en":"A"}]}\n```"""
    pairs = _parse_qa(reply)
    assert len(pairs) == 1


# --- collect with mocked http ---
class _FakeResponse:
    def __init__(self, text="", content_type="text/html; charset=UTF-8"):
        self._text = text
        self.headers = {"content-type": content_type}

    @property
    def text(self):
        return self._text


class _FakeClient:
    def __init__(self, mapping):
        self._mapping = mapping

    def get(self, url):
        if url not in self._mapping:
            raise RuntimeError("404")
        return self._mapping[url]


def test_fetch_source_writes_utf8_html(tmp_path):
    client = _FakeClient({"https://x.hu/aszf": _FakeResponse("<p>magyar szöveg</p>" * 20)})
    out = collect(
        companies=[{"company": "C1", "url": "https://x.hu/aszf"}],
        toc_dir=tmp_path,
        client=client,
    )
    assert out[0]["ok"] is True
    saved = (tmp_path / "C1" / "source.html").read_text(encoding="utf-8")
    assert saved


def test_collect_skips_non_html(tmp_path):
    client = _FakeClient({"https://x.hu/a.pdf": _FakeResponse("pdf", content_type="application/pdf")})
    out = collect(
        companies=[{"company": "C2", "url": "https://x.hu/a.pdf"}],
        toc_dir=tmp_path,
        client=client,
    )
    assert out[0]["ok"] is False


def test_collect_skips_unreachable(tmp_path):
    client = _FakeClient({})
    out = collect(
        companies=[{"company": "C3", "url": "https://x.hu/down"}],
        toc_dir=tmp_path,
        client=client,
    )
    assert out[0]["ok"] is False


# --- merge ---
def test_merge_upserts_row_and_keeps_existing(tmp_path):
    faq = tmp_path / "faq.csv"
    faq.write_text("Category,Question,Answer\nExisting,Preexisting question?,Existing answer.\n", encoding="utf-8")

    rows = [
        {
            "company": "C1",
            "section": "Foglalás",
            "clause_ref": "Foglalás#1",
            "question_hu": "Ké kemény-e az előleg?",
            "answer_hu": "Igen, 10-50%.",
            "question_en": "Is a deposit required?",
            "answer_en": "Yes, 10-50%.",
        }
    ]
    added = merge_bilingual(rows, faq_path=faq, dry_run=True)
    assert len(added) == 2  # EN + HU row per topic

    merge_bilingual(rows, faq_path=faq)
    content = faq.read_text(encoding="utf-8")
    assert "Is a deposit required?" in content  # EN row written
    assert "Ké kemény-e az előleg?" in content  # HU row written
    assert "Existing" in content  # existing untouched


def test_merge_knows_existing_topic(tmp_path):
    fa = tmp_path / "faq.csv"
    fa.write_text("Category,Question,Answer\nX,Is a deposit required?,Yes 10-50%.\n", encoding="utf-8")
    rows = [
        {"company": "C", "section": "s", "clause_ref": "s#1",
         "question_hu": "Eh?", "answer_hu": "ay", "question_en": "Is a deposit required?", "answer_en": "Yes."}
    ]
    added = merge_bilingual(rows, faq_path=fa, dry_run=True)
    # existing EN topic is deduped; only the new HU companion is appended
    assert added == [{"Category": "s", "Question": "Eh?", "Answer": "ay"}]


# --- pipeline ---
def test_toc_source_loads_to_duckdb(tmp_path):
    from src.pipeline import toc_source

    db_path = tmp_path / "toc_ingest.duckdb"
    pipeline = dlt.pipeline(
        pipeline_name="test_toc",
        destination=dlt.destinations.duckdb(credentials={"database": str(db_path)}),
        dataset_name="toc",
    )
    documents = [
        {"company": "C1", "url": "https://x.hu", "fetched_at": "2026-01-01 00:00:00", "content_hash": "h1", "lang": "hu"}
    ]
    faq_entries = [
        {"document_id": "Foglalás#1", "question_hu": "Qh", "answer_hu": "Ah",
         "question_en": "Qe", "answer_en": "Ae", "clause_ref": "Foglalás#1", "company": "C1"}
    ]
    info = pipeline.run(toc_source(documents, faq_entries))

    assert info.has_failed_jobs is False
    conn = duckdb.connect(str(db_path))
    try:
        count = conn.execute('SELECT COUNT(*) FROM "toc"."toc_documents_resource"').fetchone()[0]
        assert count == 1
        cols = [d[0] for d in conn.execute('DESCRIBE "toc"."toc_faq_resource"').fetchall()]
        assert "question_en" in cols
        assert "clause_ref" in cols
    finally:
        conn.close()


# --- content hash ---
def test_content_hash_stable_and_deterministic():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


# --- end-to-end with mocked llm ---
def test_extract_uses_cache_and_mocks_llm(tmp_path, monkeypatch):
    from src.extract import extract

    def fake_ask_llm(system_prompt, user_message, groq_model=None, openai_model=None):
        return {
            "response": json.dumps({"pairs": [
                {"question_hu": "Eh", "answer_hu": "Ay", "question_en": "Q", "answer_en": "A"}
            ]}),
            "model": "x", "provider": "mock", "latency": 0, "cost": 0, "tokens": {"prompt": 0, "completion": 0, "total": 0},
        }

    import src.extract as ex
    monkeypatch.setattr(ex, "ask_llm", fake_ask_llm)

    chunks = parse_html(HU_FIXTURE, company="C1", max_chunk_chars=50)
    cache = tmp_path / "extract_cache.json"
    out1 = extract(chunks[:1], toc_dir=tmp_path, cache_path=cache)
    assert out1["calls"] == 1
    assert len(out1["rows"]) == 1

    # rerun with unchanged sources => zero extra calls
    out2 = extract(chunks[:1], toc_dir=tmp_path, cache_path=cache)
    assert out2["calls"] == 0
    assert out2["cache_hits"] == 1