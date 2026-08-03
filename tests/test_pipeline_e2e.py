"""End-to-end tests for the full semantic merge pipeline (issue #45).

Tests the complete flow: cluster → gate → canonicalize → apply → dlt into duckdb.
"""

import json
import tempfile
import pathlib
import duckdb

from src.semantic_merge import semantic_merge_full
from src.clusterer import LLMClusterer, ExactClusterer
from src.pipeline import run_canonical_faq_pipeline
from src.canonicalizer import topic_key


def _row(company, question_en, question_hu=None, answer_en="Answer.", answer_hu=None, clause_ref=None, url=None, section=None):
    """Helper to create an extraction-schema test row."""
    return {
        "company": company,
        "question_en": question_en,
        "question_hu": question_hu,
        "answer_en": answer_en,
        "answer_hu": answer_hu if answer_hu is not None else "",
        "clause_ref": clause_ref or f"{company}/1",
        "url": url or f"https://{company}.hu/aszf",
        "section": section or "Terms & Conditions",
    }


def _llm_reply(assignments):
    """Helper to create a mock LLM reply."""
    return {
        "response": json.dumps({"assignments": assignments}),
        "model": "x", "provider": "mock", "latency": 0, "cost": 0,
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
    }


def _make_smart_mock_clusterer(assignments_map):
    """Create an LLMClusterer with a mocked ask_llm that returns assignments based on question text.
    
    This ensures consistent clustering across re-runs where existing faq.csv rows
    are also included in the clustering input.
    """
    def fake_ask_llm(system_prompt, user_message, groq_model=None, openai_model=None):
        # Parse the questions from user_message
        lines = user_message.strip().split('\n')
        questions = []
        for line in lines:
            if line.strip() and line[0].isdigit():
                # Format: '0. question text'
                parts = line.split('. ', 1)
                if len(parts) == 2:
                    questions.append(parts[1])
        
        # Build assignments based on question text
        result_assignments = {}
        for i, q in enumerate(questions):
            # Find matching assignment
            for key, topic in assignments_map.items():
                if key in q or q in key:  # Simple matching
                    result_assignments[str(i)] = topic
                    break
            else:
                result_assignments[str(i)] = 'unknown'
        
        return _llm_reply(result_assignments)
    return LLMClusterer(ask_llm=fake_ask_llm)


def _mock_clusterer_for_rows(rows, topic_assignments):
    """Create an LLMClusterer with a mocked ask_llm that returns the given topic assignments.
    
    This simpler mock is for single-run tests where the input rows are known and fixed.
    """
    def fake_ask_llm(system_prompt, user_message, groq_model=None, openai_model=None):
        return _llm_reply(topic_assignments)
    return LLMClusterer(ask_llm=fake_ask_llm)


def test_e2e_paraphrase_fold_two_companies_en_hu_pair(tmp_path):
    """Two paraphrased questions from two companies fold to one topic/pair in temp faq.csv."""
    # Two companies, paraphrased questions about deposit, both bilingual
    rows = [
        _row("C1", "How much is the deposit?", question_hu="Mekkora a foglaló?", answer_en="20% deposit.", answer_hu="20% foglaló.", section="Booking & Reservations", clause_ref="C1/deposit#1", url="https://c1.hu/aszf"),
        _row("C2", "Do I need to pay a deposit and when?", question_hu="Kell foglalót fizetni?", answer_en="Yes, 30% deposit.", answer_hu="Igen, 30% foglaló.", section="Booking & Reservations", clause_ref="C2/deposit#2", url="https://c2.hu/aszf"),
    ]
    clusterer = _mock_clusterer_for_rows(rows, {"0": "deposit", "1": "deposit"})

    faq_path = tmp_path / "faq.csv"
    db_path = tmp_path / "faq_ingestion.duckdb"

    # Run semantic merge full pipeline
    topics, written_rows, discarded_rows, kept_rows = semantic_merge_full(
        rows=rows,
        faq_path=faq_path,
        clusterer=clusterer,
        dry_run=False,
    )

    # Assert: one topic produced (two paraphrases folded)
    assert len(topics) == 1
    topic = topics[0]
    assert topic.question_en == "How much is the deposit?"
    assert topic.question_hu == "Mekkora a foglaló?"
    assert "figures vary across companies" in topic.answer_en
    assert "20% to 30%" in topic.answer_en
    assert topic.answer_hu == "20% foglaló."
    assert topic.category == "Booking & Reservations"

    # Assert: faq.csv has 2 rows (EN + HU)
    assert len(written_rows) == 2
    assert written_rows[0]["Question"] == "How much is the deposit?"
    assert written_rows[0]["Answer"] == topic.answer_en
    assert written_rows[1]["Question"] == "Mekkora a foglaló?"
    assert written_rows[1]["Answer"] == "20% foglaló."
    assert written_rows[0]["Category"] == written_rows[1]["Category"] == "Booking & Reservations"

    # Assert: no clause dumps discarded
    assert len(discarded_rows) == 0

    # Load provenance into duckdb
    pipeline, info = run_canonical_faq_pipeline(topics, kept_rows, db_path=db_path)

    # Verify duckdb row's sources lists both companies with clause_ref and url
    conn = duckdb.connect(str(db_path))
    try:
        result = conn.execute('SELECT topic_key, category, question_en, answer_en, question_hu, answer_hu, sources FROM "canonical_faq"."canonical_faq_resource"').fetchall()
        assert len(result) == 1
        row = result[0]
        assert row[0] == topic_key("How much is the deposit?")
        assert row[1] == "Booking & Reservations"
        assert row[2] == "How much is the deposit?"
        assert "figures vary across companies" in row[3]
        assert row[4] == "Mekkora a foglaló?"
        assert row[5] == "20% foglaló."

        sources = json.loads(row[6])
        assert len(sources) == 2
        companies = {s["company"] for s in sources}
        assert companies == {"C1", "C2"}
        for s in sources:
            assert "clause_ref" in s
            assert "url" in s
            if s["company"] == "C1":
                assert s["clause_ref"] == "C1/deposit#1"
                assert s["url"] == "https://c1.hu/aszf"
            else:
                assert s["clause_ref"] == "C2/deposit#2"
                assert s["url"] == "https://c2.hu/aszf"
    finally:
        conn.close()


def test_e2e_idempotent_rerun_byte_identical_faq_and_same_canonical_count(tmp_path):
    """Re-running produces byte-identical temp faq.csv and same canonical_faq_resource row count.
    
    Uses ExactClusterer for determinism (as required by spec: --skip-llm path).
    """
    rows = [
        _row("C1", "How much is the deposit?", question_hu="Mekkora a foglaló?", answer_en="20% deposit.", answer_hu="20% foglaló.", section="Booking & Reservations", clause_ref="C1/deposit#1", url="https://c1.hu/aszf"),
        _row("C2", "Do I need to pay a deposit and when?", question_hu="Kell foglalót fizetni?", answer_en="Yes, 30% deposit.", answer_hu="Igen, 30% foglaló.", section="Booking & Reservations", clause_ref="C2/deposit#2", url="https://c2.hu/aszf"),
    ]
    clusterer = ExactClusterer()

    faq_path = tmp_path / "faq.csv"
    db_path = tmp_path / "faq_ingestion.duckdb"

    # First run
    topics1, written_rows1, discarded_rows1, kept_rows1 = semantic_merge_full(
        rows=rows,
        faq_path=faq_path,
        clusterer=clusterer,
        dry_run=False,
    )
    content1 = faq_path.read_bytes()
    pipeline1, info1 = run_canonical_faq_pipeline(topics1, kept_rows1, db_path=db_path)

    # Second run (using the same faq.csv as input)
    topics2, written_rows2, discarded_rows2, kept_rows2 = semantic_merge_full(
        rows=rows,
        faq_path=faq_path,
        clusterer=clusterer,
        dry_run=False,
    )
    content2 = faq_path.read_bytes()
    pipeline2, info2 = run_canonical_faq_pipeline(topics2, kept_rows2, db_path=db_path)

    # Assert: byte-identical faq.csv
    assert content1 == content2, "Re-run did not produce byte-identical faq.csv"

    # Assert: same written rows
    assert len(written_rows1) == len(written_rows2)
    for r1, r2 in zip(written_rows1, written_rows2):
        assert r1 == r2

    # Assert: same canonical_faq_resource row count
    conn = duckdb.connect(str(db_path))
    try:
        count1 = conn.execute('SELECT COUNT(*) FROM "canonical_faq"."canonical_faq_resource"').fetchone()[0]
        # The second run uses merge disposition on topic_key, so count should be same
        count2 = conn.execute('SELECT COUNT(*) FROM "canonical_faq"."canonical_faq_resource"').fetchone()[0]
        assert count1 == count2 == 2  # ExactClusterer produces 2 topics (no folding)
    finally:
        conn.close()


def test_e2e_clause_dump_discarded_sources_still_correct(tmp_path):
    """When a clause-dump row is present, surviving topic's sources still reference correct companies (no off-by-N)."""
    # Three rows: two valid paraphrases + one clause dump
    dump_answer = "A" * 100 + ". " + "B" * 100 + "."  # >200 chars, multi-sentence = clause dump
    rows = [
        _row("C1", "How much is the deposit?", question_hu="Mekkora a foglaló?", answer_en="20% deposit.", answer_hu="20% foglaló.", section="Booking & Reservations", clause_ref="C1/deposit#1", url="https://c1.hu/aszf"),
        _row("C2", "Do I need to pay a deposit and when?", question_hu="Kell foglalót fizetni?", answer_en="Yes, 30% deposit.", answer_hu="Igen, 30% foglaló.", section="Booking & Reservations", clause_ref="C2/deposit#2", url="https://c2.hu/aszf"),
        _row("C3", "What is the full legal text of the deposit clause?", answer_en=dump_answer, section="Legal", clause_ref="C3/legal#1", url="https://c3.hu/aszf"),  # clause dump
    ]
    # Only the first two rows cluster together; the third is a clause dump and gets filtered out
    clusterer = _mock_clusterer_for_rows(rows, {"0": "deposit", "1": "deposit", "2": "legal_dump"})

    faq_path = tmp_path / "faq.csv"
    db_path = tmp_path / "faq_ingestion.duckdb"

    topics, written_rows, discarded_rows, kept_rows = semantic_merge_full(
        rows=rows,
        faq_path=faq_path,
        clusterer=clusterer,
        dry_run=False,
    )

    # Assert: one topic produced (two paraphrases folded), clause dump discarded
    assert len(topics) == 1
    assert len(discarded_rows) == 1
    assert discarded_rows[0]["company"] == "C3"

    # Assert: faq.csv has 2 rows (EN + HU) for the deposit topic
    assert len(written_rows) == 2

    # Load provenance into duckdb
    pipeline, info = run_canonical_faq_pipeline(topics, kept_rows, db_path=db_path)

    # Verify duckdb row's sources lists ONLY the two valid companies (C1, C2), not C3
    conn = duckdb.connect(str(db_path))
    try:
        result = conn.execute('SELECT topic_key, sources FROM "canonical_faq"."canonical_faq_resource"').fetchall()
        assert len(result) == 1
        row = result[0]
        sources = json.loads(row[1])
        assert len(sources) == 2, f"Expected 2 sources (C1, C2), got {len(sources)}: {sources}"
        companies = {s["company"] for s in sources}
        assert companies == {"C1", "C2"}, f"Sources should reference C1 and C2 only, got {companies}"
        for s in sources:
            assert "clause_ref" in s
            assert "url" in s
            if s["company"] == "C1":
                assert s["clause_ref"] == "C1/deposit#1"
                assert s["url"] == "https://c1.hu/aszf"
            else:
                assert s["clause_ref"] == "C2/deposit#2"
                assert s["url"] == "https://c2.hu/aszf"
    finally:
        conn.close()


def test_e2e_exact_clusterer_deterministic_no_llm(tmp_path):
    """Same test with ExactClusterer (deterministic, no LLM calls)."""
    rows = [
        _row("C1", "How much is the deposit?", question_hu="Mekkora a foglaló?", answer_en="20% deposit.", answer_hu="20% foglaló.", section="Booking & Reservations", clause_ref="C1/deposit#1", url="https://c1.hu/aszf"),
        _row("C2", "Do I need to pay a deposit and when?", question_hu="Kell foglalót fizetni?", answer_en="Yes, 30% deposit.", answer_hu="Igen, 30% foglaló.", section="Booking & Reservations", clause_ref="C2/deposit#2", url="https://c2.hu/aszf"),
    ]
    clusterer = ExactClusterer()

    faq_path = tmp_path / "faq.csv"
    db_path = tmp_path / "faq_ingestion.duckdb"

    topics, written_rows, discarded_rows, kept_rows = semantic_merge_full(
        rows=rows,
        faq_path=faq_path,
        clusterer=clusterer,
        dry_run=False,
    )

    # ExactClusterer uses exact question_en matching, so these won't fold
    # (different question_en strings) - this is expected behavior
    assert len(topics) == 2
    assert len(written_rows) == 4  # 2 topics × 2 rows (EN+HU each)

    # Load provenance
    pipeline, info = run_canonical_faq_pipeline(topics, kept_rows, db_path=db_path)

    conn = duckdb.connect(str(db_path))
    try:
        count = conn.execute('SELECT COUNT(*) FROM "canonical_faq"."canonical_faq_resource"').fetchone()[0]
        assert count == 2
    finally:
        conn.close()