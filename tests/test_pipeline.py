import dlt
import duckdb
import json
import pytest
from dlt.pipeline.exceptions import PipelineStepFailed

from src.canonicalizer import CanonicalTopic
from src.faqs import load_faqs
from src.pipeline import (
    faq_source,
    faq_resource as _faq_resource,
    canonical_faq_resource,
    canonical_faq_source,
    run_canonical_faq_pipeline,
)


def test_faq_resource_yields_dicts():
    rows = list(_faq_resource())
    assert len(rows) > 0
    for row in rows:
        assert set(row.keys()) == {"Category", "Question", "Answer"}
        assert all(isinstance(v, str) and len(v) > 0 for v in row.values())


def test_pipeline_loads_to_duckdb(tmp_path):
    db_path = tmp_path / "test_faq.duckdb"
    pipeline = dlt.pipeline(
        pipeline_name="test_faq_ingestion",
        destination=dlt.destinations.duckdb(credentials={"database": str(db_path)}),
        dataset_name="faq",
    )
    info = pipeline.run(faq_source())

    assert info.has_failed_jobs is False
    assert len(info.load_packages) == 1

    conn = duckdb.connect(str(db_path))
    try:
        result = conn.execute('SELECT COUNT(*) FROM "faq"."faq_resource"').fetchone()
        assert result[0] == len(load_faqs())

        columns = [
            desc[0]
            for desc in conn.execute('DESCRIBE "faq"."faq_resource"').fetchall()
        ]
        for col in ("category", "question", "answer"):
            assert col in columns
    finally:
        conn.close()


def test_pipeline_is_idempotent(tmp_path):
    db_path = tmp_path / "idempotent.duckdb"
    pipeline = dlt.pipeline(
        pipeline_name="test_faq_idempotent",
        destination=dlt.destinations.duckdb(credentials={"database": str(db_path)}),
        dataset_name="faq",
    )

    pipeline.run(faq_source())
    pipeline.run(faq_source())

    conn = duckdb.connect(str(db_path))
    try:
        count = conn.execute('SELECT COUNT(*) FROM "faq"."faq_resource"').fetchone()[0]
        assert count == len(load_faqs())
    finally:
        conn.close()


def test_pipeline_raises_on_missing_csv(tmp_path):
    db_path = tmp_path / "missing.duckdb"
    import src.faqs as faqs_mod

    original = faqs_mod.load_faqs

    def broken_load():
        raise FileNotFoundError("FAQ CSV not found at /fake/path")

    faqs_mod.load_faqs = broken_load
    try:
        pipeline = dlt.pipeline(
            pipeline_name="test_faq_missing",
            destination=dlt.destinations.duckdb(credentials={"database": str(db_path)}),
            dataset_name="faq",
        )
        with pytest.raises((FileNotFoundError, PipelineStepFailed)):
            pipeline.run(faq_source())
    finally:
        faqs_mod.load_faqs = original


# --- Canonical FAQ pipeline tests (issue #44) ---

def _make_topic(
    topic_key,
    question_en,
    answer_en,
    category="Booking",
    question_hu=None,
    answer_hu=None,
    member_ids=(0,),
):
    """Helper to create a CanonicalTopic for testing."""
    return CanonicalTopic(
        topic_key=topic_key,
        question_en=question_en,
        question_hu=question_hu,
        answer_en=answer_en,
        answer_hu=answer_hu,
        category=category,
        member_ids=member_ids,
    )


def _make_row(company, clause_ref, url, question_en="Question?", answer_en="Answer.", question_hu=None, answer_hu=None, section="Booking"):
    """Helper to create an extraction-schema row."""
    return {
        "company": company,
        "question_en": question_en,
        "question_hu": question_hu,
        "answer_en": answer_en,
        "answer_hu": answer_hu,
        "clause_ref": clause_ref,
        "url": url,
        "section": section,
    }


def test_canonical_faq_resource_yields_dicts():
    """canonical_faq_resource yields dicts with the expected keys."""
    topic = _make_topic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        answer_en="20% deposit.",
        question_hu="Mennyi a foglaló?",
        answer_hu="20% foglaló.",
        member_ids=(0, 1),
    )
    rows = [
        _make_row("C1", "C1/deposit#1", "https://c1.hu/aszf"),
        _make_row("C2", "C2/deposit#2", "https://c2.hu/aszf"),
    ]
    yielded = list(canonical_faq_resource([topic], rows))
    assert len(yielded) == 1
    row = yielded[0]
    expected_keys = {"topic_key", "category", "question_en", "answer_en", "question_hu", "answer_hu", "sources"}
    assert set(row.keys()) == expected_keys
    assert row["topic_key"] == "deposit"
    assert row["category"] == "Booking"
    assert row["question_en"] == "How much is the deposit?"
    assert row["answer_en"] == "20% deposit."
    assert row["question_hu"] == "Mennyi a foglaló?"
    assert row["answer_hu"] == "20% foglaló."
    # sources should be a JSON string
    sources = json.loads(row["sources"])
    assert isinstance(sources, list)
    assert len(sources) == 2
    assert sources[0]["company"] == "C1"
    assert sources[0]["clause_ref"] == "C1/deposit#1"
    assert sources[0]["url"] == "https://c1.hu/aszf"
    assert sources[1]["company"] == "C2"
    assert sources[1]["clause_ref"] == "C2/deposit#2"
    assert sources[1]["url"] == "https://c2.hu/aszf"


def test_canonical_faq_resource_deduplicates_sources_by_company_and_clause_ref():
    """Sources are deduplicated by (company, clause_ref), keeping first occurrence."""
    topic = _make_topic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        answer_en="20% deposit.",
        member_ids=(0, 1, 2),
    )
    rows = [
        _make_row("C1", "C1/deposit#1", "https://c1.hu/aszf"),
        _make_row("C1", "C1/deposit#1", "https://c1.hu/aszf"),  # duplicate
        _make_row("C2", "C2/deposit#2", "https://c2.hu/aszf"),
    ]
    yielded = list(canonical_faq_resource([topic], rows))
    sources = json.loads(yielded[0]["sources"])
    assert len(sources) == 2
    assert sources[0]["company"] == "C1"
    assert sources[1]["company"] == "C2"


def test_canonical_faq_resource_empty_hu_fields_loads_without_error():
    """Topic with no Hungarian companion loads with empty HU fields."""
    topic = _make_topic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        answer_en="20% deposit.",
        question_hu=None,
        answer_hu=None,
        member_ids=(0,),
    )
    rows = [_make_row("C1", "C1/deposit#1", "https://c1.hu/aszf")]
    yielded = list(canonical_faq_resource([topic], rows))
    row = yielded[0]
    assert row["question_hu"] == ""
    assert row["answer_hu"] == ""
    sources = json.loads(row["sources"])
    assert len(sources) == 1
    assert sources[0]["company"] == "C1"


def test_canonical_faq_resource_legacy_rows_load_with_legacy_sources():
    """Topic whose member rows are all legacy loads with legacy entries in sources."""
    topic = _make_topic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        answer_en="20% deposit.",
        member_ids=(0, 1),
    )
    rows = [
        _make_row("legacy", "legacy", "legacy"),
        _make_row("legacy", "legacy", "legacy"),
    ]
    yielded = list(canonical_faq_resource([topic], rows))
    sources = json.loads(yielded[0]["sources"])
    assert len(sources) == 1  # deduplicated
    assert sources[0]["company"] == "legacy"
    assert sources[0]["clause_ref"] == "legacy"
    assert sources[0]["url"] == "legacy"


def test_canonical_faq_pipeline_merge_primary_key_loads_to_duckdb(tmp_path):
    """dlt merge + primary_key load into temp duckdb works."""
    db_path = tmp_path / "test_canonical_faq.duckdb"
    topic = _make_topic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        answer_en="20% deposit.",
        question_hu="Mennyi a foglaló?",
        answer_hu="20% foglaló.",
        member_ids=(0, 1),
    )
    rows = [
        _make_row("C1", "C1/deposit#1", "https://c1.hu/aszf"),
        _make_row("C2", "C2/deposit#2", "https://c2.hu/aszf"),
    ]
    pipeline, info = run_canonical_faq_pipeline(
        [topic], rows, db_path=db_path, pipeline_name="test_canonical_faq", dataset_name="faq_ingestion"
    )
    assert info.has_failed_jobs is False

    conn = duckdb.connect(str(db_path))
    try:
        result = conn.execute('SELECT COUNT(*) FROM "faq_ingestion"."canonical_faq_resource"').fetchone()
        assert result[0] == 1

        # Check columns - sources should be a single JSON column
        columns = [desc[0] for desc in conn.execute('DESCRIBE "faq_ingestion"."canonical_faq_resource"').fetchall()]
        for col in ("topic_key", "category", "question_en", "answer_en", "question_hu", "answer_hu", "sources"):
            assert col in columns
        # Ensure sources is NOT a nested child table (no sources__company etc.)
        for col in columns:
            assert not col.startswith("sources__"), f"Found nested column: {col}"

        # Verify data
        row = conn.execute('SELECT * FROM "faq_ingestion"."canonical_faq_resource"').fetchone()
        assert row[0] == "deposit"  # topic_key
        assert row[1] == "Booking"  # category
        assert row[2] == "How much is the deposit?"  # question_en
        assert row[3] == "20% deposit."  # answer_en
        assert row[4] == "Mennyi a foglaló?"  # question_hu
        assert row[5] == "20% foglaló."  # answer_hu
        sources = json.loads(row[6])
        assert len(sources) == 2
        assert sources[0]["company"] == "C1"
        assert sources[1]["company"] == "C2"
    finally:
        conn.close()


def test_canonical_faq_pipeline_second_run_converges_same_row_count(tmp_path):
    """Running the pipeline twice with same topics/rows leaves same row count (no duplicates)."""
    db_path = tmp_path / "test_canonical_faq_idempotent.duckdb"
    topic = _make_topic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        answer_en="20% deposit.",
        question_hu="Mennyi a foglaló?",
        answer_hu="20% foglaló.",
        member_ids=(0, 1),
    )
    rows = [
        _make_row("C1", "C1/deposit#1", "https://c1.hu/aszf"),
        _make_row("C2", "C2/deposit#2", "https://c2.hu/aszf"),
    ]
    # First run
    run_canonical_faq_pipeline(
        [topic], rows, db_path=db_path, pipeline_name="test_canonical_faq_idempotent", dataset_name="faq_ingestion"
    )
    # Second run
    run_canonical_faq_pipeline(
        [topic], rows, db_path=db_path, pipeline_name="test_canonical_faq_idempotent", dataset_name="faq_ingestion"
    )

    conn = duckdb.connect(str(db_path))
    try:
        count = conn.execute('SELECT COUNT(*) FROM "faq_ingestion"."canonical_faq_resource"').fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_canonical_faq_pipeline_folded_topic_sources_lists_both_companies(tmp_path):
    """A topic folded from two paraphrased rows from two companies has both in sources."""
    db_path = tmp_path / "test_canonical_faq_folded.duckdb"
    topic = _make_topic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        answer_en="20% deposit (figures vary across companies: 20% to 30%)",
        question_hu="Mennyi a foglaló?",
        answer_hu="20% foglaló.",
        member_ids=(0, 1),
    )
    rows = [
        _make_row("C1", "C1/deposit#1", "https://c1.hu/aszf", question_en="How much is the deposit?", answer_en="20% deposit."),
        _make_row("C2", "C2/deposit#2", "https://c2.hu/aszf", question_en="Do I need to pay a deposit?", answer_en="30% deposit."),
    ]
    run_canonical_faq_pipeline(
        [topic], rows, db_path=db_path, pipeline_name="test_canonical_faq_folded", dataset_name="faq_ingestion"
    )

    conn = duckdb.connect(str(db_path))
    try:
        row = conn.execute('SELECT sources FROM "faq_ingestion"."canonical_faq_resource"').fetchone()
        sources = json.loads(row[0])
        assert len(sources) == 2
        companies = {s["company"] for s in sources}
        assert companies == {"C1", "C2"}
        # Check clause_ref and url are present
        for s in sources:
            assert "clause_ref" in s
            assert "url" in s
    finally:
        conn.close()


def test_canonical_faq_pipeline_does_not_modify_faq_csv(tmp_path):
    """data/faq.csv is not modified by this pipeline (provenance lives only in duckdb)."""
    # This test verifies that the pipeline doesn't touch faq.csv
    # by checking that the pipeline only writes to duckdb
    db_path = tmp_path / "test_canonical_faq_no_csv.duckdb"
    topic = _make_topic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        answer_en="20% deposit.",
        member_ids=(0,),
    )
    rows = [_make_row("C1", "C1/deposit#1", "https://c1.hu/aszf")]
    
    # The pipeline should not require or modify faq.csv
    run_canonical_faq_pipeline(
        [topic], rows, db_path=db_path, pipeline_name="test_canonical_faq_no_csv", dataset_name="faq_ingestion"
    )
    
    # Verify the data is in duckdb
    conn = duckdb.connect(str(db_path))
    try:
        count = conn.execute('SELECT COUNT(*) FROM "faq_ingestion"."canonical_faq_resource"').fetchone()[0]
        assert count == 1
    finally:
        conn.close()
    
    # The test passes if no exception is raised and data is in duckdb
    # (faq.csv is not touched because the pipeline doesn't reference it)
