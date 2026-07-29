import dlt
import duckdb
import pytest
from dlt.pipeline.exceptions import PipelineStepFailed

from src.faqs import load_faqs
from src.pipeline import faq_source, faq_resource as _faq_resource


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
