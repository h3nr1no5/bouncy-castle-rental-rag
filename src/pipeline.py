import pathlib

import dlt


_DB_DIR = pathlib.Path(__file__).resolve().parents[1] / "db"


@dlt.resource(write_disposition="replace", columns=[{"name": "Category", "data_type": "text"}, {"name": "Question", "data_type": "text"}, {"name": "Answer", "data_type": "text"}])
def faq_resource():
    from src.faqs import load_faqs
    faqs = load_faqs()
    for row in faqs:
        yield row


@dlt.source
def faq_source():
    return faq_resource


@dlt.resource(
    write_disposition="replace",
    columns=[
        {"name": "company", "data_type": "text"},
        {"name": "url", "data_type": "text"},
        {"name": "fetched_at", "data_type": "timestamp"},
        {"name": "content_hash", "data_type": "text"},
        {"name": "lang", "data_type": "text"},
    ],
)
def toc_documents_resource(rows):
    for row in rows:
        yield row


@dlt.resource(
    write_disposition="replace",
    columns=[
        {"name": "document_id", "data_type": "text"},
        {"name": "question_hu", "data_type": "text"},
        {"name": "answer_hu", "data_type": "text"},
        {"name": "question_en", "data_type": "text"},
        {"name": "answer_en", "data_type": "text"},
        {"name": "clause_ref", "data_type": "text"},
        {"name": "company", "data_type": "text"},
    ],
)
def toc_faq_resource(rows):
    for row in rows:
        yield row


@dlt.source
def toc_source(documents, faq_entries):
    return [
        toc_documents_resource(documents),
        toc_faq_resource(faq_entries),
    ]


def run_toc_pipeline(documents, faq_entries, destination="duckdb", dataset_name="toc", pipeline_name="faq_ingestion", db_path=None):
    if destination == "duckdb":
        if db_path is None:
            db_path = _DB_DIR / f"{pipeline_name}.duckdb"
        db_path = pathlib.Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        destination = dlt.destinations.duckdb(credentials={"database": str(db_path)})
    pipeline = dlt.pipeline(
        pipeline_name=pipeline_name,
        destination=destination,
        dataset_name=dataset_name,
    )
    info = pipeline.run(toc_source(documents, faq_entries))
    return pipeline, info


def run_pipeline(destination="duckdb", dataset_name="faq", pipeline_name="faq_ingestion", db_path=None):
    if destination == "duckdb":
        if db_path is None:
            db_path = _DB_DIR / f"{pipeline_name}.duckdb"
        db_path = pathlib.Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        destination = dlt.destinations.duckdb(credentials={"database": str(db_path)})
    pipeline = dlt.pipeline(
        pipeline_name=pipeline_name,
        destination=destination,
        dataset_name=dataset_name,
    )
    info = pipeline.run(faq_source())
    return pipeline, info
