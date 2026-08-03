import json
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


def _build_sources_for_topic(topic, rows):
    """Build deduplicated sources list for a CanonicalTopic from its member_ids."""
    seen = set()
    sources = []
    for member_id in topic.member_ids:
        if 0 <= member_id < len(rows):
            row = rows[member_id]
            company = row.get("company", "")
            clause_ref = row.get("clause_ref", "")
            url = row.get("url", "")
            key = (company, clause_ref)
            if key not in seen:
                seen.add(key)
                sources.append({"company": company, "clause_ref": clause_ref, "url": url})
    return sources


@dlt.resource(
    write_disposition="merge",
    primary_key=["topic_key"],
    columns=[
        {"name": "topic_key", "data_type": "text"},
        {"name": "category", "data_type": "text"},
        {"name": "question_en", "data_type": "text"},
        {"name": "answer_en", "data_type": "text"},
        {"name": "question_hu", "data_type": "text"},
        {"name": "answer_hu", "data_type": "text"},
        {"name": "sources", "data_type": "json"},
    ],
)
def canonical_faq_resource(topics, rows):
    """Yield one dict per canonical topic with provenance sources.

    Args:
        topics: List of CanonicalTopic objects from semantic_merge.semantic_topics
        rows: The combined input rows (existing faq.csv extraction rows + new rows)
              that topics.member_ids index into.
    """
    for topic in topics:
        sources = _build_sources_for_topic(topic, rows)
        # Serialize sources to JSON string for dlt complex type (duckdb JSON column)
        sources_json = json.dumps(sources, ensure_ascii=False)
        yield {
            "topic_key": topic.topic_key,
            "category": topic.category,
            "question_en": topic.question_en or "",
            "answer_en": topic.answer_en or "",
            "question_hu": topic.question_hu or "",
            "answer_hu": topic.answer_hu or "",
            "sources": sources_json,
        }


@dlt.source
def canonical_faq_source(topics, rows):
    return canonical_faq_resource(topics, rows)


def run_canonical_faq_pipeline(
    topics,
    rows,
    destination="duckdb",
    dataset_name="canonical_faq",
    pipeline_name="faq_ingestion",
    db_path=None,
):
    """Run the canonical FAQ pipeline with merge disposition on topic_key.

    Mirrors run_toc_pipeline conventions. Defaults to db/faq_ingestion.duckdb.
    """
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
    info = pipeline.run(canonical_faq_source(topics, rows))
    return pipeline, info
