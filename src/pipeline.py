import pathlib

import dlt


_TMP_DIR = pathlib.Path(__file__).resolve().parents[1] / ".tmp"


@dlt.resource(write_disposition="replace", columns=[{"name": "Category", "data_type": "text"}, {"name": "Question", "data_type": "text"}, {"name": "Answer", "data_type": "text"}])
def faq_resource():
    from src.faqs import load_faqs
    faqs = load_faqs()
    for row in faqs:
        yield row


@dlt.source
def faq_source():
    return faq_resource


def run_pipeline(destination="duckdb", dataset_name="faq", pipeline_name="faq_ingestion", db_path=None):
    if destination == "duckdb":
        if db_path is None:
            db_path = _TMP_DIR / f"{pipeline_name}.duckdb"
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
