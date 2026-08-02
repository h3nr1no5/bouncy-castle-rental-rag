import argparse
import datetime
import logging
import pathlib
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("toc")

ROOT = pathlib.Path(__file__).resolve().parent
DB_DIR = ROOT / "db"
TOC_DIR = DB_DIR / "toc"


def _collect():
    from src.collect import collect

    logger.info("Stage 1/5: collect")
    results = collect(toc_dir=TOC_DIR)
    ok = [r for r in results if r["ok"]]
    for r in results:
        logger.info("  %s %s %s", "OK " if r["ok"] else "SKIP", r["company"], r.get("path") or r.get("error"))
    return results, bool(ok)


def _parse(ok_results):
    from src.parse import parse_html

    logger.info("Stage 2/5: parse")
    chunks = []
    for r in ok_results:
        if not r.get("ok") or not r.get("path"):
            logger.info("  skip (collect failed): %s", r["company"])
            continue
        src = pathlib.Path(r["path"])
        html = src.read_text(encoding="utf-8")
        rows = parse_html(html, company=r["company"])
        chunks.extend(rows)
        logger.info("  %s -> %d chunks", r["company"], len(rows))
    return chunks


def _extract(chunks):
    from src.extract import extract

    logger.info("Stage 3/5: extract (LLM, cached)")
    out = extract(chunks, toc_dir=TOC_DIR)
    logger.info("  %d rows extracted, %d fresh LLM calls, %d cache hits", len(out["rows"]), out["calls"], out["cache_hits"])
    return out["rows"]


def _merge(rows):
    from src.merge_bilingual import merge_bilingual

    logger.info("Stage 4/5: merge into faq.csv")
    added = merge_bilingual(rows)
    logger.info("  appended %d new bilingual rows", len(added))
    return added


def _build_indexes():
    from src.faqs import load_faqs
    from src.ingest import build_indexes

    logger.info("Stage 5/5: build/refresh indexes")
    faqs = load_faqs()
    paths = build_indexes(faqs=faqs, force=True)
    logger.info("  indexed %d faqs -> %s", len(faqs), paths["docs_path"])
    return paths


def main():
    parser = argparse.ArgumentParser(description="Bilingual T&C ingestion pipeline")
    parser.add_argument("--pipeline", action="store_true", help="also load into duckdb 'toc' dataset via dlt")
    parser.add_argument("--skip-llm", action="store_true", help="use only cached extractions (no LLM calls)")
    args = parser.parse_args()

    ok_results, any_ok = _collect()
    if not any_ok:
        logger.error("All sources failed to collect; aborting.")
        sys.exit(1)

    chunks = _parse(ok_results)
    logger.info("Total chunks: %d", len(chunks))

    if args.skip_llm:
        from src.extract import _extract_single  # noqa: F401
        extracted_rows = _cached_only(chunks)
    else:
        extracted_rows = _extract(chunks)

    if not extracted_rows:
        logger.warning("No extractions produced any rows.")

    added = _merge(extracted_rows)
    _build_indexes()

    if args.pipeline:
        _load_into_dlt(chunks, extracted_rows)

    logger.info("Done. appended_rows=%d", len(added))


def _cached_only(chunks):
    from src.extract import _load_cache, _row_from_pair, content_hash

    cache = _load_cache(TOC_DIR / "extract_cache.json")
    rows = []
    hit = 0
    for section in chunks:
        key = content_hash(section["clause_text"])
        cache_key = f'{section["company"]}::{section["section"]}::{key}'
        cached = cache.get(cache_key)
        if cached is None:
            logger.info("  (cache miss, skipping) %s/%s", section["company"], section["section"])
            continue
        hit += 1
        for pair in cached:
            rows.append(_row_from_pair(section, pair))
    logger.info("  cached-only: %d rows from %d cache hits", len(rows), hit)
    return rows


def _load_into_dlt(chunks, faq_rows):
    from src.pipeline import run_toc_pipeline

    now = datetime.datetime.now()
    # one document record per company+url (derive url from source path parent)
    documents = {}
    for c in chunks:
        key = c["company"]
        if key not in documents:
            documents[key] = {
                "company": key,
                "url": f"db/toc/{key}/source.html",
                "fetched_at": now,
                "content_hash": c.get("clause_ref", ""),
                "lang": "hu",
            }
    faq_entries = [
        {
            "document_id": r.get("clause_ref", ""),
            "question_hu": r.get("question_hu", ""),
            "answer_hu": r.get("answer_hu", ""),
            "question_en": r.get("question_en", ""),
            "answer_en": r.get("answer_en", ""),
            "clause_ref": r.get("clause_ref", ""),
            "company": r.get("company", ""),
        }
        for r in faq_rows
    ]
    pipeline, info = run_toc_pipeline(list(documents.values()), faq_entries)
    logger.info("Loaded into duckdb dataset 'toc': %s", info)


if __name__ == "__main__":
    main()