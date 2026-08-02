import argparse
import datetime
import logging
import os
import pathlib

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("toc")

ROOT = pathlib.Path(__file__).resolve().parent
DB_DIR = ROOT / "db"
TOC_DIR = DB_DIR / "toc"
TOC_EN_DIR = DB_DIR / "toc_en"
COMPANIES_HU = ROOT / "data" / "companies.json"
COMPANIES_EN = ROOT / "data" / "companies_en.json"


def _collect(toc_dir=TOC_DIR, companies_path=None):
    from src.collect import collect

    logger.info("Stage 1/5: collect")
    results = collect(toc_dir=toc_dir, companies_path=companies_path)
    for r in results:
        logger.info("  %s %s %s", "OK " if r["ok"] else "SKIP", r["company"], r.get("path") or r.get("error"))
    return results, bool([r for r in results if r["ok"]])


def _parse(ok_results):
    from src.extract import content_hash
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
        source_hash = content_hash(html)
        fallback_url = f"{src.parent.parent.name}/{src.parent.name}/{r['company']}/source.html"
        for row in rows:
            row["url"] = r.get("url", fallback_url)
            row["content_hash"] = source_hash
        chunks.extend(rows)
        logger.info("  %s -> %d chunks", r["company"], len(rows))
    return chunks


def _extract(chunks, toc_dir=TOC_DIR):
    from src.extract import extract

    logger.info("Stage 3/5: extract (LLM, cached)")
    out = extract(chunks, toc_dir=toc_dir)
    logger.info("  %d rows extracted, %d fresh LLM calls, %d cache hits", len(out["rows"]), out["calls"], out["cache_hits"])
    return out["rows"]


def _extract_en(chunks, toc_dir=TOC_EN_DIR):
    from src.extract import extract_en

    logger.info("Stage 3/5: extract EN (LLM, cached)")
    out = extract_en(chunks, toc_dir=toc_dir)
    logger.info("  %d EN rows extracted, %d fresh LLM calls, %d cache hits", len(out["rows"]), out["calls"], out["cache_hits"])
    return out["rows"]


def _merge(rows):
    from src.merge_bilingual import merge_bilingual

    logger.info("Stage 4/5: merge into faq.csv")
    added = merge_bilingual(rows)
    logger.info("  appended %d new bilingual rows", len(added))
    return added


def _merge_en(rows):
    from src.merge_bilingual import merge_en

    logger.info("Stage 4/5: merge EN track into faq.csv")
    added = merge_en(rows)
    logger.info("  appended %d new English rows", len(added))
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
    parser.add_argument("--hu", action="store_true", help="run only the Hungarian track")
    parser.add_argument("--en", action="store_true", help="run only the English track")
    args = parser.parse_args()

    tracks = []
    if args.en and not args.hu:
        tracks = ["en"]
    elif args.hu and not args.en:
        tracks = ["hu"]
    else:
        tracks = ["hu", "en"]

    for lang in tracks:
        _run_track(lang, skip_llm=args.skip_llm, to_dlt=args.pipeline)

    _build_indexes()


def _run_track(lang, skip_llm=False, to_dlt=False):
    if lang == "en":
        toc_dir, companies_path, extract_fn, merge_fn, cached_only_fn = (
            TOC_EN_DIR, COMPANIES_EN, _extract_en, _merge_en, _cached_only_en
        )
    else:
        toc_dir, companies_path, extract_fn, merge_fn, cached_only_fn = (
            TOC_DIR, COMPANIES_HU, _extract, _merge, _cached_only
        )

    logger.info("=== Track: %s ===", lang.upper())
    ok_results, any_ok = _collect(toc_dir=toc_dir, companies_path=companies_path)
    if not any_ok:
        logger.error("[%s] All sources failed to collect; aborting this track.", lang)
        return

    chunks = _parse(ok_results)
    logger.info("[%s] Total chunks: %d", lang, len(chunks))

    if skip_llm:
        extracted_rows = cached_only_fn(chunks, toc_dir=toc_dir)
    else:
        extracted_rows = extract_fn(chunks, toc_dir=toc_dir)

    if not extracted_rows:
        logger.warning("[%s] No extractions produced any rows.", lang)

    added = merge_fn(extracted_rows)

    if to_dlt:
        _load_into_dlt(chunks, extracted_rows, lang=lang)

    logger.info("[%s] Done. appended_rows=%d", lang, len(added))


def _cached_only(chunks, toc_dir=TOC_DIR):
    from src.extract import _load_cache, _row_from_pair, content_hash

    cache = _load_cache(toc_dir / "extract_cache.json")
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


def _cached_only_en(chunks, toc_dir=TOC_EN_DIR):
    from src.extract import _load_cache, _row_from_en_pair, content_hash

    cache = _load_cache(toc_dir / "extract_cache.json")
    rows = []
    hit = 0
    for chunk in chunks:
        key = content_hash(chunk["clause_text"])
        cache_key = f'{chunk["company"]}::{chunk["section"]}::{key}'
        cached = cache.get(cache_key)
        if cached is None:
            logger.info("  (cache miss, skipping) %s/%s", chunk["company"], chunk["section"])
            continue
        hit += 1
        for pair in cached:
            rows.append(_row_from_en_pair(chunk, pair))
    logger.info("  cached-only: %d EN rows from %d cache hits", len(rows), hit)
    return rows


def _load_into_dlt(chunks, faq_rows, lang="hu"):
    from src.pipeline import run_toc_pipeline

    now = datetime.datetime.now()
    # one document record per company+url (url and content_hash carried from collect/parse)
    documents = {}
    for c in chunks:
        key = c["company"]
        if key not in documents:
            documents[key] = {
                "company": key,
                "url": c.get("url", f"db/toc/{key}/source.html"),
                "fetched_at": now,
                "content_hash": c.get("content_hash", ""),
                "lang": lang,
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
    logger.info("Loaded into duckdb dataset 'toc' [%s]: %s", lang, info)


if __name__ == "__main__":
    main()