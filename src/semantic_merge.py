"""Semantic merge apply stage: gate → cluster → canonicalize → rewrite faq.csv idempotently.

This module orchestrates the full semantic merge pipeline:
1. Filter clause dumps (clause_gate.filter_clause_dumps)
2. Cluster remaining rows (clusterer.cluster)
3. Canonicalize each cluster (canonicalizer.canonicalize_cluster)
4. Deduplicate by topic_key, keeping the existing-canonical representative
5. Order by minimum member input index for idempotent positional ids
6. Write bilingual EN+HU pairs (or EN-only) to faq.csv

The rewritten file has exactly the header ``Category,Question,Answer`` and no id column.
Positional ``faq_{i}`` ids (0-based row index) are stable across re-runs for
deterministic clusterers.

Migration note for ground_truth.csv:
------------------------------------
Merging reduces the row count, so positional ids **will** renumber on the first
live rewrite. The current ``data/ground_truth.csv`` (referencing ``faq_0``,
``faq_2``, ``faq_5``, …) cannot remain valid without regeneration.
Regenerate it by running::

    uv run python generate_ground_truth.py

after the rewrite. This rebuilds ``document_id`` values against the new file
and overwrites cleanly (no accumulation).
"""

import csv
import pathlib
import re
from typing import Optional

from src.canonicalizer import CanonicalTopic, canonicalize_cluster, topic_key
from src.clause_gate import filter_clause_dumps
from src.clusterer import Clusterer, TopicCluster, default_clusterer, ExactClusterer
from src.merge_bilingual import REQUIRED_COLUMNS, normalize_accent

DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
DEFAULT_FAQ_PATH = DEFAULT_DATA_DIR / "faq.csv"

# Hungarian diacritics for language detection
_HU_DIACRITICS = re.compile(r"[áéíóöőúüűÁÉÍÓÖŐÚÜŰ]")


def _is_hungarian(text: str) -> bool:
    """Check if text contains Hungarian diacritics."""
    return bool(_HU_DIACRITICS.search(text))


def _faq_rows_to_extraction(rows: list[dict]) -> list[dict]:
    """Convert faq.csv rows (Category, Question, Answer) to extraction-schema rows.

    The pairing heuristic: within each Category, consecutive rows are paired as
    EN+HU companions ONLY when one is clearly English and the other clearly
    Hungarian (detected via Hungarian diacritics: á, é, í, ó, ö, ő, ú, ü, ű).
    If both rows appear to be the same language (both EN or both HU), they are
    treated as separate rows.

    A row is classified as Hungarian if its Question contains Hungarian diacritics.
    The first row of a valid EN+HU pair becomes question_en/answer_en, the second
    becomes question_hu/answer_hu. Rows that cannot be paired become EN-only
    (if no diacritics) or HU-only with question_en fallback (if diacritics present).

    Rows with only question_hu (no EN) are normalized so question_en falls back
    to question_hu (mirrors clusterer._en_question).
    """
    # Group by category, preserving file order within each category
    by_category = {}
    for row in rows:
        cat = row.get("Category", "").strip()
        if not cat:
            continue
        by_category.setdefault(cat, []).append(row)

    extraction_rows = []
    for cat, cat_rows in by_category.items():
        i = 0
        while i < len(cat_rows):
            row1 = cat_rows[i]
            q1 = (row1.get("Question") or "").strip()
            a1 = (row1.get("Answer") or "").strip()

            # Check if there's a next row with same category
            if i + 1 < len(cat_rows):
                row2 = cat_rows[i + 1]
                q2 = (row2.get("Question") or "").strip()
                a2 = (row2.get("Answer") or "").strip()

                is_hu_1 = _is_hungarian(q1)
                is_hu_2 = _is_hungarian(q2)

                # Only pair as EN+HU if one is clearly EN and the other clearly HU
                if is_hu_1 != is_hu_2:  # XOR - exactly one is Hungarian
                    if not is_hu_1 and is_hu_2:
                        # Clear EN + HU pair (EN first)
                        extraction_rows.append({
                            "company": "legacy",
                            "question_en": q1,
                            "question_hu": q2,
                            "answer_en": a1,
                            "answer_hu": a2,
                            "clause_ref": "legacy",
                            "url": "legacy",
                            "section": cat,
                        })
                    else:
                        # HU followed by EN - swap to make EN first
                        extraction_rows.append({
                            "company": "legacy",
                            "question_en": q2,
                            "question_hu": q1,
                            "answer_en": a2,
                            "answer_hu": a1,
                            "clause_ref": "legacy",
                            "url": "legacy",
                            "section": cat,
                        })
                    i += 2
                else:
                    # Both same language - treat as separate rows
                    # Process row1
                    if is_hu_1:
                        extraction_rows.append({
                            "company": "legacy",
                            "question_en": q1,  # fallback
                            "question_hu": q1,
                            "answer_en": a1,
                            "answer_hu": a1,
                            "clause_ref": "legacy",
                            "url": "legacy",
                            "section": cat,
                        })
                    else:
                        extraction_rows.append({
                            "company": "legacy",
                            "question_en": q1,
                            "question_hu": None,
                            "answer_en": a1,
                            "answer_hu": None,
                            "clause_ref": "legacy",
                            "url": "legacy",
                            "section": cat,
                        })
                    i += 1
            else:
                # Single row - determine if it's EN or HU
                is_hu = _is_hungarian(q1)
                if is_hu:
                    # HU-only row: question_en falls back to question_hu
                    extraction_rows.append({
                        "company": "legacy",
                        "question_en": q1,  # fallback
                        "question_hu": q1,
                        "answer_en": a1,
                        "answer_hu": a1,
                        "clause_ref": "legacy",
                        "url": "legacy",
                        "section": cat,
                    })
                else:
                    # EN-only row
                    extraction_rows.append({
                        "company": "legacy",
                        "question_en": q1,
                        "question_hu": None,
                        "answer_en": a1,
                        "answer_hu": None,
                        "clause_ref": "legacy",
                        "url": "legacy",
                        "section": cat,
                    })
                i += 1

    return extraction_rows


def _extraction_rows_to_faq_rows(topics: list[CanonicalTopic]) -> list[dict]:
    """Convert CanonicalTopic objects to faq.csv rows (Category, Question, Answer).

    Each bilingual topic produces two rows: EN row followed by HU row.
    EN-only topics produce one row.
    """
    faq_rows = []
    for topic in topics:
        # EN row
        if topic.question_en and topic.answer_en:
            faq_rows.append({
                "Category": topic.category,
                "Question": topic.question_en,
                "Answer": topic.answer_en,
            })
        # HU row (if bilingual)
        if topic.question_hu and topic.answer_hu:
            faq_rows.append({
                "Category": topic.category,
                "Question": topic.question_hu,
                "Answer": topic.answer_hu,
            })
    return faq_rows


def _load_faq_rows(faq_path: Optional[pathlib.Path]) -> list[dict]:
    """Load existing faq.csv rows as list of dicts with Category, Question, Answer.

    Returns empty list if faq_path is None or file doesn't exist.
    """
    if faq_path is None:
        return []
    path = pathlib.Path(faq_path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and list(reader.fieldnames) != REQUIRED_COLUMNS:
            raise ValueError(f"Unexpected columns {list(reader.fieldnames)} in {path}")
        return [dict(r) for r in reader]


def _write_faq_rows(faq_rows: list[dict], faq_path: Optional[pathlib.Path]) -> None:
    """Write faq.csv rows to file, creating parent directory if needed."""
    if faq_path is None:
        faq_path = DEFAULT_FAQ_PATH
    path = pathlib.Path(faq_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in faq_rows:
            writer.writerow({k: row.get(k, "") for k in REQUIRED_COLUMNS})
    # Ensure trailing newline (csv module does this, but be explicit)
    with open(path, "rb") as f:
        f.seek(0, 2)
        if f.tell() > 0:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                with open(path, "ab") as f:
                    f.write(b"\n")


def _normalize_question_en(row: dict) -> None:
    """Ensure question_en falls back to question_hu if empty (mutates row)."""
    if not row.get("question_en") and row.get("question_hu"):
        row["question_en"] = row["question_hu"]


def semantic_topics(
    rows: list[dict],
    faq_path: Optional[pathlib.Path] = None,
    clusterer: Optional[Clusterer] = None,
) -> list[CanonicalTopic]:
    """Run gate → cluster → canonicalize and return CanonicalTopic objects.

    Pure function — no file writes. Returns topics in deterministic order
    (ascending minimum member input index).

    Args:
        rows: New extraction-schema rows (keys: company, question_en, question_hu,
              answer_en, answer_hu, clause_ref, url, section).
        faq_path: Path to existing faq.csv. If provided, its content is merged
                  into the clustering input (whole-file rewrite semantics).
                  If None, no existing file is loaded.
        clusterer: Injectable clusterer (defaults to default_clusterer()).

    Returns:
        List of CanonicalTopic objects, one per distinct meaning, ordered by
        minimum member input index. Each topic carries topic_key and member_ids
        for provenance tracking (#44).
    """
    if clusterer is None:
        clusterer = default_clusterer()

    # Load existing faq.csv and convert to extraction schema
    existing_faq_rows = _load_faq_rows(faq_path)
    existing_extraction_rows = _faq_rows_to_extraction(existing_faq_rows)

    # Combine: existing file rows first (in file order), then new rows (in given order)
    all_input_rows = existing_extraction_rows + rows

    # Normalize question_en fallback for all rows
    for row in all_input_rows:
        _normalize_question_en(row)

    # Step 1: Filter clause dumps
    kept_rows, discarded_rows = filter_clause_dumps(all_input_rows)

    # Step 2: Cluster
    clusters = clusterer.cluster(kept_rows)

    # Step 3: Canonicalize each cluster
    canonical_topics = []
    for cluster in clusters:
        topic = canonicalize_cluster(cluster, kept_rows)
        canonical_topics.append(topic)

    # Step 4: Deduplicate by topic_key, keeping existing-canonical representative
    # Existing topics (from faq_path) have priority - their member indices are lower
    # since existing_extraction_rows come first in all_input_rows
    seen_keys = {}
    deduped_topics = []
    for topic in canonical_topics:
        key = topic.topic_key
        if key not in seen_keys:
            seen_keys[key] = topic
            deduped_topics.append(topic)
        # If key already seen, the first one (lower min member index) wins -
        # this is the existing-canonical-wins behavior

    # Step 5: Order by minimum member input index (already in this order due to
    # clusterer returning sorted clusters and canonicalize preserving order)
    # But ensure explicit sort for determinism
    deduped_topics.sort(key=lambda t: min(t.member_ids) if t.member_ids else float('inf'))

    return deduped_topics


def semantic_merge(
    rows: list[dict],
    faq_path: Optional[pathlib.Path] = None,
    clusterer: Optional[Clusterer] = None,
    dry_run: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run the full semantic merge pipeline and rewrite faq.csv.

    Orchestration order:
    1. filter_clause_dumps(all_input_rows)
    2. clusterer.cluster(kept)
    3. canonicalize_cluster(cluster, kept) per cluster
    4. Dedupe by topic_key (existing-canonical-wins)
    5. Order by minimum member input index
    6. Write (or return) faq.csv rows

    Args:
        rows: New extraction-schema rows.
        faq_path: Path to faq.csv. If None, no existing file is loaded.
                  Rewritten in place when not dry_run.
        clusterer: Injectable clusterer (default: default_clusterer()).
        dry_run: If True, return what would be written without modifying the file.

    Returns:
        Tuple of (written_rows, discarded_rows) where:
        - written_rows: List of dicts with Category, Question, Answer that were/would be written
        - discarded_rows: List of extraction-schema rows that were filtered as clause dumps
    """
    # Use faq_path as-is (None means no existing file)
    topics = semantic_topics(rows, faq_path, clusterer)
    written_rows = _extraction_rows_to_faq_rows(topics)

    # Get discarded rows by re-running the gate on all input
    existing_faq_rows = _load_faq_rows(faq_path)
    existing_extraction_rows = _faq_rows_to_extraction(existing_faq_rows)
    all_input_rows = existing_extraction_rows + rows
    for row in all_input_rows:
        _normalize_question_en(row)
    _, discarded_rows = filter_clause_dumps(all_input_rows)

    if not dry_run:
        if faq_path is None:
            faq_path = DEFAULT_FAQ_PATH
        _write_faq_rows(written_rows, faq_path)

    return written_rows, discarded_rows


if __name__ == "__main__":
    import json
    import sys

    rows_path = sys.argv[1] if len(sys.argv) > 1 else None
    if rows_path:
        with open(rows_path, encoding="utf-8") as f:
            rows_in = json.load(f)
        written, discarded = semantic_merge(rows_in, dry_run=True)
        print(f"Would write {len(written)} rows, discard {len(discarded)} clause dumps")
        for row in written:
            print(f"  {row['Category']} | {row['Question'][:60]}...")