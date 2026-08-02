import csv
import pathlib
import re
import unicodedata

DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
DEFAULT_FAQ_PATH = DEFAULT_DATA_DIR / "faq.csv"

REQUIRED_COLUMNS = ["Category", "Question", "Answer"]


def normalize_accent(string):
    """Fold accents/diacritics and lowercase, so 'foglalás' == 'foglalas'."""
    s = unicodedata.normalize("NFD", string).encode("ascii", "ignore").decode("ascii")
    return s.lower()


def _load_csv(path):
    path = pathlib.Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and list(reader.fieldnames) != REQUIRED_COLUMNS:
            raise ValueError(f"Unexpected columns {list(reader.fieldnames)} in {path}")
        return [dict(r) for r in reader]


def _ensure_trailing_newline(path):
    """Append-mode writes need the existing file to end with a newline, otherwise
    the first appended row gets glued onto the final existing line."""
    if not path.exists():
        return
    with open(path, "rb") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            return
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            with open(path, "ab") as f:
                f.write(b"\n")


def _existing_question_keys(path):
    keys = set()
    for row in _load_csv(path):
        q = (row.get("Question") or "").strip()
        if q:
            keys.add(normalize_accent(q))
        a = (row.get("Answer") or "").strip()
        if a:
            keys.add(normalize_accent(a))
    return keys


def _money_values(text):
    return [int(m) for m in re.findall(r"\d{2,5}\s*(?:Ft|HUF|EUR|€)", text)]


def _figure_spans(text):
    """Yield (value, unit) tokens for percentages and money amounts in ``text``."""
    for m in re.finditer(r"(\d{1,4})\s*(%)|(\d{2,5})\s*(Ft|HUF|EUR|€)", text):
        value = m.group(1) or m.group(3)
        unit = m.group(2) or m.group(4)
        yield int(value), unit


def _figure_ranges(answers):
    """Return e.g. ``["10% to 50%", "50Ft to 500Ft"]`` for units that diverge.

    A unit recurs across the folded answers but is *not* constant, so its spread
    (the very thing a single-company answer would hide) is expressed explicitly.
    """
    by_unit = {}
    for ans in answers:
        for value, unit in _figure_spans(ans):
            by_unit.setdefault(unit, []).append(value)
    spans = []
    for unit in sorted(by_unit):
        values = sorted(by_unit[unit])
        if values and values[0] != values[-1]:
            spans.append(f"{values[0]}{unit} to {values[-1]}{unit}")
    return spans


def _pick_category(sections):
    counts = {}
    for s in sections:
        s = (s or "").strip()
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "Terms & Conditions"
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


CATEGORY_MAX_LEN = 80


def _clean_category(label):
    """Trim a raw section label to a bounded, presentable Category.

    Extraction sometimes surfaces a whole parsed ``section`` (menu lists, long
    GDPR clause text) as the Category. Cap it on a word boundary so the column
    stays a short label instead of a wall of text, with a fallback for long
    single-token labels.
    """
    label = " ".join((label or "").split())
    if len(label) <= CATEGORY_MAX_LEN:
        return label or "Terms & Conditions"
    clipped = label[: CATEGORY_MAX_LEN + 1]
    head = clipped.rsplit(" ", 1)[0]
    head = head.rstrip(",;:-")
    if not head:
        head = clipped[:CATEGORY_MAX_LEN]
    return (head + " …").strip()


def _fold_answers(answers_en):
    """Fold divergent figures into consensus wording that captures the divergence."""
    if not answers_en:
        return ""
    if len(answers_en) == 1:
        return answers_en[0]
    spans = _figure_ranges(answers_en)
    if not spans:
        # figures agree across companies (or there are none) — keep the base answer
        return answers_en[0]
    base = max(answers_en, key=len)
    spread = " and ".join(spans)
    return f"{base.strip()} (figures vary across companies: {spread})"


def fold_duplicate(faq_rows):
    """Group extracted rows by topic and fold divergent figures.

    Returns a list of {Category, Question, Answer} dicts with **two rows per distinct
    topic** sharing a single Category: an EN row (question_en/answer_en) and a HU row
    (question_hu/answer_hu).
    """
    groups = {}
    for row in faq_rows:
        # Hungarian is the stable cross-company source key; the same Hungarian
        # question can surface as differently phrased English translations, so
        # grouping on the Hungarian question first prevents HU-duplicate rows.
        q = (row.get("question_hu") or row.get("question_en") or "").strip()
        topic = normalize_accent(q)
        groups.setdefault(topic, {"company": row.get("company"), "rows": []})
        groups[topic]["rows"].append(row)

    out = []
    for topic in sorted(groups):
        grp = groups[topic]
        rows = grp["rows"]
        category = _clean_category(_pick_category([r.get("section") for r in rows]))
        question_en = rows[0].get("question_en") or rows[0].get("question_hu") or ""
        question_hu = rows[0].get("question_hu") or rows[0].get("question_en") or ""
        answers_en = [r["answer_en"] for r in rows if r.get("answer_en")]
        answers_hu = [r["answer_hu"] for r in rows if r.get("answer_hu")]
        answer_en = _fold_answers(answers_en) or " ".join(answers_hu) or ""
        answer_hu = " ".join(answers_hu) or answer_en
        out.append({"Category": category, "Question": question_en, "Answer": answer_en})
        out.append({"Category": category, "Question": question_hu, "Answer": answer_hu})
    return out


def merge_bilingual(faq_rows, faq_path=None, dry_run=False):
    """Append newly extracted bilingual rows to data/faq.csv.

    Existing rows are never modified/removed (document ids faq_{i} stay stable). A new
    topic is added as two rows (EN + HU) sharing a single Category, deduped accent-insensitively
    against existing Questions/Answers. Returns the list of appended rows.
    """
    if faq_path is None:
        faq_path = DEFAULT_FAQ_PATH
    path = pathlib.Path(faq_path)

    existing_keys = load_question_keys(path)
    foldable = fold_topics(faq_rows)

    new_rows = []
    for row in foldable:
        if normalize_accent(row["Question"]) in existing_keys:
            continue
        if normalize_accent(row["Answer"]) in existing_keys:
            continue
        new_rows.append(row)

    if dry_run:
        return new_rows

    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_trailing_newline(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        for row in new_rows:
            writer.writerow({k: row.get(k, "") for k in REQUIRED_COLUMNS})
    return new_rows


def load_question_keys(path):
    return _existing_question_keys(path)


def merge_en(faq_rows, faq_path=None, dry_run=False):
    """Append standalone English rows (from the EN track, issue #38) to data/faq.csv.

    Each row carries only ``question_en``/``answer_en`` and becomes a **single** row
    (no HU companion) tagged with its section as Category. Append-only and deduped
    accent-insensitively against existing Questions/Answers, matching the conservative
    policy of :func:`merge_bilingual`. Returns the list of appended rows.
    """
    if faq_path is None:
        faq_path = DEFAULT_FAQ_PATH
    path = pathlib.Path(faq_path)

    existing_keys = _existing_question_keys(path)

    new_rows = []
    for row in faq_rows:
        q = (row.get("question_en") or "").strip()
        a = (row.get("answer_en") or "").strip()
        if not q or not a:
            continue
        if normalize_accent(q) in existing_keys:
            continue
        if normalize_accent(a) in existing_keys:
            continue
        category = _clean_category(row.get("section") or "Terms & Conditions")
        candidate = {"Category": category, "Question": q, "Answer": a}
        new_rows.append(candidate)
        existing_keys.add(normalize_accent(q))
        existing_keys.add(normalize_accent(a))

    if dry_run:
        return new_rows

    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_trailing_newline(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        for row in new_rows:
            writer.writerow({k: row.get(k, "") for k in REQUIRED_COLUMNS})
    return new_rows


def fold_topics(faq_rows):
    return fold_duplicate(faq_rows)


if __name__ == "__main__":
    import json
    import sys

    rows_path = sys.argv[1] if len(sys.argv) > 1 else None
    if rows_path:
        with open(rows_path, encoding="utf-8") as f:
            rows_in = json.load(f)
        added = merge_bilingual(rows_in, dry_run=True)
        print(f"{len(added)} new bilingual rows ready to append")