import hashlib
import json
import logging
import pathlib

from pydantic import BaseModel, ValidationError

from src.llm import ask_llm

logger = logging.getLogger(__name__)

DEFAULT_TOC_DIR = pathlib.Path(__file__).resolve().parents[1] / "db" / "toc"

SYSTEM_PROMPT = (
    "You extract bilingual Q&A content from the Terms & Conditions of a Hungarian "
    "bouncy-castle rental company. Given a section of the terms, produce at most 3 "
    "concrete, useful customer questions with their answers, in Hungarian and English. "
    "Preserve concrete facts exactly: figures, fees, percentages, deposit amounts, "
    "weather/rain rules, delivery and pickup times. Do not invent facts not present in "
    "the text. Respond ONLY with a JSON object of the form "
    '{"pairs":[{"question_hu":..., "answer_hu":..., "question_en":..., "answer_en":...}]}.'
)


class QAPair(BaseModel):
    question_hu: str
    answer_hu: str
    question_en: str
    answer_en: str


def content_hash(text):
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_code_fences(text):
    text = text.strip()
    if text.startswith("```") or "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
    return text.lstrip("json").strip()


def _parse_qa(reply):
    """Best-effort parse of the LLM reply into a list of QAPair dicts."""
    text = strip_code_fences(reply)

    def to_pairs(obj):
        items = obj
        if isinstance(items, dict):
            items = items.get("pairs") or items.get("items") or [obj]
        if items is None:
            items = []
        if isinstance(items, dict):
            items = [items]
        pairs = []
        for raw in items:
            qa = QAPair(**raw)
            pairs.append(qa.model_dump())
        return pairs

    try:
        return to_pairs(json.loads(text))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return []


def _extract_section(section):
    """Ask the LLM once for a section; returns (pairs, ok)."""
    last_error = None
    for _attempt in range(3):
        try:
            result = ask_llm(
                system_prompt=SYSTEM_PROMPT,
                user_message=section["clause_text"],
            )
            pairs = _parse_qa(result["response"])
            if pairs:
                return pairs, True
            last_error = ValueError("empty/invalid QA output")
        except Exception as e:  # noqa: BLE001
            last_error = e
    logger.warning("SKIP %s/%s after retries: %s", section.get("company"), section.get("section"), last_error)
    return None, False


def _load_cache(cache_path):
    if cache_path and cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def _dump_cache(cache, cache_path):
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(cache_path)


def _row_from_pair(section, pair):
    return {
        "company": section["company"],
        "section": section["section"],
        "clause_ref": section["clause_ref"],
        "question_hu": pair["question_hu"],
        "answer_hu": pair["answer_hu"],
        "question_en": pair["question_en"],
        "answer_en": pair["answer_en"],
    }


def extract(chunks, toc_dir=None, cache_path=None):
    """Extract bilingual QA pairs from parsed chunks.

    Results are cached keyed by (company, section, content_hash) so reruns with
    unchanged sources issue zero extra LLM calls.
    """
    if toc_dir is None:
        toc_dir = DEFAULT_TOC_DIR
    toc_dir = pathlib.Path(toc_dir)
    if cache_path is None:
        cache_path = toc_dir / "extract_cache.json"

    cache = _load_cache(cache_path)
    calls = 0
    cache_hits = 0
    rows = []
    for section in chunks:
        key = content_hash(section["clause_text"])
        cache_key = f'{section["company"]}::{section["section"]}::{key}'
        cached = cache.get(cache_key, None)
        if cached is not None:
            cache_hits += 1
            for pair in cached:
                rows.append(_row_from_pair(section, pair))
            continue
        pairs, ok = _extract_section(section)
        if not ok:
            continue
        calls += 1
        serialized = [dict(p) for p in pairs]
        cache[cache_key] = serialized
        for pair in serialized:
            rows.append(_row_from_pair(section, pair))
    _dump_cache(cache, cache_path)
    return {"rows": rows, "calls": calls, "cache_hits": cache_hits}


_extract_single = _extract_section