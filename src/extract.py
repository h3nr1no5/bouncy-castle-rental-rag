import hashlib
import json
import logging
import pathlib

from pydantic import BaseModel, ValidationError

from src.llm import ask_llm

logger = logging.getLogger(__name__)

DEFAULT_TOC_DIR = pathlib.Path(__file__).resolve().parents[1] / "db" / "toc"
DEFAULT_TOC_EN_DIR = pathlib.Path(__file__).resolve().parents[1] / "db" / "toc_en"

SYSTEM_PROMPT = (
    "You extract bilingual Q&A content from the Terms & Conditions of a Hungarian "
    "bouncy-castle rental company. Given a section of the terms, produce at most 3 "
    "concrete, useful customer questions and their answers, in Hungarian and English. "
    "Preserve concrete facts exactly: figures, fees, percentages, deposit amounts, "
    "weather/rain conditions, delivery and pickup times. Do not invent facts not present in "
    "the text. Respond ONLY with a JSON object of the form "
    '{"pairs":[{"question_hu":..., "answer_hu":..., "question_en":..., "answer_en":...}]}.'
)

EN_SYSTEM_PROMPT = (
    "You extract Q&A content from the Terms & Conditions of an English-language "
    "bouncy-castle rental company. Given a section of the terms, produce at most 3 "
    "concrete, useful customer questions and their answers, in English. "
    "Preserve concrete facts, figures, fees, percentages, deposit amounts, "
    "weather/rain conditions, delivery and pickup times. Do not invent facts not present in "
    "the text. Respond ONLY with a JSON object of the form "
    '{"pairs":[{"question_en":..., "answer_en":...}]}.'
)


class QAPair(BaseModel):
    question_hu: str
    answer_hu: str
    question_en: str
    answer_en: str


class ENQAPair(BaseModel):
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


def _parse_qa(reply, model=QAPair):  # type: ignore[type-arg]
    """Best-effort parse of the LLM reply into a list of pair dicts.

    ``model`` chooses the pair schema; bilingual by default, single-language EN
    via ``ENQAPair``.
    """
    text = strip_code_fences(reply)

    def to_pairs(obj):
        items = obj
        if isinstance(items, dict):
            items = items.get("pairs") or items.get("items") or items.get("questions") or [obj]
        if items is None:
            items = []
        if isinstance(items, dict):
            items = [items]
        pairs = []
        for raw in items:
            qa = model(**raw)
            pairs.append(qa.model_dump())
        return pairs

    try:
        return to_pairs(json.loads(text))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return []


def _extract_section(section, model=QAPair, system_prompt=SYSTEM_PROMPT):
    """Ask the LLM once for a section; returns (pairs, ok).

    ``model``/``system_prompt`` select the language schema (bilingual HU by default,
    single-language EN via ``ENQAPair``/``EN_SYSTEM_PROMPT``).
    """
    last_error = None
    for _attempt in range(3):
        try:
            result = ask_llm(
                system_prompt=system_prompt,
                user_message=section["clause_text"],
            )
            pairs = _parse_qa(result["response"], model=model)
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


def _row_from_en_pair(section, pair):
    return {
        "company": section["company"],
        "section": section["section"],
        "clause_ref": section["clause_ref"],
        "question_en": pair["question_en"],
        "answer_en": pair["answer_en"],
    }


def extract(chunks, toc_dir=None, cache_path=None):
    """Extract bilingual QA pairs from parsed chunks.

    Results are cached keyed by (company, section, content_hash) so reruns with
    unchanged sources issue zero extra LLM calls.
    """
    return _extract_impl(
        chunks,
        toc_dir=toc_dir,
        cache_path=cache_path,
        default_toc_dir=DEFAULT_TOC_DIR,
        model=QAPair,
        system_prompt=SYSTEM_PROMPT,
        row_fn=_row_from_pair,
    )


def extract_en(chunks, toc_dir=None, cache_path=None):
    """Extract single-language EN QA pairs from English-language parsed chunks.

    Grounds English Q&A in real English Terms & Conditions (issue #38). Cached
    independently under ``db/toc_en/extract_cache.json`` so the HU cache stays
    isolated.
    """
    return _extract_impl(
        chunks,
        toc_dir=toc_dir,
        cache_path=cache_path,
        default_toc_dir=DEFAULT_TOC_EN_DIR,
        model=ENQAPair,
        system_prompt=EN_SYSTEM_PROMPT,
        row_fn=_row_from_en_pair,
    )


def _extract_impl(chunks, toc_dir, cache_path, default_toc_dir, model, system_prompt, row_fn):
    """Shared extract/caching loop used by both the HU and EN tracks."""
    if toc_dir is None:
        toc_dir = default_toc_dir
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
                rows.append(row_fn(section, pair))
            continue
        pairs, ok = _extract_section(section, model=model, system_prompt=system_prompt)
        if not ok:
            continue
        calls += 1
        serialized = [dict(p) for p in pairs]
        cache[cache_key] = serialized
        for pair in serialized:
            rows.append(row_fn(section, pair))
        # Incrementally persist so an interrupted run keeps its progress and a
        # rerun skips already-extracted chunks instead of redoing all LLM calls.
        _dump_cache(cache, cache_path)
    return {"rows": rows, "calls": calls, "cache_hits": cache_hits}  # type: ignore[list-item]


_extract_single = _extract_section