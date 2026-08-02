import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple

from src.clusterer import TopicCluster


@dataclass(frozen=True)
class CanonicalTopic:
    """A canonical topic derived from a cluster of rows."""
    topic_key: str
    question_en: str
    question_hu: Optional[str]
    answer_en: str
    answer_hu: Optional[str]
    category: str
    member_ids: Tuple[int, ...]


def topic_key(canonical_en: str) -> str:
    """Generate a deterministic slug from an English question.
    
    Rules:
    - Lowercase, punctuation stripped, whitespace → '-'
    - Diacritics stripped (e.g., 'é' → 'e')
    - Bounded to ≤64 chars, truncated at word boundary
    - Trailing hyphens stripped
    - Empty/whitespace-only input returns 'untitled'
    """
    if not canonical_en or not canonical_en.strip():
        return "untitled"
    
    # Strip diacritics and lowercase
    text = unicodedata.normalize("NFD", canonical_en).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    
    # Replace non-alphanumeric chars with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    
    # Bound to 64 chars at word boundary
    if len(text) > 64:
        # Find last space within 64 chars
        truncated = text[:64]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            text = truncated[:last_space]
        else:
            # No space found, truncate to 64 chars
            text = truncated
        text = text.rstrip("-")
    
    return text if text else "untitled"


def _clean_category(label: str) -> str:
    """Trim a raw section label to a bounded, presentable Category.
    
    Extraction sometimes surfaces a whole parsed section (menu lists, long
    GDPR clause text) as the Category. Cap it on a word boundary so the column
    stays a short label instead of a wall of text, with a fallback for long
    single-token labels.
    
    Spec requirements:
    - Trim "All Products" prefix
    - Split on dash/period to get first clause
    - Bound to ≤40 chars
    - Fallback to "General" if empty
    """
    if not label:
        return "General"
    
    # Remove leading "All Products" prefix
    if label.startswith("All Products"):
        label = label[len("All Products"):].lstrip()
    
    # Split on dash or period to get first clause
    parts = re.split(r"[-–—.]", label, 1)
    first_part = parts[0].strip()
    
    # Clean up any remaining punctuation and normalize whitespace
    first_part = re.sub(r"\s+", " ", first_part).strip()
    
    # Bound to 40 chars at word boundary
    if len(first_part) > 40:
        words = first_part.split()
        truncated = ""
        for word in words:
            if len(truncated + " " + word if truncated else word) <= 40:
                truncated = truncated + " " + word if truncated else word
            else:
                break
        first_part = truncated.rstrip()
    
    # Fallback to "General" if empty
    return first_part if first_part else "General"


def _figure_spans(text: str) -> list[tuple[int, str]]:
    """Yield (value, unit) tokens for percentages and money amounts in ``text``."""
    spans = []
    for m in re.finditer(r"(\d{1,4})\s*(%)|(\d{2,5})\s*(Ft|HUF|EUR|€)", text):
        value = m.group(1) or m.group(3)
        unit = m.group(2) or m.group(4)
        spans.append((int(value), unit))
    return spans


def _figure_ranges(answers: list[str]) -> list[str]:
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


def _fold_answers(answers_en: list[str]) -> str:
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


def canonicalize_cluster(cluster: TopicCluster, rows: list[dict]) -> CanonicalTopic:
    """Convert a TopicCluster into a single CanonicalTopic.
    
    Args:
        cluster: The TopicCluster containing member row indices
        rows: The full list of rows (indexed by cluster.rows)
        
    Returns:
        A CanonicalTopic with canonical EN question, answer, category, etc.
        
    Raises:
        ValueError: If the cluster has no members
    """
    if not cluster.rows:
        raise ValueError("Empty cluster (no members)")
    
    # Get member rows
    member_rows = [rows[i] for i in cluster.rows]
    
    # Canonical EN question: lowest-index member's question_en
    base_idx = cluster.rows[0]
    base_row = rows[base_idx]
    question_en = base_row.get("question_en", "")
    
    # Collect answers for consensus folding
    answers_en = [r.get("answer_en", "") for r in member_rows if r.get("answer_en")]
    
    # Fold answers (handles divergent figures)
    answer_en = _fold_answers(answers_en)
    
    # Hungarian companion: if any member has question_hu, use base member's
    question_hu = None
    answer_hu = None
    for r in member_rows:
        if r.get("question_hu"):
            question_hu = r.get("question_hu", "")
            answer_hu = r.get("answer_hu", "")
            break
    
    # Category: all members share same category, clean it
    categories = [r.get("section", "") for r in member_rows if r.get("section")]
    if categories:
        # All members share the same category (per spec)
        category = categories[0]
    else:
        category = "Terms & Conditions"
    
    category = _clean_category(category)
    
    # Generate topic_key from canonical EN question
    topic_key_slug = topic_key(question_en)
    
    return CanonicalTopic(
        topic_key=topic_key_slug,
        question_en=question_en,
        question_hu=question_hu,
        answer_en=answer_en,
        answer_hu=answer_hu,
        category=category,
        member_ids=tuple(cluster.rows)
    )
