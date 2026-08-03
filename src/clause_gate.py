import re
from typing import Optional, Tuple, List, Dict


# Common abbreviations that should not count as sentence terminators.
# These are matched case-insensitively and must be followed by whitespace or end of string.
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "mr.",
    "mrs.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "jan.",
    "feb.",
    "mar.",
    "apr.",
    "jun.",
    "jul.",
    "aug.",
    "sep.",
    "oct.",
    "nov.",
    "dec.",
)


def _count_sentence_terminators(text: str) -> int:
    """Count sentence terminators in text, collapsing consecutive runs.

    A sentence terminator is ``.``, ``!``, or ``?`` followed by whitespace or
    end of string. A run of consecutive terminators (``...``, ``?!``, ``!!``)
    counts as a single terminator. Periods in known abbreviations (``e.g.``,
    ``i.e.``, etc.) are not counted as terminators.

    Args:
        text: The text to analyze (whitespace-stripped, no newlines/semicolons).

    Returns:
        Number of sentence terminators (after collapsing consecutive runs).
    """
    # Protect abbreviations by replacing their final period with a placeholder
    # that won't be matched by the terminator regex.
    protected = text
    for abbr in _ABBREVIATIONS:
        # Match abbreviation followed by whitespace or end of string
        # Replace the final period with a non-terminator character
        pattern = re.compile(re.escape(abbr) + r"(?=\s|$)", re.IGNORECASE)
        protected = pattern.sub(abbr[:-1] + "\u200B", protected)

    # Find all terminator positions: . ! ? followed by optional closing quote
    # (ASCII " ' or Unicode " " ' ') then whitespace or end of string.
    # This regex naturally handles consecutive terminators correctly:
    # - "..." -> only the last . is followed by whitespace/end, so 1 match
    # - "?! " -> only the ! is followed by whitespace, so 1 match
    # - ". . " -> both . are followed by whitespace, so 2 matches
    # - '." ' or '?" ' -> terminator followed by quote then whitespace counts as 1
    terminator_pattern = re.compile(r"[.!?](?=[\"'\u2018\u2019\u201c\u201d]?(?:\s|$))")
    matches = list(terminator_pattern.finditer(protected))

    return len(matches)


def is_clause_dump(answer: Optional[str], *, threshold: int = 200) -> bool:
    """Flag answers that are long legal-clause dumps.

    Returns ``True`` iff the whitespace-stripped answer is longer than ``threshold``
    (strict ``>``) AND the answer is multi-sentence.

    Multi-sentence is defined as any of:
      a) at least two sentence terminators
      b) the text contains a semicolon ``;``
      c) the text contains a newline (``\\n`` or ``\\r``)

    A sentence terminator is ``.``, ``!``, or ``?`` followed by whitespace or by
    the end of the string. A run of consecutive terminators (``...``, ``?!``,
    ``!!``) counts as a **single** terminator.

    A period not followed by whitespace (e.g. in ``5.5``) is not a terminator.
    Common abbreviations (``e.g.``, ``i.e.``, etc.) are not counted as terminators.

    Args:
        answer: The answer text to evaluate.
        threshold: Minimum length (exclusive) for a dump. Default 200.

    Returns:
        ``True`` if the answer is a clause dump, ``False`` otherwise.
    """
    if answer is None:
        return False

    stripped = answer.strip()
    if not stripped:
        return False

    # Length check (strict >)
    if len(stripped) <= threshold:
        return False

    # Normalize line endings for newline/semicolon checks
    normalized = stripped.replace("\r\n", "\n").replace("\r", "\n")

    # Check for semicolon or newline (multi-sentence indicators)
    if ";" in normalized or "\n" in normalized:
        return True

    # Count sentence terminators
    terminator_count = _count_sentence_terminators(normalized)

    # Multi-sentence if at least two terminators
    return terminator_count >= 2


def filter_clause_dumps(rows: List[Dict], *, threshold: int = 200) -> Tuple[List[Dict], List[Dict]]:
    """Separate rows into kept and discarded based on clause dump detection.

    Returns ``(kept_rows, discarded_rows)`` where:
      - ``kept_rows`` contains rows that are NOT clause dumps
      - ``discarded_rows`` contains rows that ARE clause dumps

    Concatenating ``kept_rows + discarded_rows`` reproduces the input rows in
    their original relative order, and no row appears in both lists.

    A row is discarded iff ANY non-empty answer field it carries satisfies
    ``is_clause_dump``:
      - ``answer_en``
      - ``answer_hu``
      - ``Answer`` (faq.csv-style rows)

    Rows with no non-empty answer field are kept.

    Only answer fields are evaluated; ``question_en`` or ``section`` never cause
    a discard.

    The input list is not mutated; the same unmodified dict objects are returned
    in the two output lists.

    Args:
        rows: List of row dictionaries to filter.
        threshold: Length threshold for clause dump detection. Default 200.

    Returns:
        Tuple of (kept_rows, discarded_rows).
    """
    kept_rows = []
    discarded_rows = []

    for row in rows:
        # Check all possible answer fields
        is_dump = False

        # Check answer_en
        answer_en = row.get("answer_en")
        if answer_en:
            if is_clause_dump(answer_en, threshold=threshold):
                is_dump = True

        # Check answer_hu
        if not is_dump:
            answer_hu = row.get("answer_hu")
            if answer_hu:
                if is_clause_dump(answer_hu, threshold=threshold):
                    is_dump = True

        # Check Answer (faq.csv-style rows)
        if not is_dump:
            answer = row.get("Answer")
            if answer:
                if is_clause_dump(answer, threshold=threshold):
                    is_dump = True

        if is_dump:
            discarded_rows.append(row)
        else:
            kept_rows.append(row)

    return kept_rows, discarded_rows