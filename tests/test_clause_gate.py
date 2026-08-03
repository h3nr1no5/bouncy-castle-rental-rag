import pytest
from src.clause_gate import is_clause_dump, filter_clause_dumps


def test_is_clause_dump_none_empty():
    """None, empty string, and whitespace-only input all return False."""
    assert is_clause_dump(None) is False
    assert is_clause_dump("") is False
    assert is_clause_dump("   ") is False
    assert is_clause_dump("\t\n") is False


def test_is_clause_dump_short_answers():
    """Short answers are never dumps."""
    assert is_clause_dump("Yes, a deposit is required.") is False
    assert is_clause_dump("A") is False
    assert is_clause_dump("Short answer") is False


def test_is_clause_dump_long_single_sentence():
    """Long single-sentence answers are not dumps."""
    # Exactly 200 chars (should not be a dump)
    assert is_clause_dump("A" * 200) is False
    # 201 chars single sentence (should not be a dump)
    assert is_clause_dump("A" * 201) is False
    # 250 chars with period (should not be a dump)
    assert is_clause_dump("A" * 250 + ".") is False
    # 220 chars with exclamation (should not be a dump)
    assert is_clause_dump("A" * 220 + "!") is False
    # 250 chars with ellipsis (should not be a dump)
    assert is_clause_dump("A" * 250 + "...") is False


def test_is_clause_dump_long_multi_sentence():
    """Long multi-sentence answers ARE dumps."""
    # 100 chars + ". " + 100 chars + "." (201 total, 2 terminators)
    assert is_clause_dump("A" * 100 + ". " + "B" * 100 + ".") is True
    # Same with exclamation
    assert is_clause_dump("A" * 100 + "! " + "B" * 100 + "!") is True
    # Same with question mark
    assert is_clause_dump("A" * 100 + "? " + "B" * 100 + "?") is True


def test_is_clause_dump_newline_semicolon():
    """Newline and semicolon signals."""
    assert is_clause_dump("A" * 150 + "\n" + "B" * 100) is True
    assert is_clause_dump("A" * 150 + "; " + "B" * 100) is True


def test_is_clause_dump_boundary():
    """Boundary behavior: 200-char two-sentence answer is False, 201-char is True."""
    # 200 chars with two sentences (98 + 2 + 99 + 1 = 200)
    two_sentence_200 = "A" * 98 + ". " + "B" * 99 + "."
    assert len(two_sentence_200) == 200
    assert is_clause_dump(two_sentence_200) is False
    # 201 chars with two sentences (99 + 2 + 99 + 1 = 201)
    two_sentence_201 = "A" * 99 + ". " + "B" * 99 + "."
    assert len(two_sentence_201) == 201
    assert is_clause_dump(two_sentence_201) is True


def test_is_clause_dump_threshold_override():
    """Threshold override works correctly."""
    # 250-char two-sentence answer (123 + 2 + 124 + 1 = 250)
    two_sentence_250 = "A" * 123 + ". " + "B" * 124 + "."
    assert len(two_sentence_250) == 250
    # At default 200, should be True
    assert is_clause_dump(two_sentence_250) is True
    # With threshold=300, should be False
    assert is_clause_dump(two_sentence_250, threshold=300) is False


def test_is_clause_dump_abbreviation():
    """Abbreviation markers add at most one break."""
    # Single abbreviation marker "e.g. " + 205 chars
    assert is_clause_dump("e.g. " + "A" * 205) is False
    # Multiple abbreviation markers should still be False
    assert is_clause_dump("e.g. " + "A" * 100 + ". " + "i.e. " + "B" * 100) is False


def test_is_clause_dump_consecutive_terminators():
    """Consecutive terminators count as a single terminator."""
    # "A"*100 + "..." + " " + "B"*100 + "." (2 terminators: ... counts as 1, final . counts as 1)
    assert is_clause_dump("A" * 100 + "... " + "B" * 100 + ".") is True
    # "A"*100 + "?! " + "B"*100 + "." (2 terminators: ?! counts as 1, final . counts as 1)
    assert is_clause_dump("A" * 100 + "?! " + "B" * 100 + ".") is True


def test_filter_clause_dumps_basic():
    """Basic filtering test."""
    rows = [
        {"answer_en": "Short answer"},
        {"answer_en": "A" * 100 + ". " + "B" * 100 + "."},
        {"answer_hu": "Short answer hu"},
        {"answer_hu": "C" * 100 + ". " + "D" * 100 + "."},
        {"Answer": "Short answer csv"},
        {"Answer": "E" * 100 + ". " + "F" * 100 + "."},
        {"question_en": "Long question but short answer", "answer_en": "Short"},
        {"section": "Long section but short answer", "answer_en": "Short"},
        {"answer_en": ""},  # Empty answer
        {"answer_en": None},  # None answer
    ]

    kept, discarded = filter_clause_dumps(rows)

    # Check that all rows are accounted for
    assert len(kept) + len(discarded) == len(rows)

    # Check that no row appears in both lists
    kept_indices = {id(row) for row in kept}
    discarded_indices = {id(row) for row in discarded}
    assert kept_indices.isdisjoint(discarded_indices)

    # Check that relative order is preserved within kept and within discarded
    # (kept rows appear in same relative order as in input, same for discarded)
    kept_original_indices = [rows.index(row) for row in kept]
    discarded_original_indices = [rows.index(row) for row in discarded]
    assert kept_original_indices == sorted(kept_original_indices)
    assert discarded_original_indices == sorted(discarded_original_indices)

    # Check specific rows
    # Rows with clause dumps (indices 1, 3, 5): 3 rows discarded
    # Rows without (indices 0, 2, 4, 6, 7, 8, 9): 7 rows kept
    assert len(kept) == 7
    assert len(discarded) == 3

    # Verify which rows are kept/discarded
    kept_answers = [
        row.get("answer_en") or row.get("answer_hu") or row.get("Answer") or ""
        for row in kept
    ]
    discarded_answers = [
        row.get("answer_en") or row.get("answer_hu") or row.get("Answer") or ""
        for row in discarded
    ]

    assert "Short answer" in kept_answers
    assert "Short answer hu" in kept_answers
    assert "Short answer csv" in kept_answers
    assert "Short" in kept_answers  # From the question/section rows with short answers

    assert any("A" * 100 + ". " + "B" * 100 + "." in ans for ans in discarded_answers)
    assert any("C" * 100 + ". " + "D" * 100 + "." in ans for ans in discarded_answers)
    assert any("E" * 100 + ". " + "F" * 100 + "." in ans for ans in discarded_answers)


def test_filter_clause_dumps_threshold():
    """Threshold override in filter_clause_dumps."""
    # 250-char two-sentence answer
    rows = [
        {"answer_en": "A" * 123 + ". " + "B" * 124 + "."},  # 250 chars, 2 sentences
    ]

    # Default threshold (200) should discard
    kept, discarded = filter_clause_dumps(rows)
    assert len(discarded) == 1
    assert len(kept) == 0

    # Higher threshold (300) should keep
    kept, discarded = filter_clause_dumps(rows, threshold=300)
    assert len(kept) == 1
    assert len(discarded) == 0


def test_filter_clause_dumps_no_answer_fields():
    """Rows with no non-empty answer fields are kept."""
    rows = [
        {"question_en": "Some question", "section": "Some section"},
        {"question_hu": "Valami kérdés"},
        {"company": "Test Corp"},
    ]

    kept, discarded = filter_clause_dumps(rows)

    assert len(kept) == 3
    assert len(discarded) == 0
    assert kept == rows


def test_filter_clause_dumps_mixed_fields():
    """Rows with multiple answer fields - any dump field triggers discard."""
    rows = [
        {"answer_en": "Short", "answer_hu": "A" * 100 + ". " + "B" * 100 + "."},
        {"answer_en": "A" * 100 + ". " + "B" * 100 + ".", "answer_hu": "Short"},
        {"answer_en": "Short", "answer_hu": "Short"},
    ]

    kept, discarded = filter_clause_dumps(rows)

    # First two rows should be discarded (one dump field each)
    assert len(discarded) == 2
    assert len(kept) == 1

    # Third row should be kept (both short)
    assert kept[0]["answer_en"] == "Short"
    assert kept[0]["answer_hu"] == "Short"


def test_filter_clause_dumps_preserves_objects():
    """Input list is not mutated; same dict objects returned."""
    original_row = {"answer_en": "A" * 100 + ". " + "B" * 100 + "."}
    rows = [original_row]

    kept, discarded = filter_clause_dumps(rows)

    # Check that the same object is returned (in discarded since it's a clause dump)
    assert discarded[0] is original_row
    assert len(kept) == 0

    # Check that original list is unchanged
    assert rows == [original_row]


def test_is_clause_dump_terminator_before_closing_quote():
    """Terminator followed by closing quote counts as sentence boundary."""
    # Two sentences, second ends with ." (period then quote)
    # 120 + 2 + 120 + 2 = 244 chars (>200), 2 terminators
    text = "A" * 120 + ". " + "B" * 120 + '."'
    assert len(text) == 244
    assert is_clause_dump(text) is True

    # Same with ?" (question mark then quote)
    text = "A" * 120 + "? " + "B" * 120 + '?"'
    assert len(text) == 244
    assert is_clause_dump(text) is True

    # Same with !" (exclamation then quote)
    text = "A" * 120 + "! " + "B" * 120 + '!"'
    assert len(text) == 244
    assert is_clause_dump(text) is True

    # Single sentence ending with ." should NOT be a dump
    text = "A" * 250 + '."'
    assert is_clause_dump(text) is False

    # Terminator followed by single quote
    text = "A" * 120 + ". " + "B" * 120 + ".'"
    assert is_clause_dump(text) is True

    # Terminator followed by quote then whitespace
    text = "A" * 120 + '. " ' + "B" * 120 + "."
    assert is_clause_dump(text) is True

    # Unicode smart quotes (U+201C/U+201D for double, U+2018/U+2019 for single)
    text = "A" * 120 + ". " + "B" * 120 + ".\u201d"  # .”
    assert is_clause_dump(text) is True
    text = "A" * 120 + "? " + "B" * 120 + "?\u201d"  # ?”
    assert is_clause_dump(text) is True
    text = "A" * 120 + "! " + "B" * 120 + "!\u201d"  # !”
    assert is_clause_dump(text) is True
    text = "A" * 120 + ". " + "B" * 120 + ".\u2019"  # .’
    assert is_clause_dump(text) is True