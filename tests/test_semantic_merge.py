"""Tests for src/semantic_merge.py — mirrors style of test_clusterer.py, test_canonicalizer.py, test_clause_gate.py."""

import json
import tempfile
import pathlib
from src.semantic_merge import (
    semantic_topics,
    semantic_merge,
    _faq_rows_to_extraction,
    _extraction_rows_to_faq_rows,
    _load_faq_rows,
    _write_faq_rows,
)
from src.clusterer import ExactClusterer, LLMClusterer, TopicCluster
from src.canonicalizer import CanonicalTopic, topic_key
from src.clause_gate import is_clause_dump
from src.merge_bilingual import REQUIRED_COLUMNS, normalize_accent


def _row(company, question_en, question_hu=None, answer_en="Answer.", answer_hu=None, clause_ref=None, url=None, section=None):
    """Helper to create an extraction-schema test row."""
    return {
        "company": company,
        "question_en": question_en,
        "question_hu": question_hu,
        "answer_en": answer_en,
        "answer_hu": answer_hu if answer_hu is not None else "",
        "clause_ref": clause_ref or f"{company}/1",
        "url": url or f"https://{company}.hu/aszf",
        "section": section or "Terms & Conditions",
    }


def _faq_row(category, question, answer):
    """Helper to create a faq.csv-style row."""
    return {"Category": category, "Question": question, "Answer": answer}


def _llm_reply(assignments):
    """Helper to create a mock LLM reply (mirrors test_clusterer.py)."""
    return {
        "response": json.dumps({"assignments": assignments}),
        "model": "x", "provider": "mock", "latency": 0, "cost": 0,
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
    }


def _mock_clusterer_for_rows(rows, topic_assignments):
    """Create an LLMClusterer with a mocked ask_llm that returns the given topic assignments.
    
    Args:
        rows: The list of rows that will be clustered
        topic_assignments: Dict mapping row index (as string) to topic label
    
    Returns:
        LLMClusterer with fake ask_llm that returns the assignments
    """
    def fake_ask_llm(system_prompt, user_message, groq_model=None, openai_model=None):
        return _llm_reply(topic_assignments)
    return LLMClusterer(ask_llm=fake_ask_llm)


# --- EN/HU pairing round-trip from faq.csv ---

def test_faq_rows_to_extraction_pairs_en_hu():
    """Consecutive rows with same Category become EN+HU pair."""
    faq_rows = [
        _faq_row("Booking", "How much is the deposit?", "20% deposit."),
        _faq_row("Booking", "Mennyi a foglaló?", "20% foglaló."),
        _faq_row("Cancellation", "Can I cancel?", "Yes, 14 days notice."),
    ]
    extraction = _faq_rows_to_extraction(faq_rows)
    assert len(extraction) == 2
    # First pair: EN + HU
    assert extraction[0]["question_en"] == "How much is the deposit?"
    assert extraction[0]["question_hu"] == "Mennyi a foglaló?"
    assert extraction[0]["answer_en"] == "20% deposit."
    assert extraction[0]["answer_hu"] == "20% foglaló."
    assert extraction[0]["section"] == "Booking"
    # Second row: EN-only (odd count in category)
    assert extraction[1]["question_en"] == "Can I cancel?"
    assert extraction[1]["question_hu"] is None
    assert extraction[1]["answer_en"] == "Yes, 14 days notice."
    assert extraction[1]["answer_hu"] is None
    assert extraction[1]["section"] == "Cancellation"


def test_faq_rows_to_extraction_preserves_category():
    """Category is preserved as section in extraction rows."""
    faq_rows = [
        _faq_row("Setup Requirements", "How much space?", "5m x 5m."),
        _faq_row("Setup Requirements", "Mennyi hely kell?", "5m x 5m."),
    ]
    extraction = _faq_rows_to_extraction(faq_rows)
    assert extraction[0]["section"] == "Setup Requirements"


def test_extraction_rows_to_faq_rows_bilingual():
    """CanonicalTopic with HU produces two faq rows (EN then HU)."""
    topic = CanonicalTopic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        question_hu="Mennyi a foglaló?",
        answer_en="20% deposit.",
        answer_hu="20% foglaló.",
        category="Booking & Reservations",
        member_ids=(0, 1),
    )
    faq_rows = _extraction_rows_to_faq_rows([topic])
    assert len(faq_rows) == 2
    assert faq_rows[0] == {"Category": "Booking & Reservations", "Question": "How much is the deposit?", "Answer": "20% deposit."}
    assert faq_rows[1] == {"Category": "Booking & Reservations", "Question": "Mennyi a foglaló?", "Answer": "20% foglaló."}


def test_extraction_rows_to_faq_rows_en_only():
    """CanonicalTopic without HU produces one faq row."""
    topic = CanonicalTopic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        question_hu=None,
        answer_en="20% deposit.",
        answer_hu=None,
        category="Booking & Reservations",
        member_ids=(0,),
    )
    faq_rows = _extraction_rows_to_faq_rows([topic])
    assert len(faq_rows) == 1
    assert faq_rows[0] == {"Category": "Booking & Reservations", "Question": "How much is the deposit?", "Answer": "20% deposit."}


def test_extraction_rows_to_faq_rows_en_fallback_to_hu_answer():
    """Bilingual topic with no EN answer falls back to HU answer for EN row."""
    topic = CanonicalTopic(
        topic_key="deposit",
        question_en="How much is the deposit?",
        question_hu="Mennyi a foglaló?",
        answer_en="",  # No EN answer
        answer_hu="20% foglaló.",
        category="Booking & Reservations",
        member_ids=(0, 1),
    )
    faq_rows = _extraction_rows_to_faq_rows([topic])
    assert len(faq_rows) == 2
    # EN row should have HU answer as fallback
    assert faq_rows[0] == {"Category": "Booking & Reservations", "Question": "How much is the deposit?", "Answer": "20% foglaló."}
    # HU row unchanged
    assert faq_rows[1] == {"Category": "Booking & Reservations", "Question": "Mennyi a foglaló?", "Answer": "20% foglaló."}


def test_extraction_rows_to_faq_rows_skips_empty_question_or_answer():
    """Topics with empty question or answer are not written."""
    topic1 = CanonicalTopic(
        topic_key="empty-q",
        question_en="",
        question_hu=None,
        answer_en="Some answer.",
        answer_hu=None,
        category="Cat",
        member_ids=(0,),
    )
    topic2 = CanonicalTopic(
        topic_key="empty-a",
        question_en="Valid question?",
        question_hu=None,
        answer_en="",
        answer_hu=None,
        category="Cat",
        member_ids=(1,),
    )
    faq_rows = _extraction_rows_to_faq_rows([topic1, topic2])
    assert len(faq_rows) == 0


# --- semantic_topics: gate → cluster → canonicalize ---

def test_semantic_topics_paraphrase_fold_two_companies_en_hu_pair():
    """Two paraphrased questions from two companies → one topic, EN+HU pair, shared Category."""
    rows = [
        _row("C1", "How much is the deposit?", question_hu="Mekkora a foglaló?", answer_en="20% deposit.", answer_hu="20% foglaló.", section="Booking & Reservations"),
        _row("C2", "Do I need to pay a deposit and when?", question_hu="Kell foglalót fizetni?", answer_en="Yes, 30% deposit.", answer_hu="Igen, 30% foglaló.", section="Booking & Reservations"),
    ]
    clusterer = _mock_clusterer_for_rows(rows, {"0": "deposit", "1": "deposit"})
    topics = semantic_topics(rows, clusterer=clusterer)
    assert len(topics) == 1
    topic = topics[0]
    assert topic.question_en == "How much is the deposit?"
    assert topic.question_hu == "Mekkora a foglaló?"
    assert "figures vary across companies" in topic.answer_en
    assert "20% to 30%" in topic.answer_en
    assert topic.answer_hu == "20% foglaló."
    assert topic.category == "Booking & Reservations"
    assert len(topic.category) <= 40
    assert topic.topic_key == topic_key("How much is the deposit?")


def test_semantic_topics_en_only_topic_one_row():
    """EN-only topic (no HU) produces one CanonicalTopic with no HU fields."""
    rows = [
        _row("C1", "Is a deposit required?", question_hu=None, answer_en="Yes, 20%.", answer_hu=None, section="Booking"),
        _row("C2", "Do I have to pay a deposit?", question_hu=None, answer_en="Yes.", answer_hu=None, section="Booking"),
    ]
    clusterer = _mock_clusterer_for_rows(rows, {"0": "deposit", "1": "deposit"})
    topics = semantic_topics(rows, clusterer=clusterer)
    assert len(topics) == 1
    topic = topics[0]
    assert topic.question_en == "Is a deposit required?"
    assert topic.question_hu is None
    assert topic.answer_hu is None
    assert topic.category == "Booking"


def test_semantic_topics_clause_dump_excluded_from_rows_and_returned_as_discarded():
    """Clause-dump rows in new input are excluded from topics and returned as discarded."""
    dump_answer = "A" * 100 + ". " + "B" * 100 + "."  # >200 chars, multi-sentence
    rows = [
        _row("C1", "How much is the deposit?", answer_en="20% deposit."),
        _row("C2", "What is the full legal text?", answer_en=dump_answer),  # clause dump
    ]
    topics = semantic_topics(rows, clusterer=ExactClusterer())
    assert len(topics) == 1
    assert topics[0].question_en == "How much is the deposit?"

    # Verify discarded via semantic_merge
    written, discarded = semantic_merge(rows, clusterer=ExactClusterer(), dry_run=True)
    assert len(written) == 1
    assert len(discarded) == 1
    assert discarded[0]["company"] == "C2"
    assert is_clause_dump(discarded[0]["answer_en"], threshold=200)


def test_semantic_topics_clause_dump_excluded_from_existing_file():
    """Clause-dump rows already in faq.csv are cleaned out by the rewrite."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        # Normal row
        writer.writerow({"Category": "Booking", "Question": "How much is the deposit?", "Answer": "20% deposit."})
        # Clause dump row (long multi-sentence)
        dump_answer = "A" * 100 + ". " + "B" * 100 + "."
        writer.writerow({"Category": "Legal", "Question": "Full terms?", "Answer": dump_answer})
        faq_path = pathlib.Path(f.name)

    try:
        # Run semantic_merge with no new rows - should clean out the dump
        written, discarded = semantic_merge([], faq_path=faq_path, clusterer=ExactClusterer(), dry_run=True)
        assert len(written) == 1
        assert written[0]["Question"] == "How much is the deposit?"
        assert len(discarded) == 1
        # Discarded rows are extraction-schema, key is "answer_en" not "Answer"
        assert is_clause_dump(discarded[0]["answer_en"], threshold=200)
    finally:
        faq_path.unlink()


def test_semantic_topics_messy_category_cleaning():
    """Category is cleaned via canonicalizer._clean_category (≤40 chars, prefix trimmed)."""
    rows = [
        _row("C1", "Question 1", section="All ProductsBouncy Castle Hire..."),
        _row("C2", "Question 2", section="All ProductsBouncy Castle Hire..."),
    ]
    clusterer = _mock_clusterer_for_rows(rows, {"0": "topic", "1": "topic"})
    topics = semantic_topics(rows, clusterer=clusterer)
    assert len(topics) == 1
    assert topics[0].category == "Bouncy Castle Hire"
    assert len(topics[0].category) <= 40


def test_semantic_topics_deterministic_ordering_by_min_member_index():
    """Topics ordered by ascending minimum member input index."""
    rows = [
        _row("C1", "Question A", section="Cat A"),  # index 0
        _row("C2", "Question B", section="Cat B"),  # index 1
        _row("C3", "Question C", section="Cat C"),  # index 2
    ]
    topics = semantic_topics(rows, clusterer=ExactClusterer())
    assert len(topics) == 3
    assert topics[0].question_en == "Question A"
    assert topics[1].question_en == "Question B"
    assert topics[2].question_en == "Question C"
    assert min(topics[0].member_ids) == 0
    assert min(topics[1].member_ids) == 1
    assert min(topics[2].member_ids) == 2


def test_semantic_topics_existing_faq_rows_come_first_in_input_order():
    """Existing faq.csv rows precede new rows in input order (for id stability)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow({"Category": "Existing", "Question": "Existing question?", "Answer": "Existing answer."})
        faq_path = pathlib.Path(f.name)

    try:
        new_rows = [_row("C1", "New question?", answer_en="New answer.", section="New")]
        topics = semantic_topics(new_rows, faq_path=faq_path, clusterer=ExactClusterer())
        # Existing row should be first topic (member index 0)
        assert topics[0].question_en == "Existing question?"
        assert min(topics[0].member_ids) == 0
        # New row should be second (member index 1)
        assert topics[1].question_en == "New question?"
        assert min(topics[1].member_ids) == 1
    finally:
        faq_path.unlink()


# --- semantic_merge: write / dry_run / idempotency ---

def test_semantic_merge_dry_run_leaves_file_untouched():
    """dry_run=True returns what would be written without modifying the file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow({"Category": "Original", "Question": "Original?", "Answer": "Original answer."})
        faq_path = pathlib.Path(f.name)

    original_content = faq_path.read_text(encoding="utf-8")

    try:
        rows = [_row("C1", "New question?", answer_en="New answer.", section="New")]
        written, discarded = semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=True)

        # File unchanged
        assert faq_path.read_text(encoding="utf-8") == original_content
        # But written rows returned
        assert len(written) == 2  # existing + new
    finally:
        faq_path.unlink()


def test_semantic_merge_rewrites_file_in_place():
    """File is rewritten in place, not appended."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow({"Category": "Old", "Question": "Old question?", "Answer": "Old answer."})
        faq_path = pathlib.Path(f.name)

    try:
        rows = [_row("C1", "New question?", answer_en="New answer.", section="New")]
        written, _ = semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)

        # File rewritten with only merged output (whole-file rewrite)
        content = faq_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert lines[0] == "Category,Question,Answer"
        assert len(lines) == 3  # header + 2 data rows (existing + new, different topics)
        # Both old and new rows should be present since they are different topics
        assert "Old question?" in content
        assert "New question?" in content
    finally:
        faq_path.unlink()


def test_semantic_merge_output_header_exactly_required_columns():
    """Rewritten file has exactly Category,Question,Answer header."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow({"Category": "Cat", "Question": "Q?", "Answer": "A."})
        faq_path = pathlib.Path(f.name)

    try:
        semantic_merge([], faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        with open(faq_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == REQUIRED_COLUMNS
    finally:
        faq_path.unlink()


def test_semantic_merge_re_run_byte_identical():
    """Re-run proof: semantic_merge(rows) writes F1; semantic_merge(rows, faq_path=F1) writes F2; F1 == F2."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        faq_path = pathlib.Path(f.name)

    try:
        rows = [
            _row("C1", "How much is the deposit?", question_hu="Mennyi a foglaló?", answer_en="20% deposit.", answer_hu="20% foglaló.", section="Booking"),
            _row("C2", "Do I need to pay a deposit?", question_hu="Kell foglaló?", answer_en="30% deposit.", answer_hu="30% foglaló.", section="Booking"),
        ]

        # First run
        written1, _ = semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        content1 = faq_path.read_bytes()

        # Second run (using F1 as input)
        written2, _ = semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        content2 = faq_path.read_bytes()

        assert content1 == content2
        assert len(written1) == len(written2)
        # Row count unchanged
        assert len(written1) == 4  # 2 topics × 2 rows (EN+HU) = 4 rows
    finally:
        faq_path.unlink()


def test_semantic_merge_same_faq_ids_across_runs():
    """faq_{i} ids (positional) resolve to same Question/Answer in both runs."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        faq_path = pathlib.Path(f.name)

    try:
        rows = [
            _row("C1", "Question A?", answer_en="Answer A.", section="Cat"),
            _row("C2", "Question B?", answer_en="Answer B.", section="Cat"),
        ]

        semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        faqs1 = _load_faq_rows(faq_path)

        semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        faqs2 = _load_faq_rows(faq_path)

        assert len(faqs1) == len(faqs2)
        for i, (r1, r2) in enumerate(zip(faqs1, faqs2)):
            assert r1["Question"] == r2["Question"], f"Row {i} (faq_{i}) Question differs"
            assert r1["Answer"] == r2["Answer"], f"Row {i} (faq_{i}) Answer differs"
            assert r1["Category"] == r2["Category"], f"Row {i} (faq_{i}) Category differs"
    finally:
        faq_path.unlink()


def test_semantic_merge_existing_canonical_wins_no_double_fold():
    """When topic_key exists in faq_path, stored row is emitted verbatim (no re-folding).

    Uses mocked LLMClusterer so the third paraphrase genuinely clusters with
    the existing topic, exercising the absorption path. The stored answer must
    not be double-folded, and re-run must be byte-identical.
    """
    def fake_ask_llm(system_prompt, user_message, groq_model=None, openai_model=None):
        # All four rows (existing EN + existing HU + 2 new) assigned to same topic
        return _llm_reply({"0": "deposit", "1": "deposit", "2": "deposit", "3": "deposit"})

    clusterer = LLMClusterer(ask_llm=fake_ask_llm)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        # Pre-existing canonical EN+HU pair (already folded answer)
        writer.writerow({"Category": "Booking", "Question": "How much is the deposit?", "Answer": "20% deposit (figures vary across companies: 20% to 30%)"})
        writer.writerow({"Category": "Booking", "Question": "Mennyi a foglaló?", "Answer": "20% foglaló."})
        faq_path = pathlib.Path(f.name)

    try:
        # Add two new paraphrases of the same topic (both will cluster with existing)
        new_rows = [
            _row("C2", "Do I need to pay a deposit?", answer_en="30% deposit.", section="Booking"),
            _row("C3", "What is the deposit amount?", answer_en="25% deposit.", section="Booking"),
        ]
        written1, _ = semantic_merge(new_rows, faq_path=faq_path, clusterer=clusterer, dry_run=False)
        content1 = faq_path.read_text(encoding="utf-8")

        # Stored answer should be preserved verbatim (no double-fold)
        assert "figures vary across companies: 20% to 30%" in content1
        assert content1.count("figures vary across companies") == 1
        # Should not contain the new figures (25%) in the folded answer
        # (30% is already in the stored answer as "20% to 30%", so only check 25%)
        assert "25%" not in content1

        # Re-run: must be byte-identical (F1 == F2)
        written2, _ = semantic_merge(new_rows, faq_path=faq_path, clusterer=clusterer, dry_run=False)
        content2 = faq_path.read_text(encoding="utf-8")
        assert content1 == content2, "Re-run not byte-identical (double-fold occurred)"
        assert len(written1) == len(written2) == 2  # 1 bilingual topic = 2 rows (EN + HU)
    finally:
        faq_path.unlink()


def test_semantic_merge_empty_rows_reproduces_file_byte_identically():
    """semantic_merge([], faq_path=<existing merged file>) reproduces the file byte-identically."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow({"Category": "Cat1", "Question": "Q1?", "Answer": "A1."})
        writer.writerow({"Category": "Cat2", "Question": "Q2?", "Answer": "A2."})
        faq_path = pathlib.Path(f.name)

    original_content = faq_path.read_bytes()

    try:
        written, discarded = semantic_merge([], faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        new_content = faq_path.read_bytes()
        assert new_content == original_content
        assert len(written) == 2
        assert len(discarded) == 0
    finally:
        faq_path.unlink()


def test_semantic_merge_empty_rows_no_file_creates_header_only():
    """semantic_merge([]) with no file creates a header-only Category,Question,Answer file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        faq_path = pathlib.Path(f.name)
    faq_path.unlink()  # Remove so file doesn't exist

    try:
        written, discarded = semantic_merge([], faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        assert faq_path.exists()
        content = faq_path.read_text(encoding="utf-8")
        assert content == "Category,Question,Answer\n"
        assert len(written) == 0
        assert len(discarded) == 0
    finally:
        if faq_path.exists():
            faq_path.unlink()


def test_semantic_merge_all_dumps_creates_header_only():
    """If every input row is a clause dump, output is header-only and all rows returned as discarded."""
    dump_answer = "A" * 100 + ". " + "B" * 100 + "."
    rows = [
        _row("C1", "Dump 1?", answer_en=dump_answer),
        _row("C2", "Dump 2?", answer_en=dump_answer),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        faq_path = pathlib.Path(f.name)
    faq_path.unlink()

    try:
        written, discarded = semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        content = faq_path.read_text(encoding="utf-8")
        assert content == "Category,Question,Answer\n"
        assert len(written) == 0
        assert len(discarded) == 2
    finally:
        if faq_path.exists():
            faq_path.unlink()


def test_semantic_merge_rows_with_no_question_dropped():
    """Rows with no question text at all (neither EN nor HU) are dropped."""
    rows = [
        _row("C1", "Valid question?", answer_en="Valid answer."),
        {"company": "C2", "answer_en": "Answer but no question"},  # No question_en or question_hu
    ]
    topics = semantic_topics(rows, clusterer=ExactClusterer())
    assert len(topics) == 1
    assert topics[0].question_en == "Valid question?"


def test_semantic_merge_creates_parent_directory():
    """faq_path's parent directory is created if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        faq_path = pathlib.Path(tmpdir) / "subdir" / "faq.csv"
        rows = [_row("C1", "Question?", answer_en="Answer.", section="Cat")]
        written, _ = semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        assert faq_path.exists()
        content = faq_path.read_text(encoding="utf-8")
        assert "Question?" in content


def test_semantic_merge_utf8_trailing_newline():
    """UTF-8 + trailing newline conventions followed."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        faq_path = pathlib.Path(f.name)
    faq_path.unlink()

    try:
        rows = [_row("C1", "Question?", answer_en="Answer.", section="Cat")]
        semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        content = faq_path.read_bytes()
        assert content.endswith(b"\n")
        # Valid UTF-8
        content.decode("utf-8")
    finally:
        if faq_path.exists():
            faq_path.unlink()


# --- Provenance for #44 ---

def test_semantic_topics_returns_canonical_topic_with_topic_key_and_member_ids():
    """semantic_topics returns CanonicalTopic objects with topic_key and member_ids for provenance."""
    rows = [
        _row("C1", "How much is the deposit?", clause_ref="C1/deposit#1", url="https://c1.hu/aszf", section="Booking"),
        _row("C2", "Do I need to pay a deposit?", clause_ref="C2/deposit#2", url="https://c2.hu/aszf", section="Booking"),
    ]
    clusterer = _mock_clusterer_for_rows(rows, {"0": "deposit", "1": "deposit"})
    topics = semantic_topics(rows, clusterer=clusterer)
    assert len(topics) == 1
    topic = topics[0]
    assert hasattr(topic, "topic_key")
    assert hasattr(topic, "member_ids")
    assert topic.topic_key == topic_key("How much is the deposit?")
    assert topic.member_ids == (0, 1)


# --- EN/HU pairing round-trip (full pipeline) ---

def test_semantic_merge_en_hu_pairing_round_trip():
    """Bilingual topic stored as EN row + HU row round-trips to topic with both question_en and question_hu."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        # EN row
        writer.writerow({"Category": "Booking", "Question": "How much is the deposit?", "Answer": "20% deposit."})
        # HU row (same category, consecutive)
        writer.writerow({"Category": "Booking", "Question": "Mennyi a foglaló?", "Answer": "20% foglaló."})
        faq_path = pathlib.Path(f.name)

    try:
        # Run semantic_merge with no new rows - should preserve the pair
        written, _ = semantic_merge([], faq_path=faq_path, clusterer=ExactClusterer(), dry_run=True)
        assert len(written) == 2
        assert written[0]["Question"] == "How much is the deposit?"
        assert written[0]["Answer"] == "20% deposit."
        assert written[1]["Question"] == "Mennyi a foglaló?"
        assert written[1]["Answer"] == "20% foglaló."
        assert written[0]["Category"] == written[1]["Category"] == "Booking"
    finally:
        faq_path.unlink()


def test_semantic_topics_question_en_fallback_to_question_hu():
    """Rows with only question_hu (no EN) get question_en = question_hu."""
    rows = [
        {"company": "C1", "question_hu": "Mennyi a foglaló?", "answer_hu": "20% foglaló.", "section": "Booking"},
    ]
    topics = semantic_topics(rows, clusterer=ExactClusterer())
    assert len(topics) == 1
    assert topics[0].question_en == "Mennyi a foglaló?"
    assert topics[0].question_hu == "Mennyi a foglaló?"


def test_semantic_merge_bilingual_no_en_answer_fallback():
    """Bilingual topic with no EN answer: EN row falls back to HU answer, HU row written normally."""
    def fake_ask_llm(system_prompt, user_message, groq_model=None, openai_model=None):
        return _llm_reply({"0": "deposit", "1": "deposit"})

    clusterer = LLMClusterer(ask_llm=fake_ask_llm)

    # Two bilingual rows with empty answer_en and non-empty answer_hu
    rows = [
        _row("C1", "How much is the deposit?", question_hu="Mennyi a foglaló?", answer_en="", answer_hu="20% foglaló.", section="Booking"),
        _row("C2", "Do I need to pay a deposit?", question_hu="Kell foglaló?", answer_en="", answer_hu="30% foglaló.", section="Booking"),
    ]
    written, _ = semantic_merge(rows, clusterer=clusterer, dry_run=True)

    # Should produce 2 rows: EN (with HU answer fallback) + HU
    assert len(written) == 2
    en_row = written[0]
    hu_row = written[1]
    assert en_row["Question"] == "How much is the deposit?"
    assert en_row["Answer"] == "20% foglaló."  # Fallback to first member's HU answer
    assert hu_row["Question"] == "Mennyi a foglaló?"
    assert hu_row["Answer"] == "20% foglaló."
    assert en_row["Category"] == hu_row["Category"] == "Booking"


# --- Additional edge cases ---

def test_semantic_merge_discarded_rows_are_extraction_schema():
    """Discarded rows returned as extraction-schema dicts (for #45 audit trail)."""
    dump_answer = "A" * 100 + ". " + "B" * 100 + "."
    rows = [
        _row("C1", "Valid?", answer_en="Valid."),
        _row("C2", "Dump?", answer_en=dump_answer),
    ]
    written, discarded = semantic_merge(rows, clusterer=ExactClusterer(), dry_run=True)
    assert len(discarded) == 1
    d = discarded[0]
    assert "company" in d
    assert "question_en" in d
    assert "answer_en" in d
    assert "clause_ref" in d
    assert "url" in d
    assert "section" in d


def test_semantic_merge_multiple_topics_deterministic_order():
    """Multiple distinct topics maintain deterministic order across runs."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        faq_path = pathlib.Path(f.name)

    try:
        rows = [
            _row("C1", "Question A?", answer_en="Answer A.", section="Cat A"),
            _row("C2", "Question B?", answer_en="Answer B.", section="Cat B"),
            _row("C3", "Question C?", answer_en="Answer C.", section="Cat C"),
        ]
        semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        content1 = faq_path.read_text(encoding="utf-8")

        semantic_merge(rows, faq_path=faq_path, clusterer=ExactClusterer(), dry_run=False)
        content2 = faq_path.read_text(encoding="utf-8")

        assert content1 == content2
    finally:
        faq_path.unlink()


import csv  # needed for some tests above