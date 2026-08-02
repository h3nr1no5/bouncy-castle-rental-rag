import pytest
from src.canonicalizer import (
    CanonicalTopic,
    topic_key,
    canonicalize_cluster,
    _clean_category,
    _figure_spans,
    _figure_ranges,
    _fold_answers,
)
from src.clusterer import TopicCluster


def _row(company, question_en, question_hu=None, answer_en="Answer.", answer_hu=None, clause_ref=None, url=None, section=None):
    """Helper to create a test row."""
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


def test_topic_key_deterministic():
    """topic_key is deterministic — same input always yields same slug."""
    assert topic_key("How much is the deposit?") == "how-much-is-the-deposit"
    assert topic_key("How much is the deposit?") == topic_key("How much is the deposit?")
    assert topic_key("Café hire?") == "cafe-hire"
    assert topic_key("Café hire?") == topic_key("Cafe hire?")


def test_topic_key_examples():
    """Specific examples from the spec."""
    assert topic_key("How much is the deposit?") == "how-much-is-the-deposit"
    assert topic_key("Café hire?") == "cafe-hire"


def test_topic_key_diacritics():
    """Diacritics are stripped."""
    assert topic_key("Café hire?") == "cafe-hire"
    assert topic_key("Naïve approach") == "naive-approach"
    assert topic_key("Zoë's test") == "zoe-s-test"


def test_topic_key_length_bounded():
    """Output length is bounded (≤ 64 chars)."""
    long_question = "A" * 100
    result = topic_key(long_question)
    assert len(result) <= 64
    # Should be truncated to 64 chars (no word boundary to truncate at)
    assert len(result) == 64
    assert result == "a" * 64


def test_topic_key_word_boundary_truncation():
    """Word-boundary truncation works correctly (spaces already converted to hyphens)."""
    # Example from QA: long question that should truncate at word boundary
    long_question = "What is the maximum deposit amount that a customer is required to pay when booking a bouncy castle rental for a weekend party?"
    result = topic_key(long_question)
    # Should truncate at word boundary, not mid-word
    assert result == "what-is-the-maximum-deposit-amount-that-a-customer-is-required"
    assert len(result) <= 64
    # Should not end with hyphen
    assert not result.endswith("-")
    # Should be all lowercase with hyphens
    assert all(c.islower() or c == '-' for c in result)


def test_topic_key_empty_fallback():
    """Empty/whitespace-only input returns 'untitled'."""
    assert topic_key("") == "untitled"
    assert topic_key("   ") == "untitled"
    assert topic_key("\t\n") == "untitled"


def test_clean_category_examples():
    """Category cleaning examples from the spec."""
    assert _clean_category("All ProductsBouncy Castle Hire...") == "Bouncy Castle Hire"
    assert _clean_category("Terms & Conditions - Hire in Didcot...") == "Terms & Conditions"


def test_clean_category_length_bounded():
    """Cleaned category is bounded (≤ 40 chars)."""
    long_label = "A" * 100
    result = _clean_category(long_label)
    assert len(result) <= 40


def test_clean_category_fallback():
    """Empty cleaning yields 'General'."""
    assert _clean_category("") == "General"
    assert _clean_category("   ") == "General"


def test_figure_spans():
    """Extract figure spans from text."""
    assert _figure_spans("10% to 50%") == [(10, "%"), (50, "%")]
    assert _figure_spans("50Ft to 500Ft") == [(50, "Ft"), (500, "Ft")]
    assert _figure_spans("No figures here") == []
    assert _figure_spans("Price: 100€ and 200€") == [(100, "€"), (200, "€")]


def test_figure_ranges():
    """Calculate ranges for divergent figures."""
    answers = ["10% to 50%", "30% to 70%"]
    assert _figure_ranges(answers) == ["10% to 70%"]
    
    answers = ["50Ft to 500Ft", "100Ft to 200Ft"]
    assert _figure_ranges(answers) == ["50Ft to 500Ft"]
    
    answers = ["10%", "20%", "30%"]
    assert _figure_ranges(answers) == ["10% to 30%"]
    
    answers = ["No figures", "Just text"]
    assert _figure_ranges(answers) == []


def test_fold_answers_agreed_figures():
    """When all members' answers agree (no differing figures), keep base answer."""
    answers = ["The deposit is 20%."]
    assert _fold_answers(answers) == "The deposit is 20%."
    
    answers = ["Price: 100€.", "Price: 100€."]
    assert _fold_answers(answers) == "Price: 100€."


def test_fold_answers_divergent_figures():
    """When answers contain differing numeric figures, add variation note."""
    answers = ["Price: 100€.", "Price: 200€."]
    result = _fold_answers(answers)
    assert "figures vary across companies:" in result
    assert "100€ to 200€" in result
    
    answers = ["10% discount", "20% discount"]
    result = _fold_answers(answers)
    assert "figures vary across companies:" in result
    assert "10% to 20%" in result


def test_fold_answers_single_member():
    """Single-member cluster yields that member's answer with no note."""
    answers = ["Single answer."]
    assert _fold_answers(answers) == "Single answer."


def test_fold_answers_empty():
    """Empty answers list returns empty string."""
    assert _fold_answers([]) == ""


def test_canonicalize_cluster_single_member():
    """Single-member cluster yields that member's question and answer with no note."""
    rows = [_row("C1", "How much is the deposit?", answer_en="20% deposit.")]
    cluster = TopicCluster(id="test", rows=[0])
    result = canonicalize_cluster(cluster, rows)
    
    assert result.question_en == "How much is the deposit?"
    assert result.answer_en == "20% deposit."
    # When question_hu is not provided (None), result should be None (EN-only topic)
    assert result.question_hu is None
    assert result.answer_hu is None
    assert result.category == "Terms & Conditions"
    assert result.member_ids == (0,)


def test_canonicalize_cluster_empty_cluster():
    """Empty cluster (no members) raises ValueError."""
    rows = []
    cluster = TopicCluster(id="test", rows=[])
    with pytest.raises(ValueError, match="Empty cluster"):
        canonicalize_cluster(cluster, rows)


def test_canonicalize_cluster_hu_companion():
    """Hungarian companion rides along when present."""
    rows = [
        _row("C1", "How much is the deposit?", question_hu="Mekkora a foglaló?", answer_hu="20% foglaló."),
        _row("C2", "How much is the deposit?", question_hu="Mennyi a deposit?", answer_hu="30% deposit."),
    ]
    cluster = TopicCluster(id="test", rows=[0, 1])
    result = canonicalize_cluster(cluster, rows)
    
    assert result.question_en == "How much is the deposit?"
    assert result.question_hu == "Mekkora a foglaló?"
    assert result.answer_hu == "20% foglaló."


def test_canonicalize_cluster_en_only():
    """EN-only topics are permitted (no HU companion)."""
    rows = [
        _row("C1", "How much is the deposit?", question_hu=None, answer_hu=None),
        _row("C2", "How much is the deposit?", question_hu=None, answer_hu=None),
    ]
    cluster = TopicCluster(id="test", rows=[0, 1])
    result = canonicalize_cluster(cluster, rows)
    
    assert result.question_en == "How much is the deposit?"
    # When question_hu is None, it should be None in the result
    assert result.question_hu is None
    assert result.answer_hu is None


def test_canonicalize_cluster_category_cleaning():
    """Category is cleaned and bounded."""
    rows = [
        _row("C1", "Question 1", section="All ProductsBouncy Castle Hire..."),
        _row("C2", "Question 2", section="All ProductsBouncy Castle Hire..."),
    ]
    cluster = TopicCluster(id="test", rows=[0, 1])
    result = canonicalize_cluster(cluster, rows)
    
    assert result.category == "Bouncy Castle Hire"
    assert len(result.category) <= 40


def test_canonicalize_cluster_divergent_figures():
    """Divergent figures yield 'figures vary across companies: X% to Y%'."""
    rows = [
        _row("C1", "How much is the deposit?", answer_en="20% deposit."),
        _row("C2", "How much is the deposit?", answer_en="30% deposit."),
    ]
    cluster = TopicCluster(id="test", rows=[0, 1])
    result = canonicalize_cluster(cluster, rows)
    
    assert "figures vary across companies:" in result.answer_en
    assert "20% to 30%" in result.answer_en


def test_canonicalize_cluster_deterministic():
    """Calling canonicalize_cluster twice with same inputs returns identical output."""
    rows = [
        _row("C1", "How much is the deposit?", answer_en="20% deposit."),
        _row("C2", "How much is the deposit?", answer_en="30% deposit."),
    ]
    cluster = TopicCluster(id="test", rows=[0, 1])
    
    result1 = canonicalize_cluster(cluster, rows)
    result2 = canonicalize_cluster(cluster, rows)
    
    assert result1 == result2
    assert result1.topic_key == result2.topic_key
    assert result1.answer_en == result2.answer_en


def test_canonical_topic_dataclass():
    """CanonicalTopic dataclass has expected fields."""
    topic = CanonicalTopic(
        topic_key="test-key",
        question_en="Test question",
        question_hu="Teszt kérdés",
        answer_en="Test answer",
        answer_hu="Teszt válasz",
        category="Test Category",
        member_ids=(0, 1, 2),
    )
    
    assert topic.topic_key == "test-key"
    assert topic.question_en == "Test question"
    assert topic.question_hu == "Teszt kérdés"
    assert topic.answer_en == "Test answer"
    assert topic.answer_hu == "Teszt válasz"
    assert topic.category == "Test Category"
    assert topic.member_ids == (0, 1, 2)
