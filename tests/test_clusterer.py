import json

from src.clusterer import Clusterer, ExactClusterer, LLMClusterer, TopicCluster, default_clusterer
from src.merge_bilingual import normalize_accent


def _row(company, question_en, question_hu=None, answer_en="Answer.", answer_hu="Valasz.", clause_ref=None, url=None):
    return {
        "company": company,
        "question_en": question_en,
        "question_hu": question_hu or question_en,
        "answer_en": answer_en,
        "answer_hu": answer_hu,
        "clause_ref": clause_ref or f"{company}/1",
        "url": url or f"https://{company}.hu/aszf",
    }


def _llm_reply(assignments):
    return {
        "response": json.dumps({"assignments": assignments}),
        "model": "x", "provider": "mock", "latency": 0, "cost": 0,
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
    }


# --- protocol shape ---
def test_clusterer_protocol_declared():
    assert isinstance(ExactClusterer(), Clusterer)
    assert isinstance(LLMClusterer(), Clusterer)
    assert isinstance(default_clusterer(), LLMClusterer)


def test_topic_cluster_holds_id_and_member_indices():
    tc = TopicCluster(id="deposit", rows=[0, 2])
    assert tc.id == "deposit"
    assert tc.rows == [0, 2]


# --- ExactClusterer ---
def test_exact_clusterer_collapses_identical_questions():
    rows = [
        _row("C1", "How much is the deposit?", clause_ref="C1/deposit#1", url="https://c1.hu/aszf"),
        _row("C2", "How much is the deposit?", clause_ref="C2/deposit#2", url="https://c2.hu/aszf"),
        _row("C3", "What is the weather policy?", clause_ref="C3/weather#3"),
    ]
    clusters = ExactClusterer().cluster(rows)

    assert len(clusters) == 2
    deposit = [c for c in clusters if c.id == normalize_accent("How much is the deposit?")]
    assert len(deposit) == 1
    assert sorted(deposit[0].rows) == [0, 1]

    # member provenance survives
    members = [rows[i] for i in deposit[0].rows]
    assert {r["company"] for r in members} == {"C1", "C2"}
    assert {r["clause_ref"] for r in members} == {"C1/deposit#1", "C2/deposit#2"}
    assert {r["url"] for r in members} == {"https://c1.hu/aszf", "https://c2.hu/aszf"}


def test_exact_clusterer_is_accent_insensitive():
    rows = [
        _row("C1", "Foglalás feltételei?"),
        _row("C2", "Foglalas feltetelei?"),
        _row("C3", "Lemondás szabályai?"),
    ]
    clusters = ExactClusterer().cluster(rows)
    assert len(clusters) == 2
    fold = [c for c in clusters if c.id == "foglalas feltetelei?"]
    assert fold and sorted(fold[0].rows) == [0, 1]


def test_exact_clusterer_groups_by_english_question_not_hungarian():
    # Same EN meaning, different HU phrasing → one cluster (HU rides along).
    rows = [
        _row("C1", "How much is the deposit?", question_hu="Mekkora a foglaló?"),
        _row("C2", "How much is the deposit?", question_hu="Mennyi a deposit?"),
    ]
    clusters = ExactClusterer().cluster(rows)
    assert len(clusters) == 1
    assert sorted(clusters[0].rows) == [0, 1]
    # both Hungarian companions are carried by their members
    assert len({rows[i]["question_hu"] for i in clusters[0].rows}) == 2


def test_exact_clusterer_skips_rows_without_question():
    rows = [
        _row("C1", "How much is the deposit?"),
        {"company": "C2"},
    ]
    clusters = ExactClusterer().cluster(rows)
    assert len(clusters) == 1
    assert clusters[0].rows == [0]


# --- LLMClusterer (mocked ask_llm, no network) ---
def test_llm_clusterer_merges_paraphrased_questions_from_two_companies():
    rows = [
        _row("C1", "How much is the deposit and when is it due?", clause_ref="C1/deposit#1", url="https://c1.hu/aszf"),
        _row("C2", "Do I need to pay a deposit and when?", clause_ref="C2/deposit#2", url="https://c2.hu/aszf"),
        _row("C3", "Can I cancel after booking?", clause_ref="C3/cancel#3", url="https://c3.hu/aszf"),
    ]

    def fake_ask_llm(system_prompt, user_message, groq_model=None, openai_model=None):
        assert "0. How much is the deposit" in user_message
        assert "1. Do I need to pay a deposit" in user_message
        return _llm_reply({
            "0": "deposit and due date",
            "1": "deposit and due date",
            "2": "cancellation policy",
        })

    clusters = LLMClusterer(ask_llm=fake_ask_llm).cluster(rows)

    assert len(clusters) == 2
    deposit = [c for c in clusters if c.id == "deposit-and-due-date"]
    assert len(deposit) == 1
    assert sorted(deposit[0].rows) == [0, 1]

    members = [rows[i] for i in deposit[0].rows]
    assert {r["company"] for r in members} == {"C1", "C2"}
    assert {r["clause_ref"] for r in members} == {"C1/deposit#1", "C2/deposit#2"}
    assert {r["url"] for r in members} == {"https://c1.hu/aszf", "https://c2.hu/aszf"}


def test_llm_clusterer_accepts_code_fenced_reply():
    rows = [
        _row("C1", "How much is the deposit?"),
        _row("C2", "Do I need to pay a deposit and when?"),
    ]

    def fake_ask_llm(system_prompt, user_message, groq_model=None, openai_model=None):
        return {"response": "```json\n" + json.dumps({"assignments": {"0": "deposit", "1": "deposit"}}) + "\n```",
                "model": "x", "provider": "mock", "latency": 0, "cost": 0,
                "tokens": {"prompt": 0, "completion": 0, "total": 0}}

    clusters = LLMClusterer(ask_llm=fake_ask_llm).cluster(rows)
    assert len(clusters) == 1
    assert sorted(clusters[0].rows) == [0, 1]


def test_llm_clusterer_falls_back_to_exact_on_malformed_reply():
    rows = [
        _row("C1", "How much is the deposit?", clause_ref="C1/1"),
        _row("C2", "Do I need to pay a deposit and when?", clause_ref="C2/1"),
        _row("C3", "How much is the deposit?", clause_ref="C3/1"),
    ]

    def fake_ask_llm(system_prompt, user_message, groq_model=None, openai_model=None):
        return {"response": "not json at all", "model": "x", "provider": "mock",
                "latency": 0, "cost": 0, "tokens": {"prompt": 0, "completion": 0, "total": 0}}

    clusters = LLMClusterer(ask_llm=fake_ask_llm).cluster(rows)
    # identical EN questions still collapse; the paraphrase stays separate
    assert len(clusters) == 2
    exact = [c for c in clusters if c.id == normalize_accent("How much is the deposit?")]
    assert len(exact) == 1
    assert sorted(exact[0].rows) == [0, 2]


def test_llm_clusterer_falls_back_to_exact_on_llm_exception():
    def boom(system_prompt, user_message, groq_model=None, openai_model=None):
        raise RuntimeError("no network")

    rows = [
        _row("C1", "How much is the deposit?"),
        _row("C2", "How much is the deposit?"),
    ]
    clusters = LLMClusterer(ask_llm=boom).cluster(rows)
    assert len(clusters) == 1
    assert sorted(clusters[0].rows) == [0, 1]


def test_llm_clusterer_empty_input():
    assert LLMClusterer(ask_llm=lambda *a, **k: None).cluster([]) == []


def test_llm_clusterer_en_only_rows_cluster():
    # EN-only topics are allowed: rows without a HU companion still cluster by EN.
    rows = [
        {"company": "E1", "question_en": "Is a deposit required?", "answer_en": "Yes, 20%.",
         "clause_ref": "E1/deposit#1", "url": "https://e1.co.uk/terms"},
        {"company": "E2", "question_en": "Do I have to pay a deposit?", "answer_en": "Yes.",
         "clause_ref": "E2/deposit#2", "url": "https://e2.co.uk/terms"},
    ]

    def fake_ask_llm(system_prompt, user_message, groq_model=None, openai_model=None):
        return _llm_reply({"0": "deposit", "1": "deposit"})

    clusters = LLMClusterer(ask_llm=fake_ask_llm).cluster(rows)
    assert len(clusters) == 1
    assert sorted(clusters[0].rows) == [0, 1]
    members = [rows[i] for i in clusters[0].rows]
    assert {r["clause_ref"] for r in members} == {"E1/deposit#1", "E2/deposit#2"}
    assert {r["url"] for r in members} == {"https://e1.co.uk/terms", "https://e2.co.uk/terms"}
