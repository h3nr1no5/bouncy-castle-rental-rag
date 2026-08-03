#!/usr/bin/env python
"""
Evaluate multi-turn answer quality for conversational memory.

This script measures whether threading the conversation history into the
answer-generation prompt improves follow-up answer quality. For every item in
data/ground_truth_multi_turn.csv it generates two answers to the follow-up
question:

- no-history arm: `answer_question(follow_up, history=None)`
- history arm:    `answer_question(follow_up, history=history)`

Both arms use **identical retrieval**: query rewriting is disabled in both arms
(`rewrite_enabled=False, history_rewrite_enabled=False`), so the raw follow-up
question is the search query in both cases and the only difference between the
arms is the conversation history in the answer prompt.

Each generated answer is scored by an LLM-as-a-judge (via `ask_llm()`) on
coherence and relevance (1-5). The same judge prompt and judge model are used
for both arms so the comparison is fair. The script reports per-arm mean
scores, n / n_valid, the delta between arms, and a verdict line.

Usage:
    uv run python evaluate_multi_turn_answer.py
"""

import csv
import os
import pathlib
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm.auto import tqdm

from src.config import load_tuned_params
from src.evaluate_llm import _parse_scores
from src.ingest import DEFAULT_BM25_PATH, DEFAULT_FAISS_PATH, DEFAULT_DOCS_PATH
from src.llm import ask_llm
from src.rag import answer_question

DEFAULT_GROUND_TRUTH_PATH = (
    pathlib.Path(__file__).resolve().parents[0] / "data" / "ground_truth_multi_turn.csv"
)

REQUIRED_COLUMNS = ("prior_user_turns", "follow_up_question", "document_id")

MULTI_TURN_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of a multi-turn RAG (Retrieval-Augmented Generation) assistant for a bouncy castle rental FAQ.
You will be given a conversation history, a follow-up question, the retrieved FAQ context, and the assistant's answer to the follow-up question.
Rate the answer on two axes from 1 to 5:

1. **Coherence** — Does the answer make sense in the context of the ongoing conversation?
   5 = Fully coherent; correctly uses the history to resolve pronouns, references, and follow-up intent
   4 = Mostly coherent, minor reference or context issues
   3 = Partially coherent; misses some conversational context
   2 = Poorly connected to the conversation; ignores relevant history
   1 = Incoherent or contradicts the conversation

2. **Relevance** — How well does the answer address the follow-up question?
   5 = Fully answers the follow-up question with accurate, directly useful information
   4 = Mostly answers the follow-up question, minor gaps
   3 = Partially answers but misses key points
   2 = Tangentially related but does not really answer the follow-up question
   1 = Completely irrelevant or unhelpful

Return your scores as JSON only, with no additional text:
{"coherence": <1-5>, "relevance": <1-5>, "explanation": "<brief reason>"}"""


def load_multi_turn_ground_truth(path=None):
    """
    Load multi-turn ground truth from CSV.

    Equivalent loader to ``evaluate_multi_turn_rewrite.load_multi_turn_ground_truth``
    (same default path, same per-item ``question`` / ``document_id`` / ``history``
    contract, ``prior_user_turns`` split on ``;`` into user messages) that also
    tolerates malformed rows. Rows with a blank ``follow_up_question``, a blank
    or missing ``document_id``, or a missing required column are skipped and
    counted in ``skipped``. Rows with an empty ``prior_user_turns`` load with an
    empty ``history`` and are counted in ``no_history``.

    Args:
        path: Path to the ground-truth CSV (defaults to
            ``<repo root>/data/ground_truth_multi_turn.csv``).

    Returns:
        A dict with keys:
        - ``items``: list of ``{"conversation_id", "question", "document_id",
          "history"}`` where ``history`` is a list of
          ``{"role": "user", "content": str}`` dicts.
        - ``skipped``: number of rows skipped due to blank/missing required fields.
        - ``no_history``: number of loaded rows with an empty ``prior_user_turns``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    if path is None:
        path = DEFAULT_GROUND_TRUTH_PATH
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found at {path}")

    items = []
    skipped = 0
    no_history = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(col not in row for col in REQUIRED_COLUMNS):
                skipped += 1
                continue

            follow_up = (row.get("follow_up_question") or "").strip()
            if not follow_up:
                skipped += 1
                continue

            document_id = (row.get("document_id") or "").strip()
            if not document_id:
                skipped += 1
                continue

            prior_turns = [
                t.strip()
                for t in (row.get("prior_user_turns") or "").split(";")
                if t.strip()
            ]
            history = [{"role": "user", "content": t} for t in prior_turns]
            if not history:
                no_history += 1

            items.append({
                "conversation_id": (row.get("conversation_id") or "").strip(),
                "question": follow_up,
                "document_id": document_id,
                "history": history,
            })

    return {"items": items, "skipped": skipped, "no_history": no_history}


def _format_history_lines(history):
    """Format history entries as 'User: ...' / 'Assistant: ...' lines."""
    lines = []
    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content.strip()}")
    return lines


def _format_contexts(contexts):
    """Format retrieved FAQ contexts for the judge prompt."""
    if not contexts:
        return "No relevant FAQ entries found."
    return "\n\n".join(
        f"[{ctx['category']}] {ctx['question']}\n{ctx['answer']}"
        for ctx in contexts
    )


def build_judge_user_message(question, history, contexts, answer):
    """
    Build the judge user message for one generated answer.

    Includes the conversation history, the follow-up question, the retrieved
    FAQ context and the generated answer. The history is included for both arms
    so the judge can assess coherence of an answer that was generated without
    history as well.
    """
    history_lines = _format_history_lines(history)
    history_text = "\n".join(history_lines) if history_lines else "(no prior conversation)"
    context_text = _format_contexts(contexts)
    return (
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        f"Retrieved FAQ context:\n{context_text}\n\n"
        f"Generated answer: {answer}"
    )


def _is_valid_score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 1 <= value <= 5


def _run_arm(question, history, arm_kwargs, judge_system_prompt):
    """
    Run a single arm: generate an answer via ``answer_question`` and score it
    with the LLM-as-a-judge. Any RAG or judge failure is recorded in the
    returned dict with ``error=True`` and does not raise.
    """
    try:
        rag_result = answer_question(question, history=history, **arm_kwargs)
    except Exception as e:
        return {
            "error": True,
            "error_message": f"RAG pipeline failed: {e}",
            "answer": None,
            "contexts": None,
            "coherence": None,
            "relevance": None,
            "explanation": None,
            "rag_model": None,
            "rag_provider": None,
            "rag_latency": None,
            "rag_tokens": None,
            "rag_cost": None,
            "judge_model": None,
            "judge_provider": None,
            "judge_latency": None,
            "judge_tokens": None,
            "judge_cost": None,
        }

    try:
        judge_result = ask_llm(
            system_prompt=judge_system_prompt,
            user_message=build_judge_user_message(
                question, history, rag_result["contexts"], rag_result["answer"]
            ),
            groq_model=arm_kwargs.get("groq_model"),
            openai_model=arm_kwargs.get("openai_model"),
        )
        parsed = _parse_scores(judge_result["response"])
        coherence = parsed.get("coherence")
        relevance = parsed.get("relevance")
        if not (_is_valid_score(coherence) and _is_valid_score(relevance)):
            return {
                "error": True,
                "error_message": "Judge returned unparseable scores",
                "answer": rag_result["answer"],
                "contexts": rag_result["contexts"],
                "coherence": None,
                "relevance": None,
                "explanation": parsed.get("explanation", ""),
                "rag_model": rag_result["model"],
                "rag_provider": rag_result["provider"],
                "rag_latency": rag_result["latency"],
                "rag_tokens": rag_result["tokens"],
                "rag_cost": rag_result["cost"],
                "judge_model": None,
                "judge_provider": None,
                "judge_latency": None,
                "judge_tokens": None,
                "judge_cost": None,
            }
        return {
            "error": False,
            "error_message": None,
            "answer": rag_result["answer"],
            "contexts": rag_result["contexts"],
            "coherence": coherence,
            "relevance": relevance,
            "explanation": parsed.get("explanation", ""),
            "rag_model": rag_result["model"],
            "rag_provider": rag_result["provider"],
            "rag_latency": rag_result["latency"],
            "rag_tokens": rag_result["tokens"],
            "rag_cost": rag_result["cost"],
            "judge_model": judge_result["model"],
            "judge_provider": judge_result["provider"],
            "judge_latency": judge_result["latency"],
            "judge_tokens": judge_result["tokens"],
            "judge_cost": judge_result["cost"],
        }
    except Exception as e:
        return {
            "error": True,
            "error_message": f"Judge LLM call failed: {e}",
            "answer": rag_result["answer"],
            "contexts": rag_result["contexts"],
            "coherence": None,
            "relevance": None,
            "explanation": None,
            "rag_model": rag_result["model"],
            "rag_provider": rag_result["provider"],
            "rag_latency": rag_result["latency"],
            "rag_tokens": rag_result["tokens"],
            "rag_cost": rag_result["cost"],
            "judge_model": None,
            "judge_provider": None,
            "judge_latency": None,
            "judge_tokens": None,
            "judge_cost": None,
        }


def run_evaluation(
    ground_truth,
    k=None,
    bm25_path=None,
    faiss_path=None,
    docs_path=None,
    groq_model=None,
    openai_model=None,
    judge_system_prompt=None,
):
    """
    Run both arms for every ground-truth item and score the answers.

    Both arms use the raw follow-up question for retrieval (query rewriting is
    forced off via ``rewrite_enabled=False, history_rewrite_enabled=False``), so
    retrieved contexts are identical and the only difference is the conversation
    history in the answer prompt. The same judge prompt and judge model are used
    for both arms.

    Per-item RAG/judge failures are recorded as errors and excluded from the
    aggregates; the returned report always has a well-formed structure.

    Args:
        ground_truth: list of items with ``question``, ``document_id`` and
            ``history`` keys (as returned by ``load_multi_turn_ground_truth``).

    Returns:
        A report dict with ``details``, per-arm aggregates (``without_history``,
        ``with_history``), deltas and a verdict line.
    """
    if judge_system_prompt is None:
        judge_system_prompt = MULTI_TURN_JUDGE_SYSTEM_PROMPT

    arm_kwargs = {
        "k": k,
        "bm25_path": bm25_path,
        "faiss_path": faiss_path,
        "docs_path": docs_path,
        "groq_model": groq_model,
        "openai_model": openai_model,
        # Force identical retrieval in both arms: the raw follow-up question is
        # the search query, regardless of tuned_params.json.
        "rewrite_enabled": False,
        "history_rewrite_enabled": False,
    }

    details = []
    for item in tqdm(ground_truth, desc="Evaluating multi-turn answers"):
        question = item["question"]
        history = item["history"]
        details.append({
            "conversation_id": item.get("conversation_id"),
            "question": question,
            "document_id": item.get("document_id"),
            "history": history,
            "has_history": bool(history),
            "without_history": _run_arm(question, None, arm_kwargs, judge_system_prompt),
            "with_history": _run_arm(question, history, arm_kwargs, judge_system_prompt),
        })

    without_history = compute_aggregates(details, "without_history")
    with_history = compute_aggregates(details, "with_history")
    delta_coherence, delta_relevance = compute_delta(without_history, with_history)

    return {
        "n": len(details),
        "without_history": without_history,
        "with_history": with_history,
        "delta_coherence": delta_coherence,
        "delta_relevance": delta_relevance,
        "verdict": compute_verdict(delta_coherence),
        "details": details,
    }


def compute_aggregates(details, arm_key):
    """
    Aggregate per-arm scores from the per-item details.

    Entries with ``error=True`` (RAG or judge failure) are excluded from the
    mean scores; ``n`` counts every item the arm was attempted on and ``n_valid``
    only the successfully scored ones.
    """
    entries = [d[arm_key] for d in details]
    n = len(entries)
    valid = [e for e in entries if not e["error"]]
    n_valid = len(valid)

    if n_valid:
        mean_coherence = round(sum(e["coherence"] for e in valid) / n_valid, 4)
        mean_relevance = round(sum(e["relevance"] for e in valid) / n_valid, 4)
    else:
        mean_coherence = None
        mean_relevance = None

    total_rag_cost = sum(
        (e.get("rag_cost") or 0.0) for e in entries
    )
    total_judge_cost = sum(
        (e.get("judge_cost") or 0.0) for e in entries
    )
    total_latency = sum(
        (e.get("rag_latency") or 0.0) + (e.get("judge_latency") or 0.0)
        for e in entries
    )
    total_tokens = sum(
        ((e.get("rag_tokens") or {}).get("total") or 0)
        + ((e.get("judge_tokens") or {}).get("total") or 0)
        for e in entries
    )

    return {
        "n": n,
        "n_valid": n_valid,
        "mean_coherence": mean_coherence,
        "mean_relevance": mean_relevance,
        "total_cost": round(total_rag_cost + total_judge_cost, 6),
        "total_rag_cost": round(total_rag_cost, 6),
        "total_judge_cost": round(total_judge_cost, 6),
        "total_latency": round(total_latency, 3),
        "total_tokens": total_tokens,
    }


def compute_delta(without_history, with_history):
    """
    Return ``(delta_coherence, delta_relevance)`` = with-history mean minus
    without-history mean, or ``(None, None)`` when either arm has no valid
    scores.
    """
    if not without_history["n_valid"] or not with_history["n_valid"]:
        return (None, None)
    return (
        round(with_history["mean_coherence"] - without_history["mean_coherence"], 4),
        round(with_history["mean_relevance"] - without_history["mean_relevance"], 4),
    )


def compute_verdict(delta):
    """
    Return a verdict line for the coherence delta: whether threading history
    into the answer prompt improves / degrades / leaves unchanged follow-up
    answer quality.
    """
    if delta is None:
        return "leaves follow-up answer quality unchanged (insufficient valid scores to determine)"
    if delta > 0:
        return "improves follow-up answer quality"
    if delta < 0:
        return "degrades follow-up answer quality"
    return "leaves follow-up answer quality unchanged"


def _fmt_score(value):
    return "-" if value is None else f"{value:.2f}"


def _fmt_cost(value):
    return "-" if value is None else f"${value:.6f}"


def print_report(report, k=None):
    """Print the side-by-side report and per-item details."""
    print("\n" + "=" * 70)
    print("MULTI-TURN ANSWER QUALITY EVALUATION (conversational memory)")
    print("=" * 70)
    if k is not None:
        print(f"Retrieval k={k} (identical for both arms, query rewriting disabled)")

    without = report["without_history"]
    with_ = report["with_history"]

    print("\nPer-arm scores (1-5, higher is better):")
    print(f"{'Metric':<22} {'Without history':>16} {'With history':>16} {'Delta':>10}")
    print("-" * 70)
    print(
        f"{'Coherence (mean)':<22} {_fmt_score(without['mean_coherence']):>16} "
        f"{_fmt_score(with_['mean_coherence']):>16} {report['delta_coherence']:>+10.4f}"
        if report["delta_coherence"] is not None
        else
        f"{'Coherence (mean)':<22} {_fmt_score(without['mean_coherence']):>16} "
        f"{_fmt_score(with_['mean_coherence']):>16} {'n/a':>10}"
    )
    if report["delta_relevance"] is not None:
        print(
            f"{'Relevance (mean)':<22} {_fmt_score(without['mean_relevance']):>16} "
            f"{_fmt_score(with_['mean_relevance']):>16} {report['delta_relevance']:>+10.4f}"
        )
    else:
        print(
            f"{'Relevance (mean)':<22} {_fmt_score(without['mean_relevance']):>16} "
            f"{_fmt_score(with_['mean_relevance']):>16} {'n/a':>10}"
        )

    print(f"\n{'':<22} {'Without history':>16} {'With history':>16}")
    print("-" * 70)
    print(f"{'n':<22} {without['n']:>16} {with_['n']:>16}")
    print(f"{'n_valid':<22} {without['n_valid']:>16} {with_['n_valid']:>16}")
    print(f"{'Total cost':<22} {_fmt_cost(without['total_cost']):>16} {_fmt_cost(with_['total_cost']):>16}")
    print(f"{'Total latency (s)':<22} {without['total_latency']:>16.3f} {with_['total_latency']:>16.3f}")
    print(f"{'Total tokens':<22} {without['total_tokens']:>16} {with_['total_tokens']:>16}")

    print("\nVERDICT: threading history into the answer prompt "
          f"{report['verdict']}")

    print("\n" + "=" * 70)
    print("PER-ITEM DETAILS")
    print("=" * 70)
    for i, detail in enumerate(report["details"], start=1):
        question = detail["question"]
        history_lines = _format_history_lines(detail["history"])
        history_text = ", ".join(history_lines) if history_lines else "(no history)"
        print(f"\n[{i}] conversation_id={detail.get('conversation_id')} "
              f"has_history={detail['has_history']}")
        print(f"    Follow-up: {question}")
        print(f"    History: {history_text}")
        for arm_name, arm_key in (("without history", "without_history"),
                                  ("with history", "with_history")):
            arm = detail[arm_key]
            if arm["error"]:
                print(f"    {arm_name:>16}: ERROR - {arm['error_message']}")
                continue
            print(
                f"    {arm_name:>16}: coherence={arm['coherence']} "
                f"relevance={arm['relevance']} ({arm['judge_provider']}/{arm['judge_model']})"
            )
            print(f"    {'answer':>16}: {arm['answer']}")


def main(ground_truth_path=None):
    """Run the multi-turn answer quality evaluation."""
    # Check for API keys
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not groq_key and not openai_key:
        print("No API keys found (GROQ_API_KEY or OPENAI_API_KEY).")
        print("Skipping evaluation - multi-turn answer quality evaluation requires an LLM.")
        print("Set GROQ_API_KEY or OPENAI_API_KEY to run the evaluation.")
        return 0

    print("Loading ground truth...")
    try:
        loaded = load_multi_turn_ground_truth(ground_truth_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    items = loaded["items"]
    print(
        f"Loaded {len(items)} follow-up questions "
        f"(skipped {loaded['skipped']} rows, {loaded['no_history']} with no history)"
    )

    params = load_tuned_params()
    k = params.get("k", 5)

    report = run_evaluation(items, k=k, bm25_path=DEFAULT_BM25_PATH,
                            faiss_path=DEFAULT_FAISS_PATH,
                            docs_path=DEFAULT_DOCS_PATH)

    print_report(report, k=k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
