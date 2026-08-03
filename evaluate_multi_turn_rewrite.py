#!/usr/bin/env python
"""
Evaluate multi-turn query rewriting vs raw follow-up question retrieval.

This script evaluates both pipelines against a multi-turn ground-truth CSV
(default: data/ground_truth_multi_turn.csv, overridable with --ground-truth):
- Raw follow-up: search(follow_up_question, ...)
- Multi-turn rewritten: search(rewrite_query_with_history(follow_up_question, history), ...)

Reports hit rate@k and MRR@k for both approaches.

Usage:
    uv run python evaluate_multi_turn_rewrite.py
    uv run python evaluate_multi_turn_rewrite.py --ground-truth data/ground_truth_multi_turn_generated.csv
"""

import os
import sys
import csv
import pathlib

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluate import evaluate_retrieval
from src.rag import rewrite_query_with_history
from src.search import search
from src.ingest import DEFAULT_BM25_PATH, DEFAULT_FAISS_PATH, DEFAULT_DOCS_PATH
from src.config import load_tuned_params

DEFAULT_GROUND_TRUTH_PATH = (
    pathlib.Path(__file__).resolve().parents[0] / "data" / "ground_truth_multi_turn.csv"
)

REQUIRED_COLUMNS = ("prior_user_turns", "follow_up_question", "document_id")


def search_raw(query, k=5, **kwargs):
    """Search using the raw follow-up question."""
    return search(query, k=k, **kwargs)


def search_multi(query, k=5, **kwargs):
    """Search using the multi-turn rewritten query.

    History resolution is positional (index-based): ``evaluate_retrieval`` calls
    this function exactly once per ground-truth item, in order, so a per-run call
    index maps to the item at that position. Question-text matching is
    deliberately avoided because follow-up question texts can collide (e.g. the
    generated set's five "How much space do I need?" rows, each with a distinct
    history). ``search_multi.ground_truth`` and ``search_multi.cursor`` are set
    by ``main()`` before each ``evaluate_retrieval`` run.
    """
    index = search_multi.cursor
    search_multi.cursor += 1
    ground_truth = search_multi.ground_truth
    if 0 <= index < len(ground_truth):
        history = ground_truth[index]["history"]
    else:
        history = []

    rewritten = rewrite_query_with_history(query, history)
    return search(rewritten, k=k, **kwargs)


def load_multi_turn_ground_truth(path=None):
    """Load multi-turn ground truth from CSV.

    Tolerates malformed rows: rows missing a required column
    (``prior_user_turns``, ``follow_up_question``, ``document_id``), with a blank
    ``follow_up_question``, or with a blank ``document_id`` are skipped and
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


def main(ground_truth_path=None):
    # Check for API keys
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not groq_key and not openai_key:
        print("No API keys found (GROQ_API_KEY or OPENAI_API_KEY).")
        print("Skipping evaluation - multi-turn query rewriting requires an LLM.")
        print("Set GROQ_API_KEY or OPENAI_API_KEY to run the evaluation.")
        return 0

    gt_path = pathlib.Path(ground_truth_path) if ground_truth_path else DEFAULT_GROUND_TRUTH_PATH

    print("Loading ground truth...")
    try:
        loaded = load_multi_turn_ground_truth(ground_truth_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    ground_truth = loaded["items"]
    print(f"Loaded {len(ground_truth)} queries from {gt_path}")
    if loaded["skipped"]:
        print(
            f"Skipped {loaded['skipped']} malformed rows "
            f"(missing required columns or blank follow-up question)"
        )

    if not ground_truth:
        print(
            f"Error: no valid multi-turn rows in {gt_path} - expected columns "
            f"'prior_user_turns', 'follow_up_question', 'document_id' with a "
            f"non-blank follow_up_question",
            file=sys.stderr,
        )
        return 1

    # Attach ground truth to search_multi for positional history lookup
    search_multi.ground_truth = ground_truth

    # Get index paths from tuned params or use defaults
    params = load_tuned_params()

    search_kwargs = {
        "bm25_path": DEFAULT_BM25_PATH,
        "faiss_path": DEFAULT_FAISS_PATH,
        "docs_path": DEFAULT_DOCS_PATH,
    }

    k = params.get("k", 5)
    print(f"\nEvaluating with k={k}...")

    # Evaluate raw follow-up pipeline
    print("\n--- Raw Follow-up Pipeline ---")
    raw_report = evaluate_retrieval(
        ground_truth=ground_truth,
        k=k,
        search_fn=search_raw,
        **search_kwargs,
    )
    print(f"Hit Rate@{k}: {raw_report['hit_rate']:.4f}")
    print(f"MRR@{k}:      {raw_report['mrr']:.4f}")

    # Evaluate multi-turn rewritten pipeline
    print("\n--- Multi-turn Rewritten Pipeline ---")
    search_multi.cursor = 0
    multi_report = evaluate_retrieval(
        ground_truth=ground_truth,
        k=k,
        search_fn=search_multi,
        **search_kwargs,
    )
    print(f"Hit Rate@{k}: {multi_report['hit_rate']:.4f}")
    print(f"MRR@{k}:      {multi_report['mrr']:.4f}")

    # Side-by-side comparison
    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Raw Follow-up':>14} {'Multi-turn':>12} {'Delta':>10}")
    print("-" * 60)

    hr_delta = multi_report['hit_rate'] - raw_report['hit_rate']
    mrr_delta = multi_report['mrr'] - raw_report['mrr']

    print(f"{'Hit Rate@' + str(k):<20} {raw_report['hit_rate']:>14.4f} {multi_report['hit_rate']:>12.4f} {hr_delta:>+10.4f}")
    print(f"{'MRR@' + str(k):<20} {raw_report['mrr']:>14.4f} {multi_report['mrr']:>12.4f} {mrr_delta:>+10.4f}")

    print("-" * 60)

    # Determine winner
    if hr_delta > 0 and mrr_delta > 0:
        winner = "Multi-turn rewritten query (improves both metrics)"
        recommended_flag = True
    elif hr_delta < 0 and mrr_delta < 0:
        winner = "Raw follow-up question (rewriting degrades both metrics)"
        recommended_flag = False
    elif hr_delta > 0 and mrr_delta < 0:
        winner = "Mixed: Multi-turn improves hit rate but degrades MRR"
        recommended_flag = False
    elif hr_delta < 0 and mrr_delta > 0:
        winner = "Mixed: Multi-turn improves MRR but degrades hit rate"
        recommended_flag = False
    else:
        winner = "Tie: No difference in metrics"
        recommended_flag = False

    print(f"\nWinner: {winner}")
    print(f"Recommended history_rewrite_enabled: {recommended_flag}")

    # Show example rewrites
    print("\n" + "=" * 60)
    print("EXAMPLE REWRITES (first 10 queries)")
    print("=" * 60)
    for i, item in enumerate(ground_truth[:10]):
        prior_turns = [h["content"] for h in item["history"]]
        raw = item["question"]
        rewritten = rewrite_query_with_history(raw, item["history"])
        print(f"\n{i+1}. Prior turns: {prior_turns}")
        print(f"   Raw follow-up:  {raw}")
        print(f"   Multi-turn rewritten: {rewritten}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate multi-turn query rewriting vs raw follow-up retrieval."
    )
    parser.add_argument(
        "--ground-truth",
        dest="ground_truth_path",
        default=None,
        help="Path to a multi-turn ground-truth CSV "
             "(default: data/ground_truth_multi_turn.csv)",
    )
    args = parser.parse_args()
    sys.exit(main(ground_truth_path=args.ground_truth_path))