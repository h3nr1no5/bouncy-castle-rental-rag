#!/usr/bin/env python
"""
Evaluate multi-turn query rewriting vs raw follow-up question retrieval.

This script evaluates both pipelines against a multi-turn ground-truth CSV
(default: data/ground_truth_multi_turn_generated.csv, overridable with --ground-truth):
- Raw follow-up: search(follow_up_question, ...)
- Multi-turn rewritten: search(rewrite_query_with_history(follow_up_question, history), ...)

Reports hit rate@k, MRR@k, Recall@k, Precision@k and nDCG@k for both
approaches, plus per-FAQ and per-category breakdowns for single-k runs.
k can be overridden with --k, swept with --k-sweep, and the number of
evaluated rows capped with --limit.

Usage:
    uv run python evaluate_multi_turn_rewrite.py
    uv run python evaluate_multi_turn_rewrite.py --ground-truth data/ground_truth_multi_turn.csv
    uv run python evaluate_multi_turn_rewrite.py --k 10
    uv run python evaluate_multi_turn_rewrite.py --k-sweep 1,3,5,10
    uv run python evaluate_multi_turn_rewrite.py --limit 50
"""

import argparse
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
from src.faqs import load_faqs

DEFAULT_GROUND_TRUTH_PATH = (
    pathlib.Path(__file__).resolve().parents[0] / "data" / "ground_truth_multi_turn_generated.csv"
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
            ``<repo root>/data/ground_truth_multi_turn_generated.csv``).

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


def _positive_int(value):
    """argparse type: a positive integer (rejects 0, negatives, non-integers)."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"invalid positive int value: '{value}'")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer, got {ivalue}"
        )
    return ivalue


def _k_sweep_list(value):
    """argparse type: a comma-separated list of positive integers, e.g. "1,3,5,10"."""
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    if not parts:
        raise argparse.ArgumentTypeError(
            f"invalid k-sweep value: '{value}' (expected e.g. '1,3,5,10')"
        )
    ks = []
    for part in parts:
        try:
            k = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid k-sweep value: '{value}' (k '{part}' is not an integer)"
            )
        if k <= 0:
            raise argparse.ArgumentTypeError(
                f"invalid k-sweep value: '{value}' (k must be a positive integer)"
            )
        ks.append(k)
    return ks


def _print_arm_metrics(report, k):
    """Print the aggregate metric block for one arm."""
    print(f"Hit Rate@{k}: {report['hit_rate']:.4f}")
    print(f"MRR@{k}:      {report['mrr']:.4f}")
    print(f"Recall@{k}:   {report['recall']:.4f}")
    print(f"Precision@{k}: {report['precision']:.4f}")
    print(f"nDCG@{k}:     {report['ndcg']:.4f}")


def _print_side_by_side(k, raw_report, multi_report):
    """Print the side-by-side comparison table for one k."""
    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Raw Follow-up':>14} {'Multi-turn':>12} {'Delta':>10}")
    print("-" * 60)

    for label, key in (
        ("Hit Rate", "hit_rate"),
        ("MRR", "mrr"),
        ("Recall", "recall"),
        ("Precision", "precision"),
        ("nDCG", "ndcg"),
    ):
        raw = raw_report[key]
        multi = multi_report[key]
        delta = multi - raw
        print(
            f"{label + '@' + str(k):<20} {raw:>14.4f} {multi:>12.4f} {delta:>+10.4f}"
        )

    print("-" * 60)


def _load_faq_categories(path=None):
    """Map document ids (faq_N) to FAQ categories.

    Categories are derived from data/faq.csv by row index: ``faq_N`` maps to the
    Category of row N, matching the id assignment in ``src/ingest.py``
    (``{"id": f"faq_{i}", ...}`` for i, row in enumerate(faqs)).
    """
    faqs = load_faqs(path)
    return {f"faq_{i}": row["Category"] for i, row in enumerate(faqs)}


def _group_by_document_ids(details):
    """Group evaluation details by each detail's relevant document id(s).

    A detail whose relevant ids span multiple FAQs contributes to each group,
    so per-group counts may exceed the total row count (each row is counted
    once per distinct FAQ it answers).
    """
    groups = {}
    for detail in details:
        ids = detail["document_id"]
        if isinstance(ids, str):
            ids = [ids]
        for doc_id in ids:
            groups.setdefault(doc_id, []).append(detail)
    return groups


def _print_per_faq_table(report, arm_label):
    """Print the per-FAQ breakdown for one arm's report."""
    groups = _group_by_document_ids(report["details"])
    print(f"\nPer-FAQ breakdown ({arm_label})")
    header = (
        f"{'FAQ':<20} {'count':>6} {'hit':>8} {'mrr':>8} "
        f"{'recall':>8} {'precision':>8} {'ndcg':>8}"
    )
    print(header)
    print("-" * len(header))
    for doc_id in sorted(groups):
        details = groups[doc_id]
        count = len(details)
        hit = sum(d["hit"] for d in details) / count
        mrr = sum(d["mrr"] for d in details) / count
        recall = sum(d["recall"] for d in details) / count
        precision = sum(d["precision"] for d in details) / count
        ndcg = sum(d["ndcg"] for d in details) / count
        print(
            f"{doc_id:<20} {count:>6} {hit:>8.4f} {mrr:>8.4f} "
            f"{recall:>8.4f} {precision:>8.4f} {ndcg:>8.4f}"
        )


def _print_per_category_table(report, arm_label, category_map):
    """Print the per-category breakdown for one arm's report.

    A row whose relevant ids span multiple categories contributes to each
    category group (mirroring _group_by_document_ids), so per-group counts may
    exceed the total row count. Document ids that cannot be mapped to a FAQ row
    are grouped under a visible "(unknown)" bucket instead of crashing.
    """
    groups = {}
    for detail in report["details"]:
        ids = detail["document_id"]
        if isinstance(ids, str):
            ids = [ids]
        for doc_id in ids:
            category = category_map.get(doc_id, "(unknown)")
            groups.setdefault(category, []).append(detail)

    print(f"\nPer-Category breakdown ({arm_label})")
    header = (
        f"{'Category':<30} {'count':>6} {'hit':>8} {'mrr':>8} "
        f"{'recall':>8} {'precision':>8} {'ndcg':>8}"
    )
    print(header)
    print("-" * len(header))
    for category in sorted(groups):
        details = groups[category]
        count = len(details)
        hit = sum(d["hit"] for d in details) / count
        mrr = sum(d["mrr"] for d in details) / count
        recall = sum(d["recall"] for d in details) / count
        precision = sum(d["precision"] for d in details) / count
        ndcg = sum(d["ndcg"] for d in details) / count
        print(
            f"{category:<30} {count:>6} {hit:>8.4f} {mrr:>8.4f} "
            f"{recall:>8.4f} {precision:>8.4f} {ndcg:>8.4f}"
        )


def main(ground_truth_path=None, k=None, k_sweep=None, limit=None):
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
    # --limit caps the number of evaluated rows. Slice BEFORE attaching the list
    # to search_multi.ground_truth so the multi-turn arm still resolves each
    # row's history positionally over exactly the evaluated subset.
    if limit is not None:
        ground_truth = ground_truth[:limit]
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

    # Resolve the k value(s): explicit --k, else --k-sweep, else tuned params.
    if k_sweep is not None:
        ks = k_sweep
    elif k is not None:
        ks = [k]
    else:
        ks = [params.get("k", 5)]

    reports = []
    for kk in ks:
        print(f"\nEvaluating with k={kk}...")

        # Evaluate raw follow-up pipeline
        print("\n--- Raw Follow-up Pipeline ---")
        raw_report = evaluate_retrieval(
            ground_truth=ground_truth,
            k=kk,
            search_fn=search_raw,
            **search_kwargs,
        )
        _print_arm_metrics(raw_report, kk)

        # Evaluate multi-turn rewritten pipeline
        print("\n--- Multi-turn Rewritten Pipeline ---")
        search_multi.cursor = 0
        multi_report = evaluate_retrieval(
            ground_truth=ground_truth,
            k=kk,
            search_fn=search_multi,
            **search_kwargs,
        )
        _print_arm_metrics(multi_report, kk)

        reports.append((kk, raw_report, multi_report))

        # Side-by-side comparison
        _print_side_by_side(kk, raw_report, multi_report)

    # Determine winner from the last evaluated k (hit rate and MRR only).
    _, raw_report, multi_report = reports[-1]
    hr_delta = multi_report['hit_rate'] - raw_report['hit_rate']
    mrr_delta = multi_report['mrr'] - raw_report['mrr']

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

    # Per-FAQ / per-category breakdowns are printed only for single-k runs
    # (default or --k); in --k-sweep mode only the aggregate metrics and the
    # comparison are printed per k.
    if k_sweep is None:
        category_map = _load_faq_categories()
        for kk, raw_report, multi_report in reports:
            _print_per_faq_table(raw_report, "Raw Follow-up")
            _print_per_faq_table(multi_report, "Multi-turn Rewritten")
            _print_per_category_table(raw_report, "Raw Follow-up", category_map)
            _print_per_category_table(multi_report, "Multi-turn Rewritten", category_map)

    # Show example rewrites (covers only the evaluated subset when --limit is set)
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
    parser = argparse.ArgumentParser(
        description="Evaluate multi-turn query rewriting vs raw follow-up retrieval."
    )
    parser.add_argument(
        "--ground-truth",
        dest="ground_truth_path",
        default=None,
        help="Path to a multi-turn ground-truth CSV "
             "(default: data/ground_truth_multi_turn_generated.csv)",
    )
    k_group = parser.add_mutually_exclusive_group()
    k_group.add_argument(
        "--k",
        dest="k",
        type=_positive_int,
        default=None,
        help="Override k for both arms (default: k from tuned_params.json)",
    )
    k_group.add_argument(
        "--k-sweep",
        dest="k_sweep",
        type=_k_sweep_list,
        default=None,
        help="Comma-separated list of k values to evaluate, e.g. '1,3,5,10'",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=_positive_int,
        default=None,
        help="Evaluate only the first N valid rows (default: all rows)",
    )
    args = parser.parse_args()
    sys.exit(main(
        ground_truth_path=args.ground_truth_path,
        k=args.k,
        k_sweep=args.k_sweep,
        limit=args.limit,
    ))