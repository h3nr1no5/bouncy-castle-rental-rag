#!/usr/bin/env python
"""
Evaluate multi-turn query rewriting vs raw follow-up question retrieval.

This script evaluates both pipelines against data/ground_truth_multi_turn.csv:
- Raw follow-up: search(follow_up_question, ...)
- Multi-turn rewritten: search(rewrite_query_with_history(follow_up_question, history), ...)

Reports hit rate@k and MRR@k for both approaches.
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


def search_raw(query, k=5, **kwargs):
    """Search using the raw follow-up question."""
    return search(query, k=k, **kwargs)


def search_multi(query, k=5, **kwargs):
    """Search using the multi-turn rewritten query."""
    # Find the ground truth item for this query to get its history
    for item in search_multi.ground_truth:
        if item["question"] == query:
            history = item["history"]
            break
    else:
        history = []
    
    rewritten = rewrite_query_with_history(query, history)
    return search(rewritten, k=k, **kwargs)


def load_multi_turn_ground_truth(path=None):
    """Load multi-turn ground truth from CSV."""
    if path is None:
        path = pathlib.Path(__file__).resolve().parents[0] / "data" / "ground_truth_multi_turn.csv"
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found at {path}")
    
    ground_truth = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prior_turns = [t.strip() for t in row["prior_user_turns"].split(";") if t.strip()]
            history = [{"role": "user", "content": t} for t in prior_turns]
            ground_truth.append({
                "question": row["follow_up_question"],
                "document_id": row["document_id"],
                "history": history,
            })
    return ground_truth


def main():
    # Check for API keys
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not groq_key and not openai_key:
        print("No API keys found (GROQ_API_KEY or OPENAI_API_KEY).")
        print("Skipping evaluation - multi-turn query rewriting requires an LLM.")
        print("Set GROQ_API_KEY or OPENAI_API_KEY to run the evaluation.")
        return 0
    
    print("Loading ground truth...")
    ground_truth = load_multi_turn_ground_truth()
    print(f"Loaded {len(ground_truth)} queries")
    
    # Attach ground truth to search_multi for history lookup
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
    sys.exit(main())