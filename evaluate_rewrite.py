#!/usr/bin/env python
"""
Evaluate query rewriting vs raw query retrieval.

This script evaluates both pipelines against data/ground_truth.csv:
- Raw query: search(question, ...)
- Rewritten query: search(rewrite_query(question), ...)

Reports hit rate@k and MRR@k for both approaches.
"""

import os
import sys
from functools import partial

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluate import evaluate_retrieval, load_ground_truth
from src.rag import rewrite_query
from src.search import search


def search_raw(query, k=5, **kwargs):
    """Search using the raw query."""
    return search(query, k=k, **kwargs)


def search_rewritten(query, k=5, **kwargs):
    """Search using the rewritten query."""
    rewritten = rewrite_query(query)
    return search(rewritten, k=k, **kwargs)


def main():
    # Check for API keys
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not groq_key and not openai_key:
        print("No API keys found (GROQ_API_KEY or OPENAI_API_KEY).")
        print("Skipping evaluation - query rewriting requires an LLM.")
        print("Set GROQ_API_KEY or OPENAI_API_KEY to run the evaluation.")
        return 0
    
    print("Loading ground truth...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} queries")
    
    # Get index paths from tuned params or use defaults
    from src.config import load_tuned_params
    params = load_tuned_params()
    
    # We need to pass the index paths to search
    # The search function has defaults, but let's be explicit
    from src.ingest import DEFAULT_BM25_PATH, DEFAULT_FAISS_PATH, DEFAULT_DOCS_PATH
    
    search_kwargs = {
        "bm25_path": DEFAULT_BM25_PATH,
        "faiss_path": DEFAULT_FAISS_PATH,
        "docs_path": DEFAULT_DOCS_PATH,
    }
    
    k = params.get("k", 5)
    print(f"\nEvaluating with k={k}...")
    
    # Evaluate raw query pipeline
    print("\n--- Raw Query Pipeline ---")
    raw_report = evaluate_retrieval(
        ground_truth=ground_truth,
        k=k,
        search_fn=search_raw,
        **search_kwargs,
    )
    print(f"Hit Rate@{k}: {raw_report['hit_rate']:.4f}")
    print(f"MRR@{k}:      {raw_report['mrr']:.4f}")
    
    # Evaluate rewritten query pipeline
    print("\n--- Rewritten Query Pipeline ---")
    rewritten_report = evaluate_retrieval(
        ground_truth=ground_truth,
        k=k,
        search_fn=search_rewritten,
        **search_kwargs,
    )
    print(f"Hit Rate@{k}: {rewritten_report['hit_rate']:.4f}")
    print(f"MRR@{k}:      {rewritten_report['mrr']:.4f}")
    
    # Side-by-side comparison
    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Raw Query':>12} {'Rewritten':>12} {'Delta':>10}")
    print("-" * 60)
    
    hr_delta = rewritten_report['hit_rate'] - raw_report['hit_rate']
    mrr_delta = rewritten_report['mrr'] - raw_report['mrr']
    
    print(f"{'Hit Rate@' + str(k):<20} {raw_report['hit_rate']:>12.4f} {rewritten_report['hit_rate']:>12.4f} {hr_delta:>+10.4f}")
    print(f"{'MRR@' + str(k):<20} {raw_report['mrr']:>12.4f} {rewritten_report['mrr']:>12.4f} {mrr_delta:>+10.4f}")
    
    print("-" * 60)
    
    # Determine winner
    if hr_delta > 0 and mrr_delta > 0:
        winner = "Rewritten query (improves both metrics)"
        recommended_flag = True
    elif hr_delta < 0 and mrr_delta < 0:
        winner = "Raw query (rewriting degrades both metrics)"
        recommended_flag = False
    elif hr_delta > 0 and mrr_delta < 0:
        winner = "Mixed: Rewritten improves hit rate but degrades MRR"
        recommended_flag = False  # Conservative: don't enable if MRR drops
    elif hr_delta < 0 and mrr_delta > 0:
        winner = "Mixed: Rewritten improves MRR but degrades hit rate"
        recommended_flag = False  # Conservative: don't enable if hit rate drops
    else:
        winner = "Tie: No difference in metrics"
        recommended_flag = False  # Default to raw on tie
    
    print(f"\nWinner: {winner}")
    print(f"Recommended rewrite_enabled: {recommended_flag}")
    
    # Show some example rewrites
    print("\n" + "=" * 60)
    print("EXAMPLE REWRITES (first 10 queries)")
    print("=" * 60)
    for i, item in enumerate(ground_truth[:10]):
        original = item["question"]
        rewritten = rewrite_query(original)
        print(f"\n{i+1}. Original:  {original}")
        print(f"   Rewritten: {rewritten}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())