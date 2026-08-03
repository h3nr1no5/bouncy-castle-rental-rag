import csv
import math
import pathlib

from tqdm.auto import tqdm

DEFAULT_GROUND_TRUTH_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "ground_truth.csv"


def load_ground_truth(path=None):
    if path is None:
        path = DEFAULT_GROUND_TRUTH_PATH
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found at {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{"question": row["question"], "document_id": row["document_id"]} for row in reader]


def compute_hit_rate(results, relevant_ids, k=None):
    if k is not None:
        results = results[:k]
    retrieved_ids = {r["id"] for r in results}
    return 1.0 if any(doc_id in retrieved_ids for doc_id in relevant_ids) else 0.0


def compute_mrr(results, relevant_ids, k=None):
    candidates = results[:k] if k is not None else results
    for rank, r in enumerate(candidates, start=1):
        if r["id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_recall(results, relevant_ids, k=None):
    """Recall@k = |relevant ∩ retrieved[:k]| / |relevant|.

    Returns 0.0 when there are no relevant ids (avoiding a division by zero).
    """
    if k is not None:
        results = results[:k]
    retrieved_ids = {r["id"] for r in results}
    if not relevant_ids:
        return 0.0
    return len(retrieved_ids & set(relevant_ids)) / len(relevant_ids)


def compute_precision(results, relevant_ids, k=None):
    """Precision@k = |relevant ∩ retrieved[:k]| / min(k, len(retrieved[:k])).

    ``results[:k]`` already has length ``min(k, len(results))``, so the
    denominator is simply the number of retrieved results. Returns 0.0 when no
    results were retrieved (avoiding a division by zero).
    """
    if k is not None:
        results = results[:k]
    if not results:
        return 0.0
    retrieved_ids = {r["id"] for r in results}
    return len(retrieved_ids & set(relevant_ids)) / len(results)


def compute_ndcg(results, relevant_ids, k=None):
    """nDCG@k with binary relevance.

    DCG@k = Σ rel_i / log2(i+1) over ranks 1..min(k, n_results) and
    IDCG@k = Σ_{i=1}^{min(k, |relevant|)} 1/log2(i+1), nDCG = DCG / IDCG.
    Returns 0.0 when IDCG is 0 (no relevant ids), avoiding a division by zero.
    """
    if k is not None:
        results = results[:k]
    relevant = set(relevant_ids)

    dcg = 0.0
    for i, r in enumerate(results, start=1):
        if r["id"] in relevant:
            dcg += 1.0 / math.log2(i + 1)

    n_relevant = len(relevant)
    if k is not None:
        n_relevant = min(k, n_relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _item_relevant_ids(item):
    """Return the list of relevant document ids for a ground-truth item.

    Supports a single ``document_id`` (string or list) and an explicit
    ``document_ids`` list, so an item answerable by more than one FAQ entry is
    scored fairly: hit if ANY relevant id is retrieved, MRR from the best rank
    among them. Items carrying a single ``document_id`` behave exactly as
    before.
    """
    ids = item.get("document_ids", item.get("document_id"))
    if ids is None:
        raise KeyError("ground-truth item has neither 'document_id' nor 'document_ids'")
    if isinstance(ids, str):
        return [ids]
    return list(ids)


def evaluate_retrieval(ground_truth=None, k=5, search_fn=None, **kwargs):
    if ground_truth is None:
        ground_truth = load_ground_truth()

    if search_fn is None:
        from src.search import search
        search_fn = search

    hit_rates = []
    mrrs = []
    recalls = []
    precisions = []
    ndcgs = []
    details = []

    for item in tqdm(ground_truth, desc="Retrieval"):
        results = search_fn(item["question"], k=k, **kwargs)
        relevant_ids = _item_relevant_ids(item)
        hr = compute_hit_rate(results, relevant_ids, k=k)
        m = compute_mrr(results, relevant_ids, k=k)
        r = compute_recall(results, relevant_ids, k=k)
        p = compute_precision(results, relevant_ids, k=k)
        nd = compute_ndcg(results, relevant_ids, k=k)
        hit_rates.append(hr)
        mrrs.append(m)
        recalls.append(r)
        precisions.append(p)
        ndcgs.append(nd)
        details.append({
            "question": item["question"],
            "hit": hr == 1.0,
            "mrr": round(m, 4),
            "recall": round(r, 4),
            "precision": round(p, 4),
            "ndcg": round(nd, 4),
            "document_id": item.get("document_id", item.get("document_ids")),
            "retrieved_ids": [r["id"] for r in results],
        })

    n = len(ground_truth)
    return {
        "k": k,
        "hit_rate": round(sum(hit_rates) / n, 4) if n else 0.0,
        "mrr": round(sum(mrrs) / n, 4) if n else 0.0,
        "recall": round(sum(recalls) / n, 4) if n else 0.0,
        "precision": round(sum(precisions) / n, 4) if n else 0.0,
        "ndcg": round(sum(ndcgs) / n, 4) if n else 0.0,
        "n": n,
        "details": details,
    }


if __name__ == "__main__":
    report = evaluate_retrieval()
    print(f"Hit Rate@{report['k']}: {report['hit_rate']}")
    print(f"MRR@{report['k']}: {report['mrr']}")
    print(f"Recall@{report['k']}: {report['recall']}")
    print(f"Precision@{report['k']}: {report['precision']}")
    print(f"nDCG@{report['k']}: {report['ndcg']}")
    print(f"Evaluated on {report['n']} queries")
