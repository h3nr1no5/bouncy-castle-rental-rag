import csv
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


def evaluate_retrieval(ground_truth=None, k=5, search_fn=None, **kwargs):
    if ground_truth is None:
        ground_truth = load_ground_truth()

    if search_fn is None:
        from src.search import search
        search_fn = search

    hit_rates = []
    mrrs = []
    details = []

    for item in tqdm(ground_truth, desc="Retrieval"):
        results = search_fn(item["question"], k=k, **kwargs)
        relevant_ids = [item["document_id"]]
        hr = compute_hit_rate(results, relevant_ids, k=k)
        m = compute_mrr(results, relevant_ids, k=k)
        hit_rates.append(hr)
        mrrs.append(m)
        details.append({
            "question": item["question"],
            "hit": hr == 1.0,
            "mrr": round(m, 4),
            "document_id": item["document_id"],
            "retrieved_ids": [r["id"] for r in results],
        })

    n = len(ground_truth)
    return {
        "k": k,
        "hit_rate": round(sum(hit_rates) / n, 4) if n else 0.0,
        "mrr": round(sum(mrrs) / n, 4) if n else 0.0,
        "n": n,
        "details": details,
    }


if __name__ == "__main__":
    report = evaluate_retrieval()
    print(f"Hit Rate@{report['k']}: {report['hit_rate']}")
    print(f"MRR@{report['k']}: {report['mrr']}")
    print(f"Evaluated on {report['n']} queries")
