import csv
import pathlib

DEFAULT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "faq.csv"


def load_faqs(path=None):
    if path is None:
        path = DEFAULT_PATH
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAQ CSV not found at {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
    return rows


if __name__ == "__main__":
    faqs = load_faqs()
    print(f"Rows: {len(faqs)}")
    if faqs:
        print(f"Columns: {', '.join(faqs[0].keys())}")
        missing = {k: sum(1 for row in faqs if not row.get(k)) for k in faqs[0]}
        print(f"Missing values: {missing}")
