import json
import logging
import pathlib

from dlt.sources.helpers import requests

logger = logging.getLogger(__name__)

DEFAULT_COMPANIES_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "companies.json"
DEFAULT_TOC_DIR = pathlib.Path(__file__).resolve().parents[1] / "db" / "toc"


def load_companies(path=None):
    if path is None:
        path = DEFAULT_COMPANIES_PATH
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"companies.json not found at {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        if not isinstance(entry, dict) or "company" not in entry or "url" not in entry:
            raise ValueError(f"Invalid companies.json entry: {entry}")
    return data


def _looks_like_pdf(url):
    return url.lower().split("?", 1)[0].endswith((".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg"))


def fetch_source(client, entry):
    company = entry["company"]
    url = entry["url"]
    if _looks_like_pdf(url):
        raise ValueError(f"{company}: URL looks like a binary/PDF document, skipping: {url}")
    response = client.get(url)
    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        raise ValueError(f"{company}: unexpected content-type {content_type!r} for {url}")
    body = response.text
    stripped = body.strip()
    if len(stripped) < 200:
        raise ValueError(f"{company}: page appears empty/JS-only ({len(stripped)} chars) for {url}")
    return stripped


def collect(companies=None, toc_dir=None, companies_path=None, client=None):
    if companies is None:
        companies = load_companies(companies_path)
    if toc_dir is None:
        toc_dir = DEFAULT_TOC_DIR
    toc_dir = pathlib.Path(toc_dir)

    if client is None:
        client = requests.Client(
            request_timeout=20,
            raise_for_status=True,
            request_max_attempts=3,
        )

    results = []
    for entry in companies:
        company = entry["company"]
        url = entry["url"]
        out_dir = toc_dir / company
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            html = fetch_source(client, entry)
        except Exception as e:
            logger.warning("SKIP %s: %s", company, e)
            results.append({"company": company, "url": url, "ok": False, "error": str(e), "path": None})
            continue
        out_path = out_dir / "source.html"
        out_path.write_text(html, encoding="utf-8")
        results.append({"company": company, "url": url, "ok": True, "error": None, "path": str(out_path)})
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for r in collect(companies_path=DEFAULT_COMPANIES_PATH):
        status = "OK " if r["ok"] else "ERR"
        print(f"{status} {r['company']}: {r.get('path') or r.get('error')}")