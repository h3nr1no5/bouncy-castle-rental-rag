import pathlib
import re
from html.parser import HTMLParser

HUNGARIAN_HEADINGS = [
    "foglal", "átvétel", "atvetel", "lemond", "hiba", "kár", "kar",
    "javítás", "csere", "biztons", "biztonsági", "elszámolás", "fizetés",
    "visszaszállítás", "szállítás", "telepítés", "szerződés", "feltétel",
    "kötelezettség", "felelősség", "jótállás", "garancia", "adatok",
    "üzemeltetés", "foglalás", "pénztár", "törlés", "módosítás", "díj",
]

DEFAULT_TOC_DIR = pathlib.Path(__file__).resolve().parents[1] / "db" / "toc"


class _TextToTextParser(HTMLParser):
    """A minimal HTML tokenizer that drops tags/scripts/styles but keeps visible text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1
        elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "tr"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip_depth:
            self._skip_depth -= 1
        elif tag in ("p", "li", "h1", "h2", "h3", "tr", "div"):
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self):
        text = "".join(self._parts)
        text = re.sub(r"[ \t\xa0]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()


def strip_tags(html):
    parser = _TextToTextParser()
    parser.feed(html)
    parser.close()
    return parser.get_text()


def _is_heading_line(line):
    lowered = line.lower()
    if not lowered:
        return False
    if not line[0].isalpha():
        return False
    return any(h in lowered for h in HUNGARIAN_HEADINGS)


def _find_heading_lines(lines):
    return [i for i, ln in enumerate(lines) if _is_heading_line(ln)]


def _split_into_chunks(text, max_chunk_chars):
    """Split text on Hungarian heading boundaries, then by roughly equal chunks.

    Returns a list of (section_title, chunk_text).
    """
    lines = [ln.strip() for ln in text.splitlines()]
    boundary_lines = set(_find_heading_lines(lines))

    sections = []
    current = []

    def flush():
        body = "\n".join(current).strip()
        if body:
            marker = " ".join(current[0].split()) if current else ""
            sections.append((marker, body))

    for i, ln in enumerate(lines):
        if i in boundary_lines:
            flush()
            current = [ln]
        else:
            current.append(ln)
    flush()

    result = []
    for title, body in sections:
        if len(body) <= max_chunk_chars:
            result.append((title, body))
            continue
        words = body.split(" ")
        cur = []
        cur_len = 0
        for w in words:
            if cur and (cur_len + len(w) + 1) > max_chunk_chars:
                result.append((title, " ".join(cur)))
                cur = []
                cur_len = 0
            cur.append(w)
            cur_len += len(w) + 1
        if cur:
            result.append((title, " ".join(cur)))
    return result


def parse_html(html, company="unknown", max_chunk_chars=1400):
    text = strip_tags(html)
    if not text:
        return []
    chunks = _split_into_chunks(text, max_chunk_chars=max_chunk_chars)
    return [
        {
            "company": company,
            "section": section or "Feltételek",
            "clause_ref": f"{section or 'feltetelek'}#{i + 1}",
            "clause_text": chunk,
        }
        for i, (section, chunk) in enumerate(chunks)
    ]


if __name__ == "__main__":
    import sys

    company = sys.argv[1] if len(sys.argv) > 1 else None
    if company is None:
        sources = list(DEFAULT_TOC_DIR.glob("*/source.html"))
    else:
        sources = [DEFAULT_TOC_DIR / company / "source.html"]
    for src in sources:
        html = src.read_text(encoding="utf-8")
        rows = parse_html(html, company=src.parent.name)
        print(f"{src.parent.name}: {len(rows)} chunks")