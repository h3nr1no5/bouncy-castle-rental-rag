import pytest
from src.faqs import load_faqs


def test_load_faqs_returns_non_empty():
    faqs = load_faqs()
    assert len(faqs) > 0


def test_every_row_has_expected_non_empty_keys():
    expected_keys = {"Category", "Question", "Answer"}
    faqs = load_faqs()
    for row in faqs:
        assert set(row.keys()) == expected_keys
        for v in row.values():
            assert isinstance(v, str) and len(v) > 0


def test_raises_on_missing_csv(tmp_path):
    missing = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError, match="FAQ CSV not found"):
        load_faqs(path=missing)
