import json
from pathlib import Path

from src.config import DEFAULT_TUNED_PARAMS_PATH, load_tuned_params

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNED_PARAMS_FILE = PROJECT_ROOT / "tuned_params.json"

REQUIRED_KEYS = ["k", "rrf_k", "cat_weight", "bm25_k1", "bm25_b", "rewrite_enabled"]


def test_tuned_params_file_exists_at_repo_root():
    assert TUNED_PARAMS_FILE.is_file()


def test_tuned_params_file_has_required_keys():
    with open(TUNED_PARAMS_FILE) as f:
        params = json.load(f)
    assert set(params.keys()) == set(REQUIRED_KEYS)


def test_load_tuned_params_matches_committed_file():
    with open(TUNED_PARAMS_FILE) as f:
        params = json.load(f)
    assert load_tuned_params() == params


def test_load_tuned_params_default_path_is_repo_root():
    assert DEFAULT_TUNED_PARAMS_PATH == TUNED_PARAMS_FILE


def test_load_tuned_params_falls_back_when_file_missing(tmp_path):
    params = load_tuned_params(tmp_path / "missing.json")
    assert params == {
        "k": 5,
        "rrf_k": 60,
        "cat_weight": 0,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "rewrite_enabled": False,
    }


def test_load_tuned_params_overlays_partial_file(tmp_path):
    config = tmp_path / "partial.json"
    config.write_text(json.dumps({"k": 3, "bm25_k1": 2.0}))
    params = load_tuned_params(config)
    assert params["k"] == 3
    assert params["bm25_k1"] == 2.0
    assert params["rrf_k"] == 60
    assert params["cat_weight"] == 0
    assert params["bm25_b"] == 0.75
    assert params["rewrite_enabled"] is False
