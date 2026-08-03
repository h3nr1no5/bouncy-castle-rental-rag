import json
import pathlib

DEFAULT_TUNED_PARAMS_PATH = pathlib.Path(__file__).resolve().parents[1] / "tuned_params.json"

_FALLBACK_DEFAULTS = {
    "k": 5,
    "rrf_k": 60,
    "cat_weight": 0,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "rewrite_enabled": False,
    "history_rewrite_enabled": True,
    "history_turns": 4,
}


def load_tuned_params(path=None):
    if path is None:
        path = DEFAULT_TUNED_PARAMS_PATH
    path = pathlib.Path(path)

    params = dict(_FALLBACK_DEFAULTS)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key in _FALLBACK_DEFAULTS:
            if key in data:
                params[key] = data[key]
    return params
