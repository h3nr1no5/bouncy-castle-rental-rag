import json
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.extract import strip_code_fences
from src.llm import ask_llm as _default_ask_llm
from src.merge_bilingual import normalize_accent

CLUSTER_SYSTEM_PROMPT = (
    "You group paraphrased customer questions from the Terms & Conditions of "
    "bouncy-castle rental companies into topics. Questions with the same meaning, "
    "even when worded differently, belong to the same topic. Keep each topic "
    "label short and concrete. Respond ONLY with a JSON object mapping each "
    'question id to its topic label, e.g. '
    '{"assignments":{"0":"deposit amount","1":"deposit amount","2":"weather cancellation"}}.'
)

CLUSTER_USER_TEMPLATE = (
    "Group these questions by meaning:\n{questions}\n\n"
    "Return only the assignments JSON object."
)


@dataclass(frozen=True)
class TopicCluster:
    """Member row indices plus a stable, deterministic cluster id."""

    id: str
    rows: list[int]


@runtime_checkable
class Clusterer(Protocol):
    def cluster(self, rows: list[dict]) -> list[TopicCluster]: ...


def _en_question(row):
    """Cluster identity is driven by the English question; the Hungarian companion rides along."""
    return (row.get("question_en") or row.get("question_hu") or "").strip()


def _slugify(text):
    text = normalize_accent(text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "topic"


class ExactClusterer:
    """Deterministic fallback: group rows whose accent-folded EN question is identical.

    Preserves today's ``merge_bilingual.fold_duplicate`` behavior as a strategy.
    """

    def cluster(self, rows):
        groups = {}
        for idx, row in enumerate(rows):
            key = normalize_accent(_en_question(row))
            if key:
                groups.setdefault(key, []).append(idx)
        return [
            TopicCluster(id=key, rows=members)
            for key, members in sorted(groups.items())
        ]


class LLMClusterer:
    """Default strategy: batch ``ask_llm`` call assigns paraphrased EN questions to topics.

    ``ask_llm`` is injectable so tests never touch the network. A single batch call
    maps each question index to a topic label; identical labels form one cluster.
    If the LLM reply cannot be parsed, grouping falls back to :class:`ExactClusterer`.
    """

    def __init__(self, ask_llm=None, groq_model=None, openai_model=None):
        self._ask_llm = ask_llm or _default_ask_llm
        self._groq_model = groq_model
        self._openai_model = openai_model

    def cluster(self, rows):
        questions = [_en_question(row) for row in rows]
        valid = [(idx, q) for idx, q in enumerate(questions) if q]
        if not valid:
            return []

        assignments = self._assign([q for _, q in valid])
        if assignments is None:
            return ExactClusterer().cluster(rows)

        groups = {}
        for (idx, _q), topic in zip(valid, assignments):
            groups.setdefault(topic, []).append(idx)
        return [
            TopicCluster(id=topic, rows=members)
            for topic, members in sorted(groups.items())
        ]

    def _assign(self, questions):
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions))
        try:
            result = self._ask_llm(
                system_prompt=CLUSTER_SYSTEM_PROMPT,
                user_message=CLUSTER_USER_TEMPLATE.format(questions=numbered),
                groq_model=self._groq_model,
                openai_model=self._openai_model,
            )
            return self._parse_assignments(result.get("response", ""), len(questions))
        except Exception:
            return None

    @staticmethod
    def _parse_assignments(reply, count):
        obj = json.loads(strip_code_fences(reply))
        assignments = obj.get("assignments") or obj.get("topics") or obj
        if not isinstance(assignments, dict):
            raise ValueError("assignments must be a JSON object")
        labels = []
        for i in range(count):
            label = assignments.get(str(i), assignments.get(i))
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"missing or invalid assignment for question {i}")
            labels.append(_slugify(label))
        return labels


def default_clusterer():
    """Factory returning the default (LLM) clustering strategy."""
    return LLMClusterer()
