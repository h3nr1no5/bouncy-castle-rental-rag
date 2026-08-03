"""Generate an LLM-written multi-turn ground-truth set for evaluating history-aware query rewriting.

For each FAQ entry in ``data/faq.csv`` the LLM writes natural multi-turn customer
conversations that end in a pronoun/ellipsis follow-up whose answer is exactly
that FAQ entry (e.g. "How much for a weekend?"). The script writes a single CSV,
``data/ground_truth_multi_turn_generated.csv``, with columns
``conversation_id, prior_user_turns, follow_up_question, document_id`` so that
``load_multi_turn_ground_truth()`` in ``evaluate_multi_turn_rewrite.py`` can
consume it with zero code changes.

The script mirrors the structure and failure handling of
``generate_ground_truth.py``: OpenAI structured output (``responses.parse``
with a pydantic model), a 4-attempt exponential-backoff retry loop, parallel
generation via ``ThreadPoolExecutor(max_workers=8)``, a ``tqdm`` progress bar,
and CSV output through the stdlib ``csv`` module. Re-running overwrites the
output file cleanly (no rows accumulate).

Usage::

    uv run python generate_ground_truth_multi_turn.py
"""

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from src.faqs import load_faqs

MODEL = "gpt-5.4-mini"
INPUT_PRICE = 0.75 / 1_000_000
OUTPUT_PRICE = 4.50 / 1_000_000

#: How many conversations the model is asked to write per FAQ (mirrors the
#: single-turn set's 5 variants per FAQ).
CONVERSATIONS_PER_FAQ = 5
#: Upper bound on prior user turns per conversation (the ``history_turns``
#: bound from #52).
MAX_HISTORY_TURNS = 4
#: Number of generation attempts per FAQ before giving up.
MAX_ATTEMPTS = 4

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "data", "ground_truth_multi_turn_generated.csv"
)

CSV_COLUMNS = [
    "conversation_id",
    "prior_user_turns",
    "follow_up_question",
    "document_id",
]

SYSTEM_PROMPT = f"""You are a data augmentation assistant. Given a FAQ entry from a bouncy-castle rental company, generate {CONVERSATIONS_PER_FAQ} natural multi-turn customer conversations that end with a follow-up question whose answer is exactly that FAQ entry.

Each conversation consists of 1 to {MAX_HISTORY_TURNS} prior user turns followed by a final follow-up question. The prior user turns are things a customer says while planning or asking about a rental; the follow-up is the last user message and refers back to the earlier context using pronouns, deictic references, or ellipsis so it is ambiguous without the history (e.g. "How much for a weekend?").

Rules:
- The answer to the follow-up question must be exactly the given FAQ entry's Answer
- The prior user turns must NOT already state or reveal the answer
- The follow-up must use pronouns/ellipsis so it cannot be answered from the follow-up text alone (do not repeat the FAQ's topic keywords or the exact question wording)
- Vary the number of prior user turns between 1 and {MAX_HISTORY_TURNS}; include a good mix of conversations with 2, 3, or {MAX_HISTORY_TURNS} prior turns, not only 1
- Prior user turns should be short, natural customer messages (questions, statements, small clarifications) that set up the follow-up
- Do not use semicolons in any turn
- Return ONLY a JSON object with a "conversations" array of {CONVERSATIONS_PER_FAQ} objects, each with "prior_user_turns" (array of strings) and "follow_up_question" (string)"""


class Conversation(BaseModel):
    prior_user_turns: list[str]
    follow_up_question: str


class Conversations(BaseModel):
    conversations: list[Conversation]


def build_rows(conversations, faq_index, start_conversation_id=1):
    """Turn parsed conversations into CSV rows for one FAQ.

    Sanitizes the model output so the file always satisfies the loader's
    contract in ``evaluate_multi_turn_rewrite.py``:

    - ``prior_user_turns`` is ``;``-separated (so no segment may contain a
      semicolon), capped at ``MAX_HISTORY_TURNS`` segments
    - no cell is empty: blank prior turns are dropped, and a conversation
      with no remaining prior turns or a blank follow-up is skipped
    - ``document_id`` follows the ``faq_{index}`` convention from
      ``load_faqs()`` order

    Returns ``(rows, next_conversation_id)`` where ``rows`` is a list of dicts
    with keys ``CSV_COLUMNS`` and conversation ids are unique and increasing.
    """
    doc_id = f"faq_{faq_index}"
    rows = []
    conversation_id = start_conversation_id
    for conv in conversations:
        prior_turns = [t.replace(";", ",").strip() for t in conv.prior_user_turns if t.strip()]
        follow_up = conv.follow_up_question.replace(";", ",").strip()
        if not prior_turns or not follow_up:
            continue
        prior_turns = prior_turns[:MAX_HISTORY_TURNS]
        rows.append(
            {
                "conversation_id": str(conversation_id),
                "prior_user_turns": ";".join(prior_turns),
                "follow_up_question": follow_up,
                "document_id": doc_id,
            }
        )
        conversation_id += 1
    return rows, conversation_id


def _generate_for_faq(client, faq, index):
    text = f"Category: {faq['Category']}\nQuestion: {faq['Question']}\nAnswer: {faq['Answer']}"

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                text_format=Conversations,
            )
            conversations = response.output[0].content[0].parsed.conversations
            if not conversations:
                raise ValueError("empty conversation list")
            usage = response.usage
            return {
                "conversations": conversations,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }
        except Exception as e:
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2**attempt)
            else:
                print(f"Failed for FAQ {index} ({faq['Question']}): {e}")
    return {"conversations": [], "input_tokens": 0, "output_tokens": 0}


def write_csv(output_path, rows):
    """Write the generated rows to ``output_path`` as UTF-8 CSV (overwrites)."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    load_dotenv()
    # Check the key before constructing the client: the installed openai
    # version raises on construction when no key is present, and we must
    # exit before touching the output file.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        return

    faqs = load_faqs()
    client = OpenAI(api_key=api_key)

    total_input_tokens = 0
    total_output_tokens = 0
    results_by_index = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_generate_for_faq, client, faq, i): i
            for i, faq in enumerate(faqs)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating conversations"):
            faq_index = futures[future]
            result = future.result()
            results_by_index[faq_index] = result
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]

    rows = []
    conversation_id = 1
    failures = 0
    for i in range(len(faqs)):
        result = results_by_index[i]
        if not result["conversations"]:
            failures += 1
            continue
        new_rows, conversation_id = build_rows(
            result["conversations"], i, start_conversation_id=conversation_id
        )
        rows.extend(new_rows)

    input_cost = total_input_tokens * INPUT_PRICE
    output_cost = total_output_tokens * OUTPUT_PRICE

    print(f"FAQs processed: {len(faqs)}, conversations generated: {len(rows)}, failures: {failures}")
    print(f"Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}")
    print(f"Estimated cost: ${input_cost + output_cost:.4f} (${input_cost:.4f} input + ${output_cost:.4f} output)")

    if not rows:
        print("No conversations were generated; leaving existing output file untouched.")
        return

    write_csv(OUTPUT_PATH, rows)
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
