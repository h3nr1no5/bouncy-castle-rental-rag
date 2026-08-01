import sys

from src.llm import ask_llm
from src.search import search

SYSTEM_PROMPT_TEMPLATE = """You are a helpful rental FAQ assistant for a bouncy castle rental company.
Answer the user's question based on the following FAQ entries.

If the FAQ entries do not contain enough information to answer the question, say so clearly.

FAQ entries:
{contexts}"""


def _format_contexts(contexts):
    if not contexts:
        return "No relevant FAQ entries found."
    lines = []
    for i, ctx in enumerate(contexts, 1):
        lines.append(f"{i}. Category: {ctx['category']}")
        lines.append(f"   Question: {ctx['question']}")
        lines.append(f"   Answer: {ctx['answer']}")
    return "\n".join(lines)


def answer_question(
    question,
    k=None,
    bm25_path=None,
    faiss_path=None,
    docs_path=None,
    groq_model=None,
    openai_model=None,
):
    contexts = search(question, k=k, bm25_path=bm25_path, faiss_path=faiss_path, docs_path=docs_path)

    formatted = _format_contexts(contexts)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(contexts=formatted)

    result = ask_llm(
        system_prompt=system_prompt,
        user_message=question,
        groq_model=groq_model,
        openai_model=openai_model,
    )

    return {
        "answer": result["response"],
        "contexts": contexts,
        "model": result["model"],
        "provider": result["provider"],
        "latency": result["latency"],
        "cost": result["cost"],
        "tokens": result["tokens"],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.rag '<question>'")
        sys.exit(1)
    question = sys.argv[1]
    result = answer_question(question)
    print(result["answer"])
