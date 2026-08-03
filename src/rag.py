import sys

from src.config import load_tuned_params
from src.llm import ask_llm
from src.search import search

SYSTEM_PROMPT_TEMPLATE = """You are a helpful rental FAQ assistant for a bouncy castle rental company.
Answer the user's question based on the following FAQ entries.

If the FAQ entries do not contain enough information to answer the question, say so clearly.

FAQ entries:
{contexts}"""

SYSTEM_PROMPT_WITH_HISTORY_TEMPLATE = """You are a helpful rental FAQ assistant for a bouncy castle rental company.
Answer the user's question based on the following FAQ entries.

If the FAQ entries do not contain enough information to answer the question, say so clearly.

Conversation history:
{history}

FAQ entries:
{contexts}"""

REWRITE_PROMPT_TEMPLATE = """You are a search query optimizer for a bouncy castle rental FAQ system.
Rewrite the user's question into a search-friendly query that will retrieve the most relevant FAQ entries.

Guidelines:
- Expand abbreviations and synonyms (e.g., "bday" → "birthday", "bouncy" → "inflatable bounce house")
- Add domain vocabulary (e.g., "setup" → "delivery setup installation", "cost" → "price rental fee")
- Normalize phrasing (e.g., "how much?" → "what is the rental price cost")
- Keep it concise but comprehensive
- Return ONLY the rewritten query text, nothing else

User question: {question}

Rewritten query:"""

REWRITE_WITH_HISTORY_PROMPT_TEMPLATE = """You are a search query optimizer for a bouncy castle rental FAQ system.
Rewrite the user's latest question into a standalone search query that resolves pronouns and ellipsis using the conversation history.

Guidelines:
- Use the conversation history to understand context (pronouns, references, follow-up topics)
- Expand abbreviations and synonyms (e.g., "bday" → "birthday", "bouncy" → "inflatable bounce house")
- Add domain vocabulary (e.g., "setup" → "delivery setup installation", "cost" → "price rental fee")
- Normalize phrasing (e.g., "how much?" → "what is the rental price cost")
- Keep it concise but comprehensive
- Return ONLY the rewritten query text, nothing else

Conversation history:
{history}

Latest user question: {question}

Rewritten standalone query:"""


def _format_contexts(contexts):
    if not contexts:
        return "No relevant FAQ entries found."
    lines = []
    for i, ctx in enumerate(contexts, 1):
        lines.append(f"{i}. Category: {ctx['category']}")
        lines.append(f"   Question: {ctx['question']}")
        lines.append(f"   Answer: {ctx['answer']}")
    return "\n".join(lines)


def rewrite_query(question, groq_model=None, openai_model=None):
    """
    Rewrite a user's raw question into a search-friendly query using an LLM.
    
    Args:
        question: The user's raw question (string)
        groq_model: Optional Groq model to use
        openai_model: Optional OpenAI model to use
    
    Returns:
        A rewritten search query string. Falls back to the original question
        if the rewrite fails, returns empty, or the input is invalid.
    """
    # Handle non-string or empty/whitespace input
    if not isinstance(question, str) or not question.strip():
        return question
    
    # Build the rewrite prompt
    rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(question=question.strip())
    
    try:
        result = ask_llm(
            system_prompt="You are a search query optimizer. Return only the rewritten query.",
            user_message=rewrite_prompt,
            groq_model=groq_model,
            openai_model=openai_model,
        )
        
        # Handle None result or missing/non-string response
        if result is None:
            return question
        
        response = result.get("response")
        if not isinstance(response, str):
            return question
        
        rewritten = response.strip()
        
        # Fall back to original if rewrite is empty or whitespace-only
        if not rewritten:
            return question
        
        return rewritten
    
    except Exception:
        # Any exception (including RuntimeError, ValueError, AttributeError, etc.)
        # fall back to original question
        return question


def _format_history_for_prompt(history, max_turns):
    """
    Format conversation history for the rewrite prompt.
    
    Args:
        history: List of dicts with 'role' and 'content' keys, in chronological order.
        max_turns: Maximum number of prior turns to include.
    
    Returns:
        Formatted history string for the prompt, or empty string if no valid history.
    """
    if not history:
        return ""
    
    # Filter valid entries: must have 'role' and 'content' as strings, role must be 'user' or 'assistant'
    valid_entries = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str):
            continue
        valid_entries.append({"role": role, "content": content.strip()})
    
    if not valid_entries:
        return ""
    
    # Take only the most recent max_turns entries
    recent_entries = valid_entries[-max_turns:]
    
    # Format as "User: ..." / "Assistant: ..."
    lines = []
    for entry in recent_entries:
        role_label = "User" if entry["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {entry['content']}")
    
    return "\n".join(lines)


def rewrite_query_with_history(question, history, groq_model=None, openai_model=None, history_turns=4):
    """
    Rewrite a user's latest question into a standalone search query using conversation history.
    
    Args:
        question: The user's latest question (string) - not duplicated in history
        history: List of dicts in chronological order: [{"role": "user"|"assistant", "content": str}, ...]
        groq_model: Optional Groq model to use
        openai_model: Optional OpenAI model to use
        history_turns: Maximum number of prior conversation turns to include
    
    Returns:
        A rewritten standalone search query string. Falls back to the original question
        (or single-turn rewrite) if history is empty, rewrite fails, or returns empty.
    """
    # Handle non-string or empty/whitespace input
    if not isinstance(question, str) or not question.strip():
        return question
    
    # Format history for prompt
    formatted_history = _format_history_for_prompt(history, history_turns)
    
    # If no valid history, degrade to single-turn rewrite (which itself degrades to raw question)
    if not formatted_history:
        return rewrite_query(question, groq_model=groq_model, openai_model=openai_model)
    
    # Build the multi-turn rewrite prompt
    rewrite_prompt = REWRITE_WITH_HISTORY_PROMPT_TEMPLATE.format(
        history=formatted_history,
        question=question.strip()
    )
    
    try:
        result = ask_llm(
            system_prompt="You are a search query optimizer. Return only the rewritten query.",
            user_message=rewrite_prompt,
            groq_model=groq_model,
            openai_model=openai_model,
        )
        
        # Handle None result or missing/non-string response
        if result is None:
            return rewrite_query(question, groq_model=groq_model, openai_model=openai_model)
        
        response = result.get("response")
        if not isinstance(response, str):
            return rewrite_query(question, groq_model=groq_model, openai_model=openai_model)
        
        rewritten = response.strip()
        
        # Fall back to single-turn rewrite if rewrite is empty or whitespace-only
        if not rewritten:
            return rewrite_query(question, groq_model=groq_model, openai_model=openai_model)
        
        return rewritten
    
    except Exception:
        # Any exception falls back to single-turn rewrite
        return rewrite_query(question, groq_model=groq_model, openai_model=openai_model)


def answer_question(
    question,
    history=None,
    k=None,
    bm25_path=None,
    faiss_path=None,
    docs_path=None,
    groq_model=None,
    openai_model=None,
    rewrite_enabled=None,
    history_rewrite_enabled=None,
):
    # Load config to check if query rewriting is enabled. The rewrite flags can
    # be overridden per call (e.g. evaluations that must fix retrieval across
    # arms); None falls back to tuned_params.json.
    params = load_tuned_params()
    if rewrite_enabled is None:
        rewrite_enabled = params.get("rewrite_enabled", False)
    if history_rewrite_enabled is None:
        history_rewrite_enabled = params.get("history_rewrite_enabled", False)
    history_turns = params.get("history_turns", 4)
    
    # Determine the search query (multi-turn rewrite, single-turn rewrite, or raw)
    if history_rewrite_enabled and history:
        search_query = rewrite_query_with_history(
            question, 
            history, 
            groq_model=groq_model, 
            openai_model=openai_model,
            history_turns=history_turns
        )
    elif rewrite_enabled:
        search_query = rewrite_query(question, groq_model=groq_model, openai_model=openai_model)
    else:
        search_query = question
    
    # Retrieve using the (potentially rewritten) query
    contexts = search(search_query, k=k, bm25_path=bm25_path, faiss_path=faiss_path, docs_path=docs_path)

    formatted = _format_contexts(contexts)

    # Include the conversation history in the answer prompt whenever valid history
    # is provided (bounded to the last `history_turns` messages). Answer memory is
    # independent of the query-rewriting flags. With no valid history, the prompt
    # stays identical to the single-turn prompt (no history section).
    formatted_history = _format_history_for_prompt(history, history_turns)
    if formatted_history:
        system_prompt = SYSTEM_PROMPT_WITH_HISTORY_TEMPLATE.format(
            history=formatted_history,
            contexts=formatted,
        )
    else:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(contexts=formatted)

    # The raw question is still sent to the LLM for the final answer
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
