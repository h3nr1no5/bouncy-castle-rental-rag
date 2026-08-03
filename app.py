import os
from typing import Literal, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.db import get_session_history, init_db, log_interaction, update_feedback
from src.rag import answer_question

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="Bouncy Castle FAQ Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: Union[list[dict], None] = Field(default=None)
    session_id: Union[str, None] = Field(default=None)


class FeedbackRequest(BaseModel):
    interaction_id: int = Field(..., gt=0)
    feedback: Literal["up", "down"]


_db_inited = False


def _log_interaction(question, answer, metadata, session_id=None):
    global _db_inited
    if not os.environ.get("DATABASE_URL"):
        return None
    try:
        if not _db_inited:
            init_db()
            _db_inited = True
        return log_interaction(question, answer, metadata=metadata, session_id=session_id)
    except Exception:
        return None


def _load_session_history(session_id):
    """Load the stored turns for a session, or return None if the database is
    unavailable. Mirrors the graceful fallback in ``_log_interaction``."""
    global _db_inited
    if not os.environ.get("DATABASE_URL"):
        return None
    try:
        if not _db_inited:
            init_db()
            _db_inited = True
        return get_session_history(session_id)
    except Exception:
        return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        if req.session_id is not None and len(req.session_id) > 255:
            raise HTTPException(status_code=422, detail="session_id must be at most 255 characters")

        # Strip and normalize the session id; absent/empty/whitespace means "no session".
        raw_session_id = (req.session_id or "").strip()
        session_id = raw_session_id or None

        history = req.history
        if session_id:
            # Server-side history takes precedence over the client-sent history.
            # On DB failure _load_session_history returns None and we degrade to
            # the client-sent history (or empty history).
            server_history = _load_session_history(session_id)
            if server_history is not None:
                history = server_history

        result = answer_question(question=req.question, history=history)
        result["interaction_id"] = _log_interaction(
            question=req.question,
            answer=result["answer"],
            session_id=session_id,
            metadata={
                "provider": result["provider"],
                "model": result["model"],
                "tokens": result["tokens"],
                "latency": result["latency"],
                "cost": result["cost"],
            },
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to get an answer: {e}")


@app.get("/api/chat/{session_id}/history")
def chat_history(session_id: str):
    if not session_id.strip():
        raise HTTPException(status_code=422, detail="session_id is required")
    if len(session_id) > 255:
        raise HTTPException(status_code=422, detail="session_id must be at most 255 characters")
    history = _load_session_history(session_id.strip())
    return history if history is not None else []


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    try:
        update_feedback(req.interaction_id, req.feedback)
    except ValueError:
        raise HTTPException(status_code=404, detail="Interaction not found")
    return {"status": "ok", "interaction_id": req.interaction_id, "feedback": req.feedback}


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "ui"), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
