import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.db import init_db, log_interaction, update_feedback
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


class FeedbackRequest(BaseModel):
    interaction_id: int = Field(..., gt=0)
    feedback: Literal["up", "down"]


_db_inited = False


def _log_interaction(question, answer, metadata):
    global _db_inited
    if not os.environ.get("DATABASE_URL"):
        return None
    try:
        if not _db_inited:
            init_db()
            _db_inited = True
        return log_interaction(question, answer, metadata=metadata)
    except Exception:
        return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        result = answer_question(question=req.question)
        result["interaction_id"] = _log_interaction(
            question=req.question,
            answer=result["answer"],
            metadata={
                "provider": result["provider"],
                "model": result["model"],
                "tokens": result["tokens"],
                "latency": result["latency"],
                "cost": result["cost"],
            },
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to get an answer: {e}")


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
