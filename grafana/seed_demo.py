"""Seed `rag_logs` with deterministic demo rows for Grafana verification.

Usage:
    uv run python grafana/seed_demo.py

Deletes any previous demo rows (identified by the demo questions) before
inserting, so re-running stays deterministic.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.db import get_connection  # noqa: E402

DEMO_ROWS = [
    {
        "question": "How much does a castle cost?",
        "answer": "Prices start at $100 per day.",
        "feedback": "up",
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "tokens": {"prompt": 120, "completion": 45, "total": 165},
        "latency": 1.2,
        "cost": 0.00012,
    },
    {
        "question": "What sizes are available?",
        "answer": "We offer medium, large, and extra-large castles.",
        "feedback": "up",
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "tokens": {"prompt": 95, "completion": 30, "total": 125},
        "latency": 0.9,
        "cost": 0.00009,
    },
    {
        "question": "Do you deliver on weekends?",
        "answer": "Yes, delivery is available seven days a week.",
        "feedback": "down",
        "model": "gpt-5.4-mini",
        "provider": "openai",
        "tokens": {"prompt": 140, "completion": 55, "total": 195},
        "latency": 2.4,
        "cost": 0.00035,
    },
    {
        "question": "Is setup included?",
        "answer": "Setup and takedown are included in every booking.",
        "feedback": "up",
        "model": "gpt-5.4-mini",
        "provider": "openai",
        "tokens": {"prompt": 88, "completion": 25, "total": 113},
        "latency": 1.8,
        "cost": 0.00026,
    },
    {
        "question": "What is your cancellation policy?",
        "answer": "Free cancellation up to 48 hours before the rental.",
        "feedback": None,
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "tokens": {"prompt": 110, "completion": 38, "total": 148},
        "latency": 1.1,
        "cost": 0.00010,
    },
    {
        "question": "Are there any age restrictions?",
        "answer": "Children under 5 must be supervised at all times.",
        "feedback": "down",
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "tokens": {"prompt": 130, "completion": 42, "total": 172},
        "latency": 1.5,
        "cost": 0.00013,
    },
    {
        "question": "Can you deliver to parks?",
        "answer": "Yes, we deliver to public parks and private venues.",
        "feedback": "up",
        "model": "gpt-5.4-mini",
        "provider": "openai",
        "tokens": {"prompt": 102, "completion": 33, "total": 135},
        "latency": 2.1,
        "cost": 0.00029,
    },
    {
        "question": "How many children can fit inside?",
        "answer": "Up to 12 children can play inside at the same time.",
        "feedback": None,
        "model": "gpt-5.4-mini",
        "provider": "openai",
        "tokens": {"prompt": 115, "completion": 40, "total": 155},
        "latency": 1.7,
        "cost": 0.00024,
    },
    {
        "question": "What happens if it rains?",
        "answer": "We reschedule to a dry day free of charge.",
        "feedback": "up",
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "tokens": {"prompt": 125, "completion": 36, "total": 161},
        "latency": 1.0,
        "cost": 0.00011,
    },
    {
        "question": "Do you offer a discount for multiple days?",
        "answer": "Multi-day rentals receive a 10% discount.",
        "feedback": None,
        "model": "gpt-5.4-mini",
        "provider": "openai",
        "tokens": {"prompt": 145, "completion": 50, "total": 195},
        "latency": 2.6,
        "cost": 0.00033,
    },
    {
        "question": "Can I customise the colour scheme?",
        "answer": "Yes, custom themes are available for an additional fee.",
        "feedback": None,
        "model": "gpt-5.4-mini",
        "provider": "openai",
        "tokens": {"prompt": 310, "completion": 95, "total": 405},
        "latency": 4.2,
        "cost": 0.00048,
    },
    {
        "question": "Do you provide insurance?",
        "answer": "Public liability insurance is included in every rental.",
        "feedback": "down",
        "model": "gpt-5.4-mini",
        "provider": "openai",
        "tokens": {"prompt": 520, "completion": 140, "total": 660},
        "latency": 6.1,
        "cost": 0.00072,
    },
    {
        "question": "What happens if my booking overlaps?",
        "answer": "Overlapping bookings are rare; we confirm slots by phone.",
        "feedback": "up",
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "tokens": {"prompt": 400, "completion": 120, "total": 520},
        "latency": 5.3,
        "cost": 0.00031,
    },
    {
        "question": "Is there a booking deposit?",
        "answer": "A 20% deposit is due at the time of booking.",
        "feedback": None,
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "tokens": {"prompt": 460, "completion": 130, "total": 590},
        "latency": 7.8,
        "cost": 0.00036,
    },
]

DEMO_QUESTIONS = [row["question"] for row in DEMO_ROWS]


def main():
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM rag_logs WHERE question = ANY(%s)",
                (DEMO_QUESTIONS,),
            )
            now = datetime.now(timezone.utc)
            for i, row in enumerate(DEMO_ROWS):
                created_at = now - timedelta(minutes=len(DEMO_ROWS) - i)
                metadata = {
                    "provider": row["provider"],
                    "model": row["model"],
                    "tokens": row["tokens"],
                    "latency": row["latency"],
                    "cost": row["cost"],
                }
                cur.execute(
                    """
                    INSERT INTO rag_logs (question, answer, feedback, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        row["question"],
                        row["answer"],
                        row["feedback"],
                        json.dumps(metadata),
                        created_at,
                    ),
                )
            cur.execute("SELECT COUNT(*) FROM rag_logs")
            row = cur.fetchone()
            total = row[0] if row else 0
    print(f"Seeded {len(DEMO_ROWS)} demo rows. Total rows in rag_logs: {total}")


if __name__ == "__main__":
    main()
