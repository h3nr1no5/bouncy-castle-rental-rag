import json
import os
from typing import Any

import psycopg2
import psycopg2.extras

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_logs (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    feedback TEXT,
    metadata JSONB,
    session_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Idempotent migration for databases created before the session column existed.
ADD_SESSION_COLUMN_SQL = "ALTER TABLE rag_logs ADD COLUMN IF NOT EXISTS session_id TEXT"


def get_connection(dsn=None):
    if dsn is None:
        dsn = os.environ["DATABASE_URL"]
    return psycopg2.connect(dsn)


def init_db(conn=None, dsn=None):
    if conn is None:
        conn = get_connection(dsn=dsn)
    with conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(ADD_SESSION_COLUMN_SQL)


def log_interaction(question, answer, metadata=None, session_id=None, conn=None, dsn=None):
    if conn is None:
        conn = get_connection(dsn=dsn)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_logs (question, answer, metadata, session_id) VALUES (%s, %s, %s, %s) RETURNING id",
                (question, answer, json.dumps(metadata) if metadata else None, session_id),
            )
            row = cur.fetchone()
            return row[0] if row else None


def get_session_history(session_id, limit=20, conn=None, dsn=None):
    """Return the stored conversation turns for a session as a list of
    ``{"role": "user"|"assistant", "content": str}`` dicts in chronological
    order. One ``rag_logs`` row maps to one user turn plus one assistant turn.

    The result is bounded to the last ``limit`` turns (default 20) and is
    ``[]`` for an unknown session.
    """
    if conn is None:
        conn = get_connection(dsn=dsn)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT question, answer FROM rag_logs "
                "WHERE session_id = %s "
                "ORDER BY created_at DESC, id DESC "
                "LIMIT %s",
                (session_id, limit),
            )
            rows = cur.fetchall()
    # ``rows`` are most-recent-first; reverse to chronological order.
    turns = []
    for question, answer in reversed(rows):
        turns.append({"role": "user", "content": question})
        turns.append({"role": "assistant", "content": answer})
    return turns[-limit:]


def update_feedback(interaction_id, feedback, conn=None, dsn=None):
    if conn is None:
        conn = get_connection(dsn=dsn)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE rag_logs SET feedback = %s, updated_at = NOW() WHERE id = %s",
                (feedback, interaction_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"No interaction found with id {interaction_id}")
