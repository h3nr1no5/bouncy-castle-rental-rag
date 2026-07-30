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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


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


def log_interaction(question, answer, metadata=None, conn=None, dsn=None):
    if conn is None:
        conn = get_connection(dsn=dsn)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_logs (question, answer, metadata) VALUES (%s, %s, %s) RETURNING id",
                (question, answer, json.dumps(metadata) if metadata else None),
            )
            row = cur.fetchone()
            return row[0] if row else None


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
