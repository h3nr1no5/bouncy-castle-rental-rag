from unittest.mock import MagicMock, patch

import pytest

from src.db import get_session_history, init_db, log_interaction, update_feedback


@pytest.fixture
def mock_conn():
    return MagicMock()


class TestInitDb:
    def test_creates_table_and_adds_session_column(self, mock_conn):
        init_db(conn=mock_conn)
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert cursor.execute.call_count == 2
        create_sql = cursor.execute.call_args_list[0][0][0]
        alter_sql = cursor.execute.call_args_list[1][0][0]
        assert "CREATE TABLE IF NOT EXISTS rag_logs" in create_sql
        assert "ALTER TABLE rag_logs ADD COLUMN IF NOT EXISTS session_id TEXT" in alter_sql


class TestLogInteraction:
    def test_returns_id(self, mock_conn):
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (42,)
        result = log_interaction("What is the price?", "$100", conn=mock_conn)
        assert result == 42

    def test_inserts_question_and_answer(self, mock_conn):
        log_interaction("What is the price?", "$100", conn=mock_conn)
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        sql, params = cursor.execute.call_args[0]
        assert params[0] == "What is the price?"
        assert params[1] == "$100"

    def test_inserts_metadata_as_json(self, mock_conn):
        metadata = {"model": "gpt-5.4-mini", "tokens": 150}
        log_interaction("test", "answer", metadata=metadata, conn=mock_conn)
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        sql, params = cursor.execute.call_args[0]
        import json
        assert json.loads(params[2]) == metadata

    def test_metadata_is_none_by_default(self, mock_conn):
        log_interaction("test", "answer", conn=mock_conn)
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        sql, params = cursor.execute.call_args[0]
        assert params[2] is None

    def test_uses_connection_as_context_manager(self, mock_conn):
        log_interaction("test", "answer", conn=mock_conn)
        mock_conn.__enter__.assert_called_once()
        mock_conn.__exit__.assert_called_once()

    def test_creates_connection_when_not_provided(self):
        with patch("src.db.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn
            cursor = mock_conn.cursor.return_value.__enter__.return_value
            cursor.fetchone.return_value = (1,)
            log_interaction("test", "answer", dsn="postgres://localhost/test")
            mock_get_conn.assert_called_once_with(dsn="postgres://localhost/test")

    def test_inserts_session_id(self, mock_conn):
        log_interaction("test", "answer", session_id="abc-123", conn=mock_conn)
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        sql, params = cursor.execute.call_args[0]
        assert params[3] == "abc-123"

    def test_session_id_none_by_default(self, mock_conn):
        log_interaction("test", "answer", conn=mock_conn)
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        sql, params = cursor.execute.call_args[0]
        assert params[3] is None


class TestGetSessionHistory:
    def test_returns_turns_in_chronological_order(self, mock_conn):
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        # SQL orders most-recent-first; rows come back newest first.
        cursor.fetchall.return_value = [("Q2", "A2"), ("Q1", "A1")]
        turns = get_session_history("abc-123", conn=mock_conn)
        assert turns == [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]

    def test_filters_by_session_id(self, mock_conn):
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        get_session_history("abc-123", conn=mock_conn)
        cursor.execute.assert_called_once()
        sql, params = cursor.execute.call_args[0]
        assert "WHERE session_id = %s" in sql
        assert params[0] == "abc-123"

    def test_empty_for_unknown_session(self, mock_conn):
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        assert get_session_history("nope", conn=mock_conn) == []

    def test_bounded_to_last_20_turns(self, mock_conn):
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        # 15 rows would flatten to 30 turns; only the last 20 must be returned.
        cursor.fetchall.return_value = [(f"Q{i}", f"A{i}") for i in range(15, 0, -1)]
        turns = get_session_history("abc-123", conn=mock_conn)
        assert len(turns) == 20
        # Chronological: the first returned turn is the 6th stored row (Q6).
        assert turns[0] == {"role": "user", "content": "Q6"}
        assert turns[-1] == {"role": "assistant", "content": "A15"}

    def test_creates_connection_when_not_provided(self):
        with patch("src.db.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn
            cursor = mock_conn.cursor.return_value.__enter__.return_value
            cursor.fetchall.return_value = [("Q1", "A1")]
            get_session_history("abc-123", dsn="postgres://localhost/test")
            mock_get_conn.assert_called_once_with(dsn="postgres://localhost/test")


class TestUpdateFeedback:
    def test_updates_feedback(self, mock_conn):
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1
        update_feedback(42, "helpful", conn=mock_conn)
        cursor.execute.assert_called_once()
        sql, params = cursor.execute.call_args[0]
        assert params[0] == "helpful"
        assert params[1] == 42

    def test_uses_connection_as_context_manager(self, mock_conn):
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1
        update_feedback(42, "helpful", conn=mock_conn)
        mock_conn.__enter__.assert_called_once()
        mock_conn.__exit__.assert_called_once()

    def test_raises_when_id_not_found(self, mock_conn):
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 0
        with pytest.raises(ValueError, match="No interaction found with id 999"):
            update_feedback(999, "helpful", conn=mock_conn)

    def test_creates_connection_when_not_provided(self):
        with patch("src.db.get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn
            cursor = mock_conn.cursor.return_value.__enter__.return_value
            cursor.rowcount = 1
            update_feedback(1, "not helpful", dsn="postgres://localhost/test")
            mock_get_conn.assert_called_once_with(dsn="postgres://localhost/test")
