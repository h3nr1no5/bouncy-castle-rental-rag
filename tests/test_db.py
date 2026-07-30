from unittest.mock import MagicMock, patch

import pytest

from src.db import init_db, log_interaction, update_feedback


@pytest.fixture
def mock_conn():
    return MagicMock()


class TestInitDb:
    def test_creates_table(self, mock_conn):
        init_db(conn=mock_conn)
        mock_conn.cursor.return_value.__enter__.return_value.execute.assert_called_once()
        call_args = mock_conn.cursor.return_value.__enter__.return_value.execute.call_args
        assert "CREATE TABLE IF NOT EXISTS rag_logs" in call_args[0][0]


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
        metadata = {"model": "gpt-4", "tokens": 150}
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
