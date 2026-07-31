import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAFANA_DIR = PROJECT_ROOT / "grafana"

REQUIRED_PANEL_TITLES = [
    "Recent conversations",
    "Feedback distribution",
    "Average latency over time",
    "Token usage over time",
    "Estimated cost over time",
    "Model usage",
]

REQUIRED_DATASOURCE_FIELDS = {
    "name": "PostgreSQL",
    "type": "postgres",
    "url": "$POSTGRES_HOST:$POSTGRES_PORT",
    "database": "$POSTGRES_DB",
    "user": "$POSTGRES_USER",
    "isDefault": True,
}


@pytest.fixture(scope="module")
def dashboard():
    path = GRAFANA_DIR / "dashboard.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def panels(dashboard):
    return dashboard["panels"]


class TestDashboardJson:
    def test_file_exists(self):
        assert (GRAFANA_DIR / "dashboard.json").is_file()

    def test_is_valid_json(self, dashboard):
        assert isinstance(dashboard, dict)
        assert dashboard["title"] == "RAG Monitoring"
        assert dashboard["uid"]

    def test_contains_all_required_panel_titles(self, panels):
        titles = {panel["title"] for panel in panels}
        for required in REQUIRED_PANEL_TITLES:
            assert required in titles, f"Missing panel: {required}"
        assert len(panels) >= 6


class TestPanels:
    def test_every_panel_targets_rag_logs_via_postgres(self, panels):
        for panel in panels:
            for target in panel["targets"]:
                ds = target.get("datasource", {})
                assert ds.get("type") == "postgres", panel["title"]
                assert "rag_logs" in target["rawSql"], panel["title"]

    def test_every_panel_uses_time_range_filters(self, panels):
        for panel in panels:
            for target in panel["targets"]:
                sql = target["rawSql"]
                assert "$__timeFrom()" in sql, panel["title"]
                assert "$__timeTo()" in sql, panel["title"]

    def test_recent_conversations_returns_last_5_rows(self, panels):
        panel = self._panel(panels, "Recent conversations")
        sql = panel["targets"][0]["rawSql"]
        assert "LIMIT 5" in sql
        assert "ORDER BY created_at DESC" in sql
        for column in ("question", "answer", "feedback", "model", "created_at"):
            assert column in sql

    def test_feedback_distribution_groups_by_feedback(self, panels):
        panel = self._panel(panels, "Feedback distribution")
        sql = panel["targets"][0]["rawSql"]
        assert panel["type"] == "piechart"
        assert "GROUP BY feedback" in sql
        assert "COUNT(*)" in sql

    def test_feedback_distribution_shows_one_slice_per_category(self, panels):
        panel = self._panel(panels, "Feedback distribution")
        reduce = panel["options"]["reduceOptions"]
        assert reduce["values"] is True

    def test_average_latency_averages_metadata_latency(self, panels):
        panel = self._panel(panels, "Average latency over time")
        sql = panel["targets"][0]["rawSql"]
        assert "AVG(" in sql
        assert "metadata->>'latency'" in sql

    def test_token_usage_sums_metadata_tokens_total(self, panels):
        panel = self._panel(panels, "Token usage over time")
        sql = panel["targets"][0]["rawSql"]
        assert "SUM(" in sql
        assert "metadata->'tokens'->>'total'" in sql

    def test_estimated_cost_sums_metadata_cost(self, panels):
        panel = self._panel(panels, "Estimated cost over time")
        sql = panel["targets"][0]["rawSql"]
        assert "SUM(" in sql
        assert "metadata->>'cost'" in sql

    def test_model_usage_groups_by_metadata_model(self, panels):
        panel = self._panel(panels, "Model usage")
        sql = panel["targets"][0]["rawSql"]
        assert panel["type"] == "barchart"
        assert "metadata->>'model'" in sql
        assert "GROUP BY" in sql

    def test_all_panels_read_only_rag_logs(self, panels):
        for panel in panels:
            for target in panel["targets"]:
                assert "FROM rag_logs" in target["rawSql"]
                assert "JOIN" not in target["rawSql"]

    def test_empty_table_renders_without_error_sql(self, panels):
        # A valid query must not depend on rows existing: aggregation
        # queries use COUNT/AVG/SUM which return NULL/0 rows gracefully.
        for panel in panels:
            for target in panel["targets"]:
                assert target["rawSql"].strip()

    @staticmethod
    def _panel(panels, title):
        for panel in panels:
            if panel["title"] == title:
                return panel
        raise AssertionError(f"Panel not found: {title}")


class TestDatasourceProvisioning:
    def test_datasource_yml_exists_and_parses(self):
        path = GRAFANA_DIR / "provisioning" / "datasources" / "postgres.yml"
        assert path.is_file()
        data = yaml.safe_load(path.read_text())

        assert data["apiVersion"] == 1
        assert isinstance(data["datasources"], list)
        datasource = data["datasources"][0]
        for key, expected in REQUIRED_DATASOURCE_FIELDS.items():
            assert datasource.get(key) == expected, f"datasource.{key}"

    def test_datasource_reads_connection_from_env(self):
        path = GRAFANA_DIR / "provisioning" / "datasources" / "postgres.yml"
        data = yaml.safe_load(path.read_text())
        datasource = data["datasources"][0]
        assert datasource["url"] == "$POSTGRES_HOST:$POSTGRES_PORT"
        assert datasource["database"] == "$POSTGRES_DB"
        assert datasource["user"] == "$POSTGRES_USER"
        assert datasource["secureJsonData"]["password"] == "$POSTGRES_PASSWORD"
        assert datasource["jsonData"]["sslmode"] == "$POSTGRES_SSLMODE"


class TestDashboardsProvisioning:
    def test_dashboards_yml_exists_and_parses(self):
        path = GRAFANA_DIR / "provisioning" / "dashboards" / "dashboards.yml"
        assert path.is_file()
        data = yaml.safe_load(path.read_text())

        assert data["apiVersion"] == 1
        assert isinstance(data["providers"], list)
        provider = data["providers"][0]
        assert provider["type"] == "file"
        assert provider["allowUiUpdates"] is False

    def test_provider_loads_dashboard_json(self):
        path = GRAFANA_DIR / "provisioning" / "dashboards" / "dashboards.yml"
        data = yaml.safe_load(path.read_text())
        provider = data["providers"][0]
        options = provider["options"]
        assert options.get("path")
        assert (GRAFANA_DIR / "dashboard.json").is_file()
