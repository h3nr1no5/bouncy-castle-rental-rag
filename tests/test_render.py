from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENDER_YAML = PROJECT_ROOT / "render.yaml"
GRAFANA_DOCKERFILE = PROJECT_ROOT / "grafana" / "Dockerfile"


@pytest.fixture(scope="module")
def render():
    return yaml.safe_load(RENDER_YAML.read_text())


def _service(render, name):
    for service in render["services"]:
        if service["name"] == name:
            return service
    raise AssertionError(f"Service not found: {name}")


class TestRenderBlueprint:
    def test_render_yaml_exists(self):
        assert RENDER_YAML.is_file()

    def test_has_two_docker_web_services(self, render):
        services = render["services"]
        assert len(services) == 2
        for service in services:
            assert service["type"] == "web"
            assert service["runtime"] == "docker"
            assert service["plan"] == "free"
            assert service["region"] == "oregon"

    def test_app_service_points_at_root_dockerfile(self, render):
        app = _service(render, "bouncy-castle-rag")
        assert app["dockerfilePath"] == "./Dockerfile"
        assert app["healthCheckPath"] == "/health"

    def test_app_has_no_openai_key(self, render):
        app = _service(render, "bouncy-castle-rag")
        keys = {env["key"] for env in app["envVars"]}
        assert "OPENAI_API_KEY" not in keys

    def test_app_secrets_are_sync_false(self, render):
        app = _service(render, "bouncy-castle-rag")
        env = {e["key"]: e for e in app["envVars"]}
        assert env["GROQ_API_KEY"]["sync"] is False
        assert env["DATABASE_URL"]["sync"] is False

    def test_app_sets_port(self, render):
        app = _service(render, "bouncy-castle-rag")
        env = {e["key"]: e for e in app["envVars"]}
        assert env["PORT"]["value"] == 8000

    def test_grafana_service_uses_grafana_dockerfile(self, render):
        grafana = _service(render, "bouncy-castle-rag-grafana")
        assert grafana["dockerfilePath"] == "./grafana/Dockerfile"

    def test_grafana_listens_on_3000(self, render):
        grafana = _service(render, "bouncy-castle-rag-grafana")
        env = {e["key"]: e for e in grafana["envVars"]}
        assert env["PORT"]["value"] == 3000
        assert env["GF_SERVER_HTTP_PORT"]["value"] == 3000

    def test_grafana_admin_password_is_sync_false(self, render):
        grafana = _service(render, "bouncy-castle-rag-grafana")
        env = {e["key"]: e for e in grafana["envVars"]}
        assert env["GF_SECURITY_ADMIN_USER"]["value"] == "admin"
        assert env["GF_SECURITY_ADMIN_PASSWORD"]["sync"] is False

    def test_grafana_postgres_vars(self, render):
        grafana = _service(render, "bouncy-castle-rag-grafana")
        env = {e["key"]: e for e in grafana["envVars"]}
        assert env["POSTGRES_DB"]["value"] == "neondb"
        assert env["POSTGRES_PORT"]["value"] == 5432
        assert env["POSTGRES_SSLMODE"]["value"] == "require"
        for secret in ("POSTGRES_HOST", "POSTGRES_USER", "POSTGRES_PASSWORD"):
            assert env[secret]["sync"] is False


class TestGrafanaDockerfile:
    def test_file_exists(self):
        assert GRAFANA_DOCKERFILE.is_file()

    def test_bases_off_grafana_image(self):
        content = GRAFANA_DOCKERFILE.read_text()
        assert "FROM grafana/grafana:10.4.3" in content

    def test_copies_provisioning_and_dashboard(self):
        content = GRAFANA_DOCKERFILE.read_text()
        assert "grafana/provisioning" in content
        assert "/etc/grafana/provisioning" in content
        assert "grafana/dashboard.json" in content
        assert "/var/lib/grafana/dashboards/dashboard.json" in content

    def test_exposes_3000(self):
        content = GRAFANA_DOCKERFILE.read_text()
        assert "EXPOSE 3000" in content
