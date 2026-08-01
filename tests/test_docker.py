import os
import tomllib
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCKERFILE = PROJECT_ROOT / "Dockerfile"
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yaml"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ENTRYPOINT = PROJECT_ROOT / "docker-entrypoint.sh"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
UV_LOCK = PROJECT_ROOT / "uv.lock"

REQUIRED_INDEX_FILES = ["bm25_index.pkl", "faiss_index.bin", "ingest_docs.json"]


@pytest.fixture(scope="module")
def dockerfile():
    return DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(COMPOSE_FILE.read_text())


@pytest.fixture(scope="module")
def pyproject():
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def lock():
    return UV_LOCK.read_text()


class TestDockerfile:
    def test_file_exists(self):
        assert DOCKERFILE.is_file()

    def test_multi_stage_build(self, dockerfile):
        stages = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
        assert len(stages) >= 2, "Expected a multi-stage build with at least 2 FROM stages"
        assert any("AS builder" in s for s in stages)
        assert any("AS runtime" in s for s in stages)

    def test_builder_uses_uv_sync_frozen_no_dev(self, dockerfile):
        assert "uv sync" in dockerfile
        sync_line = next(line for line in dockerfile.splitlines() if "uv sync" in line)
        assert "--frozen" in sync_line
        assert "--no-dev" in sync_line

    def test_embeddings_use_fastembed_not_torch(self, dockerfile, pyproject):
        assert "fastembed" in str(pyproject["project"]["dependencies"])
        assert "https://download.pytorch.org/whl/cpu" not in dockerfile
        assert "torch" not in str(pyproject["project"]["dependencies"])

    def test_fastembed_is_a_direct_dependency(self, pyproject):
        deps = pyproject["project"]["dependencies"]
        assert any(dep.startswith("fastembed") for dep in deps)

    def test_no_pytorch_cpu_index_in_pyproject(self, pyproject):
        indexes = pyproject.get("tool", {}).get("uv", {}).get("index", [])
        assert all(
            i.get("url") != "https://download.pytorch.org/whl/cpu" for i in indexes
        )

    def test_lock_has_fastembed_and_no_torch(self, lock):
        blocks = lock.split("[[package]]")
        assert any(b.lstrip().startswith('name = "fastembed"') for b in blocks)
        assert not any(b.lstrip().startswith('name = "torch"') for b in blocks)

    def test_runtime_base_is_python_311_slim(self, dockerfile):
        last_stage = [line for line in dockerfile.splitlines() if line.startswith("FROM ")][-1]
        assert "python:3.11-slim" in last_stage

    def test_runs_as_non_root_app_user(self, dockerfile):
        assert "USER app" in dockerfile
        useradd_line = next(line for line in dockerfile.splitlines() if "useradd" in line)
        assert "--uid 1000" in useradd_line

    def test_sets_python_env_vars(self, dockerfile):
        assert "PYTHONUNBUFFERED=1" in dockerfile
        assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile

    def test_healthcheck_uses_urllib_with_start_period(self, dockerfile):
        healthcheck_line = next(line for line in dockerfile.splitlines() if line.startswith("HEALTHCHECK"))
        assert "--start-period" in healthcheck_line
        assert "urllib" in dockerfile

    def test_healthcheck_targets_health_endpoint(self, dockerfile):
        assert "/health" in dockerfile

    def test_prebakes_indexes_at_build_time(self, dockerfile):
        assert "HF_HOME=/app/.cache" in dockerfile
        assert "FASTEMBED_CACHE_PATH=/app/.cache/fastembed" in dockerfile
        assert "build_indexes" in dockerfile

    def test_copies_tuned_params_into_runtime(self, dockerfile):
        assert "tuned_params.json" in dockerfile
        copy_line = next(line for line in dockerfile.splitlines() if "tuned_params.json" in line)
        assert copy_line.startswith("COPY")
        assert "--chown=app:app" in copy_line

    def test_index_build_happens_before_user_switch(self, dockerfile):
        lines = dockerfile.splitlines()
        build_line = next(i for i, line in enumerate(lines) if "build_indexes" in line)
        user_line = next(i for i, line in enumerate(lines) if line.startswith("USER app"))
        assert build_line < user_line

    def test_entrypoint_is_entry(self, dockerfile):
        entrypoint_line = next(line for line in dockerfile.splitlines() if line.startswith("ENTRYPOINT"))
        assert "docker-entrypoint.sh" in entrypoint_line


class TestEntrypoint:
    def test_file_exists(self):
        assert ENTRYPOINT.is_file()

    def test_is_executable(self):
        assert os.access(ENTRYPOINT, os.X_OK)

    def test_builds_indexes_when_missing(self):
        script = ENTRYPOINT.read_text()
        assert "build_indexes" in script
        assert "src.ingest" in script

    def test_checks_all_three_index_files(self):
        script = ENTRYPOINT.read_text()
        for name in REQUIRED_INDEX_FILES:
            assert name in script

    def test_execs_uvicorn_on_port(self):
        script = ENTRYPOINT.read_text()
        assert "uvicorn app:app" in script
        assert "--host 0.0.0.0" in script
        assert "--port" in script
        assert "PORT" in script
        assert "8000" in script


class TestComposeServices:
    def test_compose_file_exists(self):
        assert COMPOSE_FILE.is_file()

    def test_compose_parses(self, compose):
        assert isinstance(compose, dict)
        assert set(compose["services"]) == {"app", "postgres", "grafana"}

    def test_app_builds_from_context(self, compose):
        assert compose["services"]["app"]["build"] == "."

    def test_app_exposes_port(self, compose):
        port = compose["services"]["app"]["ports"][0]
        assert "APP_PORT" in port
        assert port.endswith(":8000")

    def test_app_gets_keys_from_env_file(self, compose):
        env_file = compose["services"]["app"]["env_file"]
        assert any(str(entry.get("path")) == ".env" for entry in env_file)

    def test_app_database_url_points_to_postgres_host(self, compose):
        url = compose["services"]["app"]["environment"]["DATABASE_URL"]
        assert "postgres:5432" in url
        assert "rag_logs" in url

    def test_app_mounts_appdb_on_db(self, compose):
        volume = compose["services"]["app"]["volumes"][0]
        assert volume == "appdb:/app/db"

    def test_app_depends_on_postgres_and_restarts(self, compose):
        app = compose["services"]["app"]
        assert "postgres" in app["depends_on"]
        assert app["restart"] == "unless-stopped"

    def test_postgres_uses_pg16_and_rag_logs(self, compose):
        pg = compose["services"]["postgres"]
        assert pg["image"] == "postgres:16"
        assert pg["environment"]["POSTGRES_USER"] == "postgres"
        assert pg["environment"]["POSTGRES_PASSWORD"] == "postgres"
        assert pg["environment"]["POSTGRES_DB"] == "rag_logs"

    def test_postgres_ports_and_volume(self, compose):
        pg = compose["services"]["postgres"]
        assert "5432:5432" in pg["ports"]
        assert "pgdata:/var/lib/postgresql/data" in pg["volumes"]

    def test_grafana_mounts_provisioning_read_only(self, compose):
        volumes = compose["services"]["grafana"]["volumes"]
        assert "./grafana/provisioning:/etc/grafana/provisioning:ro" in volumes
        assert "./grafana/dashboard.json:/var/lib/grafana/dashboards/dashboard.json:ro" in volumes

    def test_grafana_admin_credentials_via_env(self, compose):
        env = compose["services"]["grafana"]["environment"]
        assert "GF_SECURITY_ADMIN_USER" in env
        assert "GF_SECURITY_ADMIN_PASSWORD" in env

    def test_grafana_gets_postgres_connection_from_env(self, compose):
        env = compose["services"]["grafana"]["environment"]
        assert env["POSTGRES_HOST"] == "postgres"
        assert env["POSTGRES_PORT"] == 5432
        assert env["POSTGRES_DB"] == "rag_logs"
        assert env["POSTGRES_USER"] == "postgres"
        assert env["POSTGRES_PASSWORD"] == "postgres"
        assert env["POSTGRES_SSLMODE"] == "disable"

    def test_grafana_depends_on_postgres_and_port(self, compose):
        grafana = compose["services"]["grafana"]
        assert "3000:3000" in grafana["ports"]
        assert "postgres" in grafana["depends_on"]

    def test_named_volumes_declared(self, compose):
        assert "pgdata" in compose["volumes"]
        assert "appdb" in compose["volumes"]


class TestEnvExample:
    def test_lists_required_vars(self):
        content = ENV_EXAMPLE.read_text()
        for var in (
            "GROQ_API_KEY",
            "OPENAI_API_KEY",
            "DATABASE_URL",
            "DATABASE_URL_CLOUD",
            "GRAFANA_ADMIN_USER",
            "GRAFANA_ADMIN_PASSWORD",
        ):
            assert var in content


class TestDockerignore:
    def test_excludes_sensitive_and_build_artifacts(self):
        content = DOCKERIGNORE.read_text()
        for pattern in (".env", "db/", ".venv", ".git", "tests/", "docs/"):
            assert pattern in content
