"""FastAPI smoke tests for support_ops_env."""

from fastapi.testclient import TestClient

from server.app import app


def test_root_healthcheck_returns_200() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["env_name"] == "support_ops_env"


def test_health_endpoint_returns_200() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "healthy"}


def test_reset_endpoint_returns_step_payload_shape() -> None:
    client = TestClient(app)
    response = client.post("/reset", json={"task_id": "billing_seat_adjustment", "seed": 1})
    payload = response.json()
    assert response.status_code == 200
    assert sorted(payload.keys()) == ["done", "info", "observation", "reward"]
    assert payload["observation"]["task_brief"]


def test_tasks_endpoint_returns_catalog() -> None:
    client = TestClient(app)
    response = client.get("/tasks")
    payload = response.json()
    assert response.status_code == 200
    assert "tasks" in payload
    assert len(payload["tasks"]) >= 3
    assert any(task["task_id"] == "billing_seat_adjustment" for task in payload["tasks"])


def test_console_endpoint_returns_html() -> None:
    client = TestClient(app)
    response = client.get("/console")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AegisDesk Console" in response.text
