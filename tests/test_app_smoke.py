"""FastAPI smoke tests for support_ops_env."""

from fastapi.testclient import TestClient

from server.app import app


def test_root_healthcheck_returns_200() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["env_name"] == "support_ops_env"


def test_reset_endpoint_returns_step_payload_shape() -> None:
    client = TestClient(app)
    response = client.post("/reset", json={"task_id": "billing_seat_adjustment", "seed": 1})
    payload = response.json()
    assert response.status_code == 200
    assert sorted(payload.keys()) == ["done", "info", "observation", "reward"]
    assert payload["observation"]["task_brief"]
