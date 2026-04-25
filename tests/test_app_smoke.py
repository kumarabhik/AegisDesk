"""FastAPI smoke tests for support_ops_env."""

from fastapi.testclient import TestClient

from server.app import app


def test_root_healthcheck_returns_200() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["env_name"] == "support_ops_env"


def test_root_returns_html_for_browser_accept_header() -> None:
    client = TestClient(app)
    response = client.get("/", headers={"accept": "text/html"})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AegisDesk" in response.text


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
    assert payload["observation"]["reply_requirements"]["template_id"] == "billing_credit_resolution"
    assert payload["observation"]["reply_requirements"]["checklist"]


def test_reset_endpoint_accepts_fixture_id_for_variant() -> None:
    client = TestClient(app)
    response = client.post("/reset", json={"fixture_id": "billing_seat_adjustment_v1", "seed": 1})
    payload = response.json()
    assert response.status_code == 200
    assert payload["observation"]["fixture_id"] == "billing_seat_adjustment_v1"
    assert payload["observation"]["task_id"] == "billing_seat_adjustment"


def test_tasks_endpoint_returns_catalog() -> None:
    client = TestClient(app)
    response = client.get("/tasks")
    payload = response.json()
    assert response.status_code == 200
    assert "tasks" in payload
    assert len(payload["tasks"]) == 30
    assert any(task["task_id"] == "billing_seat_adjustment" for task in payload["tasks"])
    assert payload["tasks"][0]["fixture_id"] == "billing_seat_adjustment"
    assert sum(1 for task in payload["tasks"] if task["track"] == "core") == 3
    assert sum(1 for task in payload["tasks"] if task["track"] == "v2") == 6
    assert sum(1 for task in payload["tasks"] if task["track"] == "generalization") == 18
    assert sum(1 for task in payload["tasks"] if task["track"] == "showcase") == 3
    assert sum(1 for task in payload["tasks"] if task["judged"]) == 27
    assert len({task["fixture_id"] for task in payload["tasks"]}) == 30
    assert all(task["oracle_available"] is True for task in payload["tasks"])


def test_console_endpoint_returns_html() -> None:
    client = TestClient(app)
    response = client.get("/console")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AegisDesk Console" in response.text


def test_benchmark_card_endpoint_returns_summary() -> None:
    client = TestClient(app)
    response = client.get("/benchmark-card")
    payload = response.json()
    assert response.status_code == 200
    assert payload["name"] == "AegisDesk"
    assert payload["task_counts"] == {
        "core": 3,
        "v2": 6,
        "generalization": 18,
        "showcase": 3,
        "judged_total": 27,
        "surfaced_total": 30,
    }
    assert payload["routes"]["console"] == "/console"


def test_home_endpoint_returns_html() -> None:
    client = TestClient(app)
    response = client.get("/home")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Why This Project Stands Out" in response.text
    assert "30 surfaced fixtures" in response.text


def test_trajectory_report_endpoint_returns_scored_report() -> None:
    client = TestClient(app)
    response = client.get("/trajectory-report", params={"task_id": "billing_seat_adjustment", "seed": 7})
    payload = response.json()
    assert response.status_code == 200
    assert payload["fixture_id"] == "billing_seat_adjustment"
    assert payload["task_id"] == "billing_seat_adjustment"
    assert payload["track"] == "core"
    assert payload["final_score"] >= 0.95
    assert payload["step_count"] >= 1


def test_trajectory_report_endpoint_returns_v2_report() -> None:
    client = TestClient(app)
    response = client.get("/trajectory-report", params={"task_id": "api_partner_access_audit", "seed": 7})
    payload = response.json()
    assert response.status_code == 200
    assert payload["fixture_id"] == "api_partner_access_audit"
    assert payload["task_id"] == "api_partner_access_audit"
    assert payload["track"] == "v2"
    assert payload["final_score"] >= 0.95


def test_trajectory_report_endpoint_accepts_fixture_id_for_generalization_case() -> None:
    client = TestClient(app)
    response = client.get(
        "/trajectory-report",
        params={"fixture_id": "billing_seat_adjustment_v1", "seed": 7},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["fixture_id"] == "billing_seat_adjustment_v1"
    assert payload["task_id"] == "billing_seat_adjustment"
    assert payload["track"] == "generalization"
    assert payload["final_score"] >= 0.95


def test_trajectory_viewer_endpoint_returns_html() -> None:
    client = TestClient(app)
    response = client.get("/trajectory-viewer")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AegisDesk Trajectory Viewer" in response.text
