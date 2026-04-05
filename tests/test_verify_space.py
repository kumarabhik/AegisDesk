"""Verification helper tests."""

from __future__ import annotations

import json

import pytest

import verify_space


def test_verify_space_success(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "/": {"status": "ok", "env_name": "support_ops_env"},
        "/reset": {
            "observation": {
                "inbox": [{"ticket_id": "TICKET-1001"}],
            }
        },
        "/step": {
            "done": False,
            "observation": {"active_ticket_id": "TICKET-1001"},
        },
        "/state": {
            "task_id": "billing_seat_adjustment",
            "selected_ticket_id": "TICKET-1001",
        },
    }

    def fake_request_json(
        base_url: str,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        assert base_url == "https://example.space"
        if path == "/reset":
            assert method == "POST"
            assert payload == {"task_id": "billing_seat_adjustment", "seed": 1}
        if path == "/step":
            assert method == "POST"
            assert payload == {"action_type": "open_ticket", "ticket_id": "TICKET-1001"}
        return responses[path]

    monkeypatch.setattr(verify_space, "_request_json", fake_request_json)

    summary = verify_space.verify_space(
        "https://example.space",
        "billing_seat_adjustment",
        1,
    )

    assert summary["env_name"] == "support_ops_env"
    assert summary["task_id"] == "billing_seat_adjustment"
    assert summary["selected_ticket_id"] == "TICKET-1001"


def test_verify_space_raises_on_missing_inbox(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "/": {"status": "ok", "env_name": "support_ops_env"},
        "/reset": {"observation": {"inbox": []}},
    }

    def fake_request_json(
        base_url: str,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return responses[path]

    monkeypatch.setattr(verify_space, "_request_json", fake_request_json)

    with pytest.raises(RuntimeError, match="Reset returned no inbox items"):
        verify_space.verify_space("https://example.space", "billing_seat_adjustment", 1)


def test_main_prints_json_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        verify_space,
        "verify_space",
        lambda base_url, task_id, seed: {
            "base_url": base_url,
            "task_id": task_id,
            "seed": seed,
        },
    )
    monkeypatch.setattr(
        verify_space.sys,
        "argv",
        ["verify_space.py", "--base-url", "https://example.space", "--task-id", "login_incident_triage", "--seed", "4"],
    )

    exit_code = verify_space.main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_url"] == "https://example.space"
    assert payload["task_id"] == "login_incident_triage"
    assert payload["seed"] == 4
