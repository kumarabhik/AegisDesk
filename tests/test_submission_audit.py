"""Submission audit tests."""

from __future__ import annotations

import json

import pytest

import submission_audit


def test_run_audit_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_command(command: list[str]) -> dict[str, object]:
        return {
            "command": command,
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "ok": True,
        }

    monkeypatch.setattr(submission_audit, "_run_command", fake_run_command)
    monkeypatch.setattr(
        submission_audit,
        "verify_space",
        lambda space_url, task_id, seed: {
            "base_url": space_url,
            "task_id": task_id,
            "selected_ticket_id": "TICKET-1001",
        },
    )

    audit = submission_audit.run_audit("https://example.space")

    assert audit["overall_ok"] is True
    assert audit["live_verify"]["ok"] is True
    assert audit["live_verify"]["result"]["base_url"] == "https://example.space"


def test_run_audit_reports_live_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_command(command: list[str]) -> dict[str, object]:
        return {
            "command": command,
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "ok": True,
        }

    monkeypatch.setattr(submission_audit, "_run_command", fake_run_command)

    def fake_verify_space(space_url: str, task_id: str, seed: int) -> dict[str, str]:
        raise RuntimeError("live failure")

    monkeypatch.setattr(submission_audit, "verify_space", fake_verify_space)

    audit = submission_audit.run_audit("https://example.space")

    assert audit["overall_ok"] is False
    assert audit["live_verify"]["ok"] is False
    assert audit["live_verify"]["result"]["error"] == "live failure"


def test_main_prints_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        submission_audit,
        "run_audit",
        lambda space_url: {
            "space_url": space_url,
            "overall_ok": True,
        },
    )
    monkeypatch.setattr(
        submission_audit.sys,
        "argv",
        ["submission_audit.py", "--space-url", "https://example.space"],
    )

    exit_code = submission_audit.main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["space_url"] == "https://example.space"
    assert payload["overall_ok"] is True
