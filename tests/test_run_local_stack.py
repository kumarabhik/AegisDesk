"""Tests for the one-command local launcher helper."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import run_local_stack


def test_run_local_stack_uses_existing_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_local_stack, "_health_ready", lambda base_url: True)
    monkeypatch.setattr(
        run_local_stack,
        "verify_space",
        lambda base_url, task_id, seed: {
            "base_url": base_url,
            "task_id": task_id,
            "seed": seed,
        },
    )

    summary = run_local_stack.run_local_stack()

    assert summary["server_started"] is False
    assert summary["server_stopped"] is False
    assert summary["verification"]["task_id"] == "billing_seat_adjustment"


def test_run_local_stack_starts_and_stops_server(monkeypatch: pytest.MonkeyPatch) -> None:
    health_checks = iter([False, True])

    monkeypatch.setattr(run_local_stack, "_health_ready", lambda base_url: next(health_checks))
    monkeypatch.setattr(run_local_stack, "verify_space", lambda base_url, task_id, seed: {"ok": True})
    monkeypatch.setattr(run_local_stack.time, "sleep", lambda _: None)

    fake_process = SimpleNamespace(
        pid=4321,
        terminate=lambda: None,
        wait=lambda timeout=10: 0,
        kill=lambda: None,
    )
    monkeypatch.setattr(run_local_stack.subprocess, "Popen", lambda *args, **kwargs: fake_process)

    summary = run_local_stack.run_local_stack(startup_timeout=2)

    assert summary["server_started"] is True
    assert summary["server_stopped"] is True


def test_main_prints_json_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        run_local_stack,
        "run_local_stack",
        lambda **kwargs: {"base_url": kwargs["base_url"], "server_started": False},
    )
    monkeypatch.setattr(
        run_local_stack.sys,
        "argv",
        ["run_local_stack.py", "--base-url", "http://127.0.0.1:9000"],
    )

    exit_code = run_local_stack.main()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_url"] == "http://127.0.0.1:9000"

