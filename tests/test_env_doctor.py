"""Tests for the local environment doctor helper."""

from __future__ import annotations

import json

import pytest

import env_doctor


def test_inspect_environment_reports_hf_mode_ready() -> None:
    report = env_doctor.inspect_environment(
        {
            "HF_TOKEN": "set",
            "MODEL_NAME": "Qwen/Qwen2.5-7B-Instruct-1M",
            "API_BASE_URL": "https://router.huggingface.co/v1",
        }
    )

    assert report["hf_router_mode"]["ready"] is True
    assert report["hf_router_mode"]["missing"] == []


def test_inspect_environment_reports_missing_values() -> None:
    report = env_doctor.inspect_environment({})

    assert report["hf_router_mode"]["ready"] is False
    assert "HF_TOKEN" in report["hf_router_mode"]["missing"]
    assert report["openai_compatible_mode"]["ready"] is False


def test_main_prints_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(env_doctor, "inspect_environment", lambda env=None: {"ok": True})

    exit_code = env_doctor.main()

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}
