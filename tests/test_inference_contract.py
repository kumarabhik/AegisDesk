"""Inference utility tests."""

import pytest

from inference import parse_model_action, resolve_inference_config


def test_parse_model_action_reads_plain_json() -> None:
    action = parse_model_action('{"action_type":"search_kb","query":"billing policy"}')
    assert action.action_type.value == "search_kb"
    assert action.query == "billing policy"


def test_parse_model_action_reads_embedded_json() -> None:
    action = parse_model_action(
        'Use this action: {"action_type":"open_ticket","ticket_id":"TICKET-1001"}'
    )
    assert action.action_type.value == "open_ticket"
    assert action.ticket_id == "TICKET-1001"


def test_resolve_inference_config_prefers_submission_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "submission-key")
    monkeypatch.setenv("MODEL_NAME", "submission-model")
    monkeypatch.setenv("API_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GROQ_MODEL", "groq-model")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    config = resolve_inference_config()

    assert config.api_key == "submission-key"
    assert config.model_name == "submission-model"
    assert config.api_base_url == "https://example.com/v1"
    assert config.credential_source == "OPENAI_API_KEY"


def test_resolve_inference_config_accepts_groq_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GROQ_MODEL", "llama3-8b-8192")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    config = resolve_inference_config()

    assert config.api_key == "groq-key"
    assert config.model_name == "llama3-8b-8192"
    assert config.api_base_url == "https://api.groq.com/openai/v1"
    assert config.credential_source == "GROQ_API_KEY"


def test_resolve_inference_config_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        resolve_inference_config()
