"""Inference utility tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference import (
    BENCHMARK,
    DEFAULT_MODEL_NAME,
    HF_ROUTER_BASE_URL,
    SupportAction,
    emit_end_log,
    emit_start_log,
    emit_step_log,
    format_action_str,
    format_bool,
    format_error,
    format_reward,
    format_rewards,
    main,
    parse_model_action,
    resolve_inference_config,
    run_task,
)


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


def test_resolve_inference_config_prefers_hf_router_submission_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1:fastest")
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "submission-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GROQ_MODEL", "groq-model")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    config = resolve_inference_config()

    assert config.api_key == "hf-token"
    assert config.model_name == "deepseek-ai/DeepSeek-R1:fastest"
    assert config.api_base_url == HF_ROUTER_BASE_URL
    assert config.credential_source == "HF_TOKEN"


def test_resolve_inference_config_defaults_model_name_for_hf_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("XAI_MODEL", raising=False)
    monkeypatch.delenv("GROK_MODEL", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    config = resolve_inference_config()

    assert config.api_key == "hf-token"
    assert config.model_name == DEFAULT_MODEL_NAME
    assert config.api_base_url == HF_ROUTER_BASE_URL
    assert config.credential_source == "HF_TOKEN"


def test_resolve_inference_config_respects_explicit_api_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("MODEL_NAME", "submission-model")
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")

    config = resolve_inference_config()

    assert config.api_key == "hf-token"
    assert config.model_name == "submission-model"
    assert config.api_base_url == "https://router.huggingface.co/v1"
    assert config.credential_source == "HF_TOKEN"


def test_resolve_inference_config_accepts_openai_env_when_hf_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "submission-key")
    monkeypatch.setenv("MODEL_NAME", "submission-model")
    monkeypatch.setenv("API_BASE_URL", "https://example.com/v1")

    config = resolve_inference_config()

    assert config.api_key == "submission-key"
    assert config.model_name == "submission-model"
    assert config.api_base_url == "https://example.com/v1"
    assert config.credential_source == "OPENAI_API_KEY"


def test_resolve_inference_config_accepts_groq_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
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
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN or OPENAI_API_KEY is required"):
        resolve_inference_config()


def test_format_helpers_match_required_style() -> None:
    action = SupportAction(action_type="open_ticket", ticket_id="TICKET-1001")

    assert format_bool(True) == "true"
    assert format_bool(False) == "false"
    assert format_reward(1) == "1.00"
    assert format_rewards([0, 0.125, 1]) == "0.00,0.12,1.00"
    assert format_error(None) == "null"
    assert format_action_str(action) == '{"action_type":"open_ticket","ticket_id":"TICKET-1001"}'


def test_emit_helpers_print_flat_lines(capsys: pytest.CaptureFixture[str]) -> None:
    emit_start_log(
        task_id="billing_seat_adjustment",
        benchmark=BENCHMARK,
        model_name="Qwen/Qwen2.5-7B-Instruct-1M",
    )
    emit_step_log(
        step=1,
        action_str='{"action_type":"search_kb","query":"policy"}',
        reward=0.125,
        done=False,
        error=None,
    )
    emit_end_log(
        success=True,
        steps=2,
        score=0.25,
        rewards=[0.125, 0.25],
    )

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "[START] task=billing_seat_adjustment env=support_ops_env model=Qwen/Qwen2.5-7B-Instruct-1M"
    assert lines[1] == '[STEP] step=1 action={"action_type":"search_kb","query":"policy"} reward=0.12 done=false error=null'
    assert lines[2] == "[END] success=true steps=2 score=0.25 rewards=0.12,0.25"


def test_run_task_emits_required_line_types(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_observation = SimpleNamespace(
        task_brief="Test task",
        inbox=[
            SimpleNamespace(
                ticket_id="TICKET-1001",
                priority=SimpleNamespace(value="normal"),
                status=SimpleNamespace(value="open"),
                subject="Subject",
                tags=[],
            )
        ],
        active_ticket_id=None,
        available_record_ids=[],
        focus_panel=None,
        last_action_error=None,
    )
    fake_result = SimpleNamespace(observation=fake_observation, reward=0.125, done=True)
    fake_state = SimpleNamespace(final_score=0.5, rubric_progress=0.5)

    class FakeEnv:
        def reset(self, task_id=None, seed=None):
            return SimpleNamespace(observation=fake_observation)

        def step(self, action):
            return fake_result

        def state(self):
            return fake_state

        def close(self):
            return None

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"action_type":"open_ticket","ticket_id":"TICKET-1001"}'
                        )
                    )
                ]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("inference.make_environment", lambda: FakeEnv())
    monkeypatch.setattr("inference.make_openai_client", lambda: (FakeClient(), "test-model"))

    config = SimpleNamespace(api_base_url=HF_ROUTER_BASE_URL, model_name="test-model")
    score = run_task("billing_seat_adjustment", 1, config)

    assert score == 0.5
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "[START] task=billing_seat_adjustment env=support_ops_env model=test-model"
    assert lines[1] == '[STEP] step=1 action={"action_type":"open_ticket","ticket_id":"TICKET-1001"} reward=0.12 done=true error=null'
    assert lines[2] == "[END] success=true steps=1 score=0.50 rewards=0.12"


def test_main_emits_only_start_step_end_lines(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "inference.resolve_inference_config",
        lambda: SimpleNamespace(
            api_key="hf-token",
            model_name="test-model",
            api_base_url=HF_ROUTER_BASE_URL,
            credential_source="HF_TOKEN",
        ),
    )
    monkeypatch.setattr(
        "inference.run_task",
        lambda task_id, task_seed, config: 0.25,
    )

    main()

    assert capsys.readouterr().out == ""
