import pytest
from pydantic import ValidationError

from config import Settings


def test_settings_loads_required_and_applies_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "172.16.16.100:21000")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    settings = Settings()

    assert settings.kafka_bootstrap_servers == "172.16.16.100:21000"
    assert settings.llm_base_url == "http://localhost:1234/v1"
    assert settings.llm_model == "test-model"
    assert settings.kafka_post_topic == "post-scrapper-to-analysis"
    assert settings.kafka_comment_topic == "comment-scrapper-to-analysis"
    assert settings.kafka_result_topic == "analysis-to-scrapper"
    assert settings.kafka_consumer_group == "analytics-backend"
    assert settings.vlm_base_url is None
    assert settings.whisper_model_size == "base"
    assert settings.max_frames == 32
    assert settings.max_tokens == 500
    assert settings.http_timeout_seconds == 30.0
    assert settings.retry_attempts == 3
    assert settings.retry_backoff_seconds == 1.0


def test_settings_missing_required_field_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(ValidationError):
        Settings()
