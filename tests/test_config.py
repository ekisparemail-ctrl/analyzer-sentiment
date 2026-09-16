from pathlib import Path

import pytest
from pydantic import ValidationError

from config import Settings


def test_settings_loads_required_and_applies_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "172.16.16.100:21000")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    # _env_file=None: ignore any real .env file that happens to exist in the
    # cwd this test runs from -- this test is about env-var loading, not
    # .env-file loading (see test_settings_loads_from_dotenv_file_when_present).
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.kafka_bootstrap_servers == "172.16.16.100:21000"
    assert settings.llm_base_url == "http://localhost:1234/v1"
    assert settings.llm_model == "test-model"
    assert settings.kafka_scrapper_topic == "scrapper-to-analysis"
    assert settings.kafka_result_topic == "analysis-to-scrapper"
    assert settings.kafka_consumer_group == "analytics-backend"
    assert settings.vlm_model_id == "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    assert settings.whisper_model_size == "base"
    assert settings.whisper_vad_filter is True
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
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_loads_from_dotenv_file_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression test: Settings() used to only read from the process
    # environment, never from a .env file -- so `python src/main.py` next to
    # a real .env (copied from .env.example, exactly as docs/testing-
    # guidelines.md instructs) crashed with "Field required" for every
    # required setting, even though the .env file had them filled in.
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "KAFKA_BOOTSTRAP_SERVERS=172.16.16.100:21000\n"
        "LLM_BASE_URL=http://localhost:1234/v1\n"
        "LLM_MODEL=test-model\n"
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.kafka_bootstrap_servers == "172.16.16.100:21000"
    assert settings.llm_base_url == "http://localhost:1234/v1"
    assert settings.llm_model == "test-model"
