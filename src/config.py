from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    kafka_bootstrap_servers: str
    # Topic name defaults below are placeholders -- the exact contract with the
    # Scrapper Backend is not yet confirmed (spec section 9). Override via env
    # once confirmed; no other code changes should be needed.
    kafka_request_topic: str = "analytics.requests"
    kafka_response_topic: str = "analytics.results"
    kafka_consumer_group: str = "analytics-backend"

    llm_base_url: str
    llm_model: str
    vlm_base_url: str | None = None
    vlm_model: str | None = None

    vlm_model_id: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    whisper_model_id: str = "mlx-community/whisper-large-v3-turbo"
    max_frames: int = 32
    max_tokens: int = 500

    http_timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0
