from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")

    kafka_bootstrap_servers: str
    # Confirmed against the Scrapper Backend's actual source (application.yml,
    # KafkaProducerService) -- these are the real topic names it publishes to,
    # not placeholders.
    kafka_post_topic: str = "post-scrapper-to-analysis"
    kafka_comment_topic: str = "comment-scrapper-to-analysis"
    # Still a placeholder: the Scrapper Backend has no Kafka consumer yet for
    # results coming back (spec section 9). Override via env once confirmed;
    # no other code changes should be needed.
    kafka_result_topic: str = "analysis-to-scrapper"
    kafka_consumer_group: str = "analytics-backend"

    llm_base_url: str
    llm_model: str
    vlm_base_url: str | None = None
    vlm_model: str | None = None

    whisper_model_size: str = "base"
    max_frames: int = 32
    max_tokens: int = 500

    http_timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0
