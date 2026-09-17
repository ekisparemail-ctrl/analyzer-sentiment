from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")

    kafka_bootstrap_servers: str
    # Confirmed against the Scrapper Backend's actual source (application.yml,
    # KafkaProducerService) -- the real topic name it publishes to (merged
    # from separate post/comment topics into one), not a placeholder.
    kafka_scrapper_topic: str = "scrapper-to-analysis"
    # Name finalized on our side (ours to name, mirroring the Scrapper
    # Backend's own topic-naming convention) -- still pending devops
    # provisioning it on the broker, and the Scrapper Backend still has no
    # consumer for it yet (spec section 9). Override via env if the name
    # ever changes; no other code changes should be needed.
    kafka_result_topic: str = "analysis-to-scrapper"
    kafka_consumer_group: str = "analytics-backend"

    llm_base_url: str
    llm_model: str

    # Remote HTTP VLM again (revision 1) -- the same LM Studio instance
    # already serving llm_base_url/llm_model, but kept as its own config
    # entry so the two can diverge later without a code change.
    vlm_base_url: str
    vlm_model: str

    whisper_model_size: str = "base"
    # Filters out non-speech audio (e.g. music-only segments) before
    # transcription -- faster-whisper is prone to hallucinating text over
    # such segments otherwise. Configurable in case it turns out too
    # aggressive for a given batch of content (e.g. drops quiet speech).
    whisper_vad_filter: bool = True
    # Every frame is encoded by the VLM's vision tower before generation can
    # even start, so this is the main lever over VLM response time on
    # CPU-only inference -- a real ~30s TikTok clip at the old default (32)
    # took over 20 minutes end to end. Lower = faster but coarser video
    # understanding (fewer sampled moments); tune per how much of that
    # trade-off is acceptable once real latency is measured (spec section 9).
    max_frames: int = 8
    # Grayscale mean-absolute-difference a candidate frame must exceed vs.
    # the previous *kept* candidate to be considered a keyframe.
    # video-analyzer hardcodes this at 10.0 and never actually reads its own
    # analysis_threshold/min_difference config keys -- keeping this
    # configurable avoids repeating that bug.
    keyframe_diff_threshold: float = 10.0
    max_tokens: int = 500

    # Adds `-hwaccel cuda -hwaccel_output_format cuda` to the ffmpeg decode
    # invocation (video/frames.py) when true. Off by default -- this
    # deployment has no NVIDIA GPU to test it against; exists for whichever
    # machine runs this with one.
    ffmpeg_hwaccel_cuda: bool = False

    # When true, keyframes are grouped into batches of vlm_batch_size and
    # sent as separate multi-image VLM calls; when false, every keyframe
    # goes into its own call instead.
    vlm_batch_enabled: bool = True
    # Keyframes per batch when vlm_batch_enabled is true; ignored otherwise.
    # Starting default balances per-call payload size against round trips to
    # the Mac Studio -- not measured yet.
    vlm_batch_size: int = 4

    http_timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0

    # Dev-only: also write each processed item's final AnalysisResult to a
    # local JSON file, in addition to publishing it to Kafka as before.
    # Defaults on per the directive's explicit ask; reconsider before a real
    # production rollout (unbounded local disk growth, one file per item
    # forever).
    dev_dump_output: bool = True
