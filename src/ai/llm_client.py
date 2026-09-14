import json
import time
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from schemas import AnalysisContext, EmotionResult, MotivationResult, SentimentResult


class LLMClientError(Exception):
    pass


@dataclass
class LLMConfig:
    base_url: str
    model: str
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0


@dataclass
class SentimentAnalysis:
    context: AnalysisContext
    sentiment: SentimentResult
    emotion: EmotionResult
    motivation: MotivationResult


SENTIMENT_SYSTEM_PROMPT = """You are an expert AI agent specialized in contextual sentiment \
analysis for Indonesian social media content. Given the content below, respond with ONLY a \
JSON object of this exact shape, no other text:
{
  "context": {"topic": "string", "key_themes": ["string"]},
  "sentiment": {"label": "positive|negative|neutral", "score": number, \
"indicators": ["string"]},
  "emotion": {"primary": "string", "secondary": "string", "intensity": number, \
"indicators": ["string"]},
  "motivation": {"type": "string", "confidence_score": number, \
"indicators": ["string"]}
}"""


def _chat_completion(config: LLMConfig, messages: list[dict]) -> str:
    last_error: Exception | None = None
    for attempt in range(1, config.retry_attempts + 1):
        try:
            response = httpx.post(
                f"{config.base_url}/chat/completions",
                json={"model": config.model, "messages": messages, "temperature": 0.0},
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError) as e:
            last_error = e
            if attempt < config.retry_attempts:
                time.sleep(config.retry_backoff_seconds * attempt)
    raise LLMClientError(f"LLM request failed after {config.retry_attempts} attempts: {last_error}")


def summarize_video(config: LLMConfig, vlm_summary: str, transcript: str | None) -> str:
    parts = [f"Visual description: {vlm_summary}"]
    if transcript:
        parts.append(f"Spoken transcript: {transcript}")
    prompt = (
        "Combine the following visual description and spoken transcript of a short "
        "social media video into one coherent summary of what the video contains:\n\n"
        + "\n\n".join(parts)
    )
    return _chat_completion(config, [{"role": "user", "content": prompt}])


def analyze_sentiment(config: LLMConfig, content: str) -> SentimentAnalysis:
    raw = _chat_completion(
        config,
        [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    try:
        data = json.loads(raw)
        return SentimentAnalysis(
            context=AnalysisContext(**data["context"]),
            sentiment=SentimentResult(**data["sentiment"]),
            emotion=EmotionResult(**data["emotion"]),
            motivation=MotivationResult(**data["motivation"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as e:
        raise LLMClientError(f"Failed to parse sentiment analysis response: {e}") from e
