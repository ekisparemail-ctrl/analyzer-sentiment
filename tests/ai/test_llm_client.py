import httpx
import pytest
import respx

from ai.llm_client import LLMClientError, LLMConfig, analyze_sentiment, summarize_video


def _config(**overrides: object) -> LLMConfig:
    defaults = dict(
        base_url="http://localhost:1234/v1",
        model="test-model",
        timeout_seconds=5.0,
        retry_attempts=3,
        retry_backoff_seconds=0.0,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)  # type: ignore[arg-type]


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@respx.mock
def test_summarize_video_sends_prompt_and_returns_text() -> None:
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("combined summary"))
    )

    result = summarize_video(_config(), "a person is talking", "hello everyone")

    assert result == "combined summary"
    assert route.called
    sent_body = route.calls[0].request.content
    assert b"a person is talking" in sent_body
    assert b"hello everyone" in sent_body


@respx.mock
def test_summarize_video_retries_then_succeeds() -> None:
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json=_chat_response("combined summary")),
        ]
    )

    result = summarize_video(_config(), "a person is talking", None)

    assert result == "combined summary"
    assert route.call_count == 2


@respx.mock
def test_summarize_video_raises_after_exhausting_retries() -> None:
    route = respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(500)
    )

    with pytest.raises(LLMClientError):
        summarize_video(_config(retry_attempts=2), "a person is talking", None)

    assert route.call_count == 2


@respx.mock
def test_analyze_sentiment_parses_structured_json_response() -> None:
    payload = {
        "context": {"topic": "politics", "key_themes": ["korupsi"]},
        "sentiment": {"label": "negative", "score": -0.8, "indicators": ["marah"]},
        "emotion": {
            "primary": "anger",
            "secondary": "disgust",
            "intensity": 0.9,
            "indicators": ["MAMPUS"],
        },
        "motivation": {"type": "criticizing", "confidence_score": 0.7, "indicators": ["skandal"]},
    }
    import json

    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response(json.dumps(payload)))
    )

    result = analyze_sentiment(_config(), "some social media content")

    assert result.context.topic == "politics"
    assert result.sentiment.label == "negative"
    assert result.emotion.primary == "anger"
    assert result.motivation.type == "criticizing"


@respx.mock
def test_analyze_sentiment_raises_llm_client_error_on_malformed_json() -> None:
    respx.post("http://localhost:1234/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("not json"))
    )

    with pytest.raises(LLMClientError):
        analyze_sentiment(_config(), "some social media content")
