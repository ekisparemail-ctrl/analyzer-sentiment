import httpx
import pytest
import respx

from ai.vlm_client import VLMClientError, VLMConfig, describe_images


def _config(**overrides: object) -> VLMConfig:
    defaults = dict(
        base_url="http://localhost:8001/v1",
        model="test-vlm-model",
        timeout_seconds=5.0,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
    )
    defaults.update(overrides)
    return VLMConfig(**defaults)  # type: ignore[arg-type]


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@respx.mock
def test_describe_images_sends_prompt_and_images_and_returns_text() -> None:
    route = respx.post("http://localhost:8001/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("a meme about politics"))
    )

    result = describe_images(_config(), "Describe these images.", ["data:image/png;base64,AAA"])

    assert result == "a meme about politics"
    sent_body = route.calls[0].request.content
    assert b"data:image/png;base64,AAA" in sent_body


@respx.mock
def test_describe_images_retries_then_succeeds() -> None:
    route = respx.post("http://localhost:8001/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json=_chat_response("a meme about politics")),
        ]
    )

    result = describe_images(_config(), "Describe these images.", ["data:image/png;base64,AAA"])

    assert result == "a meme about politics"
    assert route.call_count == 2


@respx.mock
def test_describe_images_raises_after_exhausting_retries() -> None:
    route = respx.post("http://localhost:8001/v1/chat/completions").mock(
        return_value=httpx.Response(500)
    )

    with pytest.raises(VLMClientError):
        describe_images(_config(), "Describe these images.", ["data:image/png;base64,AAA"])

    assert route.call_count == 2
