import time
from dataclasses import dataclass

import httpx


class VLMClientError(Exception):
    pass


@dataclass
class VLMConfig:
    base_url: str
    model: str
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0


def describe_images(config: VLMConfig, prompt: str, image_data_urls: list[str]) -> str:
    """
    Sends a list of image data URLs and a prompt to a remote VLM endpoint
    (OpenAI-compatible vision chat completion) and returns the response text.
    """
    last_error: Exception | None = None
    for attempt in range(1, config.retry_attempts + 1):
        try:
            content: list[dict[str, str | dict[str, str]]] = [{"type": "text", "text": prompt}]
            for image_url in image_data_urls:
                content.append({"type": "image_url", "image_url": {"url": image_url}})

            response = httpx.post(
                f"{config.base_url}/chat/completions",
                json={
                    "model": config.model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.0,
                },
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError) as e:
            last_error = e
            if attempt < config.retry_attempts:
                time.sleep(config.retry_backoff_seconds * attempt)
    raise VLMClientError(f"VLM request failed after {config.retry_attempts} attempts: {last_error}")
