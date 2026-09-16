import io
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from ai.vlm_local import MAX_IMAGE_DIMENSION, LocalVLM, VLMLocalError, describe_images


def _fake_jpeg_bytes(size: tuple[int, int] = (2, 2)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color="red").save(buf, format="JPEG")
    return buf.getvalue()


def _fake_vlm(generated_text: str = "a busy street scene") -> tuple[LocalVLM, MagicMock, MagicMock]:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT_TEXT"
    processor.return_value = {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
    processor.batch_decode.return_value = [generated_text]

    model = MagicMock()
    model.generate.return_value = torch.zeros((1, 8), dtype=torch.long)

    return LocalVLM(model=model, processor=processor), model, processor


def test_describe_images_returns_generated_text() -> None:
    vlm, _model, _processor = _fake_vlm("a meme about politics")
    frames = [_fake_jpeg_bytes(), _fake_jpeg_bytes()]

    result = describe_images(vlm, "Describe these images.", frames, max_tokens=100)

    assert result == "a meme about politics"


def test_describe_images_builds_one_image_placeholder_per_frame() -> None:
    vlm, _model, processor = _fake_vlm()
    frames = [_fake_jpeg_bytes(), _fake_jpeg_bytes(), _fake_jpeg_bytes()]

    describe_images(vlm, "Describe these images.", frames, max_tokens=100)

    conversation = processor.apply_chat_template.call_args[0][0]
    content = conversation[0]["content"]
    image_items = [c for c in content if c["type"] == "image"]
    text_items = [c for c in content if c["type"] == "text"]
    assert len(image_items) == 3
    assert text_items == [{"type": "text", "text": "Describe these images."}]


def test_describe_images_passes_max_tokens_to_generate() -> None:
    vlm, model, _processor = _fake_vlm()
    frames = [_fake_jpeg_bytes()]

    describe_images(vlm, "Describe these images.", frames, max_tokens=42)

    assert model.generate.call_args.kwargs["max_new_tokens"] == 42


def test_describe_images_strips_the_prompt_tokens_before_decoding() -> None:
    vlm, model, processor = _fake_vlm()
    model.generate.return_value = torch.arange(11).reshape(1, 11)
    frames = [_fake_jpeg_bytes()]

    describe_images(vlm, "Describe these images.", frames, max_tokens=100)

    decoded_ids = processor.batch_decode.call_args[0][0]
    assert decoded_ids.shape[1] == 8  # 11 total - 3 prompt tokens


def test_describe_images_downscales_large_frames_before_inference() -> None:
    # Regression test: a real ~720x1280 video frame, sent at full
    # resolution, made LLaVA-OneVision's anyres tiling produce 36304 tokens
    # for just 8 frames against a 32768-token context limit ("Running this
    # sequence through the model will result in indexing errors"). Frames
    # must be downscaled before reaching the processor, regardless of the
    # source video's actual resolution.
    vlm, _model, processor = _fake_vlm()
    frames = [_fake_jpeg_bytes(size=(1280, 720))]

    describe_images(vlm, "Describe these images.", frames, max_tokens=100)

    passed_images = processor.call_args.kwargs["images"]
    assert len(passed_images) == 1
    assert max(passed_images[0].size) <= MAX_IMAGE_DIMENSION


def test_describe_images_wraps_failures_in_vlm_local_error() -> None:
    vlm, model, _processor = _fake_vlm()
    model.generate.side_effect = RuntimeError("out of memory")
    frames = [_fake_jpeg_bytes()]

    with pytest.raises(VLMLocalError, match="out of memory"):
        describe_images(vlm, "Describe these images.", frames, max_tokens=100)
