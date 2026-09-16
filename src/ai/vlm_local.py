import io
from dataclasses import dataclass
from typing import Any

from PIL import Image


class VLMLocalError(Exception):
    pass


# LLaVA-OneVision's "anyres" image processing tiles an image based on its
# resolution -- a full-resolution video frame (e.g. 720x1280+) can produce
# thousands of tokens per image, easily exceeding the model's context window
# with just a handful of frames (observed: 8 full-res frames -> 36304 tokens
# against a 32768 limit, "Running this sequence through the model will
# result in indexing errors"). Downscaling first keeps tiling, and therefore
# token count, bounded regardless of the source video's resolution.
MAX_IMAGE_DIMENSION = 448


@dataclass
class LocalVLM:
    model: Any
    processor: Any


def load_local_vlm(model_id: str) -> LocalVLM:
    """
    Loads the VLM and its processor once, in-process, for CPU-only inference
    (this host has no GPU) -- not exercised in unit tests (real model
    weights, slow to download/load; see plan Global Constraints).
    """
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
    return LocalVLM(model=model, processor=processor)


def describe_images(vlm: LocalVLM, prompt: str, frames: list[bytes], max_tokens: int) -> str:
    """
    Runs one multi-image chat completion locally against an already-loaded
    model/processor -- all frames in a single call, same shape as the
    previous HTTP-based describe_images() it replaces.
    """
    try:
        images = []
        for frame in frames:
            image = Image.open(io.BytesIO(frame)).convert("RGB")
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
            images.append(image)
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [{"type": "image"} for _ in images],
            }
        ]
        chat_prompt = vlm.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = vlm.processor(images=images, text=chat_prompt, return_tensors="pt")
        output_ids = vlm.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return str(vlm.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]).strip()
    except Exception as e:
        raise VLMLocalError(f"Local VLM inference failed: {e}") from e
