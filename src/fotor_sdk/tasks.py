"""High-level task helpers that build payloads and call :class:`FotorClient`.

Each function accepts plain Python arguments (no Pydantic models required),
constructs the correct API payload, submits the task, and polls until done.
"""

from __future__ import annotations

from typing import Any, Callable

from .client import FotorClient
from .models import TaskResult

# ---------------------------------------------------------------------------
# API path constants (mirrors openapi/urls.py)
# ---------------------------------------------------------------------------
_IMAGE_GENERATION = "/v1/aiart/imagegeneration"
_IMG_UPSCALER = "/v1/aiart/imgupscale"
_BG_REMOVER = "/v1/aiart/backgroundremover"
_TXT2VIDEO = "/v1/aiart/text2video"
_IMG2VIDEO = "/v1/aiart/img2video"
_STARTEND2VIDEO = "/v1/aiart/startend2video"
_CHARACTER2VIDEO = "/v1/aiart/character2video"

# Default image sizes per aspect ratio.
# The API requires at least 3,686,400 total pixels (1920×1920 for 1:1).
_DEFAULT_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (1920, 1920),
    "16:9": (2560, 1440),
    "9:16": (1440, 2560),
    "4:3": (2240, 1680),
    "3:4": (1680, 2240),
    "3:2": (2400, 1600),
    "2:3": (1600, 2400),
    "21:9": (2940, 1260),
}


def _resolve_size(
    aspect_ratio: str, resolution: str
) -> tuple[int, int]:
    w, h = _DEFAULT_SIZES.get(aspect_ratio, (1920, 1920))
    if resolution == "2k":
        w, h = w * 2, h * 2
    elif resolution == "4k":
        w, h = w * 4, h * 4
    return w, h


# ---------------------------------------------------------------------------
# Image tasks
# ---------------------------------------------------------------------------

async def text2image(
    client: FotorClient,
    *,
    prompt: str,
    model_id: str,
    aspect_ratio: str = "auto",
    resolution: str = "auto",
    on_poll: Callable[[TaskResult], None] | None = None,
    **extra: Any,
) -> TaskResult:
    """Generate an image from a text prompt."""
    w, h = _resolve_size(aspect_ratio, resolution)
    payload: dict[str, Any] = {
        "width": w,
        "height": h,
        "content": [{"type": "text", "text": prompt}],
        "quality": "medium",
        **extra,
    }
    path = f"{_IMAGE_GENERATION}/{model_id}"
    return await client.submit_and_wait(path, payload, on_poll=on_poll)


async def image2image(
    client: FotorClient,
    *,
    prompt: str,
    model_id: str,
    image_urls: list[str],
    aspect_ratio: str = "auto",
    resolution: str = "auto",
    on_poll: Callable[[TaskResult], None] | None = None,
    **extra: Any,
) -> TaskResult:
    """Generate an image from text + reference image(s)."""
    if not image_urls:
        raise ValueError("image_urls must contain at least one URL")
    w, h = _resolve_size(aspect_ratio, resolution)
    content: list[dict[str, str]] = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "url": url.strip()})
    payload: dict[str, Any] = {
        "width": w,
        "height": h,
        "content": content,
        "quality": "medium",
        **extra,
    }
    path = f"{_IMAGE_GENERATION}/{model_id}"
    return await client.submit_and_wait(path, payload, on_poll=on_poll)


async def image_upscale(
    client: FotorClient,
    *,
    image_url: str,
    upscale_ratio: float = 2.0,
    on_poll: Callable[[TaskResult], None] | None = None,
) -> TaskResult:
    """Upscale an image by 2x or 4x."""
    payload = {
        "upscaling_resize": upscale_ratio,
        "userImageUrl": image_url,
        "max_image_width": 2048,
        "max_image_height": 2048,
    }
    return await client.submit_and_wait(_IMG_UPSCALER, payload, on_poll=on_poll)


async def background_remove(
    client: FotorClient,
    *,
    image_url: str,
    on_poll: Callable[[TaskResult], None] | None = None,
) -> TaskResult:
    """Remove the background from an image."""
    payload = {
        "userImageUrl": image_url,
        "action": "auto",
    }
    return await client.submit_and_wait(_BG_REMOVER, payload, on_poll=on_poll)


# ---------------------------------------------------------------------------
# Video tasks
# ---------------------------------------------------------------------------

def _video_payload(
    prompt: str,
    duration: int,
    resolution: str,
    aspect_ratio: str,
    audio_enable: bool,
    image_urls: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio if aspect_ratio != "auto" else "",
        "scenes": "normal",
        "enableAudio": audio_enable,
    }
    if image_urls is not None:
        payload["imageUrls"] = image_urls
    payload.update(extra)
    return payload


async def text2video(
    client: FotorClient,
    *,
    prompt: str,
    model_id: str,
    duration: int = 5,
    resolution: str = "1080p",
    aspect_ratio: str = "16:9",
    audio_enable: bool = False,
    on_poll: Callable[[TaskResult], None] | None = None,
    **extra: Any,
) -> TaskResult:
    """Generate a video from a text prompt."""
    payload = _video_payload(prompt, duration, resolution, aspect_ratio, audio_enable, **extra)
    path = f"{_TXT2VIDEO}/{model_id}"
    return await client.submit_and_wait(path, payload, on_poll=on_poll)


async def single_image2video(
    client: FotorClient,
    *,
    prompt: str,
    model_id: str,
    image_url: str,
    duration: int = 5,
    resolution: str = "1080p",
    aspect_ratio: str = "auto",
    audio_enable: bool = False,
    on_poll: Callable[[TaskResult], None] | None = None,
    **extra: Any,
) -> TaskResult:
    """Animate a single image into a video."""
    payload = _video_payload(
        prompt, duration, resolution, aspect_ratio, audio_enable,
        image_urls=[image_url], **extra,
    )
    path = f"{_IMG2VIDEO}/{model_id}"
    return await client.submit_and_wait(path, payload, on_poll=on_poll)


async def start_end_frame2video(
    client: FotorClient,
    *,
    prompt: str,
    model_id: str,
    start_image_url: str,
    end_image_url: str,
    duration: int = 5,
    resolution: str = "1080p",
    aspect_ratio: str = "auto",
    audio_enable: bool = False,
    on_poll: Callable[[TaskResult], None] | None = None,
    **extra: Any,
) -> TaskResult:
    """Generate a video from start and end frame images."""
    payload = _video_payload(
        prompt, duration, resolution, aspect_ratio, audio_enable,
        image_urls=[start_image_url, end_image_url], **extra,
    )
    path = f"{_STARTEND2VIDEO}/{model_id}"
    return await client.submit_and_wait(path, payload, on_poll=on_poll)


async def multiple_image2video(
    client: FotorClient,
    *,
    prompt: str,
    model_id: str,
    image_urls: list[str],
    duration: int = 5,
    resolution: str = "1080p",
    aspect_ratio: str = "auto",
    audio_enable: bool = False,
    on_poll: Callable[[TaskResult], None] | None = None,
    **extra: Any,
) -> TaskResult:
    """Generate a video from multiple reference images."""
    if not image_urls or len(image_urls) < 2:
        raise ValueError("image_urls must contain at least 2 URLs")
    payload = _video_payload(
        prompt, duration, resolution, aspect_ratio, audio_enable,
        image_urls=image_urls, **extra,
    )
    path = f"{_CHARACTER2VIDEO}/{model_id}"
    return await client.submit_and_wait(path, payload, on_poll=on_poll)
