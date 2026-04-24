"""High-level task helpers that build payloads and call :class:`FotorClient`.

Each function accepts plain Python arguments (no Pydantic models required),
constructs the correct API payload, submits the task, and polls until done.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from .client import FotorClient
from .models import TaskResult
from .model_image_size_rules import MODEL_IMAGE_SIZE_RULES

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
    "1:1": (1024, 1024),
    "16:9": (1392, 752),
    "9:16": (752, 1392),
    "4:3": (1184, 880),
    "3:4": (880, 1184),
    "3:2": (1536, 1024),
    "2:3": (1024, 1536),
    "21:9": (1568, 672),
}

_API_MIN_PIXELS = 1024 * 1024


def _resolution_multiplier(resolution: str) -> int:
    """
    Convert resolution like "1k"/"2k"/"4k" into a multiplier.

    The `preferred_size` values from model_config are treated as 1k base sizes,
    so we multiply width/height by the numeric k value.
    """
    r = resolution.lower().strip()
    if r.endswith("k"):
        # e.g. "2k" -> 2
        return int(r[:-1])
    return 1


def _parse_aspect_ratio(aspect_ratio: str) -> tuple[int, int] | None:
    # "W:H" where W is width ratio numerator and H is height ratio denominator.
    parts = aspect_ratio.split(":")
    if len(parts) != 2:
        return None
    try:
        w = int(parts[0])
        h = int(parts[1])
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return w, h


def _base_size_for_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
    if aspect_ratio in _DEFAULT_SIZES:
        return _DEFAULT_SIZES[aspect_ratio]
    parsed = _parse_aspect_ratio(aspect_ratio)
    if parsed is None:
        return 1920, 1920

    # Maintain approx. API minimum pixel budget while matching the ratio.
    # P = w*h, w/h = a/b => h = sqrt(P*b/a), w = sqrt(P*a/b)
    a, b = parsed
    h = int(math.ceil(math.sqrt(_API_MIN_PIXELS * b / a)))
    w = int(math.ceil(h * a / b))
    return w, h


def _resolve_size(
    aspect_ratio: str, resolution: str
) -> tuple[int, int]:
    w, h = _base_size_for_aspect_ratio(aspect_ratio)
    mult = _resolution_multiplier(resolution)
    return w * mult, h * mult


def _clamp_long_side(w: int, h: int, max_long_side: int) -> tuple[int, int]:
    long_side = max(w, h)
    if long_side <= max_long_side:
        return w, h
    scale = max_long_side / long_side
    # floor to avoid exceeding max_long_side due to rounding.
    w2 = max(1, int(w * scale))
    h2 = max(1, int(h * scale))
    return w2, h2


def _resolve_image_size(
    model_id: str,
    aspect_ratio: str,
    resolution: str,
) -> tuple[int, int]:
    """
    Resolve image `width/height` for `text2image` and `image2image`.

    - If a model provides `aspect_ratio.preferred_size`, we use it (1k base) and
      multiply by the chosen k-resolution (2k/3k/4k/...).
    - If a model has no `preferred_size` (e.g. seedream), compute from the chosen
      aspect ratio then clamp by `input_image_limits.max_long_side`.
    - `aspect_ratio` should be an explicit ratio (e.g. "1:1", "16:9").
    """
    if resolution == "auto":
        raise ValueError("resolution='auto' is not supported; use an explicit value like '1k'")

    rule = MODEL_IMAGE_SIZE_RULES.get(model_id)
    if rule is None:
        if aspect_ratio == "auto":
            raise ValueError("aspect_ratio='auto' is not supported; use an explicit ratio like '1:1'")
        return _resolve_size(aspect_ratio, resolution)

    aspect_supports: list[str] = rule.get("aspect_ratio_supports") or []
    preferred_size: dict[str, list[int]] = rule.get("preferred_size") or {}

    chosen_ratio = aspect_ratio.strip()
    if aspect_supports and chosen_ratio not in aspect_supports:
        chosen_ratio = "1:1"
    # Ensure we always have a 1:1 fallback when the model config is sparse,
    # but don't force 1:1 for models that intentionally have no preferred_size
    # (e.g. seedream uses ratio+clamp instead).
    if preferred_size and chosen_ratio not in preferred_size and chosen_ratio != "1:1":
        chosen_ratio = "1:1"

    res_supports: list[str] = rule.get("resolution_supports") or []
    chosen_resolution = resolution
    chosen_resolution = chosen_resolution.lower().strip()

    if res_supports:
        allowed = {s.lower() for s in res_supports}
        if chosen_resolution not in allowed:
            # SDK default is "1k", but not every model supports it (e.g. seedream).
            # In that case, fall back to the model's default (or the first supported).
            fallback = (rule.get("resolution_default") or "").lower().strip()
            if fallback and fallback in allowed:
                chosen_resolution = fallback
            else:
                chosen_resolution = next(iter(allowed))

    mult = _resolution_multiplier(chosen_resolution)

    if preferred_size:
        base = preferred_size.get(chosen_ratio) or preferred_size.get("1:1") or (1024, 1024)
        bw, bh = int(base[0]), int(base[1])
        w, h = bw * mult, bh * mult
        max_long_side = rule.get("max_long_side")
        if max_long_side:
            w, h = _clamp_long_side(w, h, int(max_long_side))
        return w, h

    # No preferred_size mapping (e.g. seedream): compute from ratio and clamp.
    w, h = _resolve_size(chosen_ratio, chosen_resolution)
    max_long_side = rule.get("max_long_side")
    if max_long_side:
        w, h = _clamp_long_side(w, h, int(max_long_side))
    return w, h


# ---------------------------------------------------------------------------
# Image tasks
# ---------------------------------------------------------------------------

async def text2image(
    client: FotorClient,
    *,
    prompt: str,
    model_id: str,
    aspect_ratio: str = "1:1",
    resolution: str = "1k",
    on_poll: Callable[[TaskResult], None] | None = None,
    **extra: Any,
) -> TaskResult:
    """Generate an image from a text prompt."""
    w, h = _resolve_image_size(model_id, aspect_ratio, resolution)
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
    aspect_ratio: str = "1:1",
    resolution: str = "1k",
    on_poll: Callable[[TaskResult], None] | None = None,
    **extra: Any,
) -> TaskResult:
    """Generate an image from text + reference image(s)."""
    if not image_urls:
        raise ValueError("image_urls must contain at least one URL")
    w, h = _resolve_image_size(model_id, aspect_ratio, resolution)
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
        "aspect_ratio": aspect_ratio if aspect_ratio != "auto" else "16:9",
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
    aspect_ratio: str = "16:9",
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
    aspect_ratio: str = "16:9",
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
    aspect_ratio: str = "16:9",
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
