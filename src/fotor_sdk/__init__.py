"""Fotor OpenAPI SDK -- lightweight, standalone, async-first."""

__version__ = "0.1.6"

from .client import FotorClient, FotorAPIError
from .models import TaskResult, TaskSpec, TaskStatus
from .runner import TaskRunner
from .tasks import (
    text2image,
    image2image,
    image_upscale,
    background_remove,
    text2video,
    single_image2video,
    start_end_frame2video,
    multiple_image2video,
)

__all__ = [
    "FotorClient",
    "FotorAPIError",
    "TaskResult",
    "TaskSpec",
    "TaskStatus",
    "TaskRunner",
    "text2image",
    "image2image",
    "image_upscale",
    "background_remove",
    "text2video",
    "single_image2video",
    "start_end_frame2video",
    "multiple_image2video",
]
