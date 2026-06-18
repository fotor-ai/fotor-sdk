from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from .client import FotorAPIError, FotorClient
from .models import TaskResult, TaskSpec, TaskStatus
from .runner import TaskRunner
from .tasks import (
    background_remove,
    image2image,
    image_upscale,
    multiple_image2video,
    single_image2video,
    start_end_frame2video,
    text2image,
    text2video,
)
from .upload import is_url, upload_image_sync


DEFAULT_ENDPOINT = "https://api-b.fotor.com"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_IMAGE_ASPECT_RATIO = "1:1"
DEFAULT_IMAGE_RESOLUTION = "2k"
DEFAULT_VIDEO_MODEL = "doubao-seedance-2-0-260128"
DEFAULT_VIDEO_ASPECT_RATIO = "16:9"
DEFAULT_VIDEO_RESOLUTION = "1080p"
DEFAULT_VIDEO_DURATION = 5
FALLBACK_MODEL_BY_TASK = {
    "text2image": {
        "gemini-3.1-flash-image-preview": "seedream-5-0-260128",
        "gpt-image-2": "gemini-3.1-flash-image-preview",
    },
    "image2image": {
        "gemini-3.1-flash-image-preview": "seedream-5-0-260128",
        "gpt-image-2": "gemini-3.1-flash-image-preview",
    },
    "text2video": {
        "seedance-1-5-pro-251215": "kling-v3",
        "doubao-seedance-2-0-260128": "kling-v3",
    },
    "single_image2video": {
        "seedance-1-5-pro-251215": "kling-v3",
        "doubao-seedance-2-0-260128": "kling-v3",
    },
    "start_end_frame2video": {
        "kling-video-o1": "viduq2-turbo",
        "doubao-seedance-2-0-260128": "viduq2-turbo",
    },
    "multiple_image2video": {
        "kling-v3-omni": "kling-video-o1",
        "doubao-seedance-2-0-260128": "kling-video-o1",
    },
}


# 进度日志全部走 stderr，保持 stdout 给最终 JSON 结果使用，方便脚本管道。
_VERBOSE = True


def _log(message: str) -> None:
    """打印一行进度信息到 stderr；--quiet 时静默。"""
    if not _VERBOSE:
        return
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _log_submit(task_type: str, **info: Any) -> None:
    """提交任务前的统一打点：列出关键参数。"""
    parts = [f"{k}={v}" for k, v in info.items() if v not in (None, "")]
    suffix = f" ({', '.join(parts)})" if parts else ""
    _log(f"submitting {task_type}{suffix}")


def _make_poll_logger(task_type: str) -> Callable[[TaskResult], None]:
    """构造一个 on_poll 回调，给 client.wait_for_task 用，每次轮询打一行。"""
    state = {"count": 0}

    def on_poll(result: TaskResult) -> None:
        state["count"] += 1
        _log(
            f"{task_type} polling #{state['count']} "
            f"task={result.task_id} status={result.status.name} "
            f"elapsed={round(result.elapsed_seconds, 1)}s"
        )

    return on_poll


def _print_json(payload: dict[str, Any], stream: TextIO) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=stream)


def _result_to_dict(result: TaskResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "status": result.status.name,
        "success": result.success,
        "result_url": result.result_url,
        "error": result.error,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "creditsIncrement": getattr(result, "creditsIncrement", None),
        "fallback_used": bool(result.metadata.get("fallback_used", False)),
        "original_model_id": result.metadata.get("original_model_id", ""),
        "fallback_model_id": result.metadata.get("fallback_model_id", ""),
    }


def _batch_result_to_dict(result: TaskResult) -> dict[str, Any]:
    data = _result_to_dict(result)
    data["tag"] = result.metadata.get("tag", "")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fotor", description="Run Fotor AI tasks")
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="suppress progress logs on stderr (stdout JSON is unaffected)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("credits", help="Show account credits")

    batch = subparsers.add_parser("batch", help="Run tasks from a JSON file concurrently")
    batch.add_argument("--file", required=True, help="JSON file containing an array of task specs")
    batch.add_argument("--concurrency", type=int, default=5, help="maximum concurrent tasks")

    text2image = subparsers.add_parser("text2image", help="Generate an image from text")
    text2image.add_argument("--prompt", required=True)
    text2image.add_argument("--model", default=DEFAULT_IMAGE_MODEL)
    text2image.add_argument("--aspect-ratio", default=DEFAULT_IMAGE_ASPECT_RATIO)
    text2image.add_argument("--resolution", default=DEFAULT_IMAGE_RESOLUTION)

    image2image_parser = subparsers.add_parser("image2image", help="Edit an image")
    image2image_parser.add_argument("--image", action="extend", nargs="+", required=True)
    image2image_parser.add_argument("--prompt", required=True)
    image2image_parser.add_argument("--model", default=DEFAULT_IMAGE_MODEL)
    image2image_parser.add_argument("--aspect-ratio", default=DEFAULT_IMAGE_ASPECT_RATIO)
    image2image_parser.add_argument("--resolution", default=DEFAULT_IMAGE_RESOLUTION)

    upscale = subparsers.add_parser("upscale", help="Upscale an image")
    upscale.add_argument("--image", required=True)
    upscale.add_argument("--ratio", type=float, default=2.0)

    bg_remove = subparsers.add_parser("bg-remove", help="Remove an image background")
    bg_remove.add_argument("--image", required=True)

    text2video_parser = subparsers.add_parser("text2video", help="Generate a video from text")
    _add_video_common_args(text2video_parser)

    image2video_parser = subparsers.add_parser("image2video", help="Animate one image into a video")
    image2video_parser.add_argument("--image", required=True)
    _add_video_common_args(image2video_parser)

    start_end = subparsers.add_parser("start-end-video", help="Generate a video from start and end frames")
    start_end.add_argument("--start-image", required=True)
    start_end.add_argument("--end-image", required=True)
    _add_video_common_args(start_end)

    multi = subparsers.add_parser("multi-image-video", help="Generate a video from multiple images")
    multi.add_argument("--image", action="extend", nargs="+", required=True)
    _add_video_common_args(multi)

    return parser


def _add_video_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=DEFAULT_VIDEO_MODEL)
    parser.add_argument("--duration", type=int, default=DEFAULT_VIDEO_DURATION)
    parser.add_argument("--resolution", default=DEFAULT_VIDEO_RESOLUTION)
    parser.add_argument("--aspect-ratio", default=DEFAULT_VIDEO_ASPECT_RATIO)
    parser.add_argument("--audio", action="store_true", dest="audio_enable")


def _is_insufficient_credits_error(error: FotorAPIError) -> bool:
    code = str(getattr(error, "code", "") or "")
    return code == "510" or "No enough credits" in str(error)


async def _run_with_fallback(task_type: str, call, params: dict[str, Any]) -> TaskResult:
    try:
        return await call(**params)
    except FotorAPIError as exc:
        if _is_insufficient_credits_error(exc):
            raise
        model_id = str(params.get("model_id", ""))
        fallback_model = FALLBACK_MODEL_BY_TASK.get(task_type, {}).get(model_id, "")
        if not fallback_model:
            raise
        _log(
            f"fallback triggered for {task_type}: {model_id} -> {fallback_model} "
            f"(reason: {exc})"
        )
        retry_params = dict(params)
        retry_params["model_id"] = fallback_model
        result = await call(**retry_params)
        result.metadata["fallback_used"] = True
        result.metadata["original_model_id"] = model_id
        result.metadata["fallback_model_id"] = fallback_model
        return result


def _resolve_image(value: str, *, task_type: str, api_key: str, endpoint: str) -> str:
    if is_url(value):
        _log(f"using image URL: {value}")
        return value
    _log(f"uploading image '{value}' (task_type={task_type}) ...")
    start = time.monotonic()
    result = upload_image_sync(
        value,
        task_type=task_type,
        api_key=api_key,
        endpoint=endpoint,
    )
    elapsed = round(time.monotonic() - start, 2)
    _log(f"uploaded '{value}' -> {result.file_url} ({elapsed}s)")
    return result.file_url


def _load_batch_specs(file_path: str) -> list[TaskSpec]:
    payload = json.loads(Path(file_path).expanduser().read_text(encoding="utf-8"))
    items = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("batch file must contain a JSON array or an object with a 'tasks' array")

    specs: list[TaskSpec] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"batch task #{index} must be an object")
        task_type = str(item.get("task_type", "")).strip()
        if not task_type:
            raise ValueError(f"batch task #{index} missing task_type")
        params = item.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"batch task #{index} params must be an object")
        specs.append(TaskSpec(task_type=task_type, params=params, tag=str(item.get("tag", ""))))
    return specs


async def _run_batch(args: argparse.Namespace, *, api_key: str, endpoint: str) -> dict[str, Any]:
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")

    specs = _load_batch_specs(args.file)
    _log(f"running batch tasks={len(specs)} concurrency={args.concurrency}")

    def on_progress(**info: Any) -> None:
        latest = info.get("latest")
        latest_status = latest.status.name if isinstance(latest, TaskResult) else "UNKNOWN"
        latest_tag = latest.metadata.get("tag", "") if isinstance(latest, TaskResult) else ""
        _log(
            f"batch progress completed={info.get('completed')}/{info.get('total')} "
            f"failed={info.get('failed')} in_progress={info.get('in_progress')} "
            f"latest={latest_tag or latest_status}"
        )

    client = FotorClient(api_key=api_key, endpoint=endpoint)
    runner = TaskRunner(client, max_concurrent=args.concurrency)
    results = await runner.run(
        specs,
        on_progress=on_progress,
        on_task_poll=_make_poll_logger("batch"),
    )
    completed = sum(1 for result in results if result.status == TaskStatus.COMPLETED)
    return {
        "summary": {
            "total": len(results),
            "completed": completed,
            "failed": len(results) - completed,
        },
        "results": [_batch_result_to_dict(result) for result in results],
    }


async def _run(args: argparse.Namespace, *, api_key: str, endpoint: str) -> TaskResult:
    _log(f"running command={args.command}")
    client = FotorClient(api_key=api_key, endpoint=endpoint)
    if args.command == "text2image":
        _log_submit(
            "text2image",
            model=args.model, aspect_ratio=args.aspect_ratio, resolution=args.resolution,
        )
        return await _run_with_fallback(
            "text2image",
            text2image,
            {
                "client": client,
                "prompt": args.prompt,
                "model_id": args.model,
                "aspect_ratio": args.aspect_ratio,
                "resolution": args.resolution,
                "on_poll": _make_poll_logger("text2image"),
            },
        )
    if args.command == "image2image":
        image_urls = [
            _resolve_image(image, task_type="image2image", api_key=api_key, endpoint=endpoint)
            for image in args.image
        ]
        _log_submit(
            "image2image",
            model=args.model, images=len(image_urls),
            aspect_ratio=args.aspect_ratio, resolution=args.resolution,
        )
        return await _run_with_fallback(
            "image2image",
            image2image,
            {
                "client": client,
                "prompt": args.prompt,
                "model_id": args.model,
                "image_urls": image_urls,
                "aspect_ratio": args.aspect_ratio,
                "resolution": args.resolution,
                "on_poll": _make_poll_logger("image2image"),
            },
        )
    if args.command == "upscale":
        image_url = _resolve_image(args.image, task_type="image_upscale", api_key=api_key, endpoint=endpoint)
        _log_submit("upscale", ratio=args.ratio)
        return await image_upscale(
            client, image_url=image_url, upscale_ratio=args.ratio,
            on_poll=_make_poll_logger("upscale"),
        )
    if args.command == "bg-remove":
        image_url = _resolve_image(args.image, task_type="background_remove", api_key=api_key, endpoint=endpoint)
        _log_submit("bg-remove")
        return await background_remove(
            client, image_url=image_url,
            on_poll=_make_poll_logger("bg-remove"),
        )
    if args.command == "text2video":
        _log_submit(
            "text2video",
            model=args.model, duration=args.duration,
            aspect_ratio=args.aspect_ratio, resolution=args.resolution,
            audio=args.audio_enable,
        )
        params = _video_params(client, args)
        params["on_poll"] = _make_poll_logger("text2video")
        return await _run_with_fallback("text2video", text2video, params)
    if args.command == "image2video":
        image_url = _resolve_image(args.image, task_type="single_image2video", api_key=api_key, endpoint=endpoint)
        _log_submit(
            "image2video",
            model=args.model, duration=args.duration,
            resolution=args.resolution, audio=args.audio_enable,
        )
        params = _video_params(client, args)
        params["image_url"] = image_url
        params["on_poll"] = _make_poll_logger("image2video")
        return await _run_with_fallback("single_image2video", single_image2video, params)
    if args.command == "start-end-video":
        params = _video_params(client, args)
        params["start_image_url"] = _resolve_image(
            args.start_image,
            task_type="start_end_frame2video",
            api_key=api_key,
            endpoint=endpoint,
        )
        params["end_image_url"] = _resolve_image(
            args.end_image,
            task_type="start_end_frame2video",
            api_key=api_key,
            endpoint=endpoint,
        )
        _log_submit(
            "start-end-video",
            model=args.model, duration=args.duration,
            resolution=args.resolution, audio=args.audio_enable,
        )
        params["on_poll"] = _make_poll_logger("start-end-video")
        return await _run_with_fallback("start_end_frame2video", start_end_frame2video, params)
    if args.command == "multi-image-video":
        params = _video_params(client, args)
        params["image_urls"] = [
            _resolve_image(image, task_type="multiple_image2video", api_key=api_key, endpoint=endpoint)
            for image in args.image
        ]
        _log_submit(
            "multi-image-video",
            model=args.model, images=len(params["image_urls"]),
            duration=args.duration, resolution=args.resolution, audio=args.audio_enable,
        )
        params["on_poll"] = _make_poll_logger("multi-image-video")
        return await _run_with_fallback("multiple_image2video", multiple_image2video, params)
    raise ValueError(f"Unsupported command: {args.command}")


def _get_credits(*, api_key: str, endpoint: str) -> dict[str, Any]:
    client = FotorClient(api_key=api_key, endpoint=endpoint)
    return client.get_credits_sync()


def _video_params(client: FotorClient, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "client": client,
        "prompt": args.prompt,
        "model_id": args.model,
        "duration": args.duration,
        "resolution": args.resolution,
        "aspect_ratio": args.aspect_ratio,
        "audio_enable": args.audio_enable,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    global _VERBOSE
    _VERBOSE = not getattr(args, "quiet", False)

    api_key = os.environ.get("FOTOR_OPENAPI_KEY", "")
    if not api_key:
        _print_json({"error": "FOTOR_OPENAPI_KEY not set"}, sys.stderr)
        return 1

    endpoint = os.environ.get("FOTOR_OPENAPI_ENDPOINT", DEFAULT_ENDPOINT)
    _log(f"endpoint={endpoint}")
    try:
        if args.command == "credits":
            _log("fetching credits ...")
            _print_json(_get_credits(api_key=api_key, endpoint=endpoint), sys.stdout)
            return 0
        if args.command == "batch":
            _print_json(asyncio.run(_run_batch(args, api_key=api_key, endpoint=endpoint)), sys.stdout)
            return 0
        result = asyncio.run(_run(args, api_key=api_key, endpoint=endpoint))
    except Exception as exc:  # noqa: BLE001
        _log(f"error: {exc}")
        _print_json({"error": str(exc)}, sys.stderr)
        return 1

    elapsed = round(result.elapsed_seconds, 1)
    if result.success:
        _log(f"completed in {elapsed}s -> {result.result_url}")
    else:
        _log(f"failed in {elapsed}s: status={result.status.name} error={result.error}")
    _print_json(_result_to_dict(result), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
