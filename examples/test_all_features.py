#!/usr/bin/env python3
"""Comprehensive test for every fotor-sdk feature.

Usage:
    export FOTOR_OPENAPI_KEY="your-api-key"
    python examples/test_all_features.py                  # run all tests
    python examples/test_all_features.py --only image     # image tests only
    python examples/test_all_features.py --only video     # video tests only
    python examples/test_all_features.py --only runner    # parallel runner only
    python examples/test_all_features.py --only error     # error-handling only
    python examples/test_all_features.py --only sync      # sync wrapper only

Optionally set FOTOR_OPENAPI_ENDPOINT to target a different host.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any
from dotenv import load_dotenv
load_dotenv()
from fotor_sdk import (
    FotorClient,
    FotorAPIError,
    TaskResult,
    TaskRunner,
    TaskSpec,
    TaskStatus,
    text2image,
    image2image,
    image_upscale,
    background_remove,
    text2video,
    single_image2video,
    start_end_frame2video,
    multiple_image2video,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test")

SAMPLE_IMAGE = (
    "https://images.unsplash.com/photo-1529778873920-4da4926a72c2"
    "?w=800&q=80"
)
SAMPLE_IMAGE_2 = (
    "https://images.unsplash.com/photo-1518791841217-8f162f1e1131"
    "?w=800&q=80"
)
SAMPLE_IMAGE_3 = (
    "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    "?w=800&q=80"
)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"

results_summary: list[tuple[str, str, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    status = PASS if passed else FAIL
    results_summary.append((name, status, detail))
    log.info("%s %s  %s", status, name, detail)


def poll_logger(r: TaskResult) -> None:
    log.info(
        "  ↳ polling %s  status=%-12s  elapsed=%.1fs",
        r.task_id or "n/a", r.status.name, r.elapsed_seconds,
    )


# -----------------------------------------------------------------------
# Image tests
# -----------------------------------------------------------------------

async def test_text2image(client: FotorClient) -> None:
    name = "text2image"
    try:
        result = await text2image(
            client,
            prompt="A watercolor painting of a mountain lake at sunrise",
            model_id="seedream-4-5-251128",
            resolution="1k",
            aspect_ratio="1:1",
            on_poll=poll_logger,
        )
        record(name, result.success, result.result_url or result.error or "")
    except Exception as exc:
        record(name, False, str(exc))


async def test_image2image(client: FotorClient) -> None:
    name = "image2image"
    try:
        result = await image2image(
            client,
            prompt="Transform into an oil painting style, warm tones",
            model_id="seedream-4-5-251128",
            image_urls=[SAMPLE_IMAGE],
            resolution="1k",
            aspect_ratio="1:1",
            on_poll=poll_logger,
        )
        record(name, result.success, result.result_url or result.error or "")
    except Exception as exc:
        record(name, False, str(exc))


async def test_image_upscale(client: FotorClient) -> None:
    name = "image_upscale"
    try:
        result = await image_upscale(
            client,
            image_url=SAMPLE_IMAGE,
            upscale_ratio=2.0,
            on_poll=poll_logger,
        )
        record(name, result.success, result.result_url or result.error or "")
    except Exception as exc:
        record(name, False, str(exc))


async def test_background_remove(client: FotorClient) -> None:
    name = "background_remove"
    try:
        result = await background_remove(
            client,
            image_url=SAMPLE_IMAGE,
            on_poll=poll_logger,
        )
        record(name, result.success, result.result_url or result.error or "")
    except Exception as exc:
        record(name, False, str(exc))


# -----------------------------------------------------------------------
# Video tests
# -----------------------------------------------------------------------

async def test_text2video(client: FotorClient) -> None:
    name = "text2video"
    try:
        result = await text2video(
            client,
            prompt="Gentle ocean waves at golden hour, cinematic slow motion",
            model_id="kling-v3",
            duration=5,
            resolution="1080p",
            aspect_ratio="16:9",
            on_poll=poll_logger,
        )
        record(name, result.success, result.result_url or result.error or "")
    except Exception as exc:
        record(name, False, str(exc))


async def test_single_image2video(client: FotorClient) -> None:
    name = "single_image2video"
    try:
        result = await single_image2video(
            client,
            prompt="Camera slowly zooms in, leaves gently swaying",
            model_id="kling-v3",
            image_url=SAMPLE_IMAGE,
            duration=5,
            resolution="1080p",
            on_poll=poll_logger,
        )
        record(name, result.success, result.result_url or result.error or "")
    except Exception as exc:
        record(name, False, str(exc))


async def test_start_end_frame2video(client: FotorClient) -> None:
    name = "start_end_frame2video"
    try:
        result = await start_end_frame2video(
            client,
            prompt="Smooth transition from day to night",
            model_id="kling-v3",
            start_image_url=SAMPLE_IMAGE,
            end_image_url=SAMPLE_IMAGE_2,
            duration=5,
            resolution="1080p",
            on_poll=poll_logger,
        )
        record(name, result.success, result.result_url or result.error or "")
    except Exception as exc:
        record(name, False, str(exc))


async def test_multiple_image2video(client: FotorClient) -> None:
    name = "multiple_image2video"
    try:
        result = await multiple_image2video(
            client,
            prompt="A montage of cute cats in different scenes",
            model_id="kling-v3",
            image_urls=[SAMPLE_IMAGE, SAMPLE_IMAGE_2, SAMPLE_IMAGE_3],
            duration=5,
            resolution="1080p",
            on_poll=poll_logger,
        )
        record(name, result.success, result.result_url or result.error or "")
    except Exception as exc:
        record(name, False, str(exc))


# -----------------------------------------------------------------------
# TaskRunner (parallel batch)
# -----------------------------------------------------------------------

async def test_runner(client: FotorClient) -> None:
    name = "TaskRunner.run"
    try:
        specs = [
            TaskSpec(
                "text2image",
                {"prompt": "A red rose", "model_id": "seedream-4-5-251128",
                 "resolution": "1k", "aspect_ratio": "1:1"},
                tag="rose",
            ),
            TaskSpec(
                "text2image",
                {"prompt": "A blue butterfly", "model_id": "seedream-4-5-251128",
                 "resolution": "1k", "aspect_ratio": "1:1"},
                tag="butterfly",
            ),
        ]

        progress_events: list[dict[str, Any]] = []

        def on_progress(**kw: Any) -> None:
            progress_events.append(kw)
            tag = kw["latest"].metadata.get("tag", "?")
            log.info(
                "  ↳ batch %d/%d done (ok=%d fail=%d), latest: %s",
                kw["completed"] + kw["failed"], kw["total"],
                kw["completed"], kw["failed"], tag,
            )

        runner = TaskRunner(client, max_concurrent=5)
        results = await runner.run(specs, on_progress=on_progress)

        all_ok = all(r.success for r in results)
        got_progress = len(progress_events) == len(specs)
        detail = (
            f"{sum(r.success for r in results)}/{len(results)} succeeded, "
            f"progress_events={len(progress_events)}"
        )
        record(name, all_ok and got_progress, detail)
    except Exception as exc:
        record(name, False, str(exc))


# -----------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------

async def test_error_handling(client: FotorClient) -> None:
    name = "error: bad model_id"
    try:
        await text2image(
            client,
            prompt="test",
            model_id="non-existent-model-xyz",
            resolution="1k",
            aspect_ratio="1:1",
        )
        record(name, False, "expected FotorAPIError but got success")
    except FotorAPIError as exc:
        record(name, True, f"FotorAPIError(code={exc.code!r}, msg={exc!s})")
    except Exception as exc:
        record(name, False, f"unexpected exception: {type(exc).__name__}: {exc}")

    name = "error: empty api_key"
    try:
        FotorClient(api_key="")
        record(name, False, "expected ValueError but got success")
    except ValueError:
        record(name, True, "ValueError raised as expected")
    except Exception as exc:
        record(name, False, f"unexpected: {type(exc).__name__}: {exc}")

    name = "error: image2image no urls"
    try:
        await image2image(
            client,
            prompt="test",
            model_id="seedream-4-5-251128",
            image_urls=[],
        )
        record(name, False, "expected ValueError but got success")
    except ValueError:
        record(name, True, "ValueError raised as expected")
    except Exception as exc:
        record(name, False, f"unexpected: {type(exc).__name__}: {exc}")

    name = "error: multiple_image2video <2 urls"
    try:
        await multiple_image2video(
            client,
            prompt="test",
            model_id="kling-v3",
            image_urls=["http://example.com/a.jpg"],
        )
        record(name, False, "expected ValueError but got success")
    except ValueError:
        record(name, True, "ValueError raised as expected")
    except Exception as exc:
        record(name, False, f"unexpected: {type(exc).__name__}: {exc}")

    name = "error: runner unknown task_type"
    try:
        runner = TaskRunner(client, max_concurrent=1)
        results = await runner.run([TaskSpec("nonexistent_task", {})])
        r = results[0]
        record(name, r.status == TaskStatus.FAILED and "Unknown" in (r.error or ""),
               r.error or "")
    except Exception as exc:
        record(name, False, f"unexpected: {type(exc).__name__}: {exc}")


# -----------------------------------------------------------------------
# Sync wrappers
# -----------------------------------------------------------------------

def test_sync_wrapper() -> None:
    name = "submit_and_wait_sync"
    api_key = os.environ.get("FOTOR_OPENAPI_KEY", "")
    endpoint = os.environ.get("FOTOR_OPENAPI_ENDPOINT", "https://api.fotor.com")
    client = FotorClient(api_key=api_key, endpoint=endpoint)
    try:
        result = client.submit_and_wait_sync(
            "/v1/aiart/imagegeneration/seedream-4-5-251128",
            {
                "width": 1024,
                "height": 1024,
                "content": [{"type": "text", "text": "A simple red circle on white"}],
                "quality": "medium",
            },
        )
        record(name, result.status == TaskStatus.COMPLETED,
               result.result_url or result.error or "")
    except Exception as exc:
        record(name, False, str(exc))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

TEST_GROUPS: dict[str, list[str]] = {
    "image": [
        "text2image", "image2image", "image_upscale", "background_remove",
    ],
    "video": [
        "text2video", "single_image2video",
        "start_end_frame2video", "multiple_image2video",
    ],
    "runner": ["runner"],
    "error": ["error"],
    "sync": ["sync"],
}

ASYNC_DISPATCH: dict[str, Any] = {
    "text2image": test_text2image,
    "image2image": test_image2image,
    "image_upscale": test_image_upscale,
    "background_remove": test_background_remove,
    "text2video": test_text2video,
    "single_image2video": test_single_image2video,
    "start_end_frame2video": test_start_end_frame2video,
    "multiple_image2video": test_multiple_image2video,
    "runner": test_runner,
    "error": test_error_handling,
}


async def async_main(groups: list[str]) -> None:
    api_key = os.environ.get("FOTOR_OPENAPI_KEY", "")
    endpoint = os.environ.get("FOTOR_OPENAPI_ENDPOINT", "https://api.fotor.com")

    if not api_key:
        print("ERROR: set FOTOR_OPENAPI_KEY first", file=sys.stderr)
        sys.exit(1)

    client = FotorClient(api_key=api_key, endpoint=endpoint)

    test_names: list[str] = []
    for g in groups:
        test_names.extend(TEST_GROUPS.get(g, []))

    log.info("=" * 60)
    log.info("Fotor SDK Feature Test  (groups: %s)", ", ".join(groups))
    log.info("endpoint: %s", endpoint)
    log.info("=" * 60)

    start = time.monotonic()

    for tn in test_names:
        fn = ASYNC_DISPATCH.get(tn)
        if fn:
            log.info("--- %s ---", tn)
            await fn(client)

    if "sync" in groups:
        log.info("--- sync ---")
        test_sync_wrapper()

    elapsed = time.monotonic() - start
    print()
    print("=" * 60)
    print(f"  Test Summary  ({elapsed:.1f}s)")
    print("=" * 60)
    passed = failed = 0
    for name, status, detail in results_summary:
        print(f"  {status} {name}")
        if detail:
            print(f"       {detail}")
        if PASS in status:
            passed += 1
        else:
            failed += 1
    print("-" * 60)
    color = "\033[92m" if failed == 0 else "\033[91m"
    print(f"  {color}{passed} passed, {failed} failed\033[0m")
    print("=" * 60)

    if failed:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test all fotor-sdk features")
    parser.add_argument(
        "--only",
        choices=list(TEST_GROUPS.keys()),
        nargs="+",
        default=list(TEST_GROUPS.keys()),
        help="run only specific test groups (default: all)",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args.only))


if __name__ == "__main__":
    main()
