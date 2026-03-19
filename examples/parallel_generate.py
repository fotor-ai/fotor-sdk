#!/usr/bin/env python3
"""Example: run multiple Fotor image/video tasks in parallel with live progress.

Usage:
    export FOTOR_OPENAPI_KEY="your-api-key"
    python fotor_sdk/scripts/parallel_generate.py

Optionally set FOTOR_OPENAPI_ENDPOINT to point at a different API host.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from fotor_sdk import (
    FotorClient,
    TaskRunner,
    TaskSpec,
    TaskResult,
    TaskStatus,
    text2image,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("example")


# ---------------------------------------------------------------------------
# 1. Single task -- simplest usage
# ---------------------------------------------------------------------------

async def single_task_demo(client: FotorClient) -> None:
    log.info("=== Single task demo ===")
    result = await text2image(
        client,
        prompt="A diamond kitten on a velvet cushion, studio lighting, 4K",
        model_id="seedream-4-5-251128",
        resolution="1k",
        aspect_ratio="1:1",
        on_poll=lambda r: log.info("  polling %s  status=%s  %.1fs",
                                   r.task_id, r.status.name, r.elapsed_seconds),
    )
    log.info("Result: %s", result)


# ---------------------------------------------------------------------------
# 2. Parallel batch -- the main use-case
# ---------------------------------------------------------------------------

def progress_callback(
    total: int,
    completed: int,
    failed: int,
    in_progress: int,
    latest: TaskResult,
) -> None:
    bar_len = 30
    done_ratio = (completed + failed) / total if total else 0
    filled = int(bar_len * done_ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    tag = latest.metadata.get("tag", latest.task_id)
    status = "OK" if latest.success else latest.status.name
    print(
        f"\r[{bar}] {completed + failed}/{total}  "
        f"(ok={completed} fail={failed} run={in_progress})  "
        f"latest: {tag} -> {status}     ",
        end="",
        flush=True,
    )


async def batch_demo(client: FotorClient) -> None:
    log.info("=== Parallel batch demo ===")

    specs = [
        TaskSpec(
            task_type="text2image",
            params={
                "prompt": "A cyberpunk city skyline at sunset, neon lights, 8K",
                "model_id": "seedream-4-5-251128",
                "resolution": "1k",
                "aspect_ratio": "16:9",
            },
            tag="cyberpunk-city",
        ),
        TaskSpec(
            task_type="text2image",
            params={
                "prompt": "A serene Japanese garden in autumn, koi pond, golden leaves",
                "model_id": "seedream-4-5-251128",
                "resolution": "1k",
                "aspect_ratio": "1:1",
            },
            tag="zen-garden",
        ),
        TaskSpec(
            task_type="text2video",
            params={
                "prompt": "Ocean waves crashing on a rocky shore, cinematic drone shot",
                "model_id": "kling-v3",
                "duration": 5,
                "resolution": "1080p",
                "aspect_ratio": "16:9",
            },
            tag="ocean-waves",
        ),
    ]

    runner = TaskRunner(client, max_concurrent=5)
    start = time.monotonic()
    results = await runner.run(specs, on_progress=progress_callback)
    print()  # newline after progress bar

    elapsed = time.monotonic() - start
    log.info("Batch finished in %.1fs", elapsed)
    for r in results:
        tag = r.metadata.get("tag", r.task_id)
        if r.success:
            log.info("  [OK]   %-20s -> %s  (%.1fs)", tag, r.result_url, r.elapsed_seconds)
        else:
            log.info("  [FAIL] %-20s -> %s  (%.1fs)", tag, r.error, r.elapsed_seconds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.environ.get("FOTOR_OPENAPI_KEY", "")
    endpoint = os.environ.get("FOTOR_OPENAPI_ENDPOINT", "https://api.fotor.com")

    if not api_key:
        print("ERROR: set FOTOR_OPENAPI_KEY environment variable first", file=sys.stderr)
        sys.exit(1)

    client = FotorClient(api_key=api_key, endpoint=endpoint)

    await single_task_demo(client)
    await batch_demo(client)


if __name__ == "__main__":
    asyncio.run(main())
