"""Parallel task runner with concurrency control and status monitoring."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from .client import FotorClient
from .models import TaskResult, TaskSpec, TaskStatus
from . import tasks as _tasks

logger = logging.getLogger("fotor_sdk")

# Map task_type strings to the async functions in tasks.py
_TASK_DISPATCH: dict[str, Callable[..., Any]] = {
    "text2image": _tasks.text2image,
    "image2image": _tasks.image2image,
    "image_upscale": _tasks.image_upscale,
    "background_remove": _tasks.background_remove,
    "text2video": _tasks.text2video,
    "single_image2video": _tasks.single_image2video,
    "start_end_frame2video": _tasks.start_end_frame2video,
    "multiple_image2video": _tasks.multiple_image2video,
}


class _ProgressTracker:
    """Thread-safe progress state shared across concurrent tasks."""

    def __init__(self, total: int, on_progress: Callable[..., None] | None):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.in_progress = total
        self._on_progress = on_progress
        self._lock = asyncio.Lock()

    async def mark_done(self, result: TaskResult) -> None:
        async with self._lock:
            self.in_progress -= 1
            if result.status == TaskStatus.COMPLETED:
                self.completed += 1
            else:
                self.failed += 1
            if self._on_progress:
                self._on_progress(
                    total=self.total,
                    completed=self.completed,
                    failed=self.failed,
                    in_progress=self.in_progress,
                    latest=result,
                )

    def summary(self) -> dict[str, int]:
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
        }


class TaskRunner:
    """Execute many Fotor tasks in parallel with bounded concurrency.

    Usage::

        client = FotorClient(api_key="...")
        runner = TaskRunner(client, max_concurrent=5)

        specs = [
            TaskSpec("text2image", {"prompt": "A cat", "model_id": "seedream-4-5-251128"}),
            TaskSpec("text2video", {"prompt": "Ocean", "model_id": "kling-v3", "duration": 5}),
        ]
        results = await runner.run(specs, on_progress=print)
    """

    def __init__(self, client: FotorClient, max_concurrent: int = 5):
        self._client = client
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _execute_one(
        self,
        spec: TaskSpec,
        tracker: _ProgressTracker,
        on_task_poll: Callable[[TaskResult], None] | None,
    ) -> TaskResult:
        fn = _TASK_DISPATCH.get(spec.task_type)
        if fn is None:
            result = TaskResult(
                task_id="",
                status=TaskStatus.FAILED,
                error=f"Unknown task_type: {spec.task_type!r}. "
                      f"Available: {list(_TASK_DISPATCH)}",
            )
            await tracker.mark_done(result)
            return result

        async with self._semaphore:
            start = time.monotonic()
            try:
                result = await fn(self._client, on_poll=on_task_poll, **spec.params)
                result.metadata["tag"] = spec.tag
            except Exception as exc:
                result = TaskResult(
                    task_id="",
                    status=TaskStatus.FAILED,
                    error=str(exc),
                    elapsed_seconds=time.monotonic() - start,
                    metadata={"tag": spec.tag},
                )
            await tracker.mark_done(result)
            return result

    async def run(
        self,
        specs: list[TaskSpec],
        on_progress: Callable[..., None] | None = None,
        on_task_poll: Callable[[TaskResult], None] | None = None,
    ) -> list[TaskResult]:
        """Run all *specs* in parallel (up to ``max_concurrent``).

        Parameters
        ----------
        specs:
            List of :class:`TaskSpec` describing what to execute.
        on_progress:
            Called after each task finishes with keyword args
            ``total``, ``completed``, ``failed``, ``in_progress``, ``latest``.
        on_task_poll:
            Called during polling of each individual task (per-poll-cycle).

        Returns
        -------
        list[TaskResult]
            Results in the same order as *specs*.
        """
        if not specs:
            return []

        tracker = _ProgressTracker(len(specs), on_progress)
        coros = [self._execute_one(spec, tracker, on_task_poll) for spec in specs]
        results = await asyncio.gather(*coros, return_exceptions=False)

        summary = tracker.summary()
        logger.info(
            "Batch complete: %d/%d succeeded, %d failed",
            summary["completed"], summary["total"], summary["failed"],
        )
        return list(results)

    def run_sync(
        self,
        specs: list[TaskSpec],
        on_progress: Callable[..., None] | None = None,
    ) -> list[TaskResult]:
        """Synchronous wrapper around :meth:`run`."""
        return asyncio.run(self.run(specs, on_progress=on_progress))
