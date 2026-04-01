from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

import aiohttp

from .models import TaskResult, TaskStatus

logger = logging.getLogger("fotor_sdk")

_OK_CODE = "000"
_TASK_STATUS_PATH = "/v1/aiart/tasks"
_CREDITS_PATH = "/v1/credits"

_DEFAULT_ENDPOINT = "https://api-b.fotor.com"
_DEFAULT_POLL_INTERVAL = 2.0
_DEFAULT_MAX_POLL_SECONDS = 1200


class FotorAPIError(Exception):
    """Raised when the Fotor API returns an error response."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


class FotorClient:
    """Lightweight async client for the Fotor OpenAPI.

    Only requires an API key; no S3, no credit checks, no model config.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str = _DEFAULT_ENDPOINT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        max_poll_seconds: float = _DEFAULT_MAX_POLL_SECONDS,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._poll_interval = poll_interval
        self._max_poll_seconds = max_poll_seconds

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    # ------------------------------------------------------------------
    # Core async methods
    # ------------------------------------------------------------------

    async def create_task(self, path: str, payload: dict[str, Any]) -> str:
        """Submit a task and return the ``task_id``."""
        url = f"{self._endpoint}{path}"
        logger.info("POST %s  payload=%s", url, payload)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise FotorAPIError(
                        f"HTTP {resp.status} from {url}: {body}", code=str(resp.status)
                    )
                data = await resp.json()

        if data.get("code") != _OK_CODE:
            raise FotorAPIError(data.get("msg", "unknown error"), code=data.get("code"))

        task_id = data.get("data", {}).get("taskId")
        if not task_id:
            raise FotorAPIError("API response missing data.taskId")
        logger.info("Task created: %s", task_id)
        return task_id

    async def get_task_status(self, task_id: str) -> TaskResult:
        """Query current status of *task_id* and return a ``TaskResult``."""
        url = f"{self._endpoint}{_TASK_STATUS_PATH}/{task_id}"
        logger.debug("GET %s", url)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return TaskResult(task_id=task_id, status=TaskStatus.FAILED,
                                     error=f"HTTP {resp.status}")
                data = await resp.json()

        if data.get("code") != _OK_CODE:
            return TaskResult(task_id=task_id, status=TaskStatus.FAILED,
                              error=data.get("msg", "bad code"))

        task_data = data.get("data", {})
        api_status = task_data.get("status", -1)
        credits_increment = task_data.get("creditsIncrement")
        try:
            if credits_increment is not None:
                credits_increment = int(credits_increment)
        except (TypeError, ValueError):
            credits_increment = None

        if api_status == TaskStatus.COMPLETED:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result_url=task_data.get("resultUrl"),
                creditsIncrement=credits_increment,
            )
        if api_status == TaskStatus.FAILED:
            error_msg = "NSFW_CONTENT" if task_data.get("hasHsfw") else task_data.get("msg", "task failed")
            return TaskResult(task_id=task_id, status=TaskStatus.FAILED, error=error_msg, creditsIncrement=credits_increment)
        return TaskResult(task_id=task_id, status=TaskStatus.IN_PROGRESS, creditsIncrement=credits_increment)

    async def get_credits(self) -> dict[str, Any]:
        """
        Query remaining user credits.

        API: GET /v1/credits
        Request has headers only (no body).
        """
        url = f"{self._endpoint}{_CREDITS_PATH}"
        logger.debug("GET %s", url)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise FotorAPIError(
                        f"HTTP {resp.status} from {url}: {body}", code=str(resp.status)
                    )
                data = await resp.json()

        if data.get("code") != _OK_CODE:
            raise FotorAPIError(data.get("msg", "unknown error"), code=data.get("code"))

        credits_data = data.get("data")
        if not isinstance(credits_data, dict):
            raise FotorAPIError("API response missing data for credits")
        return credits_data

    async def wait_for_task(
        self,
        task_id: str,
        on_poll: Callable[[TaskResult], None] | None = None,
    ) -> TaskResult:
        """Poll *task_id* until it completes, fails, or times out.

        *on_poll* is invoked after every poll cycle with the latest
        ``TaskResult`` so callers can implement progress UIs.
        """
        start = time.monotonic()
        while True:
            try:
                result = await self.get_task_status(task_id)
            except Exception:
                await asyncio.sleep(self._poll_interval)
                if (time.monotonic() - start) >= self._max_poll_seconds:
                    return TaskResult(task_id=task_id, status=TaskStatus.TIMEOUT,
                                      error=f"polling timed out after {self._max_poll_seconds}s")
                continue
            result.elapsed_seconds = time.monotonic() - start

            if on_poll is not None:
                on_poll(result)

            if result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return result

            if result.elapsed_seconds >= self._max_poll_seconds:
                result.status = TaskStatus.TIMEOUT
                result.error = f"polling timed out after {self._max_poll_seconds}s"
                return result

            await asyncio.sleep(self._poll_interval)

    async def submit_and_wait(
        self,
        path: str,
        payload: dict[str, Any],
        on_poll: Callable[[TaskResult], None] | None = None,
    ) -> TaskResult:
        """Convenience: create a task then poll until done."""
        start = time.monotonic()
        task_id = await self.create_task(path, payload)
        result = await self.wait_for_task(task_id, on_poll=on_poll)
        result.elapsed_seconds = time.monotonic() - start
        return result

    # ------------------------------------------------------------------
    # Sync wrappers
    # ------------------------------------------------------------------

    def create_task_sync(self, path: str, payload: dict[str, Any]) -> str:
        return asyncio.run(self.create_task(path, payload))

    def wait_for_task_sync(self, task_id: str) -> TaskResult:
        return asyncio.run(self.wait_for_task(task_id))

    def submit_and_wait_sync(self, path: str, payload: dict[str, Any]) -> TaskResult:
        return asyncio.run(self.submit_and_wait(path, payload))

    def get_credits_sync(self) -> dict[str, Any]:
        return asyncio.run(self.get_credits())