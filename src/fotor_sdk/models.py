from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class TaskStatus(IntEnum):
    UNKNOWN = -1
    IN_PROGRESS = 0
    COMPLETED = 1
    FAILED = 2
    TIMEOUT = 3
    CANCELLED = 4


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus = TaskStatus.UNKNOWN
    result_url: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == TaskStatus.COMPLETED and self.result_url is not None

    def __repr__(self) -> str:
        status_name = self.status.name
        if self.success:
            return f"TaskResult(task_id={self.task_id!r}, status={status_name}, url={self.result_url!r})"
        return f"TaskResult(task_id={self.task_id!r}, status={status_name}, error={self.error!r})"


@dataclass
class TaskSpec:
    """Describes a single task to be submitted to the runner."""
    task_type: str
    params: dict[str, Any] = field(default_factory=dict)
    tag: str = ""
