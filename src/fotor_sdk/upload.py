from __future__ import annotations

from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
from urllib import parse, request


UPLOAD_TYPE_BY_TASK = {
    "image2image": "img2img",
    "image_upscale": "img_upscale",
    "background_remove": "bg_remove",
    "single_image2video": "img2video",
    "start_end_frame2video": "img2video",
    "multiple_image2video": "img2video",
}


@dataclass(frozen=True)
class UploadResult:
    file: str
    task_type: str
    upload_type: str
    suffix: str
    file_url: str
    upload_url: str


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def infer_suffix(path: Path, explicit_suffix: str = "") -> str:
    if explicit_suffix:
        return explicit_suffix.lstrip(".").lower()
    suffix = path.suffix.lstrip(".").lower()
    if not suffix:
        raise ValueError(f"Unable to infer file suffix from path: {path}")
    return suffix


def upload_type_for_task(task_type: str) -> str:
    try:
        return UPLOAD_TYPE_BY_TASK[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type for upload mapping: {task_type}") from exc


def request_upload_ticket(
    *,
    endpoint: str,
    api_key: str,
    upload_type: str,
    suffix: str,
) -> dict[str, str]:
    query = parse.urlencode({"type": upload_type, "suffix": suffix})
    req = request.Request(
        f"{endpoint.rstrip('/')}/v1/upload/sign?{query}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "fotor-sdk",
        },
        method="GET",
    )
    with request.urlopen(req) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    data = payload["data"]
    return {
        "upload_url": data["upload_url"],
        "file_url": data["url"],
    }


def upload_file(file_path: Path, upload_url: str) -> None:
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    req = request.Request(
        upload_url,
        data=file_path.read_bytes(),
        headers={"Content-Type": content_type},
        method="PUT",
    )
    with request.urlopen(req):
        return None


def upload_image_sync(
    file_path: str | Path,
    *,
    task_type: str,
    api_key: str,
    endpoint: str,
    suffix: str = "",
) -> UploadResult:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"File not found: {path}")

    resolved_suffix = infer_suffix(path, suffix)
    upload_type = upload_type_for_task(task_type)
    ticket = request_upload_ticket(
        endpoint=endpoint,
        api_key=api_key,
        upload_type=upload_type,
        suffix=resolved_suffix,
    )
    upload_file(path, ticket["upload_url"])
    return UploadResult(
        file=str(path),
        task_type=task_type,
        upload_type=upload_type,
        suffix=resolved_suffix,
        file_url=ticket["file_url"],
        upload_url=ticket["upload_url"],
    )
