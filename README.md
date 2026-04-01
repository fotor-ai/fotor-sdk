# fotor-sdk

Lightweight, async-first Python SDK for the [Fotor OpenAPI](https://api-b.fotor.com).
Generate images and videos with a single API key -- no MCP server, no S3, no
internal services required.

## Installation

```bash
pip install fotor-sdk
```

Or install from GitHub:

```bash
pip install git+https://github.com/fotor-ai/fotor-sdk.git
```

For local development:

```bash
git clone https://github.com/fotor-ai/fotor-sdk.git
cd fotor-sdk
pip install -e .
```

## Quick Start

### Single Task

```python
import asyncio
import os
from fotor_sdk import FotorClient, text2image

async def main():
    client = FotorClient(api_key=os.environ["FOTOR_OPENAPI_KEY"])
    result = await text2image(
        client,
        prompt="A diamond kitten on velvet, studio lighting",
        model_id="seedream-4-5-251128",
        resolution="2k",
        aspect_ratio="1:1",
    )
    print(result.result_url)

asyncio.run(main())
```

### Parallel Batch with Progress

```python
import asyncio
import os
from fotor_sdk import FotorClient, TaskRunner, TaskSpec

async def main():
    client = FotorClient(api_key=os.environ["FOTOR_OPENAPI_KEY"])
    runner = TaskRunner(client, max_concurrent=5)

    specs = [
        TaskSpec("text2image", {"prompt": "A cat", "model_id": "seedream-4-5-251128"}, tag="cat"),
        TaskSpec("text2image", {"prompt": "A dog", "model_id": "seedream-4-5-251128"}, tag="dog"),
        TaskSpec("text2video", {"prompt": "Sunset", "model_id": "kling-v3", "duration": 5}, tag="sunset"),
    ]

    def on_progress(total, completed, failed, in_progress, latest):
        print(f"  {completed + failed}/{total} done, latest: {latest.metadata.get('tag')}")

    results = await runner.run(specs, on_progress=on_progress)
    for r in results:
        print(f"{r.metadata.get('tag')}: {r.status.name} -> {r.result_url}")

asyncio.run(main())
```

## Configuration

| Environment Variable | Required | Default | Description |
|---|---|---|---|
| `FOTOR_OPENAPI_KEY` | Yes | -- | Your Fotor OpenAPI key |
| `FOTOR_OPENAPI_ENDPOINT` | No | `https://api-b.fotor.com` | API base URL |

## Available Task Functions

| Function | Description |
|---|---|
| `text2image()` | Generate image from text |
| `image2image()` | Edit / multi-reference generation |
| `image_upscale()` | 2x or 4x upscale |
| `background_remove()` | Remove background |
| `text2video()` | Generate video from text |
| `single_image2video()` | Animate a single image |
| `start_end_frame2video()` | Interpolate between two frames |
| `multiple_image2video()` | Video from multiple images |

## Core Classes

### FotorClient

```python
FotorClient(
    api_key: str,
    endpoint: str = "https://api-b.fotor.com",
    poll_interval: float = 2.0,
    max_poll_seconds: float = 1200,
)
```

**Methods:**
- `await create_task(path, payload) -> task_id`
- `await get_task_status(task_id) -> TaskResult`
- `await get_credits() -> dict`
- `await wait_for_task(task_id) -> TaskResult`
- `await submit_and_wait(path, payload) -> TaskResult`
- `submit_and_wait_sync(path, payload) -> TaskResult`
- `get_credits_sync() -> dict`

### TaskRunner

```python
TaskRunner(client: FotorClient, max_concurrent: int = 5)
```

- `await run(specs, on_progress=None) -> list[TaskResult]`
- `run_sync(specs, on_progress=None) -> list[TaskResult]`

### TaskResult

```python
TaskResult(task_id, status, result_url, error, elapsed_seconds, metadata)
result.success  # True when COMPLETED with a result_url
```

### TaskSpec

```python
TaskSpec(task_type: str, params: dict, tag: str = "")
```

## Error Handling

```python
from fotor_sdk import FotorAPIError

try:
    result = await text2image(client, prompt="...", model_id="bad-model")
except FotorAPIError as e:
    print(f"API error: {e}  code={e.code}")
```

For batch runs, failed tasks appear in results with `status=FAILED` and the
`error` field populated. The runner never raises on individual task failures.

## License

MIT
