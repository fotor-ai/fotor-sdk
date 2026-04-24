import asyncio
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from fotor_sdk import FotorAPIError, FotorClient, TaskResult, TaskStatus


def _load_module(path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SequencedStatusClient(FotorClient):
    def __init__(self, responses: list[TaskResult | Exception]):
        super().__init__(api_key="test-key", poll_interval=0.0, max_poll_seconds=1.0)
        self._responses = iter(responses)

    async def get_task_status(self, task_id: str) -> TaskResult:
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class SyncWrapperClientStub:
    def submit_and_wait_sync(self, path: str, payload: dict[str, object]) -> TaskResult:
        async def _result() -> TaskResult:
            return TaskResult(task_id="demo", status=TaskStatus.COMPLETED, result_url="https://example.com/out.png")

        return asyncio.run(_result())


class WaitForTaskRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_task_retries_retryable_api_errors(self) -> None:
        client = SequencedStatusClient(
            [
                FotorAPIError("temporary upstream failure", code="503"),
                TaskResult(task_id="task-1", status=TaskStatus.IN_PROGRESS),
                TaskResult(task_id="task-1", status=TaskStatus.COMPLETED, result_url="https://example.com/final.png"),
            ]
        )

        result = await client.wait_for_task("task-1")

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result_url, "https://example.com/final.png")

    async def test_wait_for_task_fails_fast_on_non_retryable_api_errors(self) -> None:
        client = SequencedStatusClient([FotorAPIError("bad request", code="400")])

        result = await client.wait_for_task("task-2")

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("bad request", result.error or "")


class ExampleRegressionTests(unittest.TestCase):
    def test_parallel_generate_main_exits_without_name_error_when_api_key_missing(self) -> None:
        module = _load_module(ROOT / "examples" / "parallel_generate.py", "parallel_generate_example")

        with self.assertRaises(SystemExit) as ctx:
            asyncio.run(module.main())

        self.assertEqual(ctx.exception.code, 1)

    def test_sync_wrapper_example_runs_inside_async_test_harness(self) -> None:
        module = _load_module(ROOT / "examples" / "test_all_features.py", "test_all_features_example")
        module.results_summary.clear()

        module.test_sync_wrapper(SyncWrapperClientStub())

        self.assertTrue(module.results_summary)
        name, status, _detail = module.results_summary[-1]
        self.assertEqual(name, "sync_wrapper")
        self.assertIn("[PASS]", status)
