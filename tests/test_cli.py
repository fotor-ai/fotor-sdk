import pathlib
import json
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _completed_result(url: str = "https://example.com/out.png"):
    from fotor_sdk import TaskResult, TaskStatus

    return TaskResult(task_id="task-1", status=TaskStatus.COMPLETED, result_url=url, elapsed_seconds=1.23)


class UploadHelperTests(unittest.TestCase):
    def test_is_url_recognizes_http_and_https_only(self) -> None:
        from fotor_sdk.upload import is_url

        self.assertTrue(is_url("https://example.com/a.png"))
        self.assertTrue(is_url("http://example.com/a.png"))
        self.assertFalse(is_url("./local.png"))
        self.assertFalse(is_url("ftp://example.com/a.png"))

    def test_upload_type_for_task_maps_supported_task(self) -> None:
        from fotor_sdk.upload import upload_type_for_task

        self.assertEqual(upload_type_for_task("image2image"), "img2img")

    def test_upload_type_for_task_rejects_unsupported_task(self) -> None:
        from fotor_sdk.upload import upload_type_for_task

        with self.assertRaises(ValueError):
            upload_type_for_task("text2image")


class CLIShellTests(unittest.TestCase):
    def test_missing_api_key_returns_json_error(self) -> None:
        from fotor_sdk.cli import main

        stdout = StringIO()
        stderr = StringIO()
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(["text2image", "--prompt", "A cat"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn('"error": "FOTOR_OPENAPI_KEY not set"', stderr.getvalue())

    def test_pyproject_exposes_fotor_console_script(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('[project.scripts]', pyproject)
        self.assertIn('fotor = "fotor_sdk.cli:main"', pyproject)


class CLICreditsTests(unittest.TestCase):
    def test_credits_outputs_client_credits_json(self) -> None:
        from fotor_sdk import cli

        stdout = StringIO()
        with unittest.mock.patch.dict("os.environ", {"FOTOR_OPENAPI_KEY": "key"}, clear=True):
            with unittest.mock.patch.object(
                cli.FotorClient,
                "get_credits_sync",
                return_value={"remainingCredits": 123, "frozenCredits": 4},
            ) as get_credits:
                with redirect_stdout(stdout):
                    exit_code = cli.main(["credits"])

        self.assertEqual(exit_code, 0)
        get_credits.assert_called_once_with()
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {"remainingCredits": 123, "frozenCredits": 4},
        )

    def test_help_lists_credits_command(self) -> None:
        from fotor_sdk import cli

        stdout = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(stdout):
                cli.main(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("credits", stdout.getvalue())


class CLIImageCommandTests(unittest.TestCase):
    def test_text2image_uses_default_model_and_resolution(self) -> None:
        from fotor_sdk import cli

        stdout = StringIO()
        with unittest.mock.patch.dict("os.environ", {"FOTOR_OPENAPI_KEY": "key"}, clear=True):
            with unittest.mock.patch.object(cli, "text2image", return_value=_completed_result()) as task:
                with redirect_stdout(stdout):
                    exit_code = cli.main(["text2image", "--prompt", "A cat"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["result_url"], "https://example.com/out.png")
        _, kwargs = task.call_args
        self.assertEqual(kwargs["prompt"], "A cat")
        self.assertEqual(kwargs["model_id"], "gpt-image-2")
        self.assertEqual(kwargs["resolution"], "2k")

    def test_image2image_uploads_local_path_before_task(self) -> None:
        from fotor_sdk import cli
        from fotor_sdk.upload import UploadResult

        upload_result = UploadResult(
            file="/tmp/input.png",
            task_type="image2image",
            upload_type="img2img",
            suffix="png",
            file_url="https://cdn.example.com/input.png",
            upload_url="https://upload.example.com/input.png",
        )

        with unittest.mock.patch.dict("os.environ", {"FOTOR_OPENAPI_KEY": "key"}, clear=True):
            with unittest.mock.patch.object(cli, "upload_image_sync", return_value=upload_result) as upload:
                with unittest.mock.patch.object(cli, "image2image", return_value=_completed_result()) as task:
                    with redirect_stdout(StringIO()):
                        exit_code = cli.main(["image2image", "--image", "./input.png", "--prompt", "Edit it"])

        self.assertEqual(exit_code, 0)
        upload.assert_called_once()
        _, kwargs = task.call_args
        self.assertEqual(kwargs["image_urls"], ["https://cdn.example.com/input.png"])

    def test_image2image_remote_url_does_not_upload(self) -> None:
        from fotor_sdk import cli

        image_url = "https://example.com/input.png"
        with unittest.mock.patch.dict("os.environ", {"FOTOR_OPENAPI_KEY": "key"}, clear=True):
            with unittest.mock.patch.object(cli, "upload_image_sync") as upload:
                with unittest.mock.patch.object(cli, "image2image", return_value=_completed_result()) as task:
                    with redirect_stdout(StringIO()):
                        exit_code = cli.main(["image2image", "--image", image_url, "--prompt", "Edit it"])

        self.assertEqual(exit_code, 0)
        upload.assert_not_called()
        _, kwargs = task.call_args
        self.assertEqual(kwargs["image_urls"], [image_url])

    def test_image2image_accepts_space_separated_images_after_one_flag(self) -> None:
        from fotor_sdk import cli

        image_urls = ["https://example.com/input-1.png", "https://example.com/input-2.png"]
        with unittest.mock.patch.dict("os.environ", {"FOTOR_OPENAPI_KEY": "key"}, clear=True):
            with unittest.mock.patch.object(cli, "upload_image_sync") as upload:
                with unittest.mock.patch.object(cli, "image2image", return_value=_completed_result()) as task:
                    with redirect_stdout(StringIO()):
                        exit_code = cli.main(["image2image", "--image", *image_urls, "--prompt", "Edit them"])

        self.assertEqual(exit_code, 0)
        upload.assert_not_called()
        _, kwargs = task.call_args
        self.assertEqual(kwargs["image_urls"], image_urls)


class CLIVideoAndFallbackTests(unittest.TestCase):
    def test_text2video_uses_default_model(self) -> None:
        from fotor_sdk import cli

        stdout = StringIO()
        with unittest.mock.patch.dict("os.environ", {"FOTOR_OPENAPI_KEY": "key"}, clear=True):
            with unittest.mock.patch.object(cli, "text2video", return_value=_completed_result("https://example.com/out.mp4")) as task:
                with redirect_stdout(stdout):
                    exit_code = cli.main(["text2video", "--prompt", "A sunset"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["result_url"], "https://example.com/out.mp4")
        _, kwargs = task.call_args
        self.assertEqual(kwargs["prompt"], "A sunset")
        self.assertEqual(kwargs["model_id"], "doubao-seedance-2-0-260128")

    def test_multi_image_video_accepts_space_separated_images_after_one_flag(self) -> None:
        from fotor_sdk import cli

        image_urls = ["https://example.com/input-1.png", "https://example.com/input-2.png"]
        with unittest.mock.patch.dict("os.environ", {"FOTOR_OPENAPI_KEY": "key"}, clear=True):
            with unittest.mock.patch.object(cli, "upload_image_sync") as upload:
                with unittest.mock.patch.object(
                    cli,
                    "multiple_image2video",
                    return_value=_completed_result("https://example.com/out.mp4"),
                ) as task:
                    with redirect_stdout(StringIO()):
                        exit_code = cli.main(
                            [
                                "multi-image-video",
                                "--image",
                                *image_urls,
                                "--prompt",
                                "Animate them",
                            ]
                        )

        self.assertEqual(exit_code, 0)
        upload.assert_not_called()
        _, kwargs = task.call_args
        self.assertEqual(kwargs["image_urls"], image_urls)

    def test_fallback_retries_model_api_error(self) -> None:
        from fotor_sdk import cli
        from fotor_sdk import FotorAPIError

        stdout = StringIO()
        with unittest.mock.patch.dict("os.environ", {"FOTOR_OPENAPI_KEY": "key"}, clear=True):
            with unittest.mock.patch.object(
                cli,
                "text2image",
                side_effect=[FotorAPIError("primary failed", code="500"), _completed_result()],
            ) as task:
                with redirect_stdout(stdout):
                    exit_code = cli.main(["text2image", "--prompt", "A cat", "--model", "gpt-image-2"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["fallback_used"])
        self.assertEqual(payload["original_model_id"], "gpt-image-2")
        self.assertEqual(payload["fallback_model_id"], "gemini-3.1-flash-image-preview")
        self.assertEqual(task.call_count, 2)
        self.assertEqual(task.call_args_list[0].kwargs["model_id"], "gpt-image-2")
        self.assertEqual(task.call_args_list[1].kwargs["model_id"], "gemini-3.1-flash-image-preview")

    def test_insufficient_credits_does_not_retry_fallback(self) -> None:
        from fotor_sdk import cli
        from fotor_sdk import FotorAPIError

        stderr = StringIO()
        with unittest.mock.patch.dict("os.environ", {"FOTOR_OPENAPI_KEY": "key"}, clear=True):
            with unittest.mock.patch.object(
                cli,
                "text2image",
                side_effect=FotorAPIError("No enough credits", code="510"),
            ) as task:
                with redirect_stderr(stderr):
                    exit_code = cli.main(["text2image", "--prompt", "A cat", "--model", "gpt-image-2"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(task.call_count, 1)
        self.assertIn("No enough credits", stderr.getvalue())


class CLIBatchTests(unittest.TestCase):
    def test_batch_runs_task_specs_with_configured_concurrency(self) -> None:
        from fotor_sdk import cli

        class FakeTaskRunner:
            instances = []

            def __init__(self, client, max_concurrent: int = 5):
                self.client = client
                self.max_concurrent = max_concurrent
                self.specs = []
                FakeTaskRunner.instances.append(self)

            async def run(self, specs, on_progress=None, on_task_poll=None):
                from fotor_sdk import TaskResult, TaskStatus

                self.specs = specs
                results = []
                for index, spec in enumerate(specs, start=1):
                    result = TaskResult(
                        task_id=f"task-{index}",
                        status=TaskStatus.COMPLETED,
                        result_url=f"https://example.com/out-{index}.png",
                        elapsed_seconds=1.0,
                        metadata={"tag": spec.tag},
                    )
                    results.append(result)
                    if on_progress is not None:
                        on_progress(
                            total=len(specs),
                            completed=index,
                            failed=0,
                            in_progress=len(specs) - index,
                            latest=result,
                        )
                return results

        tasks = [
            {
                "task_type": "text2image",
                "tag": "cat",
                "params": {
                    "prompt": "A cat",
                    "model_id": "gpt-image-2",
                    "aspect_ratio": "1:1",
                    "resolution": "2k",
                },
            },
            {
                "task_type": "text2video",
                "tag": "sunset",
                "params": {
                    "prompt": "A sunset",
                    "model_id": "doubao-seedance-2-0-260128",
                    "duration": 5,
                    "resolution": "1080p",
                    "aspect_ratio": "16:9",
                    "audio_enable": False,
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            batch_file = pathlib.Path(tmp) / "tasks.json"
            batch_file.write_text(json.dumps(tasks), encoding="utf-8")

            stdout = StringIO()
            stderr = StringIO()
            with unittest.mock.patch.dict("os.environ", {"FOTOR_OPENAPI_KEY": "key"}, clear=True):
                with unittest.mock.patch.object(cli, "TaskRunner", FakeTaskRunner, create=True):
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        exit_code = cli.main(["batch", "--file", str(batch_file), "--concurrency", "2"])

        self.assertEqual(exit_code, 0)
        runner = FakeTaskRunner.instances[0]
        self.assertEqual(runner.max_concurrent, 2)
        self.assertEqual([spec.task_type for spec in runner.specs], ["text2image", "text2video"])
        self.assertEqual(runner.specs[0].params["prompt"], "A cat")

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["summary"], {"total": 2, "completed": 2, "failed": 0})
        self.assertEqual([result["tag"] for result in payload["results"]], ["cat", "sunset"])
        self.assertIn("batch progress", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
