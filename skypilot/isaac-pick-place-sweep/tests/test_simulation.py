"""Check simulation results and publication without importing Isaac Sim."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from run import evaluate, upload
from run import main as run_case


class SimulationTest(unittest.TestCase):
    def test_failed_upload_does_not_publish_complete(self):
        client = Mock()
        client.upload_file.side_effect = RuntimeError("S3 unavailable")
        env = {
            name: f"example-value-for-{name}"
            for name in (
                "S3_BUCKET",
                "S3_ENDPOINT_URL",
                "AWS_DEFAULT_REGION",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
            )
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"boto3": SimpleNamespace(client=lambda *a, **k: client)}),
            self.assertRaisesRegex(RuntimeError, "S3 unavailable"),
        ):
            upload(Path("result.json"), "run-1", "case-000")
        client.put_object.assert_not_called()

    def test_task_success_needs_controller_and_final_pose(self):
        target = [-0.4, 0.2, 0.02575]
        self.assertTrue(evaluate(True, target, target)["success"])
        self.assertFalse(evaluate(False, target, target)["success"])
        self.assertFalse(evaluate(True, [-0.2, 0.2, 0.02575], target)["success"])
        self.assertFalse(evaluate(True, [float("nan"), 0.2, 0.02575], target)["success"])

    def test_upload_marks_complete_last(self):
        calls = []
        client = SimpleNamespace(
            upload_file=lambda *args: calls.append(("result", args)),
            put_object=lambda **kwargs: calls.append(("complete", kwargs)),
        )
        env = {
            "S3_BUCKET": "bucket",
            "S3_ENDPOINT_URL": "https://storage.example",
            "AWS_DEFAULT_REGION": "eu-north1",
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "S3_PREFIX": "isaac-pick-place",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text("{}")
            with (
                patch.dict(os.environ, env),
                patch.dict(sys.modules, {"boto3": SimpleNamespace(client=lambda *a, **k: client)}),
            ):
                recording = Path(directory) / "trajectory.json"
                recording.write_text("{}")
                upload(result, "run-1", "case-000", recording)
        self.assertEqual([name for name, _ in calls], ["result", "result", "complete"])
        self.assertEqual(calls[0][1][2], "isaac-pick-place/run-1/case-000/trajectory.json")
        self.assertEqual(calls[1][1][2], "isaac-pick-place/run-1/case-000/result.json")
        self.assertEqual(calls[2][1]["Key"], "isaac-pick-place/run-1/case-000/COMPLETE")

    def test_upload_error_keeps_job_failed(self):
        closed = []
        app = SimpleNamespace(close=lambda: closed.append(True))
        env = {
            "S3_BUCKET": "bucket",
            "S3_ENDPOINT_URL": "https://storage.example",
            "AWS_DEFAULT_REGION": "eu-north1",
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
        }
        with tempfile.TemporaryDirectory() as directory:
            arguments = [
                "run.py",
                "--run-id",
                "run-1",
                "--case-id",
                "case-000",
                "--pick-x",
                "0.4",
                "--place-y",
                "0.2",
                "--output-dir",
                directory,
            ]
            with (
                patch.dict(os.environ, env),
                patch.dict(sys.modules, {"isaacsim": SimpleNamespace(SimulationApp=lambda _: app)}),
                patch.object(sys, "argv", arguments),
                patch("run.simulate", return_value={"success": True}),
                patch("run.upload", side_effect=RuntimeError("upload failed")),
                self.assertRaisesRegex(RuntimeError, "upload failed"),
            ):
                run_case()
        self.assertEqual(closed, [])
