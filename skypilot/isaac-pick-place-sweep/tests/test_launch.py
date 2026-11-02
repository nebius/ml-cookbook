"""Validate the customer path without submitting cloud jobs."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import launch
import sky
import yaml


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.options = SimpleNamespace(
            image="registry.example/isaac:v1",
            bucket="example-bucket",
            region="eu-north1",
            prefix="isaac-pick-place",
            record_motion=True,
        )
        self.points = launch.expand_sweep(
            json.loads(Path(launch.__file__).with_name("sweep.json").read_text())
        )

    def configs(self):
        return [
            launch.job_config(self.options, "run-1", f"case-{index:03d}", point)
            for index, point in enumerate(self.points)
        ]

    def test_grid_and_real_skypilot_task_schema(self):
        self.assertEqual(
            self.points,
            [
                {"pick_x": 0.4, "place_y": 0.2},
                {"pick_x": 0.4, "place_y": 0.22},
                {"pick_x": 0.42, "place_y": 0.2},
                {"pick_x": 0.42, "place_y": 0.22},
            ],
        )
        configs = self.configs()
        self.assertEqual(len({config["name"] for config in configs}), 4)
        for config, point in zip(configs, self.points, strict=True):
            task = sky.Task.from_yaml_config(config)
            self.assertEqual(task.envs["PICK_X"], str(point["pick_x"]))
            self.assertEqual(task.envs["PLACE_Y"], str(point["place_y"]))
            self.assertEqual(task.envs["RECORD_MOTION"], "1")
            self.assertEqual(task.envs["S3_ENDPOINT_URL"], "https://storage.eu-north1.nebius.cloud")
            resources = next(iter(task.resources))
            self.assertEqual(resources.accelerators, {"L40S": 1})
            self.assertEqual(resources.region, "eu-north1")
            self.assertFalse(resources.use_spot)
            self.assertIsNone(resources.extract_docker_image())
            self.assertIn('"$IMAGE" --run-id "$RUN_ID" --case-id "$CASE_ID"', task.run)
            self.assertIn('--pick-x "$PICK_X" --place-y "$PLACE_Y"', task.run)
            self.assertIn("--env AWS_SECRET_ACCESS_KEY", task.run)
            self.assertNotIn("--env AWS_SECRET_ACCESS_KEY=", task.run)

    def test_invalid_sweep_inputs(self):
        invalid = [
            [],
            {},
            {"pick_x": [], "place_y": [0.2]},
            {"pick_x": [True], "place_y": [0.2]},
            {"pick_x": [float("nan")], "place_y": [0.2]},
            {"pick_x": [0.4], "place_y": ["0.2"]},
            {"pick_x": [0.4], "place_y": [0.2], "unexpected": 1},
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                launch.expand_sweep(config)

    def test_submission_error_is_informative_without_exposing_credentials(self):
        args = [
            "launch.py",
            "--image",
            self.options.image,
            "--bucket",
            self.options.bucket,
            "--single",
        ]
        credentials = {
            name: f"example-value-for-{name}"
            for name in ("AWS_SECRET_ACCESS_KEY", "REGISTRY_PASSWORD")
        }
        error = RuntimeError("No project found: " + " ".join(credentials.values()))
        with (
            patch.object(sys, "argv", args),
            patch("launch.read_secrets", return_value=credentials),
            patch("launch.submit_jobs", side_effect=error),
            self.assertRaises(SystemExit) as raised,
        ):
            launch.main()
        message = str(raised.exception)
        self.assertIn("No project found", message)
        self.assertIn("[redacted]", message)
        for value in credentials.values():
            self.assertNotIn(value, message)

    def test_required_and_optional_credentials(self):
        env = {name: f"value-for-{name}" for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(launch.read_secrets(), env)
            with (
                patch.dict(os.environ, {"REGISTRY_USERNAME": "iam"}),
                self.assertRaisesRegex(ValueError, "REGISTRY_SERVER, REGISTRY_PASSWORD"),
            ):
                launch.read_secrets()
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"),
        ):
            launch.read_secrets()

    def test_dry_run_does_not_read_credentials_or_submit(self):
        for single, expected_count in (([], 4), (["--single"], 1)):
            args = [
                "launch.py",
                "--image",
                self.options.image,
                "--bucket",
                self.options.bucket,
                "--dry-run",
                *single,
            ]
            output = io.StringIO()
            with (
                patch.object(sys, "argv", args),
                redirect_stdout(output),
                patch("launch.read_secrets") as read,
                patch("launch.submit_jobs") as submit,
            ):
                launch.main()
            read.assert_not_called()
            submit.assert_not_called()
            configs = list(yaml.safe_load_all(output.getvalue().split("\n", 1)[1]))
            self.assertEqual(len(configs), expected_count)
            self.assertTrue(all(set(config["secrets"].values()) == {None} for config in configs))

    def test_four_submissions_save_ids_without_secrets(self):
        credentials = {
            name: f"example-value-for-{name}"
            for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "REGISTRY_PASSWORD")
        }
        configs = self.configs()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "run-1" / "jobs.json"
            with (
                patch("sky.jobs.launch", side_effect=[f"request-{i}" for i in range(4)]) as submit,
                patch("sky.get", side_effect=[([i + 10], None) for i in range(4)]) as get,
            ):
                launch.submit_jobs(configs, manifest, credentials)
            jobs = json.loads(manifest.read_text())
            self.assertEqual(submit.call_count, 4)
            self.assertEqual(get.call_count, 4)
            self.assertEqual([job["job_id"] for job in jobs], [10, 11, 12, 13])
            self.assertEqual(
                jobs[0]["s3_uri"], "s3://example-bucket/isaac-pick-place/run-1/case-000/"
            )
            for value in credentials.values():
                self.assertNotIn(value, manifest.read_text())
            for call in submit.call_args_list:
                self.assertEqual(
                    {
                        name: value.get_secret_value()
                        for name, value in call.args[0].secrets.items()
                    },
                    credentials,
                )

    def test_receipts_survive_submission_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "run-1" / "jobs.json"
            with (
                patch("sky.jobs.launch", side_effect=["request-0", "request-1"]) as submit,
                patch("sky.get", side_effect=[([10], None), RuntimeError("connection lost")]),
                self.assertRaisesRegex(RuntimeError, "connection lost"),
            ):
                launch.submit_jobs(self.configs(), manifest, {})
            self.assertEqual(submit.call_count, 2)
            jobs = json.loads(manifest.read_text())
            self.assertEqual(jobs[0]["job_id"], 10)
            self.assertEqual(jobs[1]["request_id"], "request-1")
            self.assertNotIn("job_id", jobs[1])
