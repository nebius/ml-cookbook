from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import chdir, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from gpu_preflight.cli import main
from gpu_preflight.image import inspect_image_reference
from gpu_preflight.profiles import load_profiles


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = load_profiles()

    def report(
        self,
        *,
        platform: str = "linux/arm64",
        reference: str = "registry.example/pytorch:1",
    ):
        return inspect_image_reference(
            reference,
            (self.profiles["gb300"],),
            platform=platform,
            scan_mode="none",
        )

    def run_with_report(
        self,
        report,
        *,
        fail_on: str = "never",
    ) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("gpu_preflight.cli.inspect_image_reference", return_value=report),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main(
                [
                    "inspect",
                    "registry.example/pytorch:1",
                    "--target",
                    "gb300",
                    "--format",
                    "json",
                    "--fail-on",
                    fail_on,
                ]
            )
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_inspect_uses_full_filesystem_scan_by_default(self) -> None:
        report = self.report()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch(
                "gpu_preflight.cli.inspect_image_reference",
                return_value=report,
            ) as inspect_image,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main(
                [
                    "inspect",
                    "registry.example/pytorch:1",
                    "--target",
                    "gb300",
                    "--platform",
                    "arm64",
                    "--format",
                    "json",
                    "--fail-on",
                    "never",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["source"], "image://registry.example/pytorch:1")
        call = inspect_image.call_args
        self.assertEqual(call.args[0], "registry.example/pytorch:1")
        self.assertEqual(call.kwargs["platform"], "linux/arm64")
        self.assertEqual(call.kwargs["scan_mode"], "filesystem")

    def test_inspect_requires_explicit_target(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["inspect", "registry.example/pytorch:1"])
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("inspect requires at least one --target", stderr.getvalue())

    def test_exhaustive_removes_only_the_cuobjdump_file_count_cap(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch(
                "gpu_preflight.cli.inspect_image_reference",
                return_value=self.report(),
            ) as inspect_image,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main(
                [
                    "inspect",
                    "registry.example/pytorch:1",
                    "--target",
                    "gb300",
                    "--exhaustive",
                    "--format",
                    "text",
                    "--fail-on",
                    "never",
                ]
            )

        self.assertEqual(exit_code, 0)
        call = inspect_image.call_args
        self.assertIsNone(call.kwargs["max_cuobjdump_files"])
        self.assertEqual(call.kwargs["max_scan_bytes"], 16384 * 1024 * 1024)
        self.assertEqual(call.kwargs["max_artifact_bytes"], 2048 * 1024 * 1024)
        self.assertEqual(call.kwargs["cuobjdump_fallback_mode"], "auto")
        self.assertEqual(call.kwargs["cuobjdump_timeout_seconds"], 30)

    def test_default_inspect_writes_all_report_formats(self) -> None:
        report = self.report()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with (
                chdir(directory),
                patch("gpu_preflight.cli.inspect_image_reference", return_value=report),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "inspect",
                        "registry.example/pytorch:1",
                        "--target",
                        "gb300",
                        "--fail-on",
                        "never",
                    ]
                )

            report_root = Path(directory) / "gpu-preflight-reports"
            report_dirs = tuple(report_root.iterdir())
            self.assertEqual(len(report_dirs), 1)
            report_dir = report_dirs[0]
            self.assertRegex(
                report_dir.name,
                r"^pytorch-1-gb300-\d{8}T\d{6}\.\d{6}Z$",
            )
            text_path = report_dir / "report.txt"
            json_path = report_dir / "report.json"
            html_path = report_dir / "report.html"
            self.assertEqual(exit_code, 0)
            self.assertEqual(text_path.read_text(encoding="utf-8"), stdout.getvalue())
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["source"],
                "image://registry.example/pytorch:1",
            )
            self.assertIn("NVIDIA GB300 NVL72", html_path.read_text(encoding="utf-8"))
            self.assertIn("Generated report bundle:", stderr.getvalue())
            self.assertIn(str(text_path), stderr.getvalue())
            self.assertIn(str(json_path), stderr.getvalue())
            self.assertIn(str(html_path), stderr.getvalue())

    def test_output_dir_customizes_default_report_bundle(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "custom-reports"
            with (
                patch(
                    "gpu_preflight.cli.inspect_image_reference",
                    return_value=self.report(),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "inspect",
                        "registry.example/pytorch:1",
                        "--target",
                        "gb300",
                        "--output-dir",
                        str(report_dir),
                        "--fail-on",
                        "never",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                {path.name for path in report_dir.iterdir()},
                {"report.txt", "report.json", "report.html"},
            )

    def test_existing_custom_report_bundle_is_protected(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "custom-reports"
            report_dir.mkdir()
            existing = report_dir / "report.json"
            existing.write_text("keep me", encoding="utf-8")
            with (
                patch("gpu_preflight.cli.inspect_image_reference") as inspect_image,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "inspect",
                        "registry.example/pytorch:1",
                        "--target",
                        "gb300",
                        "--output-dir",
                        str(report_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            inspect_image.assert_not_called()
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep me")
            self.assertIn("pass --overwrite", stderr.getvalue())

    def test_overwrite_replaces_existing_custom_report_bundle(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "custom-reports"
            report_dir.mkdir()
            existing = report_dir / "report.json"
            existing.write_text("replace me", encoding="utf-8")
            with (
                patch(
                    "gpu_preflight.cli.inspect_image_reference",
                    return_value=self.report(),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main(
                    [
                        "inspect",
                        "registry.example/pytorch:1",
                        "--target",
                        "gb300",
                        "--output-dir",
                        str(report_dir),
                        "--overwrite",
                        "--fail-on",
                        "never",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                json.loads(existing.read_text(encoding="utf-8"))["source"],
                "image://registry.example/pytorch:1",
            )

    def test_output_file_requires_explicit_single_format(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "inspect",
                    "registry.example/pytorch:1",
                    "--target",
                    "gb300",
                    "--output",
                    "report.txt",
                ]
            )
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("use --output-dir", stderr.getvalue())

    def test_failure_threshold_parameters(self) -> None:
        warning_exit, _, _ = self.run_with_report(self.report(), fail_on="warning")
        blocker_exit, _, _ = self.run_with_report(
            self.report(platform="linux/amd64"),
            fail_on="blocker",
        )
        advisory_exit, _, _ = self.run_with_report(
            self.report(platform="linux/amd64"),
            fail_on="never",
        )
        self.assertEqual(warning_exit, 2)
        self.assertEqual(blocker_exit, 2)
        self.assertEqual(advisory_exit, 0)

    def test_default_threshold_fails_on_warning(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("gpu_preflight.cli.inspect_image_reference", return_value=self.report()),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = main(
                [
                    "inspect",
                    "registry.example/pytorch:1",
                    "--target",
                    "gb300",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(exit_code, 2)

    def test_html_output_parameter_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            with patch(
                "gpu_preflight.cli.inspect_image_reference",
                return_value=self.report(),
            ):
                exit_code = main(
                    [
                        "inspect",
                        "registry.example/pytorch:1",
                        "--target",
                        "gb300",
                        "--format",
                        "html",
                        "--output",
                        str(output),
                        "--fail-on",
                        "never",
                    ]
                )
            self.assertEqual(exit_code, 0)
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("NVIDIA GB300 NVL72", rendered)
            self.assertIn("Image references", rendered)


if __name__ == "__main__":
    unittest.main()
