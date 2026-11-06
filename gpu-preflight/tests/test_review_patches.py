"""Regressions for the defects found by replaying the scanner over a public image corpus.

Each test names the image that exposed its defect. The scanner is judged on images nobody
curated for it, so the cases that matter are the ones a hand-written fixture would not
have thought to build: an executable with no extension, firmware for a processor inside
the GPU, a deliberately corrupt file a packaging tool ships on purpose.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from gpu_preflight.image import (
    DockerImageClient,
    ImageInspectionError,
    PlatformManifest,
    PulledImageFacts,
)


def commandless_facts() -> tuple[PlatformManifest, PulledImageFacts]:
    manifest = PlatformManifest(
        platform="linux/amd64",
        digest="sha256:" + "0" * 64,
        config_digest=None,
        compressed_size_bytes=0,
    )
    config = PulledImageFacts(
        platform="linux/amd64",
        digest=manifest.digest,
        environment={},
        known_packages=(),
        installed_packages=(),
        cuda_targets=(),
        cuda_versions=(),
        nccl_versions=(),
        user=None,
        entrypoint=(),
        command=(),
    )
    return manifest, config


class CommandlessImageTests(unittest.TestCase):
    """docker/docker-model-backend-llamacpp:v0.0.34-cuda declares no CMD and no ENTRYPOINT.

    docker create refuses such an image, so the scan died before reading a single file.
    The export container is never started, so any command satisfies it.
    """

    def test_export_supplies_an_entrypoint_that_cannot_run(self) -> None:
        client = DockerImageClient(cuobjdump_mode="off")
        manifest, config = commandless_facts()
        # An empty container ID aborts right after create, which is all this needs.
        with (
            patch.object(client, "_run", return_value="") as run,
            self.assertRaises(ImageInspectionError),
        ):
            client.scan_filesystem(
                "example/commandless:latest", manifest, config, max_scan_bytes=None
            )
        arguments = run.call_args.args[0]
        self.assertIn("--entrypoint", arguments)
        entrypoint = arguments[arguments.index("--entrypoint") + 1]
        self.assertTrue(entrypoint.startswith("/"))
        self.assertIn("never_executed", entrypoint)
