from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agents.bionemo.check_mcp import _token_from_environment, _validated_url


class ValidateUrlTests(unittest.TestCase):
    def test_accepts_remote_https_mcp_path(self) -> None:
        self.assertEqual(
            _validated_url("https://bionemo.example.com/mcp/"),
            "https://bionemo.example.com/mcp",
        )

    def test_accepts_loopback_http(self) -> None:
        self.assertEqual(
            _validated_url("http://127.0.0.1:8000/mcp"),
            "http://127.0.0.1:8000/mcp",
        )

    def test_rejects_remote_http(self) -> None:
        with self.assertRaisesRegex(ValueError, "use HTTPS"):
            _validated_url("http://bionemo.example.com/mcp")

    def test_rejects_credentials_in_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain credentials"):
            _validated_url("https://user:token@bionemo.example.com/mcp")

    def test_rejects_non_mcp_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact /mcp path"):
            _validated_url("https://bionemo.example.com/healthz")


class TokenTests(unittest.TestCase):
    def test_reads_token_from_environment(self) -> None:
        token = "a" * 32
        with patch.dict(os.environ, {"BIONEMO_MCP_TOKEN": token}, clear=True):
            self.assertEqual(_token_from_environment(), token)

    def test_rejects_short_token(self) -> None:
        with (
            patch.dict(os.environ, {"BIONEMO_MCP_TOKEN": "short"}, clear=True),
            self.assertRaisesRegex(ValueError, "at least 32 characters"),
        ):
            _token_from_environment()


if __name__ == "__main__":
    unittest.main()
