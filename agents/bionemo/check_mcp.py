"""Run a read-only readiness check against the Nebius BioNeMo MCP gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from urllib.parse import urlsplit

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

TOKEN_ENV = "BIONEMO_MCP_TOKEN"  # noqa: S105 - this is an environment variable name
URL_ENV = "BIONEMO_MCP_URL"
BASELINE_TOOLS = {"fleet_health", "list_models"}
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _validated_url(value: str) -> str:
    url = value.rstrip("/")
    parsed = urlsplit(url)
    if parsed.username or parsed.password:
        raise ValueError("the MCP URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("the MCP URL must not contain a query or fragment")
    if parsed.path != "/mcp":
        raise ValueError("the MCP URL must end with the exact /mcp path")
    if parsed.scheme == "https" and parsed.hostname:
        return url
    if parsed.scheme == "http" and parsed.hostname in LOOPBACK_HOSTS:
        return url
    raise ValueError("use HTTPS for remote MCP endpoints; HTTP is allowed only on loopback")


def _token_from_environment() -> str:
    token = os.environ.get(TOKEN_ENV, "")
    if len(token) < 32:
        raise ValueError(f"{TOKEN_ENV} must contain a bearer token of at least 32 characters")
    return token


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List BioNeMo MCP tools and call only the read-only list_models tool."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get(URL_ENV),
        help=f"Streamable HTTP endpoint ending in /mcp (default: ${URL_ENV})",
    )
    parser.add_argument(
        "--expected-tool",
        action="append",
        default=[],
        help="Additional tool that must be registered; repeat for multiple tools.",
    )
    return parser


async def _check(url: str, token: str, expected_tools: set[str]) -> dict[str, object]:
    headers = {"Authorization": f"Bearer {token}"}
    async with (
        httpx2.AsyncClient(headers=headers) as http_client,
        Client(streamable_http_client(url, http_client=http_client)) as client,
    ):
        tools = await client.list_tools()
        names = sorted(tool.name for tool in tools.tools)
        missing = sorted(expected_tools - set(names))
        if missing:
            raise RuntimeError(f"missing expected tools: {missing}; registered: {names}")

        result = await client.call_tool("list_models", {})
        if result.is_error:
            raise RuntimeError("the read-only list_models tool returned an error")

    return {"list_models": "ok", "registered_tools": names}


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error(f"--url or {URL_ENV} is required")

    try:
        url = _validated_url(args.url)
        token = _token_from_environment()
    except ValueError as exc:
        parser.error(str(exc))

    expected_tools = BASELINE_TOOLS | set(args.expected_tool)
    try:
        summary = asyncio.run(_check(url, token, expected_tools))
    except RuntimeError as exc:
        parser.error(str(exc))
    except Exception:
        parser.error(
            "MCP readiness check failed; verify the URL, bearer token, TLS, and gateway status"
        )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
