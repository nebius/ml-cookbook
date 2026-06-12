import json
import os
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = os.environ.get("TAVILY_BASE_URL", "https://api.tavily.com")


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable '{name}' is not set.")
    return value


def tavily_headers() -> dict[str, str]:
    api_key = require_env("TAVILY_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    project = os.environ.get("TAVILY_PROJECT")
    if project:
        headers["X-Project-ID"] = project
    return headers


def tavily_post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}",
        headers=tavily_headers(),
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def tavily_get(endpoint: str) -> dict[str, Any]:
    response = requests.get(
        f"{BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}",
        headers=tavily_headers(),
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: str, content: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def poll_research(
    request_id: str, poll_interval: int = 5, timeout_seconds: int = 900
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = tavily_get(f"research/{request_id}")
        status = response.get("status")
        if status == "completed":
            return response
        if status == "failed":
            raise RuntimeError(f"Research task failed: {response}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for research task {request_id}")
