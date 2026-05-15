# Tavily Search Recipe

This recipe runs a minimal search-backed API using Tavily Search. It is the easiest way to test an agent-style workflow on Nebius because it validates:

- secret injection
- package installation
- service startup
- external API connectivity
- HTTP access to a running workload

## Best platform

- SkyPilot for the fastest first-run experience

See [`skypilot/examples/tavily-search-api.yaml`](../../../skypilot/examples/tavily-search-api.yaml).

## Files

- [`app.py`](./app.py) - FastAPI service exposing Tavily Search

## Prerequisites

- `TAVILY_API_KEY` exported in your shell

## Run locally

```bash
cd agents/tavily
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd search
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test it:

```bash
curl http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest open-source inference servers for multi-node deployment",
    "topic": "general",
    "max_results": 5,
    "include_answer": "advanced"
  }'
```

## Expected output

The API returns:

- ranked search results
- URLs and titles
- optional generated answer
- optional raw content per result

## Why this belongs in ML Cookbook

It gives users a realistic first agent deployment without requiring a large model or complex infrastructure.
