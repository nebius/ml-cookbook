# Tavily Agent Recipes on Nebius

This section adds a small, opinionated recipe track for building agentic workflows on Nebius infrastructure using Tavily's five core capabilities:

- `search`
- `extract`
- `map`
- `crawl`
- `research`

These recipes are intentionally practical. They are designed to help users:

- experience Nebius as a platform for agent workloads
- test secrets, networking, and storage in realistic workflows
- move from a small demo to a more production-shaped pattern

## What is in this track?

| Capability | Folder | What it does | Best first platform |
| --- | --- | --- | --- |
| Search | [`search/`](./search/) | Runs a search-backed API service | SkyPilot |
| Extract | [`extract/`](./extract/) | Extracts clean content from URLs in batch | Slurm / Soperator |
| Map | [`map/`](./map/) | Discovers site structure and candidate pages | SkyPilot |
| Crawl | [`crawl/`](./crawl/) | Crawls a site and stores a reusable corpus | SkyPilot |
| Research | [`research/`](./research/) | Generates a research report with citations | SkyPilot |

All recipes share simple Python tooling in [`common.py`](./common.py) and dependencies in [`requirements.txt`](./requirements.txt).

## Prerequisites

Before running these recipes, make sure you have:

- a Tavily API key in `TAVILY_API_KEY`
- access to the target Nebius platform you want to test
- Python 3.10+ if running the scripts directly
- optional object storage or shared storage if you want to persist outputs beyond the local node

Some recipes are written to be small and cheap by default so new users can test the platform quickly.

## Recommended order

If you are new to agent recipes, use this order:

1. [`search/`](./search/) to validate secrets, service startup, and HTTP access
2. [`extract/`](./extract/) to run a batch data-processing workflow
3. [`research/`](./research/) to experience a higher-value agent workflow
4. [`map/`](./map/) to discover a site's structure
5. [`crawl/`](./crawl/) to build a reusable corpus for RAG or evaluation

## Platform recipes

This track also includes Nebius-shaped launch examples:

- [`skypilot/examples/tavily-search-api.yaml`](../../skypilot/examples/tavily-search-api.yaml)
- [`skypilot/examples/tavily-map-site.yaml`](../../skypilot/examples/tavily-map-site.yaml)
- [`skypilot/examples/tavily-crawl-site.yaml`](../../skypilot/examples/tavily-crawl-site.yaml)
- [`skypilot/examples/tavily-research-agent.yaml`](../../skypilot/examples/tavily-research-agent.yaml)
- [`slurm/tavily-extract/`](../../slurm/tavily-extract/)

## Design goals

These recipes are built to match the rest of `ml-cookbook`:

- minimal setup
- copy-paste friendly
- clear expected outputs
- easy transition from demo to larger workloads

## Future extensions

Natural follow-ups to this track include:

- a Tavily + vector database ingestion path
- a Tavily + open-weight LLM answer generation service
- a recurring research/report automation pattern
- a Kubernetes-hosted agent API for persistent serving
