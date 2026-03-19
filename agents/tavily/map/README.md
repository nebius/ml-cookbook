# Tavily Map Recipe

This recipe uses Tavily Map to discover the structure of a site before extraction or crawling. It is useful for:

- planning a docs ingestion workflow
- identifying candidate pages for a corpus
- validating the shape of a target site before a larger job

## Best platform

- SkyPilot for a quick CLI-driven test

See [`skypilot/examples/tavily-map-site.yaml`](../../../skypilot/examples/tavily-map-site.yaml).

## Run locally

```bash
cd agents/tavily
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python map/map_site.py \
  --url https://docs.nebius.com \
  --instructions "Find pages most relevant to GPU clusters, object storage, and managed Kubernetes" \
  --output outputs/map/nebius_docs_map.json
```

## Expected output

The output JSON contains:

- the base URL
- discovered URLs
- response time
- optional usage information

## Why it is useful

It gives users an easy way to experience structured site discovery without committing to a full crawl.
