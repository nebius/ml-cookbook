# Tavily Crawl Recipe

This recipe uses Tavily Crawl to build a reusable multi-page corpus from a site. It is a natural next step after `map` and is useful for:

- building a retrieval corpus
- creating evaluation datasets
- collecting documentation for downstream processing

## Best platform

- SkyPilot for a quick end-to-end run on Nebius

See [`skypilot/examples/tavily-crawl-site.yaml`](../../../skypilot/examples/tavily-crawl-site.yaml).

## Run locally

```bash
cd agents/tavily
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python crawl/crawl_site.py \
  --url https://docs.nebius.com \
  --instructions "Find pages relevant to AI infrastructure onboarding" \
  --output outputs/crawl/nebius_docs_corpus.json
```

## Expected output

The crawl output includes:

- discovered pages
- extracted content per page
- response timing
- usage details

## Why this is a good cookbook recipe

It demonstrates a more realistic data-ingestion workflow while staying easy to run.
