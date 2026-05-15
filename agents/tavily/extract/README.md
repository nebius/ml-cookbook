# Tavily Extract Recipe

This recipe uses Tavily Extract to turn a list of URLs into clean markdown or text that can later be used for:

- RAG ingestion
- offline analysis
- dataset preparation
- internal documentation corpora

## Best platform

- Slurm / Soperator for a simple batch-processing workflow

See [`slurm/tavily-extract/`](../../../slurm/tavily-extract/).

## Files

- [`extract_urls.py`](./extract_urls.py) - batch extractor
- [`sample_urls.txt`](./sample_urls.txt) - small starter input

## Run locally

```bash
cd agents/tavily
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python extract/extract_urls.py \
  --urls-file extract/sample_urls.txt \
  --output outputs/extract/results.json \
  --query "Nebius platform capabilities for machine learning teams"
```

## Expected output

The extractor writes a JSON file containing:

- successful extraction results
- cleaned page content
- failed URLs if any
- response time and usage details

## Nebius platform value

This is a good "real work" first batch job because users can test distributed or scheduled execution without needing model weights or GPUs.
