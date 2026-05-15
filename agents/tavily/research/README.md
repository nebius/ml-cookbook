# Tavily Research Recipe

This recipe generates a research report with citations using Tavily Research. It is the highest-value demo in this track because it feels like a real agent workflow instead of a raw API call.

## Best platform

- SkyPilot for a fast single-node batch experience

See [`skypilot/examples/tavily-research-agent.yaml`](../../../skypilot/examples/tavily-research-agent.yaml).

## Run locally

```bash
cd agents/tavily
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python research/research_report.py \
  --input "Compare open-source inference servers for multi-node deployment and summarize tradeoffs for ML platform teams" \
  --output-dir outputs/research
```

## Expected output

The recipe writes:

- `response.json` with the full API response
- `sources.json` with cited sources
- `report.md` with the generated report when the content is text

## Why it is useful

It gives users a concrete "agent on Nebius" outcome that is easy to demo and extend.
