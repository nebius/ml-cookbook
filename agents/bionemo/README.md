# BioNeMo NIMs from an Agent on Nebius

This recipe is the Nebius entry point for using NVIDIA BioNeMo NIMs from an
agent. It helps you choose the deployment track, verify an authenticated MCP
gateway without starting an inference job, and hand the gateway to an MCP
client.

It intentionally does not copy NVIDIA's model notebooks or scientific payload
examples. Those remain in the
[NVIDIA digital biology examples](https://github.com/NVIDIA/digital-biology-examples/tree/main/examples/nims).
The value here is the Nebius-specific path from a deployed service to a safe,
agent-ready connection.

## Choose a deployment track

| Goal | Canonical implementation | What it owns |
| --- | --- | --- |
| Run a browser-based demo on Nebius Serverless | [`serverless-ai-cookbook/life-science/bionemo-agent`](https://github.com/nebius/serverless-ai-cookbook/tree/main/life-science/bionemo-agent) | Agent image, browser UI, model-provider selection, hosted NVIDIA tools, and remote MCP integration |
| Run customer-owned BioNeMo NIMs on Nebius Managed Kubernetes | [`nebius-solutions-library/tools/bionemo-mcp`](https://github.com/nebius/nebius-solutions-library/tree/main/tools/bionemo-mcp) | NIM catalog, authenticated MCP gateway, Helm deployment, artifact storage, and client configuration |

Keep deployment code in those repositories. This page provides the common
selection and safety guidance; the readiness script below specifically checks
the customer-owned Kubernetes gateway contract. The Serverless recipe has its
own `/healthz` and `/readyz` checks for its selected backend.

## Prerequisites

- Python 3.13+
- the HTTPS URL of a BioNeMo Streamable HTTP MCP endpoint, ending in `/mcp`
- its bearer token in `BIONEMO_MCP_TOKEN`
- research-only, non-clinical use of model outputs

The check accepts plain HTTP only for loopback addresses, which supports a
local `kubectl port-forward` without weakening remote connections.

## Verify the gateway without running inference

Create an isolated environment:

```bash
cd agents/bionemo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the endpoint URL, then read the bearer token without putting it in shell
history or a process argument:

```bash
export BIONEMO_MCP_URL="https://bionemo.example.com/mcp"
read -rsp "BioNeMo MCP token: " BIONEMO_MCP_TOKEN
export BIONEMO_MCP_TOKEN
printf '\n'
```

Run the readiness check:

```bash
python check_mcp.py
```

The script opens a real MCP connection, lists the registered tools, verifies
that `list_models` and `fleet_health` are present, and calls the read-only
`list_models` tool. It never calls a model or submits a compute job. A successful
result looks like this:

```json
{
  "list_models": "ok",
  "registered_tools": ["fleet_health", "list_models", "openfold2_predict"]
}
```

Remove the token from the current shell after the check if you are not starting
an MCP client immediately:

```bash
unset BIONEMO_MCP_TOKEN
```

Tool registration is health-gated, so the exact list depends on which NIMs were
ready when the gateway started. Require additional tools when validating a
specific environment:

```bash
python check_mcp.py \
  --expected-tool openfold2_predict \
  --expected-tool evo2_run
```

## Connect an agent

Use the checked-in client examples beside the Kubernetes gateway as the
authoritative configuration for Codex, Claude Code, Cursor, and local stdio
clients. Keep the bearer token in the environment or a secret manager; do not
paste it into committed JSON or TOML.

For a browser-first event setup, follow the Serverless recipe. Its image starts
without model-provider credentials so configuration errors are visible, while
actual reasoning and BioNeMo inference require the relevant provider keys.

## Safe first workflow

Start with a public, short protein sequence and one structure-prediction model.
State the research-only and non-clinical acknowledgement explicitly. Ask the
agent to submit exactly one job, retain the returned job ID, and poll that exact
job a bounded number of times. Do not let an agent resubmit merely because a
long-running job is still queued or running.

Treat every generated structure, sequence, or molecule as a research artifact,
not a clinical result. Record the model and image version, input, returned job
ID, checksums, and confidence metrics with the output.

## Troubleshooting

- `401 Unauthorized`: the token is missing, stale, or belongs to another
  gateway. Rotate it at the deployment source and update the local environment.
- A required model tool is absent: inspect `fleet_health`, then restart the MCP
  gateway after the NIM becomes ready so every replica exposes the same tool
  surface.
- A model call is still running: poll the original job ID. Do not submit a
  replacement unless the first job reached a terminal failure state.
- An artifact link expired: use the gateway's artifact retrieval flow to mint a
  fresh, short-lived URL; do not make the artifact bucket public.
