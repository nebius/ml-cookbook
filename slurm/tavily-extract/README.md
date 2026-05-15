# Running Tavily Extract on Slurm (Soperator)

This recipe runs a small batch extraction job with Tavily Extract on a Nebius Soperator cluster.

It is a good fit for users who want to:

- test a simple scheduled workload
- produce useful artifacts without GPUs
- prepare data for downstream retrieval or evaluation jobs

## Prerequisites

- Access to a Nebius Soperator cluster
- `TAVILY_API_KEY` exported in your shell
- This repository cloned to the shared filesystem

## Submit the job

From the shared filesystem:

```bash
cd ml-cookbook/slurm/tavily-extract
sbatch run_extract.sh
```

## Expected output

The job writes extraction results to:

```bash
ml-cookbook/agents/tavily/outputs/extract/results.json
```

## Customize the URLs

Edit [`../../agents/tavily/extract/sample_urls.txt`](../../agents/tavily/extract/sample_urls.txt) to change the input URL list.
