#!/bin/bash
#SBATCH --job-name=tavily_extract
#SBATCH --output=outputs/tavily-extract-%j.out
#SBATCH --error=outputs/tavily-extract-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

set -euo pipefail

if [ -z "${TAVILY_API_KEY:-}" ]; then
  echo "TAVILY_API_KEY must be set before submitting the job."
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

python extract/extract_urls.py \
  --urls-file extract/sample_urls.txt \
  --output outputs/extract/results.json \
  --query "Nebius and Tavily documentation relevant to ML platform workflows"
