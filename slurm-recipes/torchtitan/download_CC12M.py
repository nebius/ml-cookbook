#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# Set up cache in /root/hf_cache
cache_dir = "/root/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

# Set environment variables
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = f"{cache_dir}/datasets"

print(f"Downloading CC12M dataset to: {cache_dir}")
print("This will take several hours (~6TB)...")

# Download the dataset
snapshot_download(
    repo_id="pixparse/cc12m-wds",
    repo_type="dataset",
    local_dir=f"{cache_dir}/datasets/pixparse___cc12m-wds",
    resume_download=True,
)

print("✅ Download complete!")
