import argparse
import os
from huggingface_hub import login, snapshot_download

# -----------------------------
# Download Utility
# -----------------------------
def download_model(model_name, model_dir, hf_token=None):
    if hf_token:
        login(token=hf_token)
    print(f"Downloading model {model_name} to {model_dir} (no symlinks)")
    snapshot_dir = snapshot_download(
        repo_id=model_name,
        repo_type="model",
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=hf_token
    )
    print(f"Model snapshot downloaded to {snapshot_dir}")

def download_dataset(dataset_name, data_dir, subset=None, hf_token=None):
    print(f"Downloading dataset {dataset_name} to {data_dir} (no symlinks)")
    repo_id = dataset_name if not subset else f"{dataset_name}/{subset}"
    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=data_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=hf_token
    )
    print(f"Dataset snapshot downloaded to {snapshot_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF model and dataset to shared folder")
    parser.add_argument('--model', type=str, default='distilbert-base-uncased', help='Model name')
    parser.add_argument('--dataset', type=str, default='glue', help='Dataset name')
    parser.add_argument('--subset', type=str, default='sst2', help='Dataset subset (e.g., sst2 for GLUE)')
    parser.add_argument('--shared_folder', type=str, default='/shared', help='Shared folder path')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory to store/download the model (default: <shared_folder>/model/<model_name>)')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory to store/download the dataset (default: <shared_folder>/data/<dataset>_<subset>)')
    args = parser.parse_args()

    hf_token = os.environ.get('HF_TOKEN')

    # Set default values for model_dir and data_dir using argparse defaults
    if args.model_dir is None:
        args.model_dir = os.path.join(args.shared_folder, 'model', args.model.replace('/', '_'))
    if args.data_dir is None:
        args.data_dir = os.path.join(args.shared_folder, 'data', f"{args.dataset}_{args.subset}")
    model_dir = args.model_dir
    data_dir = args.data_dir

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    download_model(args.model, model_dir, hf_token)
    download_dataset(args.dataset, data_dir, args.subset, hf_token)
