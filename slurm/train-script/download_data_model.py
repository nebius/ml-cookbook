
import argparse
import os
from huggingface_hub import login, snapshot_download
from datasets import load_dataset

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

def download_dataset(dataset_name, data_dir, hf_token=None):
    print(f"Downloading and saving dataset {dataset_name} to {data_dir} using datasets.load_dataset/save_to_disk")
    # Download the dataset (default to 'train', 'validation', 'test' splits if available)
    dataset = load_dataset(dataset_name, use_auth_token=hf_token)
    dataset.save_to_disk(data_dir)
    print(f"Dataset saved to {data_dir} (load_from_disk compatible)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF model and dataset to shared folder")
    parser.add_argument('--model', type=str, required=True, help='Model repo name (e.g., org/model_name)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset repo name (e.g., org/dataset_name)')
    parser.add_argument('--shared_folder', type=str, default='/shared', help='Shared folder path')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory to store/download the model (default: <shared_folder>/model/<model_name>)')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory to store/download the dataset (default: <shared_folder>/data/<dataset_name>)')
    args = parser.parse_args()

    hf_token = os.environ.get('HF_TOKEN')

    # Set default values for model_dir and data_dir using argparse defaults
    if args.model_dir is None:
        args.model_dir = os.path.join(args.shared_folder, 'model', args.model.replace('/', '_'))
    if args.data_dir is None:
        args.data_dir = os.path.join(args.shared_folder, 'data', args.dataset.replace('/', '_'))
    model_dir = args.model_dir
    data_dir = args.data_dir

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    download_model(args.model, model_dir, hf_token)
    download_dataset(args.dataset, data_dir, hf_token)
