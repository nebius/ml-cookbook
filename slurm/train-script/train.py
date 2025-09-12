import os
import argparse
import yaml
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, DataCollatorWithPadding
from datasets import load_from_disk
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import signal
import sys
import atexit

# -----------------------------
# Utility Functions
# -----------------------------
def setup():
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return global_rank, local_rank, world_size

def cleanup():
    """Clean up distributed environment."""
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """Save model and optimizer state."""
    sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        'model_state_dict': sd,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer, scheduler, path):
    """Load model and optimizer state."""
    checkpoint = torch.load(path, map_location='cpu')
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']

# -----------------------------
# Training Function
# -----------------------------
def train(config):
    # Register cleanup for normal exit and signals
    def handle_exit(signum=None, frame=None):
        try:
            cleanup()
        except Exception:
            pass
        if signum is not None:
            sys.exit(0)

    atexit.register(handle_exit)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(sig, handle_exit)
    # -----------------------------
    # 1. Setup DDP
    # -----------------------------
    global_rank, local_rank, world_size = setup()
    device = torch.device(f"cuda:{local_rank}")

    # Sanity checks, ensure ranks and visible GPUs align
    try:
        visible_gpu_count = torch.cuda.device_count()
    except Exception:
        visible_gpu_count = 0

    if world_size < 1:
        raise RuntimeError(f"Invalid world_size={world_size}")
    if not (0 <= local_rank < max(1, visible_gpu_count)):
        # This likely indicates a mismatch between SLURM/GPU allocation and torchrun
        raise RuntimeError(f"local_rank {local_rank} is out of range for visible GPUs ({visible_gpu_count})")

    # -----------------------------
    # 2. Load dataset and tokenizer/model
    # -----------------------------
    model_dir = config['model_dir']
    data_dir = config['data_dir']
    checkpoint_path = config['checkpoint_path']
    dataset = load_from_disk(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Tokenize dataset in-place so collator can pad tensors
    dataset = dataset.map(lambda examples: tokenizer(examples['sentence'], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=config.get('num_labels', 2))
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # -----------------------------
    # 3. Distributed Samplers for DDP
    # -----------------------------
    train_sampler = DistributedSampler(dataset['train'], num_replicas=world_size, rank=global_rank, shuffle=True)
    val_sampler = DistributedSampler(dataset['validation'], num_replicas=world_size, rank=global_rank, shuffle=False)
    train_loader = DataLoader(dataset['train'], batch_size=config['batch_size'], sampler=train_sampler, collate_fn=data_collator)
    val_loader = DataLoader(dataset['validation'], batch_size=config['batch_size'], sampler=val_sampler, collate_fn=data_collator)

    # -----------------------------
    # 4. Optimizer and Scheduler
    # -----------------------------
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['num_warmup_steps'],
        num_training_steps=config['num_epochs'] * len(train_loader)
    )

    # -----------------------------
    # 5. Optionally resume from checkpoint
    # -----------------------------
    start_epoch = 0
    if config.get('resume') and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path) + 1
        if global_rank == 0:
            print(f"[Rank {global_rank}] Resumed from checkpoint at epoch {start_epoch}")

    # -----------------------------
    # 6. Training and Validation Loop
    # -----------------------------
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        correct = 0
        total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", disable=global_rank!=0):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        # Gather train loss and accuracy from all processes
        train_loss = torch.tensor(total_loss / len(train_loader), device=device)
        train_correct = torch.tensor(correct, device=device)
        train_total = torch.tensor(total, device=device)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total, op=dist.ReduceOp.SUM)
        train_loss = train_loss.item() / world_size
        train_acc = train_correct.item() / train_total.item()

        # Validation
        model.eval()
        val_sampler.set_epoch(epoch)
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", disable=global_rank!=0):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(device)
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = torch.tensor(val_loss / len(val_loader), device=device)
        val_correct = torch.tensor(val_correct, device=device)
        val_total = torch.tensor(val_total, device=device)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total, op=dist.ReduceOp.SUM)
        val_loss = val_loss.item() / world_size
        val_acc = val_correct.item() / val_total.item()

        # Print only on rank 0
        if global_rank == 0:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
        dist.barrier()
    # After training is complete, delete the checkpoint file if it exists (only on rank 0)
    if global_rank == 0 and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"[Rank {global_rank}] Deleted checkpoint file at {checkpoint_path} after successful training.")
    cleanup()

# -----------------------------
# Main Entrypoint
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP Training Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['resume'] = args.resume

    train(config)
