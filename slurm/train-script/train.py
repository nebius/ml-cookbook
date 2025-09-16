import os
import argparse
import yaml
import time
import math
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import transformers as hf
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from datasets import load_from_disk
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import signal
import sys
import atexit
from inspect import signature
from datetime import timedelta
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

"""Training script with strict NCCL setup and AMP.
Assumes model and dataset are already downloaded to shared folder.
"""

# AMP import strategy (new API first, fallback to legacy)
try:  # New style (preferred): torch.amp
    from torch import amp as torch_amp  # type: ignore
    _AMP_USES_DEVICE_ARG = True
except Exception:  # Fallback: legacy torch.cuda.amp
    from torch.cuda import amp as torch_amp  # type: ignore
    _AMP_USES_DEVICE_ARG = False

# -----------------------------
# Utility Functions
# -----------------------------
def setup(timeout_seconds: int = 1800):
    """Initialize distributed process group with strict NCCL-only requirement.

    Assumes NVIDIA GPUs + CUDA + NCCL are always available. Fails fast otherwise.
    """
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but NCCL required. Check drivers and environment.")

    visible_gpu_count = torch.cuda.device_count()
    if local_rank >= visible_gpu_count:
        raise RuntimeError(f"LOCAL_RANK {local_rank} >= visible CUDA devices {visible_gpu_count}")

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized() and world_size > 1:
        # torch.cuda.set_device(local_rank) above already sets the correct current device
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(seconds=timeout_seconds)
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[INIT] Initialized process group backend=nccl world_size={world_size}")
    return global_rank, local_rank, world_size

def cleanup():
    """Clean up distributed environment."""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """Save model and optimizer state."""
    sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    tmp_path = path + ".tmp"
    torch.save({
        'model_state_dict': sd,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, tmp_path)
    # atomic replace
    try:
        os.replace(tmp_path, path)
    except Exception:
        # fallback to rename
        os.rename(tmp_path, path)

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
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


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
    global_rank, local_rank, world_size = setup(
        timeout_seconds=int(config.get('ddp_timeout_seconds', 1800))
    )
    device = torch.device(f"cuda:{local_rank}")

    # Optional suppression of duplicate HF warnings on non-zero ranks
    suppress_hf = bool(config.get('suppress_hf_warnings', True))
    if suppress_hf:
        try:
            from transformers.utils import logging as hf_logging  # type: ignore
            if global_rank == 0:
                # Keep default (WARNING). Users can still raise via env if desired.
                hf_logging.set_verbosity_warning()
            else:
                # Silence to ERROR so only real problems surface.
                hf_logging.set_verbosity_error()
        except Exception:
            pass

    # Set seeds (rank-aware) after we know ranks
    base_seed = int(config.get('seed', 42))
    deterministic = bool(config.get('deterministic', False))
    set_seed(base_seed + global_rank, deterministic=deterministic)

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
    # 2. Resolve paths & load dataset/model
    # -----------------------------
    # Resolve required directories from environment variables
    def _require_env(name: str):
        v = os.environ.get(name)
        if not v:
            raise ValueError(f"Required environment variable '{name}' is not set.")
        return v

    model_dir = _require_env('MODEL_DIR')
    data_dir = _require_env('DATA_DIR')
    experiments_root = _require_env('EXPERIMENTS_ROOT')
    checkpoint_path = config.get('checkpoint_path', os.path.join('.', 'checkpoint.pt'))

    dataset = load_from_disk(data_dir)

    # Detect splits: prefer 'train' and a validation split, but allow other names
    if 'train' not in dataset:
        raise RuntimeError(f"Dataset at {data_dir} does not contain a 'train' split. Found: {list(dataset.keys())}")
    val_split = config.get('validation_split', None)
    if val_split is None:
        # prefer common names
        for candidate in ('validation', 'validation_holdout', 'val', 'dev'):
            if candidate in dataset:
                val_split = candidate
                break
    if val_split is None:
        # allow training without validation if explicitly requested
        if not config.get('allow_no_validation', False):
            raise RuntimeError(f"No validation split found in dataset and 'allow_no_validation' not set. Splits: {list(dataset.keys())}")

    # Load tokenizer
    tokenizer = hf.AutoTokenizer.from_pretrained(model_dir)

    # Infer task: explicit config['task'] preferred, otherwise use model config if available
    task = config.get('task', None)
    try:
        model_cfg = hf.AutoConfig.from_pretrained(model_dir)
        if task is None:
            if getattr(model_cfg, 'is_encoder_decoder', False):
                task = 'seq2seq'
            else:
                task = 'classification'
    except Exception:
        if task is None:
            task = 'classification'

    # Detect input and label/target columns heuristically
    train_cols = dataset['train'].column_names
    # common text column candidates
    text_candidates = [config.get('text_column')] if config.get('text_column') else []
    text_candidates += ['text', 'sentence', 'input_text', 'article', 'document']
    text_column = None
    for c in text_candidates:
        if c and c in train_cols:
            text_column = c
            break
    if text_column is None:
        # fallback: pick first string column that is not an index/label
        for c in train_cols:
            if c in ('label', 'labels', 'id', 'idx'):
                continue
            # sample a few rows to see if values are strings
            try:
                v = dataset['train'][0][c]
                if isinstance(v, str) or (isinstance(v, list) and len(v) and isinstance(v[0], str)):
                    text_column = c
                    break
            except Exception:
                continue
    if text_column is None:
        raise RuntimeError(f"Could not auto-detect a text input column in dataset columns: {train_cols}; set 'text_column' in config.")

    # label/target column detection
    label_column = config.get('label_column', None)
    target_column = config.get('target_column', None)
    if task == 'seq2seq':
        # target text column candidates
        tgt_candidates = [target_column] if target_column else []
        tgt_candidates += ['target', 'summary', 'labels', 'label_text']
        for c in tgt_candidates:
            if c and c in train_cols:
                target_column = c
                break
        if target_column is None:
            raise RuntimeError("Could not detect target text column for seq2seq task; set 'target_column' in config.")
    else:
        if label_column is None:
            for c in ('label', 'labels', 'target'):
                if c in train_cols:
                    label_column = c
                    break
    # tokenized dataset path
    tokenized_dir = config.get('tokenized_dir', f"{data_dir.rstrip('/')}_tokenized")

    # Tokenization function depends on task
    if task == 'seq2seq':
        def tokenize_fn(examples):
            return tokenizer(examples[text_column], text_target=examples[target_column], truncation=True)
        token_presence_key = 'input_ids'
    else:
        def tokenize_fn(examples):
            return tokenizer(examples[text_column], truncation=True)
        token_presence_key = 'input_ids'

    # If dataset already tokenized (has input_ids), skip. Otherwise have rank 0 create a tokenized copy.
    already_tokenized = False
    try:
        already_tokenized = token_presence_key in dataset['train'].column_names
    except Exception:
        already_tokenized = False

    if not already_tokenized:
        if global_rank == 0:
            print(f"[Rank 0] Tokenizing dataset (task={task}) and saving to {tokenized_dir} ...")
            remove_cols = [text_column]
            if task == 'seq2seq' and target_column:
                remove_cols.append(target_column)
            tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=list(dict.fromkeys(remove_cols)))
            tokenized.save_to_disk(tokenized_dir)
        # Wait for rank 0 to finish tokenization and saving
        dist.barrier()
        # All ranks load the tokenized dataset from disk
        dataset = load_from_disk(tokenized_dir)

    # Data collator will be constructed after model is loaded (seq2seq collator sometimes needs model)

    # Lightweight validation on rank 0: print dataset columns and a sample, and check for unexpected string fields.
    if global_rank == 0:
        try:
            print("Dataset columns (train):", dataset['train'].column_names)
            sample = dataset['train'][0]
            print("Sample keys and types:")
            for k, v in sample.items():
                print(f"  {k}: {type(v)} -> {v if (isinstance(v, (int, float)) or (isinstance(v, list) and len(v)<=5)) else type(v)}")

            # Quick scan first 10 examples for any plain str values in tokenized fields
            bad = []
            for i in range(min(10, len(dataset['train']))):
                ex = dataset['train'][i]
                for k, v in ex.items():
                    if k == 'label':
                        continue
                    # tokenized fields should be lists/ints/floats, not plain str
                    if isinstance(v, str):
                        bad.append((i, k, v))
            if bad:
                print("Found unexpected string values in tokenized fields (first 10 examples):", bad)
                raise RuntimeError("Tokenization produced string values in token fields; check tokenizer usage and dataset columns.")
        except Exception as e:
            print("Dataset validation error:", e)
            raise

    # -----------------------------
    # Load model dynamically and build collator
    # -----------------------------
    if task == 'seq2seq':
        model = hf.AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    else:
        num_labels = int(config.get('num_labels', 2))
        model = hf.AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # Cache forward signature arg names once (post DDP wrap so .forward is the DDP wrapper -> access module.forward)
    forward_arg_names = set(signature(model.module.forward).parameters.keys()) if hasattr(model, 'module') else set(signature(model.forward).parameters.keys())

    # Build appropriate data collator
    if task == 'seq2seq':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # -----------------------------
    # 3. Distributed Samplers for DDP
    # -----------------------------
    train_split = 'train'
    val_split = val_split if val_split is not None else None
    train_sampler = DistributedSampler(dataset[train_split], num_replicas=world_size, rank=global_rank, shuffle=True)
    if val_split:
        val_sampler = DistributedSampler(dataset[val_split], num_replicas=world_size, rank=global_rank, shuffle=False)
    else:
        val_sampler = None
    num_workers = int(config.get('num_workers', 2))
    pin_memory = bool(config.get('pin_memory', True)) and torch.cuda.is_available()
    # Ensure reproducibility in DataLoader sharding
    g = torch.Generator()
    g.manual_seed(base_seed + global_rank)
    train_loader = DataLoader(
        dataset[train_split],
        batch_size=config['batch_size'],
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        dataset[val_split],
        batch_size=config['batch_size'],
        sampler=val_sampler,
        collate_fn=data_collator,
        num_workers=max(1, num_workers // 2) if val_split else 0,
        pin_memory=pin_memory,
        persistent_workers=(val_split is not None and num_workers > 0)
    ) if val_split else None

    # Fail early if loaders are empty to avoid zero-division or silent runtime errors
    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader is empty. Check dataset size, batch_size, and sampler configuration.")
    if val_loader is None:
        if not config.get('allow_no_validation', False):
            raise RuntimeError("Validation dataloader not found. Set 'allow_no_validation' in config to proceed without validation.")
    else:
        if len(val_loader) == 0:
            raise RuntimeError("Validation dataloader is empty. Provide a validation split or adjust dataset preparation.")

    # -----------------------------
    # 4. Optimizer and Scheduler
    # -----------------------------
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    grad_accum_steps = int(config.get('grad_accum_steps', 1))
    effective_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['num_warmup_steps'],
        num_training_steps=max(1, config['num_epochs'] * effective_steps_per_epoch)
    )
    use_amp = bool(config.get('use_amp', True)) and torch.cuda.is_available()
    if _AMP_USES_DEVICE_ARG:
        scaler = torch_amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = torch_amp.GradScaler(enabled=use_amp)

    # -----------------------------
    # 5. Optionally resume from checkpoint
    # -----------------------------
    start_epoch = 0
    if config.get('resume') and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path) + 1
        if global_rank == 0:
            print(f"[Rank {global_rank}] Resumed from checkpoint at epoch {start_epoch}")

    # -----------------------------
    # Prepare experiment/run directory (rank 0 only) before training loop
    writer = None
    run_dir = None
    if global_rank == 0:
        experiments_root = os.environ['EXPERIMENTS_ROOT']
        run_name = config.get('run_name')
        if not run_name:
            # derive from model basename + timestamp
            base_name = os.path.basename(model_dir.rstrip('/'))
            run_name = base_name + '-' + time.strftime('%Y%m%d-%H%M%S')
        run_dir = os.path.join(experiments_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        # If user did not set a specific final_model_dir, nest it inside run_dir
        if 'final_model_dir' not in config:
            config['final_model_dir'] = os.path.join(run_dir, 'final_model')
        # Optionally mirror checkpoint path inside run if checkpoint_path not absolute user location
        if not config.get('checkpoint_path'):
            config['checkpoint_path'] = os.path.join(run_dir, 'checkpoint.pt')
        # TensorBoard
        if bool(config.get('tensorboard_logging', True)) and SummaryWriter:
            try:
                writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tb'))
            except Exception as e:
                print(f"[Rank 0] TensorBoard writer init failed: {e}")

    # Broadcast potential updates to final_model_dir / checkpoint_path to all ranks
    if dist.is_initialized():
        # simple string broadcast via object list (only small items)
        obj = {'final_model_dir': config.get('final_model_dir'), 'checkpoint_path': config.get('checkpoint_path')}
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=0)
        config.update(obj_list[0])
        checkpoint_path = config.get('checkpoint_path', checkpoint_path)

    # -----------------------------
    # 6. Training and Validation Loop (task-agnostic)
    # -----------------------------
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        train_samples = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", disable=global_rank!=0)):
            batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
            if task == 'seq2seq':
                labels = None
                for key in ('labels', 'label', 'target_ids'):
                    if key in batch and batch[key] is not None:
                        labels = batch[key]
                        break
                inputs = {k: v for k, v in batch.items() if k not in ('labels', 'label', 'target_ids') and k in forward_arg_names}
            else:
                labels = None
                for key in ('labels', 'label'):
                    if key in batch and batch[key] is not None:
                        labels = batch[key]
                        break
                inputs = {k: v for k, v in batch.items() if k not in ('labels', 'label') and k in forward_arg_names}

            if _AMP_USES_DEVICE_ARG:
                ctx_mgr = torch_amp.autocast('cuda', enabled=use_amp)
            else:
                ctx_mgr = torch_amp.autocast(enabled=use_amp)
            with ctx_mgr:
                outputs = model(**inputs, labels=labels)
                orig_loss = outputs.loss
            # retain original loss value for reporting
            loss_value = orig_loss.item()
            loss = orig_loss / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # batch size detection
            bs = None
            for v in batch.values():
                if torch.is_tensor(v):
                    try:
                        bs = v.size(0)
                    except Exception:
                        bs = None
                    break
            if bs is None:
                bs = 1
            total_loss += loss_value * bs
            train_samples += bs

            if task != 'seq2seq' and outputs.logits is not None and labels is not None:
                preds = torch.argmax(outputs.logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

        # Aggregate training metrics across ranks by summing raw losses and counts
        # total_loss is sum of batch losses per rank; better to reduce sum and divide by global sample count
        local_loss_sum = torch.tensor(total_loss, device=device)
        local_sample_count = torch.tensor(train_samples, device=device)
        t_correct = torch.tensor(train_correct, device=device)
        t_total = torch.tensor(train_total if train_total > 0 else 0, device=device)
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sample_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_total, op=dist.ReduceOp.SUM)
        # compute averages
        train_loss = (local_loss_sum.item() / local_sample_count.item()) if local_sample_count.item() > 0 else 0.0
        train_acc = (t_correct.item() / t_total.item()) if t_total.item() > 0 else 0.0

        # Validation pass (if available)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_samples = 0
        model.eval()
        if val_loader is not None:
            val_sampler.set_epoch(epoch)
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", disable=global_rank!=0):
                    batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
                    if task == 'seq2seq':
                        labels = None
                        for key in ('labels', 'label', 'target_ids'):
                            if key in batch and batch[key] is not None:
                                labels = batch[key]
                                break
                        inputs = {k: v for k, v in batch.items() if k not in ('labels', 'label', 'target_ids') and k in forward_arg_names}
                        if _AMP_USES_DEVICE_ARG:
                            v_ctx = torch_amp.autocast('cuda', enabled=use_amp)
                        else:
                            v_ctx = torch_amp.autocast(enabled=use_amp)
                        with v_ctx:
                            outputs = model(**inputs, labels=labels)
                        # determine batch size
                        bs = None
                        for v in batch.values():
                            if torch.is_tensor(v):
                                try:
                                    bs = v.size(0)
                                except Exception:
                                    bs = None
                                break
                        if bs is None:
                            bs = 1
                        val_loss += outputs.loss.item() * bs
                        val_samples += bs
                    else:
                        labels = None
                        for key in ('labels', 'label'):
                            if key in batch and batch[key] is not None:
                                labels = batch[key]
                                break
                        inputs = {k: v for k, v in batch.items() if k not in ('labels', 'label') and k in forward_arg_names}
                        if _AMP_USES_DEVICE_ARG:
                            v_ctx = torch_amp.autocast('cuda', enabled=use_amp)
                        else:
                            v_ctx = torch_amp.autocast(enabled=use_amp)
                        with v_ctx:
                            outputs = model(**inputs, labels=labels)
                        # determine batch size
                        bs = None
                        for v in batch.values():
                            if torch.is_tensor(v):
                                try:
                                    bs = v.size(0)
                                except Exception:
                                    bs = None
                                break
                        if bs is None:
                            bs = 1
                        val_loss += outputs.loss.item() * bs
                        val_samples += bs
                        if outputs.logits is not None and labels is not None:
                            preds = torch.argmax(outputs.logits, dim=1)
                            val_correct += (preds == labels).sum().item()
                            val_total += labels.size(0)

            # aggregate validation metrics by summing raw losses and counts
            local_val_loss_sum = torch.tensor(val_loss, device=device)
            local_val_sample_count = torch.tensor(val_samples, device=device)
            v_correct = torch.tensor(val_correct, device=device)
            v_total = torch.tensor(val_total if val_total > 0 else 0, device=device)
            dist.all_reduce(local_val_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_val_sample_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(v_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(v_total, op=dist.ReduceOp.SUM)
            val_loss = (local_val_loss_sum.item() / local_val_sample_count.item()) if local_val_sample_count.item() > 0 else 0.0
            val_acc = (v_correct.item() / v_total.item()) if v_total.item() > 0 else 0.0
        else:
            val_loss = 0.0
            val_acc = 0.0

        # Print / log / checkpoint on rank 0
        if global_rank == 0:
            if val_loader is not None:
                msg = f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            else:
                msg = f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val: None"
            print(msg)
            if writer:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('accuracy/train', train_acc, epoch)
                if val_loader is not None:
                    writer.add_scalar('loss/val', val_loss, epoch)
                    writer.add_scalar('accuracy/val', val_acc, epoch)
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
        dist.barrier()
    # After training is complete, delete the checkpoint file if it exists (only on rank 0)
    if global_rank == 0:
        # Final model export if requested
        if config.get('save_final_model', True):
            final_dir = config.get('final_model_dir', os.path.join('.', 'final_model'))
            os.makedirs(final_dir, exist_ok=True)
            target_model = model.module if hasattr(model, 'module') else model
            try:
                target_model.save_pretrained(final_dir)
                tokenizer.save_pretrained(final_dir)
                print(f"[Rank 0] Saved final model + tokenizer to {final_dir}")
            except Exception as e:
                print(f"[Rank 0] Warning: failed to save final model: {e}")
        # Handle checkpoint retention policy
        if os.path.exists(checkpoint_path):
            if not config.get('keep_last_checkpoint', False):
                os.remove(checkpoint_path)
                print(f"[Rank 0] Deleted checkpoint file at {checkpoint_path} after successful training.")
            else:
                print(f"[Rank 0] Retained last checkpoint at {checkpoint_path}")
        if writer:
            writer.close()
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

    # ---- Basic config checks and concise coercion ----
    required = ['batch_size', 'learning_rate', 'num_epochs']
    for k in required:
        if k not in config:
            raise ValueError(f"Config must provide '{k}'")

    # concise coercions with simple errors
    try:
        config['batch_size'] = int(config['batch_size'])
        config['learning_rate'] = float(config['learning_rate'])
        config['num_epochs'] = int(config['num_epochs'])
        # optional fields with defaults
        config['num_warmup_steps'] = int(config.get('num_warmup_steps', 0))
        config['num_labels'] = int(config.get('num_labels', 2))
        if 'grad_accum_steps' in config:
            config['grad_accum_steps'] = max(1, int(config['grad_accum_steps']))
        config['seed'] = int(config.get('seed', 42))
        config['ddp_timeout_seconds'] = int(config.get('ddp_timeout_seconds', 1800))
        if 'num_workers' in config:
            config['num_workers'] = int(config['num_workers'])
        if 'pin_memory' in config:
            config['pin_memory'] = bool(config['pin_memory'])
        if 'deterministic' in config:
            config['deterministic'] = bool(config['deterministic'])
        if 'use_amp' in config:
            config['use_amp'] = bool(config['use_amp'])
        if 'keep_last_checkpoint' in config:
            config['keep_last_checkpoint'] = bool(config['keep_last_checkpoint'])
        if 'save_final_model' in config:
            config['save_final_model'] = bool(config['save_final_model'])
        if 'suppress_hf_warnings' in config:
            config['suppress_hf_warnings'] = bool(config['suppress_hf_warnings'])
    except Exception as e:
        raise TypeError(f"Config type error: {e}")

    # Environment-based path validation 
    for env_key in ('DATA_DIR', 'MODEL_DIR', 'EXPERIMENTS_ROOT'):
        path_val = os.environ.get(env_key)
        if not path_val or not os.path.exists(path_val):
            raise ValueError(f"Environment {env_key} missing or path does not exist: {path_val}")

    train(config)
