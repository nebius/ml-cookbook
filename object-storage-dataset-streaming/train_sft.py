#!/usr/bin/env python3
"""Fine-tune Qwen3.8-27B with QLoRA on document QA read straight from a mounted bucket.

The dataset directory is a Mountpoint for Amazon S3 mount, so this is an ordinary
TRL supervised fine-tuning run. The only storage-specific code is the Parquet
glob in ``load_documents``; everything else is the model and the trainer.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
from datasets import Image, IterableDataset, List, load_dataset
from peft import LoraConfig
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="directory that contains train/part-*.parquet, usually a Mountpoint mount",
    )
    parser.add_argument("--output-dir", default="outputs/qwen3.8-27b-docvqa-lora")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-pixels", type=int, default=65_536)
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-save", action="store_true")
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        parser.error("max-steps, batch-size, and gradient-accumulation must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    if not 0 < args.min_pixels <= args.max_pixels:
        parser.error("pixel bounds must satisfy 0 < min-pixels <= max-pixels")
    return args


def load_documents(
    data_dir: Path,
    *,
    world_size: int,
    num_workers: int,
    seed: int,
) -> IterableDataset:
    """Stream Parquet shards from the mounted directory. This is the whole storage layer."""

    files = sorted(str(path) for path in (data_dir / "train").glob("part-*.parquet"))
    if not files:
        raise SystemExit(
            f"no train/part-*.parquet under {data_dir}: is the bucket mounted, and did "
            "the mount --prefix end with '/'?"
        )
    if len(files) < world_size:
        raise SystemExit(
            f"{len(files)} shard(s) for {world_size} processes: with fewer shards than "
            "processes every process would read the whole dataset, so publish at least "
            "one shard per process"
        )
    workers = max(1, num_workers)
    if len(files) < world_size * workers or len(files) % world_size:
        print(
            f"note: {len(files)} shards for {world_size} process(es) x {workers} worker(s); "
            f"use a multiple of {world_size} and at least {world_size * workers} shards to keep "
            "every worker busy; use similar row counts per shard to balance sampling",
            flush=True,
        )
    # Randomize shard order here: datasets 5.0.1's default shuffle interleaves up to ten
    # shard streams, reducing n_shards (four sample shards become one). With fewer shards
    # than processes, Accelerate makes every process read every file. A custom shuffle
    # with max_buffer_input_shards=1 would also preserve the shard count.
    random.Random(seed).shuffle(files)  # noqa: S311 - ordering, not cryptography
    # Local Parquet files on the mount, not a Hub download, so no revision applies.
    dataset = load_dataset(  # nosec B615
        "parquet", data_files=files, split="train", streaming=True
    )
    # Repeat forever so --max-steps alone bounds the run. With a finite stream the Trainer
    # only steps the optimizer when a full accumulation window completes inside one pass,
    # so a small dataset with a large --gradient-accumulation would never train.
    return dataset.repeat(None).cast_column("images", List(Image()))


def load_model(local_rank: int):
    """Load Qwen in NF4 on this process's GPU; LoRA is added by the trainer."""

    return AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": local_rank},
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    documents = load_documents(
        args.data_dir,
        world_size=world_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = load_model(local_rank)
    model.config.text_config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=0.05,  # a float is a ratio of total steps in transformers 5
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=args.num_workers,
            dataloader_pin_memory=True,
            max_length=None,
            completion_only_loss=True,
            remove_unused_columns=False,
            shuffle_dataset=False,  # shard order is already randomized; see load_documents
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=documents,
        processing_class=processor,
        peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            exclude_modules=r"model\.visual(?:\..*)?",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            revision=MODEL_REVISION,
        ),
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    result = trainer.train()
    if trainer.accelerator.is_main_process:
        metrics = result.metrics
        peak_gib = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
        print(
            f"finished {args.max_steps} optimizer steps in {metrics['train_runtime']:.1f}s "
            f"({metrics['train_samples_per_second']:.3f} documents/s, "
            f"mean training loss {metrics['train_loss']:.4f}, "
            f"peak reserved GPU memory {peak_gib:.1f} GiB at --max-pixels {args.max_pixels})",
            flush=True,
        )
        if not args.skip_save:
            trainer.save_model(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"saved the LoRA adapter and processor to {args.output_dir}", flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
