#!/usr/bin/env python3
"""Write a small Docmatix sample as Parquet shards, ready to upload to a bucket.

Each Parquet row holds one document: its page images and one question-answer
pair in TRL's prompt-completion format. Several shards per split let every
training process and DataLoader worker read its own files. Upload the output
directory with the S3 command-line tool you already use.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Image, List, load_dataset

DATASET = "HuggingFaceM4/Docmatix"
DATASET_CONFIG = "images"
REVISION = "0725b65616e0e5f6024be10e38ddf8d8c48664fd"
SOURCE_SPLIT = "train"

MESSAGE_TYPE = pa.struct(
    [
        pa.field("role", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
    ]
)
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("images", pa.list_(pa.binary()), nullable=False),
        pa.field("prompt", pa.list_(MESSAGE_TYPE), nullable=False),
        pa.field("completion", pa.list_(MESSAGE_TYPE), nullable=False),
        pa.field(
            "chat_template_kwargs",
            pa.struct([pa.field("enable_thinking", pa.bool_(), nullable=False)]),
            nullable=False,
        ),
    ]
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-documents", type=int, required=True)
    parser.add_argument("--validation-documents", type=int, default=0)
    parser.add_argument("--documents-per-file", type=int, default=1)
    args = parser.parse_args(argv)
    if args.train_documents <= 0 or args.validation_documents < 0 or args.documents_per_file <= 0:
        parser.error("document counts must be positive (validation may be zero)")
    return args


def make_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Turn one Docmatix document into TRL's vision prompt-completion format."""

    qa = row["texts"][0]
    return {
        "images": [image["bytes"] for image in row["images"]],
        "prompt": [{"role": "user", "content": qa["user"]}],
        "completion": [{"role": "assistant", "content": qa["assistant"]}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def write_parquet_samples(destination: Path, samples: Iterable[dict[str, Any]]) -> int:
    """Write one document per row group so a reader can start before the file ends."""

    written = 0
    with pq.ParquetWriter(destination, PARQUET_SCHEMA, compression="snappy") as parquet:
        for sample in samples:
            parquet.write_table(pa.Table.from_pylist([sample], schema=PARQUET_SCHEMA))
            written += 1
    return written


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for split in ("train", "validation"):
        directory = args.output_dir / split
        if directory.exists() and any(directory.iterdir()):
            raise SystemExit(f"{directory} is not empty; choose a fresh --output-dir")

    # Keep page images encoded; the trainer decodes one document at a time.
    source = iter(
        load_dataset(
            DATASET,
            DATASET_CONFIG,
            revision=REVISION,
            split=SOURCE_SPLIT,
            streaming=True,
        ).cast_column("images", List(Image(decode=False)))
    )
    for split, document_count in (
        ("train", args.train_documents),
        ("validation", args.validation_documents),
    ):
        if not document_count:
            continue
        directory = args.output_dir / split
        directory.mkdir(parents=True, exist_ok=True)
        for part, start in enumerate(range(0, document_count, args.documents_per_file)):
            rows = min(args.documents_per_file, document_count - start)
            path = directory / f"part-{part:05d}.parquet"
            written = write_parquet_samples(path, (make_sample(next(source)) for _ in range(rows)))
            print(f"wrote {path} ({written} documents)", flush=True)
    print(
        f"Upload with: aws s3 sync {args.output_dir} s3://YOUR_BUCKET/datasets/docmatix-demo/ "
        "--endpoint-url https://storage.eu-north1.nebius.cloud",
        flush=True,
    )


if __name__ == "__main__":
    main()
    # Work around huggingface/datasets#7357: an Arrow thread can crash interpreter shutdown.
    os._exit(0)
