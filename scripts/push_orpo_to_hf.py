#!/usr/bin/env python3
"""Push ORPO preference dataset to HuggingFace Hub with Image() feature.

The aitraining VLM ORPO trainer expects `images` column to decode to PIL.
HF Datasets `Image()` feature stores bytes and decodes on access — exactly
what `trl.DataCollatorForVisionPreference` consumes.

Usage:
    HF_TOKEN=hf_xxx python scripts/push_orpo_to_hf.py \
        --repo monostate/ivf-bench-orpo-qwen9b

Reads:
    data/orpo/train.jsonl
    data/orpo/eval.jsonl
    data/raw/Images/*.png

Pushes:
    {repo} with train + eval splits, columns:
        images: list[Image]   (decodes to PIL)
        prompt: list[{role,content}]
        chosen: list[{role,content}]
        rejected: list[{role,content}]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
# HF quirk: Sequence({"role": Value, "content": Value}) transposes to
# dict-of-lists on access. We want list-of-dicts, so use [{...}] syntax.


def _load_split(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _to_dataset(rows: list[dict], project_root: Path) -> Dataset:
    """Convert JSONL rows to a HF Dataset with proper schema.

    Resolves relative image paths against project_root and casts the
    `images` column with Image() so HF decodes to PIL on access.
    """
    data = {"images": [], "prompt": [], "chosen": [], "rejected": []}
    for r in rows:
        # Resolve image paths to absolute so HF can read bytes.
        imgs = []
        for p in r["images"]:
            ap = (project_root / p).resolve() if not Path(p).is_absolute() else Path(p)
            if not ap.exists():
                raise FileNotFoundError(f"Image missing: {ap}")
            imgs.append(str(ap))
        data["images"].append(imgs)
        data["prompt"].append(r["prompt"])
        data["chosen"].append(r["chosen"])
        data["rejected"].append(r["rejected"])

    msg_list = [{"role": Value("string"), "content": Value("string")}]
    features = Features({
        "images": Sequence(Image()),
        "prompt": msg_list,
        "chosen": msg_list,
        "rejected": msg_list,
    })
    return Dataset.from_dict(data, features=features)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. monostate/ivf-bench-orpo-qwen9b")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--data-dir", default="data/orpo")
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var required")

    root = Path(args.project_root).resolve()
    data_dir = Path(args.data_dir)

    print(f"Loading {data_dir}/train.jsonl + eval.jsonl...")
    train_rows = _load_split(data_dir / "train.jsonl")
    eval_rows = _load_split(data_dir / "eval.jsonl")
    print(f"  train={len(train_rows)} eval={len(eval_rows)}")

    print("Building HF Datasets with Image() feature...")
    ds = DatasetDict({
        "train": _to_dataset(train_rows, root),
        "eval": _to_dataset(eval_rows, root),
    })

    # Sanity: confirm first row decodes
    sample = ds["train"][0]
    pil = sample["images"][0]
    print(f"  first image decodes: {type(pil).__name__} size={pil.size}")
    print(f"  prompt[0]: {sample['prompt'][0]['role']} ({len(sample['prompt'][0]['content'])} chars)")
    print(f"  chosen[0]: {sample['chosen'][0]['role']} ({len(sample['chosen'][0]['content'])} chars)")

    print(f"\nPushing to {args.repo} (private={args.private})...")
    ds.push_to_hub(args.repo, token=token, private=args.private)
    print(f"Done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
