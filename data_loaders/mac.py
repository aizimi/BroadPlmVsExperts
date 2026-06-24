from __future__ import annotations
import os
import json
from typing import Dict, Optional

import numpy as np
from datasets import DatasetDict, load_dataset
from data_loaders.common import NEG, NEU, POS, debug_label_distribution, stratified_split_indices, make_tokenize_fn

# Maps MAC textual labels to the shared 3-class integer scheme (NEG=0, NEU=1, POS=2).
MAC_LABEL_MAP = {
    "negative": NEG,
    "neutral": NEU,
    "positive": POS,
}


def load_mac_tokenized_dataset(
    csv_path: str,
    tok_name: str,
    max_length: int,
    split_seed: int,
    label_map: Optional[Dict[str, int]] = None,
    split_file: Optional[str] = None,
) -> DatasetDict:
    ds_full = load_dataset("csv", data_files=csv_path)["train"]

    if "tweets" in ds_full.column_names and "text" not in ds_full.column_names:
        ds_full = ds_full.rename_column("tweets", "text")
    if "tweet" in ds_full.column_names and "text" not in ds_full.column_names:
        ds_full = ds_full.rename_column("tweet", "text")
    if "type" in ds_full.column_names:
        ds_full = ds_full.rename_column("type", "label")

    if "text" not in ds_full.column_names or "label" not in ds_full.column_names:
        raise ValueError(
            f"MAC CSV must contain columns 'text' and 'label' (or 'tweet'/'tweets' and 'type'). Found: {ds_full.column_names}"
        )
    if label_map is None:
        label_map = MAC_LABEL_MAP

    # 1. Normalize labels
    normalized_label_map = {str(k).strip().lower(): int(v) for k, v in label_map.items()}

    def normalize_label(example):
        raw = str(example["label"]).strip().lower()
        return {"label": normalized_label_map.get(raw, -1)}

    ds_full = ds_full.map(normalize_label)
    ds_full = ds_full.filter(lambda x: x["label"] != -1)

    if split_file is None:
        split_file = f"mac_split_seed_{split_seed}.json"

    if os.path.isfile(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        train_idx = payload["train_indices"]
        val_idx = payload["validation_indices"]
        test_idx = payload["test_indices"]
        print(f"Loaded split indices from: {split_file}")
    else:
        labels = ds_full["label"]
        train_idx, val_idx, test_idx = stratified_split_indices(labels, split_seed)
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split_seed": split_seed,
                    "train_indices": train_idx,
                    "validation_indices": val_idx,
                    "test_indices": test_idx,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved split indices to: {split_file}")

    ds_splits = DatasetDict(
        {
            "train": ds_full.select(train_idx),
            "validation": ds_full.select(val_idx),
            "test": ds_full.select(test_idx),
        }
    )

    debug_label_distribution(ds_splits, "MAC", label_map)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    ds_tok = ds_splits.map(make_tokenize_fn(tokenizer, "text", max_length), batched=True, desc="Tokenising MAC")
    ds_tok = ds_tok.rename_column("label", "labels").with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return ds_tok