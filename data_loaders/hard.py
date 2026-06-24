"""HARD dataset loader.

This module provides a single entry point:
    load_hard_tokenized_dataset(...)

Design goals:
- Stable 3-class mapping from 1..5 hotel review ratings.
- Reproducible stratified splits via saved indices JSON (split_seed).
- Outputs a HuggingFace DatasetDict formatted for Torch Trainer.

Common 3-class mapping used in Arabic sentiment work:
  1-2 -> negative (0)
  3   -> neutral  (1)
  4-5 -> positive (2)

Split strategy
--------------
- HF provides train + validation + test : use all three directly.
- HF provides train + test only         : keep official test; carve val
                                          from official train (90/10 stratified).
- HF provides train only / local file   : create 80/10/10 stratified split.
All split index sets are persisted in split_file for full reproducibility.
The MAX_TRAIN cap is applied after splitting; selected indices are also
saved so repeated runs select exactly the same training examples.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import datasets as hf_datasets

from data_loaders.common import NEG, NEU, POS, debug_label_distribution, stratified_split_indices, val_split_indices, apply_train_cap, make_tokenize_fn

DatasetDict = hf_datasets.DatasetDict
load_dataset = hf_datasets.load_dataset

# Deterministic mapping for HARD star ratings:
# 1-2 -> NEG (Negative)
# 3   -> NEU (Neutral)
# 4-5 -> POS (Positive)
HARD_RATING_MAP = {
    1: NEG,
    2: NEG,
    3: NEU,
    4: POS,
    5: POS,
}


def _rating_to_3class(raw) -> int:
    """Map HARD rating variants to {0,1,2}. Returns -1 if unknown."""
    if raw is None:
        return -1

    if isinstance(raw, (int, np.integer)):
        r = int(raw)
    elif isinstance(raw, float) and not np.isnan(raw):
        r = int(raw)
    else:
        s = str(raw).strip()
        if not s:
            return -1
        try:
            r = int(float(s))
        except Exception:
            return -1

    return HARD_RATING_MAP.get(r, -1)



def load_hard_tokenized_dataset(
    *,
    data_path: Optional[str] = None,
    tok_name: str,
    max_length: int,
    split_seed: int,
    split_file: Optional[str] = None,
    text_col: str = "text",
    rating_col: str = "rating",
) -> DatasetDict:
    """Load HARD from HF or a local file, map to 3 classes, tokenize, and split.

    Parameters
    ----------
    data_path:
        Optional local HARD file (.csv/.tsv) override containing at least
        (text, rating) or (review, rating). If omitted, loads from
        Hugging Face (Elnagara/hard) and respects any official splits.
    tok_name:
        HF tokenizer name/path.
    max_length:
        Max sequence length for truncation.
    split_seed:
        Seed used ONLY for dataset split generation.
    split_file:
        JSON path to save/reuse split indices.

    Returns
    -------
    DatasetDict with train/validation/test and torch columns.
    """
    # 1. Load — detect which official splits HF provides
    _hf_has_all_three = False   # train + validation + test
    _hf_has_train_test = False  # train + test only (no validation split)

    if data_path is None:
        ds_hf = load_dataset("Elnagara/hard", "plain_text")

        if isinstance(ds_hf, hf_datasets.DatasetDict):
            available = set(ds_hf.keys())
            _val_key = "validation" if "validation" in available else ("dev" if "dev" in available else None)
            if _val_key and {"train", "test"} <= available:
                _hf_has_all_three = True
                ds_train_raw = ds_hf["train"]
                ds_val_raw   = ds_hf[_val_key]
                ds_test_raw  = ds_hf["test"]
            elif {"train", "test"} <= available:
                # Official test must not be discarded — carve val from train.
                _hf_has_train_test = True
                ds_train_raw = ds_hf["train"]
                ds_test_raw  = ds_hf["test"]
            elif "train" in available:
                ds_train_raw = ds_hf["train"]
            else:
                raise ValueError(
                    f"HARD HF dataset has no usable splits. Available: {sorted(available)}"
                )
        else:
            ds_train_raw = ds_hf
    else:
        ext = os.path.splitext(data_path)[1].lower()
        if ext not in (".csv", ".tsv"):
            raise ValueError(f"Unsupported HARD file extension: {ext}. Use .csv or .tsv")

        ds_train_raw = load_dataset(
            "csv",
            data_files=data_path,
            delimiter=("\t" if ext == ".tsv" else ","),
        )["train"]

    # 2. Normalize columns & map ratings to 3-class labels
    def _prepare_part(part):
        if text_col not in part.column_names:
            for cand in ("review", "text", "content", "comments", "comment",
                         "sentence", "body", "tweet", "tweets"):
                if cand in part.column_names:
                    part = part.rename_column(cand, text_col)
                    break

        if rating_col not in part.column_names:
            for cand in ("rating", "label", "stars", "score", "class", "polarity"):
                if cand in part.column_names:
                    part = part.rename_column(cand, rating_col)
                    break

        if text_col not in part.column_names or rating_col not in part.column_names:
            raise ValueError(
                f"HARD expected text/rating columns (e.g. review+rating). "
                f"Got: {part.column_names}"
            )

        part = part.map(lambda ex: {"label": _rating_to_3class(ex[rating_col])})
        part = part.filter(lambda x: x["label"] in (NEG, NEU, POS))
        return part

    if _hf_has_all_three:
        ds_train = _prepare_part(ds_train_raw)
        ds_val   = _prepare_part(ds_val_raw)
        ds_test  = _prepare_part(ds_test_raw)
    elif _hf_has_train_test:
        ds_train_prep = _prepare_part(ds_train_raw)
        ds_test       = _prepare_part(ds_test_raw)
    else:
        ds_all = _prepare_part(ds_train_raw)

    # 3. Resolve split_file path
    if split_file is None:
        split_dir = os.path.join("data", "splits")
        os.makedirs(split_dir, exist_ok=True)
        split_file = os.path.join(split_dir, f"hard_split_seed_{split_seed}.json")

    os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)

    # 4. Build splits — load persisted indices or generate + save
    if _hf_has_all_three:
        ds_splits = DatasetDict(
            {"train": ds_train, "validation": ds_val, "test": ds_test}
        )
        print(f"[INFO] HARD official train size: {len(ds_splits['train'])}")
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                _payload = json.load(f)
        else:
            _payload = {"split_mode": "all_official", "split_seed": split_seed}
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(_payload, f, ensure_ascii=False, indent=2)
            print(f"Saved HARD split metadata to: {split_file}")

    elif _hf_has_train_test:
        labels = ds_train_prep["label"]
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                _payload = json.load(f)
            train_idx = _payload["train_indices"]
            val_idx   = _payload["validation_indices"]
            print(f"Loaded HARD train/val split indices from: {split_file}")
        else:
            train_idx, val_idx = val_split_indices(labels, split_seed)
            _payload = {
                "split_seed": split_seed,
                "split_mode": "train_test_official",
                "train_indices": train_idx,
                "validation_indices": val_idx,
            }
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(_payload, f, ensure_ascii=False, indent=2)
            print(f"Saved HARD train/val split indices to: {split_file}")

        ds_splits = DatasetDict(
            {
                "train":      ds_train_prep.select(train_idx),
                "validation": ds_train_prep.select(val_idx),
                "test":       ds_test,
            }
        )

    else:
        labels = ds_all["label"]
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                _payload = json.load(f)
            train_idx = _payload["train_indices"]
            val_idx   = _payload["validation_indices"]
            test_idx  = _payload["test_indices"]
            print(f"Loaded HARD split indices from: {split_file}")
        else:
            train_idx, val_idx, test_idx = stratified_split_indices(labels, split_seed)
            _payload = {
                "split_seed": split_seed,
                "split_mode": "custom_80_10_10",
                "train_indices": train_idx,
                "validation_indices": val_idx,
                "test_indices": test_idx,
            }
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(_payload, f, ensure_ascii=False, indent=2)
            print(f"Saved HARD split indices to: {split_file}")

        ds_splits = DatasetDict(
            {
                "train":      ds_all.select(train_idx),
                "validation": ds_all.select(val_idx),
                "test":       ds_all.select(test_idx),
            }
        )

    # 5. Cap training size
    ds_splits, _payload = apply_train_cap(ds_splits, split_seed, split_file, _payload, "HARD")

    debug_label_distribution(ds_splits, "HARD", HARD_RATING_MAP)

    # 6. Tokenize
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    ds_tok = ds_splits.map(make_tokenize_fn(tokenizer, text_col, max_length), batched=True, desc="Tokenising HARD")
    ds_tok = ds_tok.rename_column("label", "labels").with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return ds_tok