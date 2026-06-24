"""LABR dataset loader.

This module provides a single entry point:
    load_labr_tokenized_dataset(...)

Design goals:
- Stable 3-class mapping from 1..5 star ratings.
- Reproducible stratified splits via saved indices JSON (split_seed).
- Outputs a HuggingFace DatasetDict formatted for Torch Trainer.

Common 3-class mapping used in Arabic sentiment work:
  1-2 -> negative (0)
  3   -> neutral  (1)
  4-5 -> positive (2)
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import datasets as hf_datasets
from data_loaders.common import NEG, NEU, POS, debug_label_distribution, TokenLengthProfiler, stratified_split_indices, val_split_indices, apply_train_cap, make_tokenize_fn

DatasetDict = hf_datasets.DatasetDict
load_dataset = hf_datasets.load_dataset


# Deterministic mapping for LABR star ratings:
# 1-2 -> NEG (Negative)
# 3   -> NEU (Neutral)
# 4-5 -> POS (Positive)
LABR_RATING_MAP = {
    1: NEG,
    2: NEG,
    3: NEU,
    4: POS,
    5: POS,
}


def _rating_to_3class(raw) -> int:
    """Map LABR rating variants to {0,1,2}. Returns -1 if unknown."""
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
        # handle strings like "5" or "5.0"
        try:
            r = int(float(s))
        except Exception:
            return -1

    return LABR_RATING_MAP.get(r, -1)



def load_labr_tokenized_dataset(
    *,
    data_path: Optional[str] = None,
    tok_name: str,
    max_length: int,
    split_seed: int,
    split_file: Optional[str] = None,
    text_col: str = "text",
    rating_col: str = "rating",
) -> DatasetDict:
    """Load LABR from a local file, map to 3 classes, tokenize, and split.

    Parameters
    ----------
    data_path:
        Optional local LABR file (.csv/.tsv) override containing at least (text,rating) or (review,rating).
        If omitted, the loader will fetch LABR from Hugging Face (mohamedadaly/labr) and
        then apply the same 80/10/10 stratified split logic using `split_seed`.
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
    _hf_has_all_three = False
    _hf_has_train_test = False

    if data_path is None:
        ds_hf = load_dataset("mohamedadaly/labr")

        if isinstance(ds_hf, hf_datasets.DatasetDict):
            available = set(ds_hf.keys())
            _val_key = "validation" if "validation" in available else ("dev" if "dev" in available else None)
            if _val_key and {"train", "test"} <= available:
                _hf_has_all_three = True
                ds_train_raw = ds_hf["train"]
                ds_val_raw   = ds_hf[_val_key]
                ds_test_raw  = ds_hf["test"]
            elif {"train", "test"} <= available:
                _hf_has_train_test = True
                ds_train_raw = ds_hf["train"]
                ds_test_raw  = ds_hf["test"]
            elif "train" in available:
                ds_train_raw = ds_hf["train"]
            else:
                raise ValueError(
                    f"LABR HF dataset has no usable splits. Available: {sorted(available)}. "
                    "Expected at least a 'train' split."
                )
        else:
            ds_train_raw = ds_hf
    else:
        ext = os.path.splitext(data_path)[1].lower()
        if ext not in (".csv", ".tsv"):
            raise ValueError(f"Unsupported LABR file extension: {ext}. Use .csv or .tsv")

        ds_train_raw = load_dataset(
            "csv",
            data_files=data_path,
            delimiter=("\t" if ext == ".tsv" else ","),
        )["train"]

    # 2. Normalize columns & map ratings to 3-class labels
    def _prepare_part(part):
        if text_col not in part.column_names:
            for cand in ("review", "text", "sentence", "content", "tweet", "tweets"):
                if cand in part.column_names:
                    part = part.rename_column(cand, text_col)
                    break

        if rating_col not in part.column_names:
            for cand in ("rating", "label", "stars", "score", "class"):
                if cand in part.column_names:
                    part = part.rename_column(cand, rating_col)
                    break

        if text_col not in part.column_names or rating_col not in part.column_names:
            raise ValueError(
                f"LABR expected text/rating columns. Got: {part.column_names}"
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
        split_file = os.path.join(split_dir, f"labr_split_seed_{split_seed}.json")

    os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)

    # 4. Build splits — load persisted indices or generate + save
    if _hf_has_all_three:
        ds_splits = DatasetDict(
            {"train": ds_train, "validation": ds_val, "test": ds_test}
        )
        print(f"[INFO] LABR official train size: {len(ds_splits['train'])}")
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                _payload = json.load(f)
        else:
            _payload = {"split_mode": "all_official", "split_seed": split_seed}
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(_payload, f, ensure_ascii=False, indent=2)
            print(f"Saved LABR split metadata to: {split_file}")

    elif _hf_has_train_test:
        labels = ds_train_prep["label"]
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                _payload = json.load(f)
            train_idx = _payload["train_indices"]
            val_idx   = _payload["validation_indices"]
            print(f"Loaded LABR train/val split indices from: {split_file}")
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
            print(f"Saved LABR train/val split indices to: {split_file}")

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
            print(f"Loaded LABR split indices from: {split_file}")
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
            print(f"Saved LABR split indices to: {split_file}")

        ds_splits = DatasetDict(
            {
                "train":      ds_all.select(train_idx),
                "validation": ds_all.select(val_idx),
                "test":       ds_all.select(test_idx),
            }
        )

    # 5. Cap training size
    ds_splits, _payload = apply_train_cap(ds_splits, split_seed, split_file, _payload, "LABR")

    debug_label_distribution(ds_splits, "LABR", LABR_RATING_MAP)

    # 6. Tokenize — print token-length stats before truncation (LABR has long reviews)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    TokenLengthProfiler(tokenizer, "LABR").print_report(ds_splits["train"][text_col], max_length=max_length)

    ds_tok = ds_splits.map(make_tokenize_fn(tokenizer, text_col, max_length), batched=True, desc="Tokenising LABR")
    ds_tok = ds_tok.rename_column("label", "labels").with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return ds_tok