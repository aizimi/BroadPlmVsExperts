"""ASTD dataset loader.

Split strategy
--------------
- HF provides train + (validation|dev) + test : use all three directly.
- HF provides train + test only               : keep official test; carve val
                                                from official train (90/10 stratified).
- HF provides train only / local file         : create 80/10/10 stratified split.
All split index sets are persisted in split_file for full reproducibility.
The MAX_TRAIN cap is applied after splitting; selected indices are also
saved so repeated runs select exactly the same training examples.
"""

from __future__ import annotations

import json
import os
import numpy as np
from typing import Optional

import datasets as hf_datasets

from data_loaders.common import NEG, NEU, POS, debug_label_distribution, stratified_split_indices, val_split_indices, apply_train_cap, make_tokenize_fn

DatasetDict = hf_datasets.DatasetDict
load_dataset = hf_datasets.load_dataset

# ASTD uses 4 original labels; we collapse them to 3 classes.
# Numeric encoding observed in the HF version: 0=Neutral, 1=Objective, 2=Positive, 3=Negative.
ASTD_ID_MAP = {
    0: NEU,  # Neutral
    1: NEU,  # Objective
    2: POS,  # Positive
    3: NEG,  # Negative
}

ASTD_STR_MAP = {
    "objective": NEU,
    "neutral": NEU,
    "mixed": NEU,
    "positive": POS,
    "pos": POS,
    "negative": NEG,
    "neg": NEG,
    "subjective positive": POS,
    "subjective negative": NEG,
    "subjective mixed": NEU,
}


def _normalize_label_to_3class(raw) -> int:
    """Map any ASTD label variant to {NEG=0, NEU=1, POS=2}. Returns -1 for unknowns."""
    if raw is None:
        return -1
    if isinstance(raw, (int, np.integer)):
        return ASTD_ID_MAP.get(int(raw), -1)
    s = str(raw).strip().lower().replace("\t", " ").replace("\n", " ").strip()
    return ASTD_STR_MAP.get(s, -1)



def load_astd_tokenized_dataset(
    *,
    tok_name: str,
    max_length: int,
    split_seed: int,
    data_path: Optional[str] = None,
    split_file: Optional[str] = None,
    text_col: str = "text",
    label_col: str = "label",
) -> DatasetDict:
    """Load ASTD, map to 3 classes, tokenize, and return DatasetDict.

    Parameters
    ----------
    tok_name:
        HF tokenizer name/path.
    max_length:
        Max sequence length for truncation.
    split_seed:
        Seed used ONLY for dataset split generation.
    data_path:
        Optional local file path override (.csv/.tsv/.txt).
        If omitted, loads from Hugging Face (arbml/ASTD).
    split_file:
        Optional JSON path to save/reuse split indices.
    text_col / label_col:
        Column names in the loaded Dataset.

    Returns
    -------
    DatasetDict with keys: train/validation/test
    and torch-formatted columns: input_ids, attention_mask, labels
    """
    # 1. Load — detect which official splits HF provides
    _hf_has_all_three = False
    _hf_has_train_test = False

    if data_path is None:
        ds_hf = load_dataset("arbml/ASTD")

        # R2 fix: verify ASTD_ID_MAP against the HF ClassLabel encoding at runtime.
        # If the dataset's label order ever changes, this will raise immediately
        # rather than silently corrupting labels.
        _ref_split = "train" if "train" in ds_hf else next(iter(ds_hf))
        _label_feature = ds_hf[_ref_split].features.get(label_col)
        if hasattr(_label_feature, "names"):
            hf_names = _label_feature.names
            for idx, name in enumerate(hf_names):
                expected = ASTD_STR_MAP.get(name.lower().strip())
                actual = ASTD_ID_MAP.get(idx)
                if expected is None:
                    raise RuntimeError(
                        f"ASTD ClassLabel index {idx} has name '{name}' which is absent "
                        f"from ASTD_STR_MAP. Update ASTD_STR_MAP to cover this label."
                    )
                if expected != actual:
                    raise RuntimeError(
                        f"ASTD label encoding mismatch at index {idx}: "
                        f"HF name='{name}' → ASTD_STR_MAP gives {expected}, "
                        f"but ASTD_ID_MAP gives {actual}. "
                        f"Update ASTD_ID_MAP to match the actual HF label order."
                    )
            print(f"[INFO] ASTD ClassLabel encoding verified: {dict(enumerate(hf_names))}")

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
                    f"ASTD HF dataset has no usable splits. Available: {sorted(available)}. "
                    "Expected at least a 'train' split."
                )
        else:
            ds_train_raw = ds_hf
    else:
        ext = os.path.splitext(data_path)[1].lower()
        if ext in (".csv", ".tsv"):
            ds_train_raw = load_dataset(
                "csv", data_files=data_path,
                delimiter=("\t" if ext == ".tsv" else ","),
            )["train"]
        elif ext == ".txt":
            ds_train_raw = load_dataset(
                "csv", data_files=data_path,
                delimiter="\t", column_names=["text", "rating"],
            )["train"]
        else:
            raise ValueError(f"Unsupported ASTD file extension: {ext}. Use .csv/.tsv/.txt")

    # 2. Normalize columns & map labels to 3-class
    def _prepare_part(part):
        if text_col not in part.column_names:
            for cand in ("tweet", "tweets", "text", "review", "sentence", "content"):
                if cand in part.column_names:
                    part = part.rename_column(cand, text_col)
                    break

        if label_col not in part.column_names:
            for cand in ("label", "rating", "class", "sentiment", "polarity"):
                if cand in part.column_names:
                    part = part.rename_column(cand, label_col)
                    break

        if text_col not in part.column_names or label_col not in part.column_names:
            raise ValueError(
                f"ASTD expected columns text/label (or tweet/label). Got: {part.column_names}"
            )

        def _map(ex):
            return {label_col: _normalize_label_to_3class(ex[label_col])}

        part = part.map(_map)
        part = part.filter(lambda x: x[label_col] in (NEG, NEU, POS))
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
        split_file = os.path.join(split_dir, f"astd_split_seed_{split_seed}.json")

    os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)

    # 4. Build splits — load persisted indices or generate + save
    if _hf_has_all_three:
        ds_splits = DatasetDict(
            {"train": ds_train, "validation": ds_val, "test": ds_test}
        )
        print(f"[INFO] ASTD official train size: {len(ds_splits['train'])}")
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                _payload = json.load(f)
        else:
            _payload = {"split_mode": "all_official", "split_seed": split_seed}
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(_payload, f, ensure_ascii=False, indent=2)
            print(f"Saved ASTD split metadata to: {split_file}")

    elif _hf_has_train_test:
        labels = ds_train_prep[label_col]
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                _payload = json.load(f)
            train_idx = _payload["train_indices"]
            val_idx   = _payload["validation_indices"]
            print(f"Loaded ASTD train/val split indices from: {split_file}")
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
            print(f"Saved ASTD train/val split indices to: {split_file}")

        ds_splits = DatasetDict(
            {
                "train":      ds_train_prep.select(train_idx),
                "validation": ds_train_prep.select(val_idx),
                "test":       ds_test,
            }
        )

    else:
        labels = ds_all[label_col]
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                _payload = json.load(f)
            train_idx = _payload["train_indices"]
            val_idx   = _payload["validation_indices"]
            test_idx  = _payload["test_indices"]
            print(f"Loaded ASTD split indices from: {split_file}")
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
            print(f"Saved ASTD split indices to: {split_file}")

        ds_splits = DatasetDict(
            {
                "train":      ds_all.select(train_idx),
                "validation": ds_all.select(val_idx),
                "test":       ds_all.select(test_idx),
            }
        )

    # 5. Cap training size
    ds_splits, _payload = apply_train_cap(ds_splits, split_seed, split_file, _payload, "ASTD")

    debug_label_distribution(ds_splits, "ASTD", ASTD_ID_MAP)

    # 6. Tokenize
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    ds_tok = ds_splits.map(make_tokenize_fn(tokenizer, text_col, max_length), batched=True, desc="Tokenising ASTD")
    ds_tok = ds_tok.rename_column(label_col, "labels").with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return ds_tok