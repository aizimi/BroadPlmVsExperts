"""ArSAS dataset loader.

This module provides a single entry point:
    load_arsas_tokenized_dataset(...)

Design goals:
- Reproducible splits via a saved indices JSON (split_seed).
- Robust label normalization.
- Outputs a HuggingFace DatasetDict formatted for Torch Trainer.

ArSAS (Arabic Sentiment Analysis Shared task dataset) is available on
HuggingFace at: arbml/ArSAS.

The dataset typically uses three sentiment classes:
    negative -> 0
    neutral  -> 1
    positive -> 2

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

# ArSAS numeric encoding: 0=Negative, 1=Neutral, 2=Positive. Mixed (3) is not mapped and gets dropped.
ARSAS_ID_MAP = {
    0: NEG,  # Negative
    1: NEU,  # Neutral
    2: POS,  # Positive
}


def _normalize_label_to_3class(raw) -> int:
    """Map ArSAS label variants to {0,1,2}.

    Handles:
    - strings: objective/obj, mixed, neutral, positive/pos, negative/neg
    - deterministic numeric encodings: 0=Negative, 1=Neutral, 2=Positive, 3=Mixed

    Returns -1 for unknown labels.
    """
    if raw is None:
        return -1

    if isinstance(raw, (int, np.integer)):
        return ARSAS_ID_MAP.get(int(raw), -1)

    if isinstance(raw, float) and not np.isnan(raw):
        return _normalize_label_to_3class(int(raw))

    s = str(raw).strip().lower()
    s = s.replace("\t", " ").replace("\n", " ").strip()

    if s in {"negative", "neg", "subj_negative", "subjective negative", "subjective_negative"}:
        return NEG
    if s in {"positive", "pos", "subj_positive", "subjective positive", "subjective_positive"}:
        return POS
    if s in {"objective", "obj", "neutral", "neu"}:
        return NEU

    if "neg" in s and "pos" not in s:
        return NEG
    if "pos" in s and "neg" not in s:
        return POS
    if "obj" in s or "neutral" in s:
        return NEU

    return -1




def load_arsas_tokenized_dataset(
    *,
    tok_name: str,
    max_length: int,
    split_seed: int,
    data_path: Optional[str] = None,
    split_file: Optional[str] = None,
    text_col: str = "text",
    label_col: str = "label",
) -> DatasetDict:
    """Load ArSAS, map to 3 classes, tokenize, and return DatasetDict.

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
        If omitted, loads from Hugging Face (arbml/ArSAS).
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
        ds_hf = load_dataset("arbml/ArSAS")

        # R2 fix: verify ARSAS_ID_MAP against the HF ClassLabel encoding at runtime.
        _ARSAS_EXPECTED_STR_MAP = {
            "negative": NEG, "neg": NEG,
            "neutral": NEU, "neu": NEU, "objective": NEU, "obj": NEU,
            "positive": POS, "pos": POS,
        }
        _ref_split = "train" if "train" in ds_hf else next(iter(ds_hf))
        _label_feature = ds_hf[_ref_split].features.get(label_col)
        if hasattr(_label_feature, "names"):
            hf_names = _label_feature.names
            for idx, name in enumerate(hf_names):
                expected = _ARSAS_EXPECTED_STR_MAP.get(name.lower().strip())
                actual = ARSAS_ID_MAP.get(idx)
                if expected is None:
                    print(f"[INFO] ArSAS ClassLabel index {idx} ('{name}') is not in the label map — examples will be dropped.")
                    continue
                if expected != actual:
                    raise RuntimeError(
                        f"ArSAS label encoding mismatch at index {idx}: "
                        f"HF name='{name}' → expected {expected}, "
                        f"but ARSAS_ID_MAP gives {actual}. "
                        f"Update ARSAS_ID_MAP to match the actual HF label order."
                    )
            print(f"[INFO] ArSAS ClassLabel encoding verified: {dict(enumerate(hf_names))}")

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
                    f"ArSAS HF dataset has no usable splits. Available: {sorted(available)}. "
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
            raise ValueError(f"Unsupported ArSAS file extension: {ext}. Use .csv/.tsv/.txt")

    # 2. Normalize columns & map labels to 3-class
    def _prepare_part(part):
        # ArSAS uses "tweet" / "Tweet_text" as the text column
        if text_col not in part.column_names:
            for cand in ("tweet", "Tweet_text", "tweets", "text", "review", "sentence", "content"):
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
                f"ArSAS expected columns text/label (or tweet/label). Got: {part.column_names}"
            )

        part = part.map(lambda ex: {label_col: _normalize_label_to_3class(ex[label_col])})
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
        split_file = os.path.join(split_dir, f"arsas_split_seed_{split_seed}.json")

    os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)

    # 4. Build splits — load persisted indices or generate + save
    if _hf_has_all_three:
        ds_splits = DatasetDict(
            {"train": ds_train, "validation": ds_val, "test": ds_test}
        )
        print(f"[INFO] ArSAS official train size: {len(ds_splits['train'])}")
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                _payload = json.load(f)
        else:
            _payload = {"split_mode": "all_official", "split_seed": split_seed}
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(_payload, f, ensure_ascii=False, indent=2)
            print(f"Saved ArSAS split metadata to: {split_file}")

    elif _hf_has_train_test:
        labels = ds_train_prep[label_col]
        if os.path.isfile(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                _payload = json.load(f)
            train_idx = _payload["train_indices"]
            val_idx   = _payload["validation_indices"]
            print(f"Loaded ArSAS train/val split indices from: {split_file}")
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
            print(f"Saved ArSAS train/val split indices to: {split_file}")

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
            print(f"Loaded ArSAS split indices from: {split_file}")
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
            print(f"Saved ArSAS split indices to: {split_file}")

        ds_splits = DatasetDict(
            {
                "train":      ds_all.select(train_idx),
                "validation": ds_all.select(val_idx),
                "test":       ds_all.select(test_idx),
            }
        )

    # 5. Cap training size
    ds_splits, _payload = apply_train_cap(ds_splits, split_seed, split_file, _payload, "ArSAS")

    debug_label_distribution(ds_splits, "ArSAS", ARSAS_ID_MAP)

    # 6. Tokenize
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    ds_tok = ds_splits.map(make_tokenize_fn(tokenizer, text_col, max_length), batched=True, desc="Tokenising ArSAS")
    ds_tok = ds_tok.rename_column(label_col, "labels").with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return ds_tok