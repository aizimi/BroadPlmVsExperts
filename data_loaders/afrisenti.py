from __future__ import annotations
import os
import json
from typing import Dict, Optional

import numpy as np
import datasets as hf_datasets
from datasets import DatasetDict, load_dataset

from data_loaders.common import NEG, NEU, POS, debug_label_distribution, debug_tokenized_dataset, apply_train_cap, make_tokenize_fn

# Maps AfriSenti textual labels to the shared 3-class scheme (NEG=0, NEU=1, POS=2).
AFRISENTI_LABEL_MAP = {
    "negative": NEG,
    "neutral": NEU,
    "positive": POS,
}


def _load_afrisenti_tokenized_dataset(
    lang_code: str,
    tok_name: str,
    max_length: int,
    label_map: Optional[Dict[str, int]] = None,
    split_seed: int = 42,
    split_file: Optional[str] = None,
) -> DatasetDict:
    ds = load_dataset("masakhane/afrisenti", lang_code)

    available_splits = set(ds.keys())
    _val_key = "validation" if "validation" in available_splits else ("dev" if "dev" in available_splits else None)
    if _val_key is None or "train" not in available_splits or "test" not in available_splits:
        raise ValueError(f"AfriSenti({lang_code}) missing required splits. Available: {sorted(available_splits)}")
    if _val_key != "validation":
        ds = hf_datasets.DatasetDict({
            "train":      ds["train"],
            "validation": ds[_val_key],
            "test":       ds["test"],
        })

    if split_file is None:
        split_file = f"afrisenti_{lang_code}_splits.json"

    # Ensure parent directory for split metadata exists (e.g., data/splits/)
    split_dir = os.path.dirname(split_file)
    if split_dir and not os.path.isdir(split_dir):
        os.makedirs(split_dir, exist_ok=True)

    if not os.path.isfile(split_file):
        meta = {
            "source": "masakhane/afrisenti",
            "subset": lang_code,
            "note": "Official HF splits recorded for reproducibility.",
            "splits": {},
        }
        for sp in ("train", "validation", "test"):
            part = ds[sp]
            meta["splits"][sp] = {"n": int(len(part)), "fingerprint": getattr(part, "_fingerprint", None)}
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Saved AfriSenti split metadata to: {split_file}")
    else:
        print(f"Loaded AfriSenti split metadata from: {split_file}")

    if label_map is None:
        label_map = AFRISENTI_LABEL_MAP

    normalized_label_map = {str(k).strip().lower(): int(v) for k, v in label_map.items()}

    def _map_label(x):
        s = str(x.get("label", "")).strip().lower()
        return {"label": normalized_label_map.get(s, -1)}

    ds_mapped = DatasetDict()
    for split in ("train", "validation", "test"):
        part = ds[split]

        # Canonicalize raw text column to `text`
        if "tweet" in part.column_names and "text" not in part.column_names:
            part = part.rename_column("tweet", "text")

        if "text" not in part.column_names:
            for c in ("sentence", "content"):
                if c in part.column_names:
                    part = part.rename_column(c, "text")
                    break

        if "label" not in part.column_names:
            for c in ("sentiment", "polarity", "class"):
                if c in part.column_names:
                    part = part.rename_column(c, "label")
                    break

        if "text" not in part.column_names or "label" not in part.column_names:
            raise ValueError(
                f"AfriSenti({lang_code}) expected columns 'text' and 'label'. Found: {part.column_names}"
            )

        part = part.map(_map_label)
        part = part.filter(lambda x: x["label"] in (0, 1, 2))
        ds_mapped[split] = part

    print(f"[INFO] AfriSenti({lang_code}) official train size: {len(ds_mapped['train'])}")

    # Cap training size for consistency across datasets
    _payload = {}
    if os.path.isfile(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            _payload = json.load(f)
    ds_mapped, _payload = apply_train_cap(ds_mapped, split_seed, split_file, _payload, f"AfriSenti({lang_code})")

    debug_label_distribution(ds_mapped, f"AfriSenti({lang_code})", label_map)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    ds_tok = ds_mapped.map(make_tokenize_fn(tokenizer, "text", max_length), batched=True, desc=f"Tokenising AfriSenti({lang_code})")
    ds_tok = ds_tok.rename_column("label", "labels").with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    debug_tokenized_dataset(ds_tok, f"AfriSenti({lang_code})")
    return ds_tok


def load_afrisenti_ary_tokenized_dataset(
    tok_name: str,
    max_length: int,
    label_map: Optional[Dict[str, int]] = None,
    split_seed: int = 42,
    split_file: Optional[str] = None,
) -> DatasetDict:
    """Load the Moroccan Darija (ary) subset of AfriSenti."""
    return _load_afrisenti_tokenized_dataset(
        "ary",
        tok_name=tok_name,
        max_length=max_length,
        label_map=label_map,
        split_seed=split_seed,
        split_file=split_file,
    )


def load_afrisenti_arq_tokenized_dataset(
    tok_name: str,
    max_length: int,
    label_map: Optional[Dict[str, int]] = None,
    split_seed: int = 42,
    split_file: Optional[str] = None,
) -> DatasetDict:
    """Load the Algerian Arabic (arq) subset of AfriSenti."""
    return _load_afrisenti_tokenized_dataset(
        "arq",
        tok_name=tok_name,
        max_length=max_length,
        label_map=label_map,
        split_seed=split_seed,
        split_file=split_file,
    )