#!/usr/bin/env python3
"""
scripts/report_dataset_counts.py

Extracts dataset sizes and class distributions for paper Table 1.

Paper harmonization rules applied here:
  ASTD        : Objective + Neutral -> Neutral  (4-class -> 3-class)
  ArSAS       : Mixed excluded  (not merged into Neutral)
  MACcorpus   : Mixed excluded  (not merged into Neutral)
  AfriSenti   : unchanged  (negative/neutral/positive only)
  LABR / HARD : 1-2 -> Negative, 3 -> Neutral, 4-5 -> Positive

For datasets whose saved split files were created before Mixed was excluded
(MACcorpus and ArSAS), the script regenerates an in-memory 80/10/10 stratified
split using the same split seed.  These are REPORTING-ONLY splits — no split
files are written or overwritten.

For all other datasets the saved split files are used directly.

The output clearly states which split source was used for each dataset.

Run from any directory:
    python scripts/report_dataset_counts.py [--split-seed 42]

Outputs:
    - Formatted table to stdout
    - outputs/dataset_counts.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

# Project root = parent of scripts/.  All default paths are anchored here so
# the script works regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def _proj(*parts: str) -> str:
    """Absolute path anchored at the project root."""
    return os.path.join(_PROJECT_ROOT, *parts)


from datasets import load_dataset

from data_loaders.astd import _normalize_label_to_3class as _astd_norm
from data_loaders.hard import _rating_to_3class as _hard_norm
from data_loaders.labr import _rating_to_3class as _labr_norm
from data_loaders.mac import MAC_LABEL_MAP
from data_loaders.afrisenti import AFRISENTI_LABEL_MAP
from data_loaders.common import NEG, NEU, POS, stratified_split_indices

VALID = {NEG, NEU, POS}

# ── Paper-specific ArSAS normalizer (Mixed excluded, not merged) ──────────────
# The shipped _arsas_norm maps label 3 (Mixed) -> NEU; the paper excludes it.
_ARSAS_PAPER_INT_MAP = {0: NEG, 1: NEU, 2: POS}  # 3 = Mixed -> excluded


def _arsas_paper_norm(raw) -> int:
    """ArSAS label -> {0,1,2,-1}.  Mixed (label 3) returns -1 (excluded)."""
    if raw is None:
        return -1
    if isinstance(raw, (int, np.integer)):
        return _ARSAS_PAPER_INT_MAP.get(int(raw), -1)
    if isinstance(raw, float) and not np.isnan(raw):
        return _arsas_paper_norm(int(raw))
    s = str(raw).strip().lower()
    if s in {"negative", "neg", "subjective negative", "subj_negative"}:
        return NEG
    if s in {"positive", "pos", "subjective positive", "subj_positive"}:
        return POS
    if s in {"neutral", "neu", "objective", "obj"}:
        return NEU
    # "mixed" / "subjective mixed" / anything else -> excluded
    return -1


# ── Utilities ─────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _split_path(split_dir: str, key: str, seed: int) -> str:
    return os.path.join(split_dir, f"{key}_split_seed_{seed}.json")


def _cnt(part) -> Counter:
    return Counter(int(v) for v in part["label"])


def _split_compatible(ds, payload: dict) -> bool:
    """True iff saved split indices exactly cover the filtered dataset size."""
    if "train_indices" not in payload:
        return False
    saved_total = (
        len(payload.get("train_indices", []))
        + len(payload.get("validation_indices", []))
        + len(payload.get("test_indices", []))
    )
    return len(ds) == saved_total


def _reporting_split(ds, seed: int):
    """In-memory 80/10/10 stratified split.  Not saved to any file."""
    train_idx, val_idx, test_idx = stratified_split_indices(ds["label"], seed)
    return ds.select(train_idx), ds.select(val_idx), ds.select(test_idx)


# ── Per-dataset loaders (no tokenization) ────────────────────────────────────
# Every function returns a dict:
#   display_name  str
#   split_source  "saved_experimental" | "reporting_only"
#   train         Counter
#   val           Counter
#   test          Counter
#   is_hard       bool (HARD only)
#   all_dist      Counter (HARD only – usable before cap)
#   train_before  Counter (HARD only)
#   train_after   Counter (HARD only)


def _get_astd(split_dir: str, seed: int) -> dict:
    """ASTD: Objective + Neutral -> Neutral.  Uses saved split (compatible)."""
    payload = _load_json(_split_path(split_dir, "astd", seed))
    ds_hf   = load_dataset("arbml/ASTD")
    # split_mode = custom_80_10_10: only 'train' was available on HF
    ds = ds_hf["train"]
    ds = ds.map(lambda x: {"label": _astd_norm(x["label"])})
    ds = ds.filter(lambda x: x["label"] in VALID)
    # ASTD has no Mixed class in its ClassLabel encoding; saved split is valid.
    return dict(
        display_name="ASTD",
        split_source="saved_experimental",
        train=_cnt(ds.select(payload["train_indices"])),
        val=_cnt(ds.select(payload["validation_indices"])),
        test=_cnt(ds.select(payload["test_indices"])),
    )


def _get_arsas(split_dir: str, seed: int) -> dict:
    """ArSAS: Mixed excluded per paper methodology."""
    payload = _load_json(_split_path(split_dir, "arsas", seed))
    ds_hf   = load_dataset("arbml/ArSAS")
    ds = ds_hf["train"]
    ds = ds.map(lambda x: {"label": _arsas_paper_norm(x["label"])})
    ds = ds.filter(lambda x: x["label"] in VALID)

    if _split_compatible(ds, payload):
        # Mixed was already absent when the split was created -> indices valid.
        source = "saved_experimental"
        train = _cnt(ds.select(payload["train_indices"]))
        val   = _cnt(ds.select(payload["validation_indices"]))
        test  = _cnt(ds.select(payload["test_indices"]))
    else:
        # Saved split was created with Mixed merged into Neutral; incompatible
        # after exclusion -> regenerate 80/10/10 for reporting only.
        source = "reporting_only"
        tr, vl, te = _reporting_split(ds, seed)
        train, val, test = _cnt(tr), _cnt(vl), _cnt(te)

    return dict(
        display_name="ArSAS",
        split_source=source,
        train=train,
        val=val,
        test=test,
    )


def _get_mac(split_dir: str, seed: int) -> dict:
    """MACcorpus: Mixed excluded per paper methodology."""
    payload = _load_json(_split_path(split_dir, "mac", seed))
    ds = load_dataset("csv", data_files=_proj("data", "MACcorpus.csv"))["train"]

    if "tweets" in ds.column_names and "text" not in ds.column_names:
        ds = ds.rename_column("tweets", "text")
    elif "tweet" in ds.column_names and "text" not in ds.column_names:
        ds = ds.rename_column("tweet", "text")
    if "type" in ds.column_names:
        ds = ds.rename_column("type", "label")

    # MAC_LABEL_MAP covers only negative/neutral/positive; 'mixed' -> -1 naturally.
    norm = {k.lower(): v for k, v in MAC_LABEL_MAP.items()}
    ds = ds.map(lambda x: {"label": norm.get(str(x["label"]).strip().lower(), -1)})
    ds = ds.filter(lambda x: x["label"] in VALID)
    # Filtered size is 17,444; saved split covers 18,087 (mixed was included).
    # Always regenerate reporting-only split for MAC.
    if _split_compatible(ds, payload):
        source = "saved_experimental"
        train = _cnt(ds.select(payload["train_indices"]))
        val   = _cnt(ds.select(payload["validation_indices"]))
        test  = _cnt(ds.select(payload["test_indices"]))
    else:
        source = "reporting_only"
        tr, vl, te = _reporting_split(ds, seed)
        train, val, test = _cnt(tr), _cnt(vl), _cnt(te)

    return dict(
        display_name="MACcorpus",
        split_source=source,
        train=train,
        val=val,
        test=test,
    )


def _get_labr(split_dir: str, seed: int) -> dict:
    """LABR: 1-2->NEG, 3->NEU, 4-5->POS.  Uses saved split (no Mixed)."""
    payload = _load_json(_split_path(split_dir, "labr", seed))
    ds_hf   = load_dataset("mohamedadaly/labr")
    # split_mode = train_test_official

    def _prep(part):
        if "text" not in part.column_names:
            for c in ("review", "sentence", "content", "tweet", "tweets"):
                if c in part.column_names:
                    part = part.rename_column(c, "text")
                    break
        if "rating" not in part.column_names:
            for c in ("label", "stars", "score", "class"):
                if c in part.column_names:
                    part = part.rename_column(c, "rating")
                    break
        part = part.map(lambda x: {"label": _labr_norm(x["rating"])})
        part = part.filter(lambda x: x["label"] in VALID)
        return part

    train_prep = _prep(ds_hf["train"])
    test_prep  = _prep(ds_hf["test"])

    return dict(
        display_name="LABR",
        split_source="saved_experimental",
        train=_cnt(train_prep.select(payload["train_indices"])),
        val=_cnt(train_prep.select(payload["validation_indices"])),
        test=_cnt(test_prep),
    )


def _get_hard(split_dir: str, seed: int) -> dict:
    """HARD: 1-2->NEG, 3->NEU, 4-5->POS.  Usable = before cap; cap reported separately."""
    payload = _load_json(_split_path(split_dir, "hard", seed))
    ds_hf   = load_dataset("Elnagara/hard", "plain_text")
    # split_mode = custom_80_10_10
    ds_raw = ds_hf["train"]

    if "text" not in ds_raw.column_names:
        for c in ("review", "content", "comments", "comment",
                  "sentence", "body", "tweet", "tweets"):
            if c in ds_raw.column_names:
                ds_raw = ds_raw.rename_column(c, "text")
                break

    rating_col = "rating"
    if rating_col not in ds_raw.column_names:
        for c in ("label", "stars", "score", "class", "polarity"):
            if c in ds_raw.column_names:
                ds_raw = ds_raw.rename_column(c, "rating")
                break

    ds_all = ds_raw.map(lambda x: {"label": _hard_norm(x[rating_col])})
    ds_all = ds_all.filter(lambda x: x["label"] in VALID)

    train_before = ds_all.select(payload["train_indices"])
    val          = ds_all.select(payload["validation_indices"])
    test         = ds_all.select(payload["test_indices"])
    # train_cap_indices index into train_before (same as apply_train_cap)
    train_after  = train_before.select(payload["train_cap_indices"])

    all_labels = list(train_before["label"]) + list(val["label"]) + list(test["label"])

    return dict(
        display_name="HARD",
        split_source="saved_experimental",
        is_hard=True,
        all_dist=Counter(all_labels),
        train_before=_cnt(train_before),
        train_after=_cnt(train_after),
        val=_cnt(val),
        test=_cnt(test),
    )


def _get_afrisenti(lang_code: str, display_name: str) -> dict:
    """AfriSenti: official HF splits, negative/neutral/positive only (no Mixed)."""
    ds   = load_dataset("masakhane/afrisenti", lang_code)
    norm = {k.lower(): v for k, v in AFRISENTI_LABEL_MAP.items()}

    def _prep(part):
        if "tweet" in part.column_names and "text" not in part.column_names:
            part = part.rename_column("tweet", "text")
        if "label" not in part.column_names:
            for c in ("sentiment", "polarity", "class"):
                if c in part.column_names:
                    part = part.rename_column(c, "label")
                    break
        part = part.map(lambda x: {"label": norm.get(str(x["label"]).strip().lower(), -1)})
        part = part.filter(lambda x: x["label"] in VALID)
        return part

    val_key = "validation" if "validation" in ds else "dev"
    return dict(
        display_name=display_name,
        split_source="saved_experimental",
        train=_cnt(_prep(ds["train"])),
        val=_cnt(_prep(ds[val_key])),
        test=_cnt(_prep(ds["test"])),
    )


# ── Formatting ────────────────────────────────────────────────────────────────

_SOURCE_LABEL = {
    "saved_experimental": "saved experimental split",
    "reporting_only":     "reporting-only split (Mixed excluded, 80/10/10 seed={})",
}


def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def _print_info(info: dict, seed: int) -> None:
    name    = info["display_name"]
    source  = info["split_source"]
    is_hard = info.get("is_hard", False)

    if is_hard:
        dist      = info["all_dist"]
        total     = sum(dist.values())
        n_train_b = sum(info["train_before"].values())
        n_train_a = sum(info["train_after"].values())
        n_val     = sum(info["val"].values())
        n_test    = sum(info["test"].values())
    else:
        dist   = info["train"] + info["val"] + info["test"]
        total  = sum(dist.values())
        n_train = sum(info["train"].values())
        n_val   = sum(info["val"].values())
        n_test  = sum(info["test"].values())

    src_str = _SOURCE_LABEL[source].format(seed)

    W = 58
    print(f"\n{'-'*W}")
    print(f"  {name}")
    print(f"  Split source: {src_str}")
    print(f"{'-'*W}")
    if is_hard:
        print(f"  Usable (harmonized, before cap):  {total:>8,}")
    else:
        print(f"  Usable (after harmonization):     {total:>8,}")
    print(f"  Class distribution:")
    for lbl, tag in ((NEG, "Negative"), (NEU, "Neutral "), (POS, "Positive")):
        n = dist[lbl]
        print(f"    {tag}:  {n:>7,}  ({_pct(n, total):5.1f}%)")
    print(f"  Splits:")
    if is_hard:
        print(f"    Train (before cap):  {n_train_b:>8,}")
        print(f"    Train (after cap):   {n_train_a:>8,}")
        print(f"    Validation:          {n_val:>8,}")
        print(f"    Test:                {n_test:>8,}")
    else:
        print(f"    Train:               {n_train:>8,}")
        print(f"    Validation:          {n_val:>8,}")
        print(f"    Test:                {n_test:>8,}")


def _to_csv_row(info: dict) -> dict:
    is_hard = info.get("is_hard", False)

    if is_hard:
        dist        = info["all_dist"]
        total       = sum(dist.values())
        n_train     = sum(info["train_before"].values())
        n_train_cap = sum(info["train_after"].values())
        n_val       = sum(info["val"].values())
        n_test      = sum(info["test"].values())
    else:
        dist        = info["train"] + info["val"] + info["test"]
        total       = sum(dist.values())
        n_train     = sum(info["train"].values())
        n_train_cap = ""
        n_val       = sum(info["val"].values())
        n_test      = sum(info["test"].values())

    return {
        "dataset":        info["display_name"],
        "split_source":   info["split_source"],
        "usable_total":   total,
        "neg_count":      dist[NEG],
        "neg_pct":        f"{_pct(dist[NEG], total):.2f}",
        "neu_count":      dist[NEU],
        "neu_pct":        f"{_pct(dist[NEU], total):.2f}",
        "pos_count":      dist[POS],
        "pos_pct":        f"{_pct(dist[POS], total):.2f}",
        "train_size":     n_train,
        "train_cap_size": n_train_cap,
        "val_size":       n_val,
        "test_size":      n_test,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-seed", type=int, default=42,
                    help="Seed used for splits (default: 42)")
    ap.add_argument("--split-dir", default=_proj("data", "splits"),
                    help="Directory containing saved split JSON files")
    ap.add_argument("--out-csv", default=_proj("outputs", "dataset_counts.csv"),
                    help="Output CSV path")
    args = ap.parse_args()

    seed      = args.split_seed
    split_dir = args.split_dir if os.path.isabs(args.split_dir) else _proj(args.split_dir)
    out_csv   = args.out_csv   if os.path.isabs(args.out_csv)   else _proj(args.out_csv)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f"Paper methodology dataset statistics  (split_seed={seed})")
    print("Mixed labels: excluded from MACcorpus and ArSAS.")
    print("Saved splits used where compatible; reporting-only 80/10/10 otherwise.\n")

    loaders = [
        ("ASTD",          lambda: _get_astd(split_dir, seed)),
        ("ArSAS",         lambda: _get_arsas(split_dir, seed)),
        ("AfriSenti_ARQ", lambda: _get_afrisenti("arq", "AfriSenti_ARQ")),
        ("AfriSenti_ARY", lambda: _get_afrisenti("ary", "AfriSenti_ARY")),
        ("MACcorpus",     lambda: _get_mac(split_dir, seed)),
        ("LABR",          lambda: _get_labr(split_dir, seed)),
        ("HARD",          lambda: _get_hard(split_dir, seed)),
    ]

    all_info = []
    for label, fn in loaders:
        print(f"  Loading {label} ...", end=" ", flush=True)
        info = fn()
        all_info.append(info)
        print("done")

    # ── Print table ──────────────────────────────────────────────────────────
    W = 58
    print(f"\n{'='*W}")
    print(f"  DATASET STATISTICS  (split_seed={seed}, paper methodology)")
    print(f"{'='*W}")
    for info in all_info:
        _print_info(info, seed)
    print(f"\n{'='*W}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    fieldnames = [
        "dataset", "split_source", "usable_total",
        "neg_count", "neg_pct",
        "neu_count", "neu_pct",
        "pos_count", "pos_pct",
        "train_size", "train_cap_size",
        "val_size", "test_size",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for info in all_info:
            writer.writerow(_to_csv_row(info))

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
