from __future__ import annotations

import json
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
from datasets import DatasetDict

NEG, NEU, POS = 0, 1, 2
LABEL_NAMES = ["negative", "neutral", "positive"]


def debug_label_distribution(ds: DatasetDict, dataset_name: str, label_map: dict) -> None:
    """Print pre-tokenization label distribution and effective label map."""
    for split_name in ["train", "validation", "test"]:
        if split_name in ds:
            labels = ds[split_name]["label"]
            print(f"[DEBUG] {dataset_name} {split_name} label distribution:", Counter(labels))
    print(f"[DEBUG] {dataset_name} label_map: {label_map}")


def debug_tokenized_dataset(ds: DatasetDict, dataset_name: str) -> None:
    """Print post-tokenization dataset sanity checks."""
    for split_name in ["train", "validation", "test"]:
        if split_name in ds:
            print(f"[DEBUG] {dataset_name} {split_name} size:", len(ds[split_name]))
            sample = ds[split_name][0]
            print(f"[DEBUG] {dataset_name} sample keys:", list(sample.keys()))
            print(f"[DEBUG] {dataset_name} sample label:", sample["labels"])


MAX_TRAIN = 20_000


def stratified_split_indices(labels, split_seed: int) -> Tuple[list, list, list]:
    """Create 80/10/10 train/val/test stratified indices."""
    from sklearn.model_selection import train_test_split

    idx = list(range(len(labels)))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.2, random_state=split_seed, stratify=labels)
    tmp_labels = [labels[i] for i in tmp_idx]
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=split_seed, stratify=tmp_labels)
    return train_idx, val_idx, test_idx


def val_split_indices(labels, split_seed: int) -> Tuple[list, list]:
    """Carve 10% validation from official train when a test set already exists."""
    from sklearn.model_selection import train_test_split

    idx = list(range(len(labels)))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=split_seed, stratify=labels)
    return train_idx, val_idx


def apply_train_cap(
    ds_splits,
    split_seed: int,
    split_file: str,
    payload: dict,
    dataset_name: str = "",
    label_col: str = "label",
) -> Tuple[object, dict]:
    """Cap the training split to MAX_TRAIN examples using stratified sampling."""
    if len(ds_splits["train"]) <= MAX_TRAIN:
        return ds_splits, payload
    original_size = len(ds_splits["train"])
    if "train_cap_indices" in payload:
        cap_idx = payload["train_cap_indices"]
        if len(cap_idx) != MAX_TRAIN:
            raise RuntimeError(
                f"Cached train_cap_indices has {len(cap_idx)} entries but MAX_TRAIN={MAX_TRAIN}. "
                f"Delete {split_file} and re-run to regenerate stratified cap indices."
            )
    else:
        from sklearn.model_selection import train_test_split
        idx = list(range(original_size))
        labels = ds_splits["train"][label_col]
        cap_idx, _ = train_test_split(
            idx,
            train_size=MAX_TRAIN,
            random_state=split_seed,
            stratify=labels,
        )
        payload["train_cap_indices"] = cap_idx
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    ds_splits["train"] = ds_splits["train"].select(cap_idx)
    label = f"{dataset_name} " if dataset_name else ""
    print(f"[INFO] {label}train capped to {MAX_TRAIN} (original: {original_size}; saved to {split_file}).")
    return ds_splits, payload


def make_tokenize_fn(tokenizer, text_col: str, max_length: int):
    """Return a batched tokenize function suitable for DatasetDict.map()."""
    def tokenize(batch):
        texts = [
            "" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t)
            for t in batch.get(text_col, [])
        ]
        return tokenizer(texts, truncation=True, padding=False, max_length=max_length)
    return tokenize


class TokenLengthProfiler:
    """Estimate token-length statistics and truncation impact for one dataset.

    This helper is intended for pre-training analysis so max_length choices can be
    justified empirically and reported in the paper.
    """

    def __init__(self, tokenizer, dataset_name: str, sample_size: int = 2000):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.sample_size = sample_size

    def profile(self, texts: Sequence[str], max_length: int) -> dict:
        sampled_texts = list(texts[: self.sample_size])
        sampled_texts = ["" if t is None else str(t) for t in sampled_texts]

        encodings = self.tokenizer(sampled_texts, truncation=False, padding=False)
        lengths = [len(ids) for ids in encodings["input_ids"]]

        if not lengths:
            return {
                "median": 0.0,
                "p95": 0.0,
                "max": 0,
                "trunc_rate_128": 0.0,
                "trunc_rate_256": 0.0,
                "trunc_rate_max_length": 0.0,
                "sample_size": 0,
            }

        arr = np.array(lengths)
        return {
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": int(np.max(arr)),
            "trunc_rate_128": float(100.0 * np.mean(arr > 128)),
            "trunc_rate_256": float(100.0 * np.mean(arr > 256)),
            "trunc_rate_max_length": float(100.0 * np.mean(arr > max_length)),
            "sample_size": int(len(arr)),
        }

    def print_report(self, texts: Sequence[str], max_length: int) -> None:
        stats = self.profile(texts, max_length=max_length)
        print(f"[DEBUG] {self.dataset_name} token length median: {stats['median']:.1f}")
        print(f"[DEBUG] {self.dataset_name} token length p95: {stats['p95']:.1f}")
        print(f"[DEBUG] {self.dataset_name} token length max: {stats['max']}")
        print(f"[DEBUG] {self.dataset_name} truncation rate @128: {stats['trunc_rate_128']:.2f}%")
        print(f"[DEBUG] {self.dataset_name} truncation rate @256: {stats['trunc_rate_256']:.2f}%")
        print(
            f"[DEBUG] {self.dataset_name} truncation rate @{max_length}: "
            f"{stats['trunc_rate_max_length']:.2f}%"
        )


class DatasetLengthAnalyzer:
    """Aggregate token-length statistics across datasets and render article-ready tables."""

    def __init__(self, tokenizer, sample_size: int = 2000):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.rows: List[dict] = []

    def add_dataset(
        self,
        dataset_name: str,
        texts: Sequence[str],
        domain: str,
        chosen_max_length: int,
    ) -> dict:
        profiler = TokenLengthProfiler(
            tokenizer=self.tokenizer,
            dataset_name=dataset_name,
            sample_size=self.sample_size,
        )
        stats = profiler.profile(texts, max_length=chosen_max_length)

        row = {
            "dataset": dataset_name,
            "domain": domain,
            "sample_size": stats["sample_size"],
            "median_tokens": round(stats["median"], 1),
            "p95_tokens": round(stats["p95"], 1),
            "max_tokens": stats["max"],
            "trunc_128_pct": round(stats["trunc_rate_128"], 2),
            "trunc_256_pct": round(stats["trunc_rate_256"], 2),
        }
        self.rows.append(row)
        return row

    def to_markdown(self) -> str:
        if not self.rows:
            return ""

        headers = [
            "Dataset",
            "Domain",
            "n",
            "Median",
            "P95",
            "Max",
            "Trunc@128%",
            "Trunc@256%",
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in self.rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["dataset"]),
                        str(row["domain"]),
                        str(row["sample_size"]),
                        f"{row['median_tokens']:.1f}",
                        f"{row['p95_tokens']:.1f}",
                        str(row["max_tokens"]),
                        f"{row['trunc_128_pct']:.2f}",
                        f"{row['trunc_256_pct']:.2f}",
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    def to_latex(self, caption: str = "Token-length statistics by dataset", label: str = "tab:lengths") -> str:
        if not self.rows:
            return ""

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{l l r r r r r r}",
            "\\hline",
            "Dataset & Domain & n & Median & P95 & Max & Trunc@128 & Trunc@256 \\\\",
            "\\hline",
        ]
        for row in self.rows:
            lines.append(
                f"{row['dataset']} & {row['domain']} & {row['sample_size']} & "
                f"{row['median_tokens']:.1f} & {row['p95_tokens']:.1f} & {row['max_tokens']} & "
                f"{row['trunc_128_pct']:.2f} & {row['trunc_256_pct']:.2f} \\\\",
            )
        lines.extend(
            [
                "\\hline",
                "\\end{tabular}",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\end{table}",
            ]
        )
        return "\n".join(lines)