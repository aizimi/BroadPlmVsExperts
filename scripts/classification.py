from __future__ import annotations

import os
from collections import Counter
from typing import Iterable, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from data_loaders import DATASET_REGISTRY

# Select best available device (CUDA, Apple MPS, or CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Using device: {DEVICE}")

# Country/region-level columns derived from CAMeL DIDModel26 city labels.
# This keeps Moroccan, Tunisian, Algerian, and Libyan distinct instead of collapsing
# them all into one broad Maghrebi bucket.
DIALECT_COLUMNS = ["EGY", "LEV", "GLF", "MAGHREB", "MSA"]

MAX_SAMPLE_PER_DATASET = None  # set to None to analyze the full dataset
SAMPLE_SEED = 42
BATCH_SIZE = 128

# MARBERT dialect classifier (regional dialect prediction)
MARBERT_DIALECT_MODEL = "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"


def load_marbert_classifier():
    tokenizer = AutoTokenizer.from_pretrained(MARBERT_DIALECT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MARBERT_DIALECT_MODEL)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model




def _flatten_hf_dataset(ds_obj) -> Dataset:
    """Flatten HF dataset outputs into a single Dataset deterministically."""
    if isinstance(ds_obj, Dataset):
        return ds_obj

    if isinstance(ds_obj, DatasetDict):
        return concatenate_datasets([ds_obj[k] for k in sorted(ds_obj.keys())])

    if isinstance(ds_obj, dict):
        parts = []
        for k in sorted(ds_obj.keys()):
            v = ds_obj[k]
            if isinstance(v, Dataset):
                parts.append(v)
            elif isinstance(v, DatasetDict):
                parts.extend(v[s] for s in sorted(v.keys()))
            elif isinstance(v, dict):
                nested = _flatten_hf_dataset(v)
                parts.append(nested)
        if not parts:
            raise ValueError("No dataset parts found while flattening HF dataset object.")
        return concatenate_datasets(parts)

    raise TypeError(f"Unsupported dataset object type: {type(ds_obj)}")


def _pick_text_column(column_names: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        if cand in column_names:
            return cand
    return None


def _sample_texts(texts: list[str], max_n: int = MAX_SAMPLE_PER_DATASET, seed: int = SAMPLE_SEED) -> list[str]:
    """Deterministically sample up to `max_n` texts."""
    if max_n is None or len(texts) <= max_n:
        return texts
    import random
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    chosen = rng.sample(idx, max_n)
    chosen.sort()
    return [texts[i] for i in chosen]


def _coarse_row_from_fine(row: dict) -> dict:
    """Collapse fine-grained percentages into Dialect vs MSA view."""
    non_msa_cols = [c for c in DIALECT_COLUMNS if c != "MSA"]
    dialect_pct = sum(float(row.get(c, 0.0)) for c in non_msa_cols)
    msa_pct = float(row.get("MSA", 0.0))

    dominant = max(
        [(d, float(row.get(d, 0.0))) for d in non_msa_cols],
        key=lambda x: x[1],
        default=("None", 0.0),
    )[0]

    return {
        "Dataset": row["Dataset"],
        "Dialect": round(dialect_pct, 2),
        "MSA": round(msa_pct, 2),
        "DominantDialect": dominant,
        "n_sampled": row.get("n_sampled", row.get("n_texts", 0)),
        "n_classified": row.get("n_classified", 0),
        "n_skipped": row.get("n_skipped", 0),
    }


def _load_texts_from_registry(dataset_id: str, cfg: dict) -> list[str]:
    source_type = cfg["source_type"]
    if source_type == "csv":
        path = cfg.get("default_csv_path", "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local dataset file not found: {path}")
        ds = load_dataset("csv", data_files=path)["train"]
    elif source_type == "hf":
        hf_config = cfg.get("hf_config")
        ds_obj = load_dataset(cfg["hf_dataset_id"], hf_config) if hf_config else load_dataset(cfg["hf_dataset_id"])
        ds = _flatten_hf_dataset(ds_obj)
    else:
        raise ValueError(f"Unsupported source_type '{source_type}' for dataset '{dataset_id}'")

    text_col = _pick_text_column(ds.column_names, cfg["text_columns"])
    if text_col is None:
        raise ValueError(
            f"Could not find a text column for {cfg['display_name']}. Available: {ds.column_names}"
        )

    texts = []
    for t in ds[text_col]:
        if t is None:
            continue
        s = str(t).strip()
        if s:
            texts.append(s)
    return texts


def _classify_dataset_marbert(texts: list[str], tokenizer, model) -> dict:
    predictions = []
    skipped = 0

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        try:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=1).tolist()

            id2label = model.config.id2label

            for p in preds:
                label = id2label[p]
                if label in DIALECT_COLUMNS:
                    predictions.append(label)
                else:
                    skipped += 1

        except Exception:
            skipped += len(batch)

    counts = Counter(predictions)
    total = sum(counts.values())

    row = {dialect: 0.0 for dialect in DIALECT_COLUMNS}
    row["n_texts"] = len(texts)
    row["n_classified"] = total
    row["n_skipped"] = skipped

    if total == 0:
        return row

    for dialect in DIALECT_COLUMNS:
        row[dialect] = round(100.0 * counts.get(dialect, 0) / total, 2)

    return row


def main() -> None:
    try:
        tokenizer, model = load_marbert_classifier()
    except Exception as e:
        print("\n[ERROR] MARBERT dialect classifier could not be loaded.")
        print("Ensure transformers and torch are installed:")
        print("    pip install transformers torch")
        print("\nOriginal error:")
        print(e)
        return
    rows = []

    for dataset_id, cfg in DATASET_REGISTRY.items():
        dataset_name = cfg["display_name"]
        print(f"\n[INFO] Loading {dataset_name}...")
        try:
            texts = _load_texts_from_registry(dataset_id, cfg)
            texts = _sample_texts(texts)
            stats = _classify_dataset_marbert(texts, tokenizer, model)
            row = {"Dataset": dataset_name, "n_sampled": len(texts), **stats}
            rows.append(row)
            print(
                f"[OK] {dataset_name}: {stats['n_classified']} classified / "
                f"{stats['n_texts']} texts (skipped={stats['n_skipped']})"
            )
        except Exception as e:
            print(f"[SKIP] {dataset_name}: {e}")

    if not rows:
        print("No dataset could be loaded/classified.")
        return

    df_fine = pd.DataFrame(rows)
    fine_cols = ["Dataset", *DIALECT_COLUMNS, "n_sampled", "n_texts", "n_classified", "n_skipped"]
    df_fine = df_fine[fine_cols]

    coarse_rows = [_coarse_row_from_fine(r) for r in rows]
    df_coarse = pd.DataFrame(coarse_rows)
    coarse_cols = ["Dataset", "Dialect", "MSA", "DominantDialect", "n_sampled", "n_classified", "n_skipped"]
    df_coarse = df_coarse[coarse_cols]

    print("\n=== MARBERT Dialect Distribution Table ===")
    print(df_fine[["Dataset", *DIALECT_COLUMNS]].to_markdown(index=False))

    print("\n=== MARBERT Dialect vs MSA Table ===")
    print(df_coarse[["Dataset", "Dialect", "MSA", "DominantDialect"]].to_markdown(index=False))

    out_dir = os.path.join("outputs", "classification")
    os.makedirs(out_dir, exist_ok=True)

    fine_csv = os.path.join(out_dir, "dialect_distribution_fine.csv")
    coarse_csv = os.path.join(out_dir, "dialect_distribution_coarse.csv")
    df_fine.to_csv(fine_csv, index=False, encoding="utf-8")
    df_coarse.to_csv(coarse_csv, index=False, encoding="utf-8")
    print(f"\nSaved CSV: {fine_csv}")
    print(f"Saved CSV: {coarse_csv}")


if __name__ == "__main__":
    main()
