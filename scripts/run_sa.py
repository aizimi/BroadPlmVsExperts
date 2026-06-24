# run_sa.py
# Modular sentiment training runner.
#
# Architecture:
#   data_loaders/
#     mac.py, astd.py, labr.py, afrisenti.py
#   training/
#     trainer.py, metrics.py
#   run_sa.py
#
# Responsibilities of this file:
# - Parse CLI
# - Resolve model alias
# - Call dataset loader from data_loaders/*
# - Call trainer builder from training/*
# - Train or load checkpoint
# - Evaluate on test + save artifacts
#
# Statistical tests (McNemar etc.) MUST be done in aggregate_results.py, not here.

from __future__ import annotations

import ssl as _ssl

# aiohttp calls ssl.create_default_context() at module import time. On Windows,
# this loads every cert in the system store; a single malformed cert raises
# ssl.SSLError and aborts the import. This patch skips bad certs and is a
# no-op on Linux/macOS where _load_windows_store_certs is never called.
def _safe_ldc(self, purpose=_ssl.Purpose.SERVER_AUTH,
              _orig=_ssl.SSLContext.load_default_certs):
    try:
        _orig(self, purpose)
    except _ssl.SSLError:
        pass
_ssl.SSLContext.load_default_certs = _safe_ldc
del _safe_ldc

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
import json
import argparse
import hashlib
import logging
import warnings
from collections import Counter
import platform
from datetime import datetime
from typing import Dict, Optional, Tuple

# Suppress noisy third-party warnings before any heavy imports.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer

from data_loaders import (
    load_mac_tokenized_dataset,
    load_astd_tokenized_dataset,
    load_labr_tokenized_dataset,
    load_afrisenti_ary_tokenized_dataset,
    load_afrisenti_arq_tokenized_dataset,
    load_arsas_tokenized_dataset,
    load_hard_tokenized_dataset,
    DATASET_REGISTRY,
    normalize_dataset_id,
)
from training import build_trainer, train_or_load_best

# ---- OPTIONAL DEPENDENCY CHECK -----------------------
try:
    import sklearn  # noqa: F401
except ImportError:
    raise SystemExit(
        "❗ scikit-learn is required but not installed.\n"
        "Run:  pip install scikit-learn"
    )

# -----------------------------
# Model registry (aliases)
# -----------------------------
MODEL_REGISTRY = {
    "marbert": "UBC-NLP/MARBERTv2",
    "darijabert": "SI2M-Lab/DarijaBERT",
    "arabert": "aubmindlab/bert-base-arabertv2",
    "egybert": "faisalq/EgyBERT",
    "dziribert": "alger-ia/dziribert",
}


def resolve_alias(name_or_alias: str) -> str:
    return MODEL_REGISTRY.get(name_or_alias, name_or_alias)


# -----------------------------
# Config defaults
# -----------------------------
DEFAULT_MODEL = "marbert"
DEFAULT_DATASET = "MACcorpus"

DEFAULT_CSV_PATH = "data/MACcorpus.csv"  # used only for MAC by default
DEFAULT_MAX_LENGTH = None  # resolved per dataset from DATASET_REGISTRY unless overridden by --max-length
DEFAULT_NUM_LABELS = 3

DEFAULT_SEED = 42  # training seed
DEFAULT_SPLIT_SEED = 42  # split seed

# -----------------------------
# Label mapping (stable 3-class)
# -----------------------------
LABEL_MAP: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
LABEL_NAMES = ["negative", "neutral", "positive"]


# -----------------------------
# Utility helpers
# -----------------------------
def get_device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "unknown"


def sha256_of_texts(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        s = "" if t is None else str(t)
        h.update(s.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()



def label_distribution(ds_split) -> Dict[str, int]:
    """Return label distribution for a split (expects `labels` or `label`)."""
    if "labels" in ds_split.column_names:
        y = list(ds_split["labels"])
    elif "label" in ds_split.column_names:
        y = list(ds_split["label"])
    else:
        return {}
    c = Counter(int(v) for v in y)
    return {str(k): int(v) for k, v in sorted(c.items())}


# -----------------------------
# Class weights helper
# -----------------------------
def compute_class_weights(ds_split, num_labels: int = DEFAULT_NUM_LABELS) -> np.ndarray:
    """Compute inverse-frequency class weights from the training split only."""
    if "labels" in ds_split.column_names:
        y = np.asarray(ds_split["labels"], dtype=int)
    elif "label" in ds_split.column_names:
        y = np.asarray(ds_split["label"], dtype=int)
    else:
        raise ValueError("Split does not contain 'labels' or 'label' column for class-weight computation.")

    counts = np.bincount(y, minlength=num_labels).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Cannot compute class weights: missing class in training split. Counts={counts.tolist()}")

    weights = len(y) / (num_labels * counts)
    return weights.astype(np.float32)




# -----------------------------
# Core run
# -----------------------------
def run(
        *,
        model_arg: str,
        train_seed: int,
        split_seed: int,
        dataset: str,
        csv_path: Optional[str] = None,
        out_dir: Optional[str] = None,
        max_length: Optional[int] = DEFAULT_MAX_LENGTH,
        debug: bool = False,
        class_weighted_loss: bool = False,
) -> Tuple[AutoModelForSequenceClassification, Trainer, object]:
    resolved_name = resolve_alias(model_arg)

    dataset_id = normalize_dataset_id(dataset)
    dataset_cfg = DATASET_REGISTRY[dataset_id]
    effective_max_length = max_length if max_length is not None else int(dataset_cfg["default_max_length"])

    out_dir = out_dir or f"checkpoints/{model_arg}_{dataset_id}_split_{split_seed}/seed_{train_seed}"
    ckpt_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)

    # -------- Dataset dispatch --------
    if dataset_id == "maccorpus":
        # MACcorpus: use csv_path if provided, else default to DEFAULT_CSV_PATH
        mac_path = csv_path or DEFAULT_CSV_PATH
        ds_tok = load_mac_tokenized_dataset(
            tok_name=resolved_name,
            max_length=effective_max_length,
            split_seed=split_seed,
            label_map=LABEL_MAP,
            csv_path=mac_path,
            split_file=str(_PROJECT_ROOT / "data" / "splits" / f"mac_split_seed_{split_seed}.json"),
        )

    elif dataset_id == "astd":
        # ASTD loads from Hugging Face by default; `csv_path` is an optional local override.
        ds_tok = load_astd_tokenized_dataset(
            tok_name=resolved_name,
            max_length=effective_max_length,
            split_seed=split_seed,
            data_path=csv_path,
            split_file=str(_PROJECT_ROOT / "data" / "splits" / f"astd_split_seed_{split_seed}.json"),
        )

    elif dataset_id == "arsas":
        ds_tok = load_arsas_tokenized_dataset(
            tok_name=resolved_name,
            max_length=effective_max_length,
            split_seed=split_seed,
            split_file=str(_PROJECT_ROOT / "data" / "splits" / f"arsas_split_seed_{split_seed}.json"),
        )

    elif dataset_id == "hard":
        ds_tok = load_hard_tokenized_dataset(
            tok_name=resolved_name,
            max_length=effective_max_length,
            split_seed=split_seed,
            split_file=str(_PROJECT_ROOT / "data" / "splits" / f"hard_split_seed_{split_seed}.json"),
        )

    elif dataset_id == "labr":
        # LABR loads from Hugging Face by default; `csv_path` is an optional local override.
        ds_tok = load_labr_tokenized_dataset(
            data_path=csv_path,
            tok_name=resolved_name,
            max_length=effective_max_length,
            split_seed=split_seed,
            split_file=str(_PROJECT_ROOT / "data" / "splits" / f"labr_split_seed_{split_seed}.json"),
        )

    elif dataset_id == "afrisenti_ary":
        ds_tok = load_afrisenti_ary_tokenized_dataset(
            tok_name=resolved_name,
            max_length=effective_max_length,
            label_map=LABEL_MAP,
            split_file=str(_PROJECT_ROOT / "data" / "splits" / f"afrisenti_ary_split_seed_{split_seed}.json"),
        )

    elif dataset_id == "afrisenti_arq":
        ds_tok = load_afrisenti_arq_tokenized_dataset(
            tok_name=resolved_name,
            max_length=effective_max_length,
            label_map=LABEL_MAP,
            split_file=str(_PROJECT_ROOT / "data" / "splits" / f"afrisenti_arq_split_seed_{split_seed}.json"),
        )

    else:
        raise ValueError(f"Unsupported dataset after normalization: {dataset_id}")

    if debug:
        print(
            f"[DEBUG] Dataset='{dataset_id}' sizes: "
            f"train={len(ds_tok['train'])} val={len(ds_tok['validation'])} test={len(ds_tok['test'])}"
        )
        for i in range(min(3, len(ds_tok["train"]))):
            row = ds_tok["train"][i]
            print({"labels": int(row["labels"]), "input_ids_len": len(row["input_ids"])})
        # -----------------------------
        # Save minimal debug config (even in debug mode)
        # -----------------------------
        debug_config_path = os.path.join(out_dir, "run_config_debug.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(debug_config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "mode": "debug",
                    "model_arg": model_arg,
                    "resolved_model": resolved_name,
                    "dataset": dataset_id,
                    "split_seed": split_seed,
                    "train_seed": train_seed,
                    "max_length": effective_max_length,
                    "num_labels": DEFAULT_NUM_LABELS,
                    "label_names": LABEL_NAMES,
                    "sizes": {
                        "train": len(ds_tok["train"]),
                        "validation": len(ds_tok["validation"]),
                        "test": len(ds_tok["test"]),
                    },
                    "label_distribution": {
                        "train": label_distribution(ds_tok["train"]),
                        "validation": label_distribution(ds_tok["validation"]),
                        "test": label_distribution(ds_tok["test"]),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[DEBUG] Saved debug config: {debug_config_path}")
        raise SystemExit("[DEBUG] Exiting as requested (--debug).")

    # -------- Class-weighted loss --------
    class_weights = None
    if class_weighted_loss:
        import torch
        class_weights_np = compute_class_weights(ds_tok["train"], num_labels=DEFAULT_NUM_LABELS)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
        print(f"[INFO] Using class-weighted loss with weights: {class_weights_np.tolist()}")

    # -------- Trainer --------
    trainer, _ = build_trainer(
        ds_tok=ds_tok,
        base_name=resolved_name,
        output_dir=out_dir,
        ckpt_dir=ckpt_dir,
        num_labels=DEFAULT_NUM_LABELS,
        seed=train_seed,
        tok_name=resolved_name,
        class_weights=class_weights,
    )

    model = train_or_load_best(trainer, ckpt_dir=ckpt_dir)
    return model, trainer, ds_tok


def main():
    import torch
    import transformers
    import datasets as hf_datasets
    import evaluate
    import sklearn

    parser = argparse.ArgumentParser(description="Sentiment runner (modular data_loaders/ + training/)")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model alias (marbert, darijabert, arabert, egybert, dziribert) or full HF model id",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset: MACcorpus (or mac), astd, arsas, hard, labr, afrisenti_ary, afrisenti_arq",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Training seed (varied across runs)")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED, help="Split seed (fixed split indices)")
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum tokenizer sequence length. If omitted, uses the dataset default from DATASET_REGISTRY.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Optional local dataset path override. "
            "MACcorpus defaults to data/MACcorpus.csv if omitted. "
            "ASTD, ArSAS, HARD, and LABR load from Hugging Face by default; use --csv only if you want a local file. "
            "Ignored for AfriSenti (HF official splits)."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output/checkpoint directory. Default: checkpoints/<model>_<dataset>_split_<split_seed>/seed_<seed>",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug dataset loading: print sizes + a few rows and exit (no training).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If the run directory already exists, allow loading an existing checkpoint instead of failing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If the run directory already exists, allow overwriting evaluation artifacts.",
    )


    # Add CLI flag for class-weighted loss
    parser.add_argument(
        "--class-weighted-loss",
        action="store_true",
        help="Use inverse-frequency class-weighted cross-entropy computed from the training split.",
    )

    args = parser.parse_args()

    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())
    print("Device:", "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    normalized_dataset = normalize_dataset_id(args.dataset)
    _loss_suffix = "_weighted" if args.class_weighted_loss else ""
    effective_out = args.out or f"checkpoints/{args.model}_{normalized_dataset}{_loss_suffix}_split_{args.split_seed}/seed_{args.seed}"
    sentinel = os.path.join(effective_out, "run_config.json")
    if os.path.exists(effective_out) and os.path.exists(sentinel) and not args.resume and not args.overwrite:
        raise RuntimeError(
            f"Run directory already exists: {effective_out}\n"
            "Refusing to overwrite. Use --resume or --overwrite explicitly."
        )

    _, trainer, ds_tok = run(
        model_arg=args.model,
        train_seed=args.seed,
        split_seed=args.split_seed,
        dataset=args.dataset,
        csv_path=args.csv,
        out_dir=effective_out,
        max_length=args.max_length,
        debug=args.debug,
        class_weighted_loss=args.class_weighted_loss,
    )

    # Use the trainer's output_dir to avoid recomputing paths inconsistently
    out_dir = trainer.args.output_dir
    os.makedirs(out_dir, exist_ok=True)


    # -----------------------------
    # Evaluate on TEST set + save artifacts
    # -----------------------------
    test_metrics = trainer.evaluate(ds_tok["test"], metric_key_prefix="test")
    print(test_metrics)
    if "test_f1" in test_metrics:
        print(
            f"Test: f1={test_metrics['test_f1']:.6f}, "
            f"accuracy={test_metrics.get('test_accuracy', float('nan')):.6f}"
        )

    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    pred_out = trainer.predict(ds_tok["test"], metric_key_prefix="test")
    y_true = pred_out.label_ids
    y_pred = np.argmax(pred_out.predictions, axis=1)

    present = sorted(set(int(x) for x in np.unique(y_true)))
    if present != [0, 1, 2]:
        raise RuntimeError(
            f"Test split does not contain all 3 classes. Present={present}. "
            "This is not acceptable for 3-class evaluation; regenerate splits."
        )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\nClassification report (test):")
    print(report)


    report_path = os.path.join(out_dir, "classification_report_test.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 1) Metrics JSON
    metrics_path = os.path.join(out_dir, "metrics_test.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # 2) Predictions NPZ (y_true, y_pred, logits, correct_mask)
    preds_path = os.path.join(out_dir, "predictions_test.npz")
    logits_path = os.path.join(out_dir, "logits_test.npy")
    correct_mask = (y_true == y_pred)
    correct_mask = np.asarray(correct_mask).ravel().astype(bool)

    logits = pred_out.predictions

    np.savez_compressed(
        preds_path,
        y_true=y_true,
        y_pred=y_pred,
        logits=logits,
        correct_mask=correct_mask,
    )
    np.save(logits_path, logits)

    # 2b) Predictions JSONL (readable; useful for error analysis & McNemar inputs)
    preds_jsonl_path = os.path.join(out_dir, "preds_test.jsonl")

    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    texts = list(ds_tok["test"].with_format(None)["text"]) if "text" in ds_tok["test"].column_names else [None] * len(y_true)
    with open(preds_jsonl_path, "w", encoding="utf-8") as f:
        for i, (t, yt, yp, pr) in enumerate(zip(texts, y_true, y_pred, probs)):
            rec = {
                "i": int(i),
                "text_sha256": hashlib.sha256(("" if t is None else str(t)).encode("utf-8")).hexdigest(),
                "gold": int(yt),
                "pred": int(yp),
                "prob_neg": float(pr[0]),
                "prob_neu": float(pr[1]),
                "prob_pos": float(pr[2]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3) Run config JSON (for reproducibility + aggregate_results.py)
    config_path = os.path.join(out_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_arg": args.model,
                "resolved_model": resolve_alias(args.model),
                "dataset": normalized_dataset,
                "dataset_source": (
                    "local" if normalized_dataset == "maccorpus" else ("local" if args.csv else "hf")
                ),
                "csv": (
                    (args.csv or DEFAULT_CSV_PATH)
                    if normalized_dataset == "maccorpus"
                    else (args.csv if normalized_dataset in ("astd", "labr") else None)
                ),
                "split_seed": args.split_seed,
                "train_seed": args.seed,
                "max_length": (args.max_length if args.max_length is not None else DATASET_REGISTRY[normalized_dataset]["default_max_length"]),
                "num_labels": DEFAULT_NUM_LABELS,
                "label_names": LABEL_NAMES,
                "class_weighted_loss": bool(args.class_weighted_loss),
                "training": {
                    "learning_rate": float(trainer.args.learning_rate),
                    "per_device_train_batch_size": int(trainer.args.per_device_train_batch_size),
                    "per_device_eval_batch_size": int(trainer.args.per_device_eval_batch_size),
                    "num_train_epochs": float(trainer.args.num_train_epochs),
                    "evaluation_strategy": str(getattr(trainer.args, "evaluation_strategy", "")),
                    "save_strategy": str(getattr(trainer.args, "save_strategy", "")),
                    "logging_steps": int(getattr(trainer.args, "logging_steps", 0) or 0),
                    "dataloader_num_workers": int(getattr(trainer.args, "dataloader_num_workers", 0) or 0),
                    "seed": int(trainer.args.seed),
                    "data_seed": int(getattr(trainer.args, "data_seed", trainer.args.seed)),
                    "load_best_model_at_end": bool(trainer.args.load_best_model_at_end),
                    "metric_for_best_model": str(trainer.args.metric_for_best_model),
                    "greater_is_better": bool(trainer.args.greater_is_better),
                    "class_weights": (compute_class_weights(ds_tok["train"]).tolist() if args.class_weighted_loss else None),
                },
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "device": get_device_name(),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "mps_available": bool(torch.backends.mps.is_available()),
                    "torch_cuda_version": getattr(torch.version, "cuda", None),
                    "cudnn_version": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
                    "cuda_device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                },
                "libs": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "torch": getattr(torch, "__version__", None),
                    "transformers": getattr(transformers, "__version__", None),
                    "datasets": getattr(hf_datasets, "__version__", None),
                    "evaluate": getattr(evaluate, "__version__", None),
                    "sklearn": getattr(sklearn, "__version__", None),
                },
                "dataset_fingerprint": {
                    "text_column": "text" if "text" in ds_tok["test"].column_names else None,
                    "splits": {
                        "train": {
                            "size": int(len(ds_tok["train"])),
                            "text_sha256": (
                                sha256_of_texts(list(ds_tok["train"].with_format(None)["text"])) if "text" in ds_tok["train"].column_names else None
                            ),
                            "label_dist": label_distribution(ds_tok["train"]),
                            "hf_fingerprint": getattr(ds_tok["train"], "_fingerprint", None),
                        },
                        "validation": {
                            "size": int(len(ds_tok["validation"])),
                            "text_sha256": (
                                sha256_of_texts(list(ds_tok["validation"].with_format(None)["text"])) if "text" in ds_tok["validation"].column_names else None
                            ),
                            "label_dist": label_distribution(ds_tok["validation"]),
                            "hf_fingerprint": getattr(ds_tok["validation"], "_fingerprint", None),
                        },
                        "test": {
                            "size": int(len(ds_tok["test"])),
                            "text_sha256": (
                                sha256_of_texts(list(ds_tok["test"].with_format(None)["text"])) if "text" in ds_tok["test"].column_names else None
                            ),
                            "label_dist": label_distribution(ds_tok["test"]),
                            "hf_fingerprint": getattr(ds_tok["test"], "_fingerprint", None),
                        },
                    },
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 4) Confusion matrices
    # Raw counts
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title("Confusion Matrix (test, raw counts)")
    fig.tight_layout()

    cm_path = os.path.join(out_dir, "confusion_matrix_test.png")
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    # Row-normalized (per-class recall)
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True)
    ax.set_title("Confusion Matrix (test, normalized by true class)")
    fig.tight_layout()

    cm_norm_path = os.path.join(out_dir, "confusion_matrix_test_normalized.png")
    fig.savefig(cm_norm_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {logits_path}")
    print(f"Saved: {preds_jsonl_path}")
    print(f"Saved: {config_path}")
    print(f"Saved: {cm_path}")
    print(f"Saved: {cm_norm_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
