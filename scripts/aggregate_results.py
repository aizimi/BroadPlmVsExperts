#!/usr/bin/env python3
"""aggregate_results.py

Build the main results table (mean +/- std) from saved per-run artifacts.

Expected per-run files (created by run_sa.py):
  - metrics_test.json
  - run_config.json

Directory layout (default):
  checkpoints/<model>_<dataset>_split_<split_seed>/seed_<train_seed>/

Per-run artifacts (created by run_sa.py):
  - metrics_test.json
  - run_config.json
  - predictions_test.npz (optional)
  - preds_test.jsonl (optional)

Usage:
  python aggregate_results.py --root checkpoints --out results_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
from math import sqrt
from typing import Any, Dict, List, Tuple
from typing import DefaultDict

import numpy as np
from scipy.stats import t



def _load_predictions(path: str):
    """Load y_true and y_pred from predictions_test.npz if available."""
    try:
        data = np.load(path)
        if "y_true" in data and "y_pred" in data:
            return data["y_true"], data["y_pred"]
    except Exception:
        pass
    return None, None


# --- Logits loader helper ---
def _load_logits(path: str):
    """Load logits from logits_test.npy if available."""
    try:
        return np.load(path)
    except Exception:
        return None



def _mcnemar_test(y_true_a, y_pred_a, y_true_b, y_pred_b):
    """Compute McNemar test: exact binomial when (b+c)<25, else continuity-corrected chi-squared."""
    correct_a = y_true_a == y_pred_a
    correct_b = y_true_b == y_pred_b

    b = int(np.sum((correct_a == True) & (correct_b == False)))
    c = int(np.sum((correct_a == False) & (correct_b == True)))

    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0

    # Use exact binomial test for small samples
    if n < 25:
        try:
            from scipy.stats import binomtest
            p = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
        except Exception:
            from math import comb
            k = min(b, c)
            prob = 0.0
            for i in range(0, k + 1):
                prob += comb(n, i) * (0.5 ** n)
            p = float(min(1.0, 2.0 * prob))
        return b, c, 0.0, p

    # Chi-squared approximation (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / n

    try:
        from scipy.stats import chi2 as _chi2
        p = float(_chi2.sf(chi2, df=1))
    except Exception:
        from math import erfc, sqrt as _sqrt
        p = float(erfc(_sqrt(chi2 / 2.0)))

    return b, c, chi2, p


def _bootstrap_f1_diff(y_true, y_pred_a, y_pred_b, labels=(0, 1, 2), n_samples: int = 10000, seed: int = 42):
    """Paired bootstrap test for macro-F1 difference between two systems (two-sided p-value)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def _macro_f1(y_t, y_p):
        f1s = []
        for lab in labels:
            tp = np.sum((y_t == lab) & (y_p == lab))
            fp = np.sum((y_t != lab) & (y_p == lab))
            fn = np.sum((y_t == lab) & (y_p != lab))
            if tp == 0 and (fp == 0 or fn == 0):
                f1s.append(0.0)
                continue
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1s.append(0.0)
            else:
                f1s.append(2 * precision * recall / (precision + recall))
        return float(np.mean(f1s))

    diffs = []
    idx = np.arange(n)

    for _ in range(n_samples):
        sample = rng.choice(idx, size=n, replace=True)
        f1_a = _macro_f1(y_true[sample], y_pred_a[sample])
        f1_b = _macro_f1(y_true[sample], y_pred_b[sample])
        diffs.append(f1_a - f1_b)

    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_one = float(np.mean(diffs <= 0))
    p = float(min(1.0, 2.0 * min(p_one, 1.0 - p_one)))

    return mean_diff, float(ci_low), float(ci_high), p


# ----- Paired t-test helper -----

def _paired_t_test(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired t-test p-value for arrays a and b (same length)."""
    if len(a) < 2:
        return float("nan")
    d = a - b
    n = d.shape[0]
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1))
    if std_d == 0.0:
        return 1.0
    t_stat = mean_d / (std_d / np.sqrt(n))
    try:
        from scipy.stats import t as _t
        p = float(2 * _t.sf(abs(t_stat), df=n - 1))
    except Exception:
        # Normal approximation fallback
        from math import erfc, sqrt
        z = abs(t_stat)
        p = float(erfc(z / sqrt(2.0)))
    return p


# ----- Holm-Bonferroni correction -----
def _holm_bonferroni_adjust(pvals: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    """Holm-Bonferroni adjusted p-values for a family of tests.

    Input: mapping from comparison key -> raw p-value.
    Output: mapping from comparison key -> adjusted p-value.
    """
    valid = [(k, float(v)) for k, v in pvals.items() if isinstance(v, (int, float)) and np.isfinite(v)]
    if not valid:
        return {}

    # Sort ascending by raw p-value
    valid.sort(key=lambda x: x[1])
    m = len(valid)

    adjusted_ordered = []
    running_max = 0.0
    for i, (key, p) in enumerate(valid):
        adj = (m - i) * p
        adj = min(1.0, adj)
        running_max = max(running_max, adj)
        adjusted_ordered.append((key, running_max))

    return dict(adjusted_ordered)


def _holm_bonferroni_adjust_within_dataset(pvals: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    """Holm-Bonferroni correction applied within each dataset's family of comparisons.

    The correction family is defined as all pre-specified model comparisons within
    a single dataset. This is appropriate when significance claims are dataset-specific.
    """
    by_dataset: Dict[str, Dict[Tuple[str, str], float]] = {}
    for (dataset, model), p in pvals.items():
        by_dataset.setdefault(dataset, {})[(dataset, model)] = p

    result: Dict[Tuple[str, str], float] = {}
    for group in by_dataset.values():
        result.update(_holm_bonferroni_adjust(group))
    return result


def _fmt_f1_ci(mean: float, ci95: float) -> str:
    if mean != mean or ci95 != ci95:  # NaN guard
        return "nan"
    return f"{mean:.3f}+/-{ci95:.3f}"


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_dataset_name(csv_path: str | None) -> str:
    if not csv_path:
        return "unknown"
    base = os.path.basename(csv_path)
    name, _ = os.path.splitext(base)
    return name


def _find_run_dirs(root: str) -> List[str]:
    run_dirs: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "metrics_test.json" in filenames and "run_config.json" in filenames:
            run_dirs.append(dirpath)
    return sorted(run_dirs)


def _extract_metric(metrics: Dict[str, Any], preferred_keys: Tuple[str, ...]) -> float | None:
    for k in preferred_keys:
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def aggregate(root: str) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Return mapping: (dataset, model_arg) -> list of runs."""
    runs = _find_run_dirs(root)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for rdir in runs:
        metrics_path = os.path.join(rdir, "metrics_test.json")
        config_path = os.path.join(rdir, "run_config.json")

        try:
            metrics = _load_json(metrics_path)
            config = _load_json(config_path)
        except Exception:
            continue

        # Aggregate ONLY TEST metrics for reviewer-grade reporting.
        f1 = _extract_metric(metrics, ("test_f1", "f1_test"))
        acc = _extract_metric(metrics, ("test_accuracy", "accuracy_test"))

        if f1 is None and acc is None:
            continue

        model_arg = config.get("model_arg") or config.get("resolved_model") or "unknown"
        if isinstance(model_arg, str):
            m = model_arg.strip().lower()
            if m in ("marbert", "ubc-nlp/marbertv2"):
                model_arg = "marbert"
            elif m in ("arabert", "aubmindlab/bert-base-arabertv2"):
                model_arg = "arabert"
            elif m in ("darija", "darijabert", "si2m-lab/darijabert"):
                model_arg = "darijabert"
            elif m in ("egybert", "faisalq/egybert"):
                model_arg = "egybert"
            elif m in ("dziribert", "alger-ia/dziribert"):
                model_arg = "dziribert"
            else:
                model_arg = m
        dataset = (
            config.get("dataset")
            or config.get("dataset_name")
            or _infer_dataset_name(config.get("csv"))
            or "unknown"
        )

        # Normalize dataset names to avoid splitting the same dataset across runs.
        if isinstance(dataset, str):
            d = dataset.strip().lower()
            if d in ("mac", "maccorpus"):
                dataset = "MACcorpus"
            elif d in ("astd",):
                dataset = "ASTD"
            elif d in ("labr",):
                dataset = "LABR"
            elif d in ("arsas", "ar-sas", "arbml/arsas"):
                dataset = "ArSAS"
            elif d in ("afrisenti_ary", "ary", "afrisenti-ary"):
                dataset = "AfriSenti_ARY"
            elif d in ("afrisenti_arq", "arq", "afrisenti-arq"):
                dataset = "AfriSenti_ARQ"
            elif d in ("hard",):
                dataset = "HARD"
            else:
                dataset = d

        grouped[(dataset, model_arg)].append(
            {
                "dir": rdir,
                "f1": f1,
                "acc": acc,
                "train_seed": config.get("train_seed"),
                "split_seed": config.get("split_seed"),
                "dataset_source": config.get("dataset_source"),
                "split_file": config.get("split_file"),
                "weighted": bool(config.get("class_weighted_loss", False)),
            }
        )

    return grouped


def _mean_std_ci95(values: List[float]) -> Tuple[int, float, float, float]:
    n = len(values)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    ci95 = t.ppf(0.975, df=n - 1) * (std / sqrt(n)) if n > 1 else 0.0
    return n, mean, std, ci95


def _run_full_analysis(
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]],
    label: str,
    out_csv: str,
    verbose: bool = False,
) -> None:
    """Run the full aggregation + statistical analysis pipeline on one experiment group."""

    # Ensure output sub-directories exist
    out_dir = os.path.dirname(os.path.abspath(out_csv))
    csv_dir = os.path.join(out_dir, "csv")
    tex_dir = os.path.join(out_dir, "tex")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)

    # Shared display constants — used by both Table 2 (pivot) and Table 3 (significance).
    # Update here only; both tables will stay in sync automatically.
    DATASET_ORDER = ["ASTD", "ArSAS", "AfriSenti_ARQ", "MACcorpus", "AfriSenti_ARY", "LABR", "HARD"]
    MODEL_DISPLAY = {
        "marbert":    "MARBERTv2",
        "arabert":    "AraBERTv2",
        "egybert":    "EgyBERT",
        "darijabert": "DarijaBERT",
        "dziribert":  "DziriBERT",
    }

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")

    if verbose:
        print("\n[VERBOSE] Per-run test_f1 by (dataset, model):")
        for (dataset, model), runs in sorted(grouped.items()):
            for r in sorted(
                runs,
                key=lambda x: (x.get("train_seed") if x.get("train_seed") is not None else 10**9),
            ):
                print(
                    f"- {dataset} | {model} | seed={r.get('train_seed')} | "
                    f"test_f1={r.get('f1')} | test_acc={r.get('acc')} | {r.get('dir')}"
                )
        print("")

    rows: List[Dict[str, Any]] = []
    for (dataset, model), runs in sorted(grouped.items()):
        f1s = [r["f1"] for r in runs if isinstance(r["f1"], (int, float))]
        accs = [r["acc"] for r in runs if isinstance(r["acc"], (int, float))]

        n_f1, f1_mean, f1_std, f1_ci95 = _mean_std_ci95(f1s)
        _n_acc, acc_mean, acc_std, acc_ci95 = _mean_std_ci95(accs)

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "n": int(n_f1),
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "f1_ci95": f1_ci95,
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "acc_ci95": acc_ci95,
            }
        )

    # ----- Collect per-seed F1s aligned by train_seed for paired tests -----
    seed_f1s: Dict[Tuple[str, str], Dict[int, float]] = {}
    for (dataset, model), runs in grouped.items():
        m: Dict[int, float] = {}
        for r in runs:
            f1 = r.get("f1")
            seed = r.get("train_seed")
            if isinstance(f1, (int, float)) and isinstance(seed, int):
                m[seed] = float(f1)
        if m:
            seed_f1s[(dataset, model)] = m

    # ----- Build ensemble predictions from ALL seeds (average logits) -----
    ensemble_preds = {}

    for (dataset, model), runs in grouped.items():
        logits_list = []
        y_true_ref = None

        for r in runs:
            logits_path = os.path.join(r["dir"], "logits_test.npy")
            preds_path = os.path.join(r["dir"], "predictions_test.npz")

            logits = _load_logits(logits_path)
            y_true, _ = _load_predictions(preds_path)

            if logits is None or y_true is None:
                continue

            if y_true_ref is None:
                y_true_ref = y_true
            else:
                if len(y_true_ref) != len(y_true) or not np.array_equal(y_true_ref, y_true):
                    continue

            logits_list.append(logits)

        if logits_list:
            avg_logits = np.mean(np.stack(logits_list), axis=0)
            y_pred = np.argmax(avg_logits, axis=1)
            ensemble_preds[(dataset, model)] = (y_true_ref, y_pred)

    # ----- McNemar statistical comparison (ensemble predictions across seeds) -----
    print(f"=== McNemar Tests (ensemble predictions across seeds) ===")

    mcnemar_pvals: Dict[Tuple[str, str], float] = {}
    bootstrap_pvals: Dict[Tuple[str, str], float] = {}
    # Accumulate all per-comparison stats for persistence
    sig_rows: List[Dict[str, Any]] = []

    datasets = set(k[0] for k in ensemble_preds.keys())

    for dataset in sorted(datasets):
        marbert_key = (dataset, "marbert")
        if marbert_key not in ensemble_preds:
            continue

        y_true_m, y_pred_m = ensemble_preds[marbert_key]

        for (d, model), (y_true_o, y_pred_o) in sorted(ensemble_preds.items()):
            if d != dataset or model == "marbert":
                continue

            if len(y_true_m) != len(y_true_o) or not np.array_equal(y_true_m, y_true_o):
                print(f"\nDataset: {dataset}")
                print(f"MARBERT vs {model}")
                print("McNemar skipped: y_true mismatch.")
                continue

            b, c, chi2, p = _mcnemar_test(y_true_m, y_pred_m, y_true_o, y_pred_o)

            print(f"\nDataset: {dataset}")
            print(f"MARBERT vs {model}")
            print(f"b (MARBERT correct, {model} wrong): {b}")
            print(f"c ({model} correct, MARBERT wrong): {c}")
            print(f"chi2: {chi2:.4f}")
            print(f"p-value: {p:.6f}")

            mcnemar_pvals[(dataset, model)] = float(p)

            # Bootstrap significance on macro-F1 difference
            mean_diff, ci_low, ci_high, p_boot = _bootstrap_f1_diff(y_true_m, y_pred_m, y_pred_o)
            print(f"Bootstrap delta-F1 (MARBERT - {model}): {mean_diff:.4f}")
            print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            print(f"Bootstrap p-value: {p_boot:.6f}")
            bootstrap_pvals[(dataset, model)] = float(p_boot)

            sig_rows.append({
                "dataset": dataset,
                "model_vs_marbert": model,
                "mcnemar_b": b,
                "mcnemar_c": c,
                "mcnemar_chi2": round(chi2, 6),
                "mcnemar_p": round(p, 6),
                "boot_delta_f1": round(mean_diff, 6),
                "boot_ci_low": round(ci_low, 6),
                "boot_ci_high": round(ci_high, 6),
                "boot_p": round(p_boot, 6),
                # Holm-corrected boot_p filled in below after all comparisons are done
                "boot_p_holm": None,
                "ttest_p": None,
            })

    # ----- Paired t-test across seed-level F1s (vs MARBERT) -----
    ttest_pvals: Dict[Tuple[str, str], float] = {}
    for dataset in set(k[0] for k in seed_f1s.keys()):
        marbert_key = (dataset, "marbert")
        if marbert_key not in seed_f1s:
            continue
        seeds_m = seed_f1s[marbert_key]
        for (d, model), seeds_o in seed_f1s.items():
            if d != dataset or model == "marbert":
                continue
            # align on common seeds
            common = sorted(set(seeds_m.keys()) & set(seeds_o.keys()))
            if len(common) < 2:
                continue
            a = np.array([seeds_m[s] for s in common], dtype=float)
            b_arr = np.array([seeds_o[s] for s in common], dtype=float)
            p_t = _paired_t_test(a, b_arr)
            ttest_pvals[(dataset, model)] = float(p_t)

    # ----- Multiple-comparisons correction (Holm-Bonferroni, primary test only) -----
    # Primary test: paired bootstrap on delta-macro-F1 (ensemble predictions).
    # Correction family: all pre-specified model comparisons within each dataset.
    # Secondary (McNemar) and diagnostic (paired t-test) report raw p-values only.
    bootstrap_pvals_holm = _holm_bonferroni_adjust_within_dataset(bootstrap_pvals)

    # Back-fill Holm-corrected boot_p and ttest_p into sig_rows
    for row in sig_rows:
        key = (row["dataset"], row["model_vs_marbert"])
        p_holm = bootstrap_pvals_holm.get(key)
        row["boot_p_holm"] = round(p_holm, 6) if isinstance(p_holm, float) else None
        p_t = ttest_pvals.get(key)
        row["ttest_p"] = round(p_t, 6) if isinstance(p_t, float) else None

    # ----- Paper-style summary table: F1±CI, best model, raw and Holm-corrected p-values vs MARBERT -----
    print("\n=== Main Results Table (Macro-F1 with CI95 and significance vs MARBERT) ===")

    # Determine best model per dataset by highest mean F1
    best_by_dataset: Dict[str, str] = {}
    for r in rows:
        dataset = r["dataset"]
        model = r["model"]
        f1m = r.get("f1_mean")

        if not isinstance(f1m, (int, float)):
            continue

        if dataset not in best_by_dataset:
            best_by_dataset[dataset] = model
        else:
            # compare with current best
            current_best_model = best_by_dataset[dataset]
            current_best_f1 = next(
                (x["f1_mean"] for x in rows if x["dataset"] == dataset and x["model"] == current_best_model),
                None,
            )
            if current_best_f1 is None or float(f1m) > float(current_best_f1):
                best_by_dataset[dataset] = model

    # Build rows sorted by dataset then by descending F1_mean
    results_rows: List[Tuple[str, str, str, str, str, str, str, str]] = []
    for r in sorted(rows, key=lambda x: (x["dataset"], -float(x["f1_mean"]) if isinstance(x["f1_mean"], (int, float)) else 0.0)):
        dataset = str(r["dataset"])
        model = str(r["model"])
        f1_ci = _fmt_f1_ci(float(r["f1_mean"]), float(r["f1_ci95"]))
        best_mark = "*" if best_by_dataset.get(dataset) == model else ""

        # p-values are defined for non-MARBERT models vs MARBERT
        if model == "marbert":
            p_boot_str = "-"
            p_boot_holm_str = "-"
            p_mcnemar_str = "-"
            p_t_str = "-"
        else:
            # PRIMARY: paired bootstrap on delta-macro-F1 (with within-dataset Holm correction)
            p_boot = bootstrap_pvals.get((dataset, model))
            p_boot_str = f"{p_boot:.4g}" if isinstance(p_boot, (int, float)) else "n/a"
            p_boot_holm = bootstrap_pvals_holm.get((dataset, model))
            p_boot_holm_str = f"{p_boot_holm:.4g}" if isinstance(p_boot_holm, (int, float)) else "n/a"

            # SECONDARY: McNemar on ensemble predictions (raw p, no correction)
            p_mcnemar = mcnemar_pvals.get((dataset, model))
            p_mcnemar_str = f"{p_mcnemar:.4g}" if isinstance(p_mcnemar, (int, float)) else "n/a"

            # DIAGNOSTIC: paired t-test across seeds (raw p, not reported in paper)
            p_t = ttest_pvals.get((dataset, model))
            p_t_str = f"{p_t:.4g}" if isinstance(p_t, (int, float)) else "n/a"

        results_rows.append((model, dataset, f1_ci, best_mark, p_boot_str, p_boot_holm_str, p_mcnemar_str, p_t_str))

    # Print as fixed-width text table
    col1 = max([len("Model")] + [len(x[0]) for x in results_rows])
    col2 = max([len("Dataset")] + [len(x[1]) for x in results_rows])
    col3 = max([len("F1 +/- CI95")] + [len(x[2]) for x in results_rows])
    col4 = len("Best")
    col5 = len("Boot p [PRIMARY]")
    col6 = len("Boot p Holm [PRIMARY]")
    col7 = len("McNemar p [secondary]")
    col8 = len("Paired-t p [diagnostic]")

    header_line = (
        f"{'Model'.ljust(col1)}  {'Dataset'.ljust(col2)}  {'F1 +/- CI95'.ljust(col3)}  {'Best'.ljust(col4)}  "
        f"{'Boot p [PRIMARY]'.ljust(col5)}  {'Boot p Holm [PRIMARY]'.ljust(col6)}  "
        f"{'McNemar p [secondary]'.ljust(col7)}  {'Paired-t p [diagnostic]'.ljust(col8)}"
    )
    print(header_line)
    print("-" * len(header_line))
    for model, dataset, f1_ci, best_mark, p_boot_str, p_boot_holm_str, p_mcnemar_str, p_t_str in results_rows:
        print(
            f"{model.ljust(col1)}  {dataset.ljust(col2)}  {f1_ci.ljust(col3)}  {best_mark.ljust(col4)}  "
            f"{p_boot_str.ljust(col5)}  {p_boot_holm_str.ljust(col6)}  "
            f"{p_mcnemar_str.ljust(col7)}  {p_t_str.ljust(col8)}"
        )

    print("\n[INFO] Significance testing design:")
    print("       PRIMARY  — paired bootstrap on delta-macro-F1 (ensemble predictions vs MARBERT),")
    print("                  Holm-Bonferroni corrected within each dataset's comparison family.")
    print("       SECONDARY — McNemar on ensemble predictions; raw p-value, not corrected.")
    print("       DIAGNOSTIC — paired t-test across seeds; raw p-value, not reported in paper.")
    print("       Estimand: performance difference between ensemble systems (logit-averaged across seeds),")
    print("                 not mean performance across stochastic fine-tuning runs.")

    header = [
        "dataset",
        "model",
        "n",
        "f1_mean",
        "f1_std",
        "f1_ci95",
        "acc_mean",
        "acc_std",
        "acc_ci95",
    ]

    # Derive a stem for file naming (e.g. "results_summary" or "results_summary_weighted")
    stem = os.path.splitext(os.path.basename(out_csv))[0]

    # --- Save summary CSV ---
    summary_csv = os.path.join(csv_dir, stem + ".csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(
                f"{r['dataset']},{r['model']},{r['n']},"
                f"{r['f1_mean']:.6f},{r['f1_std']:.6f},{r['f1_ci95']:.6f},"
                f"{r['acc_mean']:.6f},{r['acc_std']:.6f},{r['acc_ci95']:.6f}\n"
            )
    print("Saved CSV:", summary_csv)

    # --- Save significance tests CSV ---
    sig_stem = stem.replace("results_summary", "significance_tests")
    sig_csv = os.path.join(csv_dir, sig_stem + ".csv")
    sig_header = [
        "dataset", "model_vs_marbert",
        "mcnemar_b", "mcnemar_c", "mcnemar_chi2", "mcnemar_p",
        "boot_delta_f1", "boot_ci_low", "boot_ci_high", "boot_p", "boot_p_holm",
        "ttest_p",
    ]
    with open(sig_csv, "w", encoding="utf-8") as f:
        f.write(",".join(sig_header) + "\n")
        for r in sig_rows:
            f.write(
                f"{r['dataset']},{r['model_vs_marbert']},"
                f"{r['mcnemar_b']},{r['mcnemar_c']},{r['mcnemar_chi2']},{r['mcnemar_p']},"
                f"{r['boot_delta_f1']},{r['boot_ci_low']},{r['boot_ci_high']},"
                f"{r['boot_p']},{r['boot_p_holm'] if r['boot_p_holm'] is not None else ''},"
                f"{r['ttest_p'] if r['ttest_p'] is not None else ''}\n"
            )
    print("Saved CSV:", sig_csv)

    # --- Significance summary table (Table 3) ---
    # Stored boot_delta_f1 = MARBERT − model.  We negate for display so that
    # ΔF1 = model − MARBERTv2: negative values mean the model underperforms MARBERTv2.
    # Uses the shared DATASET_ORDER and MODEL_DISPLAY constants defined above so
    # dataset and model names are guaranteed to match Table 2 exactly.

    def _sig_row_order(r: Dict[str, Any]) -> Tuple[int, float]:
        ds = str(r["dataset"])
        idx = DATASET_ORDER.index(ds) if ds in DATASET_ORDER else len(DATASET_ORDER)
        delta = -float(r["boot_delta_f1"]) if isinstance(r["boot_delta_f1"], (int, float)) else 0.0
        return idx, delta  # within dataset: sort by ΔF1 descending (most positive first)

    sig_table_rows = sorted(sig_rows, key=_sig_row_order)

    # Save significance summary CSV
    sig_table_stem = stem.replace("results_summary", "significance_summary_table")
    sig_table_csv = os.path.join(csv_dir, sig_table_stem + ".csv")
    sig_table_header = ["Dataset", "Model", "delta_f1", "ci_low", "ci_high", "delta_f1_formatted", "p_holm", "sig"]
    with open(sig_table_csv, "w", encoding="utf-8") as f:
        f.write(",".join(sig_table_header) + "\n")
        for r in sig_table_rows:
            ds    = str(r["dataset"])
            model = MODEL_DISPLAY.get(str(r["model_vs_marbert"]), str(r["model_vs_marbert"]))
            delta = -float(r["boot_delta_f1"]) if isinstance(r["boot_delta_f1"], (int, float)) else float("nan")
            ci_lo = -float(r["boot_ci_high"])  if isinstance(r["boot_ci_high"], (int, float)) else float("nan")
            ci_hi = -float(r["boot_ci_low"])   if isinstance(r["boot_ci_low"],  (int, float)) else float("nan")
            fmt   = f"{delta:.3f} [{ci_lo:.3f}\u2013{ci_hi:.3f}]"
            p_h   = r["boot_p_holm"]
            p_str = f"{p_h:.4f}" if isinstance(p_h, (int, float)) else ""
            sig   = "\u2713" if isinstance(p_h, (int, float)) and p_h < 0.05 else "\u2717"
            f.write(f"{ds},{model},{delta:.6f},{ci_lo:.6f},{ci_hi:.6f},{fmt},{p_str},{sig}\n")
    print("Saved CSV:", sig_table_csv)

    # Save significance summary LaTeX
    sig_table_tex = os.path.join(tex_dir, sig_table_stem + ".tex")
    with open(sig_table_tex, "w", encoding="utf-8") as f:
        f.write(
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\caption{Statistical comparison against MARBERTv2 using paired bootstrap tests. "
            "$\\Delta$F1 = model $-$ MARBERTv2; negative values indicate the model underperforms MARBERTv2. "
            "95\\% bootstrap confidence intervals are reported. "
            "p-values are Holm-corrected within each dataset. "
            "$\\checkmark$ indicates $p < 0.05$.}\n"
            "\\label{tab:significance}\n"
        )
        f.write("\\begin{tabular}{l l c c c c}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Model & $\\Delta$F1 & 95\\% CI & p (Holm) & Sig. \\\\\n")
        f.write("\\midrule\n")
        for r in sig_table_rows:
            ds    = str(r["dataset"])
            model = MODEL_DISPLAY.get(str(r["model_vs_marbert"]), str(r["model_vs_marbert"]))
            delta = -float(r["boot_delta_f1"]) if isinstance(r["boot_delta_f1"], (int, float)) else float("nan")
            ci_lo = -float(r["boot_ci_high"])  if isinstance(r["boot_ci_high"], (int, float)) else float("nan")
            ci_hi = -float(r["boot_ci_low"])   if isinstance(r["boot_ci_low"],  (int, float)) else float("nan")
            p_h   = r["boot_p_holm"]
            p_str = f"{p_h:.4f}" if isinstance(p_h, (int, float)) else "---"
            sig   = "\u2713" if isinstance(p_h, (int, float)) and p_h < 0.05 else "\u2717"
            f.write(
                f"{ds} & {model} & {delta:.3f} & "
                f"[{ci_lo:.3f}, {ci_hi:.3f}] & {p_str} & {sig} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print("Saved LaTeX:", sig_table_tex)

    # --- Save main results table (Macro-F1, CI95, significance vs MARBERT) ---
    paper_stem = stem.replace("results_summary", "main_results_table")
    paper_csv = os.path.join(csv_dir, paper_stem + ".csv")
    paper_csv_header = [
        "dataset", "model", "f1_mean", "f1_ci95", "f1_formatted",
        "is_best", "boot_p", "boot_p_holm", "mcnemar_p", "ttest_p",
    ]
    with open(paper_csv, "w", encoding="utf-8") as f:
        f.write(",".join(paper_csv_header) + "\n")
        for model, dataset, f1_ci, best_mark, p_boot_str, p_boot_holm_str, p_mcnemar_str, p_t_str in results_rows:
            row_lookup = next((r for r in rows if r["dataset"] == dataset and r["model"] == model), None)
            f1_mean_val = f"{row_lookup['f1_mean']:.6f}" if row_lookup else ""
            f1_ci95_val = f"{row_lookup['f1_ci95']:.6f}" if row_lookup else ""
            f.write(
                f"{dataset},{model},{f1_mean_val},{f1_ci95_val},{f1_ci},"
                f"{'1' if best_mark == '*' else '0'},"
                f"{p_boot_str},{p_boot_holm_str},{p_mcnemar_str},{p_t_str}\n"
            )
    print("Saved CSV:", paper_csv)

    # --- Save per-seed macro F1 diagnostic table ---
    per_seed_rows: List[Dict[str, Any]] = []
    for (dataset, model), runs in sorted(grouped.items()):
        for r in sorted(runs, key=lambda x: (x.get("train_seed") if x.get("train_seed") is not None else 10**9)):
            per_seed_rows.append({
                "dataset": dataset,
                "model": model,
                "seed": r.get("train_seed"),
                "macro_f1": r["f1"],
                "accuracy": r["acc"],
            })

    per_seed_stem = stem.replace("results_summary", "per_seed_runs")
    per_seed_csv = os.path.join(csv_dir, per_seed_stem + ".csv")
    with open(per_seed_csv, "w", encoding="utf-8") as f:
        f.write("dataset,model,seed,macro_f1,accuracy\n")
        for r in per_seed_rows:
            f1_str = f"{r['macro_f1']:.6f}" if isinstance(r["macro_f1"], float) else ""
            acc_str = f"{r['accuracy']:.6f}" if isinstance(r["accuracy"], float) else ""
            f.write(f"{r['dataset']},{r['model']},{r['seed']},{f1_str},{acc_str}\n")
    print("Saved CSV:", per_seed_csv)

    # --- Pivoted wide table (Table 2): datasets × models, cell = mean [low–high] ---
    _MODEL_ORDER = ["marbert", "arabert", "egybert", "darijabert", "dziribert"]

    # Index rows by (dataset, model) for fast lookup
    row_index: Dict[Tuple[str, str], Dict[str, Any]] = {
        (r["dataset"], r["model"]): r for r in rows
    }

    # Dataset order: full canonical list first (absent datasets get all-— rows),
    # then any extra datasets found in the runs, alphabetically.
    present_datasets = {r["dataset"] for r in rows}
    all_datasets = list(DATASET_ORDER) + \
                   sorted(present_datasets - set(DATASET_ORDER))

    # Model order: canonical first (only those present), then any extras
    present_models = {r["model"] for r in rows}
    col_models = [m for m in _MODEL_ORDER if m in present_models] + \
                 sorted(present_models - set(_MODEL_ORDER))

    display_cols = [MODEL_DISPLAY.get(m, m) for m in col_models]

    def _fmt_cell(r: Dict[str, Any]) -> str:
        mean = float(r["f1_mean"])
        ci   = float(r["f1_ci95"])
        return f"{mean:.3f} [{mean - ci:.3f}\u2013{mean + ci:.3f}]"

    # Best model per dataset (highest f1_mean among present models)
    best_model: Dict[str, str] = {}
    for ds in all_datasets:
        candidates = [(m, float(row_index[(ds, m)]["f1_mean"])) for m in col_models if (ds, m) in row_index]
        if candidates:
            best_model[ds] = max(candidates, key=lambda x: x[1])[0]

    # Build pivot (plain strings); track best separately
    pivot: List[Dict[str, str]] = []
    for ds in all_datasets:
        row_dict: Dict[str, str] = {"Dataset": ds}
        for m in col_models:
            dc = MODEL_DISPLAY.get(m, m)
            key = (ds, m)
            row_dict[dc] = _fmt_cell(row_index[key]) if key in row_index else "\u2014"
        pivot.append(row_dict)

    # Save pivoted CSV — best cell prefixed with * for visibility
    pivot_stem = stem.replace("results_summary", "pivot_table")
    pivot_csv = os.path.join(csv_dir, pivot_stem + ".csv")
    with open(pivot_csv, "w", encoding="utf-8") as f:
        f.write(",".join(["Dataset"] + display_cols) + "\n")
        for row_dict in pivot:
            ds = row_dict["Dataset"]
            best_dc = MODEL_DISPLAY.get(best_model.get(ds, ""), "")
            cells = []
            for c in ["Dataset"] + display_cols:
                val = row_dict.get(c, "\u2014")
                cells.append(f"*{val}" if c == best_dc else val)
            f.write(",".join(cells) + "\n")
    print("Saved CSV:", pivot_csv)

    # Save LaTeX snippet — best cell wrapped in \textbf{}, en-dash as --
    # NOTE: Table 2 reports seed-level variability (t-CI across seeds).
    #       Significance tests (Table 3) operate on ensemble predictions.
    #       Both are correct but measure different things; caption must say so.
    pivot_tex = os.path.join(tex_dir, pivot_stem + ".tex")
    with open(pivot_tex, "w", encoding="utf-8") as f:
        # lcccc... — centred model columns, consistent with journal house style
        tex_cols = "l" + "c" * len(display_cols)
        f.write(
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\caption{Macro-F1 across datasets. Values are mean performance across "
            "fine-tuning runs with 95\\% confidence intervals computed using a "
            "t-distribution over seeds. The best-performing model for each dataset is shown in bold.}\n"
            "\\label{tab:main_results}\n"
        )
        f.write(f"\\begin{{tabular}}{{{tex_cols}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(["Dataset"] + display_cols) + " \\\\\n")
        f.write("\\midrule\n")
        for row_dict in pivot:
            ds = row_dict["Dataset"]
            best_dc = MODEL_DISPLAY.get(best_model.get(ds, ""), "")
            cells = [ds]
            for dc in display_cols:
                val = row_dict.get(dc, "\u2014")
                # en-dash → --, em-dash (absent cell) → \textemdash{}
                val_tex = val.replace("\u2013", "--").replace("\u2014", "\\textemdash{}")
                cells.append(f"\\textbf{{{val_tex}}}" if dc == best_dc else val_tex)
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print("Saved LaTeX:", pivot_tex)

    # Print pivoted table to console (* = best per row)
    print("\n=== Pivoted Results Table (Macro-F1 mean [95% CI over seeds, t-dist], * = best per dataset) ===")
    col_w = [max(len("Dataset"), max((len(r["Dataset"]) for r in pivot), default=0))]
    for dc in display_cols:
        # +1 for the * marker on the best cell
        col_w.append(max(len(dc), max((len(r.get(dc, "\u2014")) + 1 for r in pivot), default=0)))
    header_parts = ["Dataset"] + display_cols
    header_line = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(header_parts))
    print(header_line)
    print("-" * len(header_line))
    for row_dict in pivot:
        ds = row_dict["Dataset"]
        best_dc = MODEL_DISPLAY.get(best_model.get(ds, ""), "")
        cells = [ds]
        for dc in display_cols:
            val = row_dict.get(dc, "\u2014")
            cells.append(f"*{val}" if dc == best_dc else val)
        print("  ".join(cells[i].ljust(col_w[i]) for i in range(len(cells))))

    if rows:
        rows_sorted = sorted(rows, key=lambda x: (x["dataset"], -x["f1_mean"]))
        print("dataset\tmodel\tn\tf1_mean\tf1_std\tf1_ci95\tacc_mean\tacc_std\tacc_ci95")
        for r in rows_sorted:
            print(
                f"{r['dataset']}\t{r['model']}\t{r['n']}\t"
                f"{r['f1_mean']:.4f}\t{r['f1_std']:.4f}\t{r['f1_ci95']:.4f}\t"
                f"{r['acc_mean']:.4f}\t{r['acc_std']:.4f}\t{r['acc_ci95']:.4f}"
            )
    else:
        print(f"No runs found for group: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate test results (mean +/- std)")
    parser.add_argument("--root", type=str, default=str(_PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--out", type=str, default=str(_PROJECT_ROOT / "results" / "results_summary.csv"))
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run test metrics for debugging",
    )
    args = parser.parse_args()

    grouped = aggregate(args.root)

    # Split baseline vs weighted runs based on class_weighted_loss flag in run_config.json
    grouped_baseline: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_weighted: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for key, runs in grouped.items():
        for r in runs:
            if r.get("weighted"):
                grouped_weighted[key].append(r)
            else:
                grouped_baseline[key].append(r)

    out_stem, out_ext = os.path.splitext(args.out)
    out_weighted = f"{out_stem}_weighted{out_ext}"

    _run_full_analysis(grouped_baseline, "BASELINE RUNS", args.out, verbose=args.verbose)

    if grouped_weighted:
        _run_full_analysis(grouped_weighted, "WEIGHTED-LOSS ABLATION RUNS", out_weighted, verbose=args.verbose)
    else:
        print("\n[INFO] No weighted-loss runs found. Skipping ablation section.")


if __name__ == "__main__":
    main()