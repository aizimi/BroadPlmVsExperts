#!/usr/bin/env python3
"""
report_per_seed_macro_f1.py

Extracts per-seed test Macro-F1 from checkpoints/*/seed_*/metrics_test.json
and produces Appendix B outputs:

  outputs/per_seed_macro_f1.csv          (uniform + weighted rows, loss_type column)
  outputs/per_seed_macro_f1_appendix.md  (Table B1: main experiments;
                                          Table B2: ASTD class-weighted ablation)

Metric-file choice rationale
------------------------------
Each seed directory contains:
  - metrics_test.json          -> machine-readable JSON, key "test_f1" is the
                                   sklearn macro-averaged F1 on the test split,
                                   written directly by the training script.
  - classification_report_test.txt -> human-readable text, same value in the
                                       "macro avg f1-score" row, but requires
                                       regex parsing.
This script reads metrics_test.json (primary) because it is unambiguous and
avoids fragile text parsing.  The existing results/csv/per_seed_runs.csv and
results/csv/per_seed_runs_weighted.csv (produced by aggregate_results.py from
the same JSON files) are used as cross-validation references only.

Weighted ablation identification
----------------------------------
Directories matching *_astd_weighted_split_42 are identified as class-weighted
runs.  Identity is confirmed by the presence of training.class_weights in
run_config.json.  These rows are tagged loss_type=weighted in the CSV and
appear exclusively in Table B2; they are never mixed into Table B1.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RESULTS_CSV_DIR = REPO_ROOT / "results" / "csv"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# -- Paper name mappings -------------------------------------------------------
MODEL_DISPLAY = {
    "marbert":    "MARBERTv2",
    "arabert":    "AraBERTv2",
    "egybert":    "EgyBERT",
    "darijabert": "DarijaBERT",
    "dziribert":  "DziriBERT",
}

DATASET_DISPLAY = {
    "astd":          "ASTD",
    "arsas":         "ArSAS",
    "afrisenti_arq": "AfriSenti_ARQ",
    "afrisenti_ary": "AfriSenti_ARY",
    "maccorpus":     "MACcorpus",
    "labr":          "LABR",
    "hard":          "HARD",
}

DATASET_ORDER = ["ASTD", "ArSAS", "AfriSenti_ARQ", "AfriSenti_ARY",
                 "MACcorpus", "LABR", "HARD"]
MODEL_ORDER   = ["MARBERTv2", "AraBERTv2", "EgyBERT", "DarijaBERT", "DziriBERT"]

SEEDS = [42, 43, 44, 45, 46]


# -- Directory parsers ---------------------------------------------------------

def parse_exp_dir(name: str):
    """
    Parse 'arabert_afrisenti_arq_split_42' -> (model_key, dataset_key).
    Returns None for weighted/ablation runs or unrecognised names.
    """
    if "weighted" in name:
        return None
    if not name.endswith("_split_42"):
        return None
    stem = name[: -len("_split_42")]
    for model_key in MODEL_DISPLAY:
        if stem.startswith(model_key + "_"):
            dataset_key = stem[len(model_key) + 1:]
            if dataset_key in DATASET_DISPLAY:
                return model_key, dataset_key
    return None


def parse_weighted_exp_dir(name: str):
    """
    Parse 'arabert_astd_weighted_split_42' -> model_key.
    Returns None for anything else.
    """
    suffix = "_astd_weighted_split_42"
    if not name.endswith(suffix):
        return None
    model_key = name[: -len(suffix)]
    return model_key if model_key in MODEL_DISPLAY else None


# -- Extraction ----------------------------------------------------------------

def _read_metrics_test(seed_dir: Path, exp_dir_name: str, seed: int,
                       records: list, source_files: list, ambiguities: list,
                       model: str, dataset: str):
    """Shared extraction logic for one seed directory."""
    metrics_fp = seed_dir / "metrics_test.json"

    if not seed_dir.exists():
        print(f"  MISSING seed dir : {exp_dir_name}/seed_{seed}")
        return
    if not metrics_fp.exists():
        print(f"  MISSING metrics  : {metrics_fp}")
        return

    other_json = [
        f for f in seed_dir.glob("*.json")
        if f != metrics_fp
        and any(kw in f.name.lower() for kw in ("metric", "eval", "result"))
    ]
    if other_json:
        print(f"  NOTE  : {exp_dir_name}/seed_{seed} — additional JSON files "
              f"present: {[f.name for f in other_json]}. Using metrics_test.json "
              f"(contains unambiguous 'test_f1' key).")

    with open(metrics_fp, encoding="utf-8") as fh:
        data = json.load(fh)

    if "test_f1" not in data:
        ambiguities.append(
            f"Key 'test_f1' absent in {metrics_fp}. "
            f"Present keys: {list(data.keys())}"
        )
        return

    records.append({
        "dataset":     dataset,
        "model":       model,
        "seed":        seed,
        "macro_f1":    data["test_f1"],
        "source_file": str(metrics_fp),
    })
    source_files.append(str(metrics_fp))


def extract_records():
    """Extract test Macro-F1 for all main (uniform-loss) experiments."""
    records, source_files, ambiguities = [], [], []
    for exp_dir in sorted(CHECKPOINTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        parsed = parse_exp_dir(exp_dir.name)
        if parsed is None:
            continue
        model_key, dataset_key = parsed
        model   = MODEL_DISPLAY[model_key]
        dataset = DATASET_DISPLAY[dataset_key]
        for seed in SEEDS:
            _read_metrics_test(
                exp_dir / f"seed_{seed}", exp_dir.name, seed,
                records, source_files, ambiguities, model, dataset,
            )
    return records, source_files, ambiguities


def extract_weighted_records():
    """
    Extract test Macro-F1 for ASTD class-weighted ablation runs.
    Verifies that run_config.json confirms class weights were used.
    """
    records, source_files, ambiguities = [], [], []
    for exp_dir in sorted(CHECKPOINTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        model_key = parse_weighted_exp_dir(exp_dir.name)
        if model_key is None:
            continue

        # Confirm this is a weighted-loss run via run_config.json
        sample_cfg = exp_dir / "seed_42" / "run_config.json"
        if sample_cfg.exists():
            with open(sample_cfg, encoding="utf-8") as fh:
                cfg = json.load(fh)
            if not cfg.get("training", {}).get("class_weights"):
                ambiguities.append(
                    f"{exp_dir.name}: run_config.json does not confirm "
                    f"class_weights — cannot reliably identify as weighted run."
                )
                continue

        model   = MODEL_DISPLAY[model_key]
        dataset = "ASTD"
        for seed in SEEDS:
            _read_metrics_test(
                exp_dir / f"seed_{seed}", exp_dir.name, seed,
                records, source_files, ambiguities, model, dataset,
            )
    return records, source_files, ambiguities


# -- Pivot / stats -------------------------------------------------------------

def pivot_and_compute(records):
    """Pivot to wide format and compute mean / sample-std over available seeds."""
    df = pd.DataFrame(records)
    pivot = (
        df.pivot_table(
            index=["dataset", "model"],
            columns="seed",
            values="macro_f1",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={s: f"seed_{s}" for s in SEEDS})

    seed_cols = [f"seed_{s}" for s in SEEDS]
    for col in seed_cols:
        if col not in pivot.columns:
            pivot[col] = np.nan

    vals = pivot[seed_cols].values.astype(float)
    n    = (~np.isnan(vals)).sum(axis=1)

    pivot["n_available"]  = n
    pivot["mean"]         = np.nanmean(vals, axis=1)
    pivot["std"]          = np.where(
        n > 1,
        np.nanstd(vals, axis=1, ddof=1),
        np.nan,
    )

    def _missing(row):
        ms = [str(s) for s in SEEDS if np.isnan(row[f"seed_{s}"])]
        return ",".join(ms)

    pivot["missing_seeds"] = pivot.apply(_missing, axis=1)
    return pivot


def sort_paper_order(df):
    ds_rank = {d: i for i, d in enumerate(DATASET_ORDER)}
    m_rank  = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = df.copy()
    df["_ds"] = df["dataset"].map(ds_rank)
    df["_m"]  = df["model"].map(m_rank)
    df = df.sort_values(["_ds", "_m"]).drop(columns=["_ds", "_m"])
    return df.reset_index(drop=True)


# -- Validation ----------------------------------------------------------------

def _validate_pivot_vs_ref(pivot, ref_fp, label):
    if not ref_fp.exists():
        print(f"\nWARNING: {ref_fp} not found — skipping {label} validation.")
        return
    print(f"\n-- Validation vs. {ref_fp.name} "
          + "-" * max(0, 55 - len(ref_fp.name)))
    ref = pd.read_csv(ref_fp)
    ref["model_display"] = ref["model"].map(MODEL_DISPLAY)
    any_mismatch = False
    for _, row in ref.iterrows():
        ds, mdl, f1_ref = row["dataset"], row["model_display"], row["f1_mean"]
        hit = pivot[(pivot["dataset"] == ds) & (pivot["model"] == mdl)]
        if hit.empty:
            print(f"  NOT FOUND in pivot: {ds} / {mdl}")
            continue
        comp = hit["mean"].values[0]
        diff = abs(comp - f1_ref)
        if diff > 0.001:
            any_mismatch = True
            print(f"  MISMATCH: {ds:20s} / {mdl:12s}  ref={f1_ref:.6f}  "
                  f"computed={comp:.6f}  diff={diff:.6f}")
    if not any_mismatch:
        print(f"  All computed means match {ref_fp.name} within 0.001.")


def _cross_validate_vs_seed_csv(records, ref_fp, label):
    if not ref_fp.exists():
        print(f"\nWARNING: {ref_fp} not found — skipping {label} cross-validation.")
        return
    print(f"\n-- Cross-validation vs. {ref_fp.name} "
          + "-" * max(0, 45 - len(ref_fp.name)))
    ref = pd.read_csv(ref_fp)
    ref["model_display"] = ref["model"].map(MODEL_DISPLAY)
    any_mismatch = False
    for rec in records:
        ds, mdl, seed, val = rec["dataset"], rec["model"], rec["seed"], rec["macro_f1"]
        hit = ref[
            (ref["dataset"] == ds)
            & (ref["model_display"] == mdl)
            & (ref["seed"] == seed)
        ]
        if hit.empty:
            print(f"  NOT IN csv: {ds}/{mdl}/seed_{seed}")
            continue
        ref_val = hit["macro_f1"].values[0]
        diff    = abs(val - ref_val)
        if diff > 1e-4:
            any_mismatch = True
            print(f"  MISMATCH: {ds}/{mdl}/seed_{seed}  "
                  f"json={val:.6f}  csv={ref_val:.6f}  diff={diff:.6f}")
    if not any_mismatch:
        print(f"  All values match {ref_fp.name} within 1e-4.")


# -- Output writers ------------------------------------------------------------

def write_csv(pivot_uniform, pivot_weighted, path):
    """
    Write combined CSV.  loss_type column distinguishes uniform (Table B1)
    from weighted (Table B2) rows.
    """
    seed_cols  = [f"seed_{s}" for s in SEEDS]
    float_cols = seed_cols + ["mean", "std"]
    all_cols   = ["dataset", "model", "loss_type"] + seed_cols + \
                 ["mean", "std", "n_available", "missing_seeds"]

    def _prep(df, loss_type):
        out = df.copy()
        out.insert(2, "loss_type", loss_type)
        for col in float_cols:
            out[col] = out[col].apply(
                lambda x: f"{x:.6f}" if pd.notna(x) and x != "" else ""
            )
        return out[all_cols]

    combined = pd.concat(
        [_prep(pivot_uniform, "uniform"), _prep(pivot_weighted, "weighted")],
        ignore_index=True,
    )
    combined.to_csv(path, index=False)


def _md_table(pivot, caption: str, note: str) -> list:
    """Build markdown lines for one results table."""
    def fmt(x):
        try:
            v = float(x)
            return f"{v:.3f}" if not np.isnan(v) else "NA"
        except (TypeError, ValueError):
            return "NA"

    lines = [
        f"## {caption}",
        "",
        "| Dataset | Model | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean | Std |",
        "|---------|-------|--------:|--------:|--------:|--------:|--------:|-----:|----:|",
    ]
    for _, row in pivot.iterrows():
        cells = [
            row["dataset"], row["model"],
            fmt(row["seed_42"]), fmt(row["seed_43"]),
            fmt(row["seed_44"]), fmt(row["seed_45"]),
            fmt(row["seed_46"]),
            fmt(row["mean"]), fmt(row["std"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"*{note}*"]
    return lines


def write_markdown(pivot_uniform, pivot_weighted, path):
    note_main = (
        "Values are test Macro-F1 scores for individual fine-tuning seeds. "
        "Mean and standard deviation are computed over available seeds."
    )
    note_weighted = (
        "Values are test Macro-F1 scores for the class-weighted cross-entropy "
        "ablation on ASTD. Weights were computed from inverse class frequency on "
        "the training split and held constant across seeds. "
        "Mean and standard deviation are computed over available seeds. "
        "NA indicates a seed whose run did not complete."
    )

    lines = _md_table(
        pivot_uniform,
        "Table B1. Per-seed Macro-F1 results across model-dataset combinations.",
        note_main,
    )
    lines += [
        "",
        "---",
        "",
    ]
    lines += _md_table(
        pivot_weighted,
        "Table B2. Per-seed Macro-F1 results for the ASTD class-weighted loss ablation.",
        note_weighted,
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("report_per_seed_macro_f1.py")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TABLE B1 — main (uniform-loss) experiments
    # -------------------------------------------------------------------------
    print("\n[TABLE B1] Extracting uniform-loss runs")
    print("-" * 70)
    records, source_files, ambiguities = extract_records()

    if ambiguities:
        print("\nAMBIGUITY — cannot confidently identify test Macro-F1. Stopping.")
        for a in ambiguities:
            print(f"  {a}")
        sys.exit(1)

    print(f"  {len(records)} records from {len(source_files)} files.")
    print("\n  Source files:")
    for sf in sorted(source_files):
        print(f"    {sf}")

    _cross_validate_vs_seed_csv(
        records,
        RESULTS_CSV_DIR / "per_seed_runs.csv",
        "uniform",
    )

    pivot_uniform = pivot_and_compute(records)
    pivot_uniform = sort_paper_order(pivot_uniform)

    print("\n  Missing combinations (Table B1):")
    missing = pivot_uniform[pivot_uniform["missing_seeds"] != ""]
    if missing.empty:
        print("  None — all 5 seeds present for every combination.")
    else:
        for _, row in missing.iterrows():
            print(f"    {row['dataset']:20s} / {row['model']:12s}  "
                  f"missing seeds: {row['missing_seeds']}")

    _validate_pivot_vs_ref(
        pivot_uniform,
        RESULTS_CSV_DIR / "main_results_table.csv",
        "uniform",
    )

    # -------------------------------------------------------------------------
    # TABLE B2 — ASTD class-weighted ablation
    # -------------------------------------------------------------------------
    print("\n[TABLE B2] Extracting class-weighted ASTD runs")
    print("-" * 70)
    recs_w, src_w, amb_w = extract_weighted_records()

    if amb_w:
        print("\nAMBIGUITY in weighted runs — stopping.")
        for a in amb_w:
            print(f"  {a}")
        sys.exit(1)

    print(f"  {len(recs_w)} records from {len(src_w)} files.")
    print("\n  Source files:")
    for sf in sorted(src_w):
        print(f"    {sf}")

    _cross_validate_vs_seed_csv(
        recs_w,
        RESULTS_CSV_DIR / "per_seed_runs_weighted.csv",
        "weighted",
    )

    pivot_weighted = pivot_and_compute(recs_w)
    pivot_weighted = sort_paper_order(pivot_weighted)

    print("\n  Missing combinations (Table B2):")
    missing_w = pivot_weighted[pivot_weighted["missing_seeds"] != ""]
    if missing_w.empty:
        print("  None.")
    else:
        for _, row in missing_w.iterrows():
            print(f"    {row['dataset']:20s} / {row['model']:12s}  "
                  f"missing seeds: {row['missing_seeds']}")

    _validate_pivot_vs_ref(
        pivot_weighted,
        RESULTS_CSV_DIR / "main_results_table_weighted.csv",
        "weighted",
    )

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    OUTPUTS_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUTS_DIR / "per_seed_macro_f1.csv"
    md_path  = OUTPUTS_DIR / "per_seed_macro_f1_appendix.md"

    write_csv(pivot_uniform, pivot_weighted, csv_path)
    write_markdown(pivot_uniform, pivot_weighted, md_path)

    print(f"\n-- Output files " + "-" * 55)
    print(f"  CSV      : {csv_path}")
    print(f"  Markdown : {md_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
