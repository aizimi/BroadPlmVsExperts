#!/usr/bin/env python3
"""
scripts/report_secondary_diagnostics.py

Generates Appendix C: Secondary Statistical Diagnostics.
Reads existing outputs only — does not recompute model training or evaluation.

Sources:
  results/csv/significance_tests.csv   — McNemar b, c, chi2, p; stored ttest_p for validation
  results/csv/per_seed_runs.csv        — per-seed Macro-F1 for t-test computation

Outputs:
  outputs/appendix_c_secondary_diagnostics.md
  outputs/mcnemar_diagnostics.csv
  outputs/seed_ttest_diagnostics.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
_ROOT        = Path(__file__).resolve().parent.parent
SIG_CSV      = str(_ROOT / "results/csv/significance_tests.csv")
PER_SEED_CSV = str(_ROOT / "results/csv/per_seed_runs.csv")
OUT_MD       = str(_ROOT / "outputs/appendix_c_secondary_diagnostics.md")
OUT_MCNEMAR  = str(_ROOT / "outputs/mcnemar_diagnostics.csv")
OUT_TTEST    = str(_ROOT / "outputs/seed_ttest_diagnostics.csv")

# ---------------------------------------------------------------------------
# Canonical paper ordering
# ---------------------------------------------------------------------------
DATASET_ORDER    = ["ASTD", "ArSAS", "AfriSenti_ARQ", "AfriSenti_ARY",
                    "MACcorpus", "LABR", "HARD"]
COMPARATOR_ORDER = ["arabert", "egybert", "darijabert", "dziribert"]
DISPLAY = {
    "arabert":    "AraBERTv2",
    "egybert":    "EgyBERT",
    "darijabert": "DarijaBERT",
    "dziribert":  "DziriBERT",
}

SEP = "=" * 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_p(p, precision=4):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.{precision}f}"


def mcnemar_interp(b, c, p):
    if p >= 0.05:
        return "no reliable correctness-disagreement asymmetry; secondary diagnostic only"
    if b > c:
        return "MARBERTv2 has more correct-only predictions; uncorrected p < 0.05; secondary diagnostic only"
    return "Comparator has more correct-only predictions; uncorrected p < 0.05; secondary diagnostic only"


def ttest_interp(mean_delta, p):
    if p < 0.05:
        direction = "comparator lower than MARBERTv2" if mean_delta < 0 else "comparator higher than MARBERTv2"
        return f"{direction}; uncorrected p < 0.05; diagnostic only"
    return "no reliable run-level difference; uncorrected p >= 0.05; diagnostic only"


# ---------------------------------------------------------------------------
# Load source files
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("SOURCE FILES")
print(SEP)

for path in [SIG_CSV, PER_SEED_CSV]:
    if not os.path.exists(path):
        sys.exit(f"\nERROR: required file not found: {path}\n"
                 "Run from the repository root directory.")
    print(f"  Found: {path}")

sig_df  = pd.read_csv(SIG_CSV)
seed_df = pd.read_csv(PER_SEED_CSV)

print(f"\n  {SIG_CSV}: {len(sig_df)} rows, columns: {list(sig_df.columns)}")
print(f"  {PER_SEED_CSV}: {len(seed_df)} rows, columns: {list(seed_df.columns)}")

# ---------------------------------------------------------------------------
# C.1 — McNemar
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("C.1: McNemar Tests")
print(SEP)

mcnemar_rows = []
for ds in DATASET_ORDER:
    for comp in COMPARATOR_ORDER:
        sub = sig_df[(sig_df["dataset"] == ds) & (sig_df["model_vs_marbert"] == comp)]
        if sub.empty:
            continue
        if len(sub) > 1:
            sys.exit(f"CONFLICT: {len(sub)} rows for ({ds}, {comp}) in {SIG_CSV}. "
                     "Stopping — resolve conflict before proceeding.")
        r = sub.iloc[0]
        b    = int(r["mcnemar_b"])
        c    = int(r["mcnemar_c"])
        chi2 = float(r["mcnemar_chi2"])
        p    = float(r["mcnemar_p"])
        mcnemar_rows.append({
            "Dataset":        ds,
            "Comparator":     DISPLAY[comp],
            "b":              b,
            "c":              c,
            "chi2":           chi2,
            "p_value":        p,
            "interpretation": mcnemar_interp(b, c, p),
            "_key":           comp,
        })
        print(f"  {ds:20s}  {DISPLAY[comp]:12s}  b={b:4d}  c={c:4d}  "
              f"chi2={chi2:.4f}  p={fmt_p(p)}")

print(f"\n  Total McNemar rows: {len(mcnemar_rows)}")
print(f"  Source: {SIG_CSV}")

# Validation: coverage must equal evaluated pairs
expected_pairs = set(zip(sig_df["dataset"], sig_df["model_vs_marbert"]))
got_pairs      = {(r["Dataset"], r["_key"]) for r in mcnemar_rows}
if got_pairs != expected_pairs:
    missing = expected_pairs - got_pairs
    extra   = got_pairs - expected_pairs
    if missing:
        print(f"  WARNING — pairs in sig CSV not included in table: {missing}")
    if extra:
        print(f"  WARNING — extra pairs in table not in sig CSV: {extra}")
else:
    print("  Validation: McNemar table covers all evaluated pairs. OK.")

# ---------------------------------------------------------------------------
# C.2 — Paired t-tests from per-seed data
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("C.2: Seed-Level Paired t-Tests")
print(SEP)
print("  Sign convention: Delta = comparator − MARBERTv2 (negative = comparator worse)")
print()

marbert_df   = seed_df[seed_df["model"] == "marbert"]
ttest_rows   = []
ttest_missing = []

for ds in DATASET_ORDER:
    for comp in COMPARATOR_ORDER:
        # Only process pairs that appear in the significance table
        sig_sub = sig_df[(sig_df["dataset"] == ds) & (sig_df["model_vs_marbert"] == comp)]
        if sig_sub.empty:
            continue

        m_sub = (marbert_df[marbert_df["dataset"] == ds]
                 .set_index("seed")["macro_f1"])
        c_sub = (seed_df[(seed_df["dataset"] == ds) & (seed_df["model"] == comp)]
                 .set_index("seed")["macro_f1"])
        common = m_sub.index.intersection(c_sub.index).sort_values()

        if len(common) < 2:
            note = f"fewer than 2 common seeds ({list(common)})"
            print(f"  MISSING: ({ds}, {DISPLAY[comp]}) — {note}")
            ttest_missing.append({
                "Dataset":    ds,
                "Comparator": DISPLAY[comp],
                "reason":     note,
            })
            continue

        m_vals     = m_sub.loc[common].values
        c_vals     = c_sub.loc[common].values
        deltas     = c_vals - m_vals          # comparator − MARBERTv2
        t_stat, p_val = stats.ttest_rel(c_vals, m_vals)
        mean_delta = float(np.mean(deltas))

        # Cross-validate against ttest_p stored in significance_tests.csv
        stored_p = float(sig_sub.iloc[0]["ttest_p"])
        if abs(p_val - stored_p) > 1e-4:
            print(f"  WARNING: ({ds}, {comp}) recomputed p={p_val:.6f}  "
                  f"stored p={stored_p:.6f}  MISMATCH — check aggregate_results.py")
        else:
            print(f"  {ds:20s}  {DISPLAY[comp]:12s}  seeds={list(common)}  "
                  f"delta={mean_delta:+.5f}  t={t_stat:.4f}  p={p_val:.6f}  "
                  f"[stored={stored_p:.6f} ✓]")

        ttest_rows.append({
            "Dataset":        ds,
            "Comparator":     DISPLAY[comp],
            "n_seeds":        int(len(common)),
            "seeds_used":     str(list(common)),
            "mean_delta_f1":  round(mean_delta, 6),
            "t_statistic":    round(float(t_stat), 6),
            "p_value":        round(float(p_val), 6),
            "interpretation": ttest_interp(mean_delta, p_val),
            "_key":           comp,
        })

print(f"\n  Total t-test rows: {len(ttest_rows)}")
print(f"  Source: {PER_SEED_CSV}")
if ttest_missing:
    print(f"  Missing pairs ({len(ttest_missing)}):")
    for m in ttest_missing:
        print(f"    {m['Dataset']} / {m['Comparator']}: {m['reason']}")

# ---------------------------------------------------------------------------
# Write CSVs
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("WRITING OUTPUT FILES")
print(SEP)

os.makedirs("outputs", exist_ok=True)

mcnemar_csv_df = pd.DataFrame(mcnemar_rows).drop(columns=["_key"])
mcnemar_csv_df.to_csv(OUT_MCNEMAR, index=False)
print(f"\n  Saved: {OUT_MCNEMAR}")

if ttest_rows:
    ttest_csv_df = pd.DataFrame(ttest_rows).drop(columns=["_key"])
    ttest_csv_df.to_csv(OUT_TTEST, index=False)
    print(f"  Saved: {OUT_TTEST}")
else:
    print(f"  SKIPPED: {OUT_TTEST} — no computable t-test rows")

# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------
FOOTER_NOTE = (
    "> **Note:** These analyses are reported as secondary or diagnostic checks only. "
    "McNemar's test examines paired correctness disagreements, while seed-level paired "
    "t-tests summarize run-level variability. "
    "No multiple-comparison correction is applied to these diagnostic p-values. "
    "The paper's primary significance claims are based on paired bootstrap testing on "
    "ensemble predictions with Holm correction."
)


def mcnemar_md_table(rows):
    H = ("| Dataset | Comparator | *b* | *c* | χ² | *p*-value "
         "| Diagnostic interpretation |")
    S = ("|:--------|:----------:|----:|----:|---:|----------:"
         "|:--------------------------|")
    lines = [H, S]
    for r in rows:
        lines.append(
            f"| {r['Dataset']} | {r['Comparator']} | {r['b']} | {r['c']} "
            f"| {r['chi2']:.4f} | {fmt_p(r['p_value'])} "
            f"| {r['interpretation']} |"
        )
    return "\n".join(lines)


def ttest_md_table(rows):
    H = ("| Dataset | Comparator | *n* | Mean Δ Macro-F1 | *t* "
         "| *p*-value | Diagnostic interpretation |")
    S = ("|:--------|:----------:|----:|---------------:|----:"
         "|----------:|:--------------------------|")
    lines = [H, S]
    for r in rows:
        lines.append(
            f"| {r['Dataset']} | {r['Comparator']} | {r['n_seeds']} "
            f"| {r['mean_delta_f1']:+.3f} | {r['t_statistic']:.4f} "
            f"| {fmt_p(r['p_value'])} | {r['interpretation']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assemble Markdown
# ---------------------------------------------------------------------------
md = []
md.append("# Appendix C. Secondary Statistical Diagnostics")
md.append("")
md.append(FOOTER_NOTE)
md.append("")
md.append("---")
md.append("")

# C.1
md.append("## C.1 McNemar's Test on Ensemble Predictions")
md.append("")
md.append(
    "McNemar's test assesses whether two classifiers differ in their per-instance "
    "correctness on the test set. Applied here to ensemble predictions (logit-averaged "
    "across all seeds, then argmax). "
    "When b + c >= 25 the chi-squared statistic with continuity correction is used; "
    "otherwise an exact binomial test is used. "
    "No multiple-comparison correction is applied to these McNemar *p*-values. "
    "These results are secondary diagnostics alongside the paper's primary paired "
    "bootstrap analysis."
)
md.append("")
md.append("**Table C1.** McNemar diagnostic results for pairwise comparisons "
          "on ensemble predictions.")
md.append("")
md.append(mcnemar_md_table(mcnemar_rows))
md.append("")
md.append(
    f"*b* = instances where MARBERTv2 predicted correctly and comparator did not; "
    f"*c* = instances where comparator predicted correctly and MARBERTv2 did not. "
    f"No Holm correction is applied to these *p*-values. "
    f"Source: `{SIG_CSV}`."
)
md.append("")
md.append("---")
md.append("")

# C.2
md.append("## C.2 Seed-Level Paired t-Test Diagnostics")
md.append("")
md.append(
    "Paired t-tests are computed across per-seed Macro-F1 scores "
    "(five seeds per model-dataset pair, seeds 42-46). "
    "Delta Macro-F1 = comparator - MARBERTv2; negative values indicate the "
    "comparator underperforms MARBERTv2. "
    "t-statistics are recomputed from per-seed data in "
    f"`{PER_SEED_CSV}` and cross-validated against stored *p*-values in `{SIG_CSV}`. "
    "No multiple-comparison correction is applied. "
    "These results are diagnostic only and are not used for the paper's primary "
    "significance claims."
)
md.append("")

if ttest_rows:
    md.append("**Table C2.** Paired t-test diagnostics across seed-level Macro-F1 scores.")
    md.append("")
    md.append(ttest_md_table(ttest_rows))
    md.append("")
    md.append(
        f"No Holm correction is applied to these *p*-values. "
        f"Source: `{PER_SEED_CSV}`."
    )
else:
    md.append("**Table C2 could not be generated** — per-seed data was insufficient "
              "for all evaluated pairs.")
    md.append("")
    md.append("Missing pairs:")
    for m in ttest_missing:
        md.append(f"- {m['Dataset']} / {m['Comparator']}: {m['reason']}")

if ttest_missing:
    md.append("")
    md.append("**Pairs excluded from Table C2 (insufficient seed data):**")
    for m in ttest_missing:
        md.append(f"- {m['Dataset']} / {m['Comparator']}: {m['reason']}")

md.append("")
md.append("---")
md.append("")
md.append(
    "*These analyses are reported as secondary or diagnostic checks only. "
    "The paper's primary significance claims are based on paired bootstrap testing on "
    "ensemble predictions with Holm correction.*"
)

md_text = "\n".join(md)
with open(OUT_MD, "w", encoding="utf-8") as fh:
    fh.write(md_text)
print(f"  Saved: {OUT_MD}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"  McNemar rows:          {len(mcnemar_rows)}")
print(f"  t-test rows:           {len(ttest_rows)}")
print(f"  t-test missing pairs:  {len(ttest_missing)}")
print(f"\n  Output files:")
print(f"    {OUT_MD}")
print(f"    {OUT_MCNEMAR}")
if ttest_rows:
    print(f"    {OUT_TTEST}")
print()
